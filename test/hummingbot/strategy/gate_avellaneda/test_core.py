import unittest
from dataclasses import replace
from decimal import Decimal as D

from hummingbot.strategy.gate_avellaneda.core import (
    Intent, Market, Portfolio, Settings, allocate, avellaneda_quotes, candidate_universe, floor_step,
)


def market(pair="AAA-USDT", **changes):
    values = dict(pair=pair, bid=D("9.99"), ask=D("10.01"), volatility=D("0.002"),
                  kappa=D("200"), alpha=D("100"), volume=D("10000000"),
                  bid_depth=D("500"), ask_depth=D("500"), funding=D("0.0001"), trend=D("0"),
                  maker_fee=D("0.0002"), step=D("0.001"), tick=D("0.001"), minimum=D("0.001"),
                  observed_at=100)
    values.update(changes)
    return Market(**values)


class SettingsTests(unittest.TestCase):
    def test_defaults_fit_100_usdt(self):
        s = Settings()
        self.assertEqual(s.pair_margin * 2 + s.reserve, s.capital)

    def test_invalid_settings(self):
        for changes in [dict(capital=D("99")), dict(capital=D("NaN")), dict(order_quote=D("0")),
                        dict(leverage=2), dict(max_position_quote=D("35")), dict(eta=D("2")),
                        dict(failures_to_retire=0), dict(cooldown=0), dict(max_age=float("nan")), dict(min_depth=D("0"))]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                Settings(**changes)


class ScreeningTests(unittest.TestCase):
    def test_healthy_market(self):
        self.assertIsNone(market().rejection(Settings(), 100))

    def test_screen_rejections(self):
        cases = [
            (dict(bid=D("NaN")), "invalid market values"),
            (dict(observed_at=0), "stale market data"),
            (dict(observed_at=101), "stale market data"),
            (dict(enabled=False), "contract unavailable"),
            (dict(ready=False), "indicators warming up"),
            (dict(kappa=D("0")), "indicators warming up"),
            (dict(ask=D("9")), "invalid trading rules or book"),
            (dict(step=D("0")), "invalid trading rules or book"),
            (dict(minimum=D("1")), "minimum order exceeds order budget"),
            (dict(step=D("0.6"), minimum=D("0.4")), "order cannot meet minimum after rounding"),
            (dict(bid=D("9.999"), ask=D("10.001")), "book spread below fee and edge threshold"),
            (dict(bid=D("9.9"), ask=D("10.1")), "book spread too wide"),
            (dict(kappa=D("0.1")), "model spread too wide"),
            (dict(tick=D("0.1")), "price tick too coarse"),
            (dict(volatility=D("0.1")), "volatility too high"),
            (dict(trend=D("0.04")), "strong directional move"),
            (dict(funding=D("-0.003")), "funding too high"),
            (dict(volume=D("100")), "volume too low"),
            (dict(ask_depth=D("1")), "insufficient nearby depth"),
        ]
        for changes, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(market(**changes).rejection(Settings(), 100), expected)


class PortfolioTests(unittest.TestCase):
    def setUp(self):
        self.portfolio = Portfolio(Settings())
        self.markets = {p: market(p, volume=D(v)) for p, v in
                        [("A-USDT", "3000000"), ("B-USDT", "2000000"), ("C-USDT", "1000000")]}
        self.evaluate(100)

    def evaluate(self, now, fresh=True, positions=None, orders=None):
        self.markets = {p: replace(m, observed_at=now) for p, m in self.markets.items()}
        self.portfolio.evaluate(self.markets, now, fresh, positions or {}, orders or set())

    def test_select_two_and_keep_when_new_candidate_scores_higher(self):
        self.assertEqual(set(self.portfolio.slots), {"A-USDT", "B-USDT"})
        self.markets["C-USDT"] = market("C-USDT", volume=D("999999999"))
        self.evaluate(160)
        self.assertEqual(set(self.portfolio.slots), {"A-USDT", "B-USDT"})

    def test_failures_require_confirmation_and_reset(self):
        self.markets["A-USDT"] = replace(self.markets["A-USDT"], bid_depth=D("1"))
        self.evaluate(160)
        self.assertEqual(self.portfolio.slots["A-USDT"].failures, 1)
        self.markets["A-USDT"] = replace(self.markets["A-USDT"], bid_depth=D("500"))
        self.evaluate(220)
        self.assertEqual(self.portfolio.slots["A-USDT"].failures, 0)

    def test_retiring_slot_cannot_be_replaced_until_flat_and_order_free(self):
        self.markets["A-USDT"] = replace(self.markets["A-USDT"], volume=D("10"))
        for now in (160, 220, 280):
            self.evaluate(now, positions={"A-USDT": D("1")})
        self.assertEqual(self.portfolio.slots["A-USDT"].state, "retiring")
        self.evaluate(340, positions={"A-USDT": D("1")})
        self.assertNotIn("C-USDT", self.portfolio.slots)
        self.evaluate(400, orders={"A-USDT"})
        self.assertNotIn("C-USDT", self.portfolio.slots)
        self.evaluate(460)
        self.assertEqual(set(self.portfolio.slots), {"B-USDT", "C-USDT"})
        self.assertEqual(self.portfolio.cooldowns["A-USDT"], 1060)

    def test_stale_account_never_releases_retiring_slot(self):
        self.portfolio.retire("A-USDT", "test", 100)
        self.evaluate(160, fresh=False)
        self.assertIn("A-USDT", self.portfolio.slots)

    def test_data_loss_pauses_instead_of_churning(self):
        del self.markets["A-USDT"]
        for now in (160, 220, 280, 340):
            self.evaluate(now)
        self.assertEqual(self.portfolio.slots["A-USDT"].state, "active")

    def test_no_candidate_leaves_slot_empty_and_cooldown_blocks_reentry(self):
        self.portfolio.retire("A-USDT", "test", 100)
        self.markets["C-USDT"] = replace(self.markets["C-USDT"], volume=D("0"))
        self.evaluate(160)
        self.assertEqual(set(self.portfolio.slots), {"B-USDT"})
        self.evaluate(220)
        self.assertNotIn("A-USDT", self.portfolio.slots)
        self.evaluate(800)
        self.assertIn("A-USDT", self.portfolio.slots)

    def test_retirement_does_not_reset_exit_deadline(self):
        self.portfolio.retire("A-USDT", "one", 100)
        self.portfolio.retire("A-USDT", "two", 150)
        self.assertEqual(self.portfolio.slots["A-USDT"].retiring_since, 100)

    def test_disabled_entries_allow_cleanup_without_refilling(self):
        self.portfolio.retire("A-USDT", "reserve", 100)
        self.portfolio.evaluate(self.markets, 100, True, {}, set(), allow_entries=False)
        self.assertEqual(set(self.portfolio.slots), {"B-USDT"})


class QuoteTests(unittest.TestCase):
    def test_passive_prices_and_step_sizes(self):
        m = market()
        quotes = avellaneda_quotes(m, D("0"), Settings())
        self.assertEqual(len(quotes), 2)
        self.assertLessEqual(quotes[0].price, m.bid)
        self.assertGreaterEqual(quotes[1].price, m.ask)
        for q in quotes:
            self.assertFalse(q.close)
            self.assertEqual(q.amount % m.step, 0)
            self.assertEqual(q.price % m.tick, 0)
            self.assertLessEqual(q.notional, Settings().order_quote)

    def test_long_reduces_buy_size_and_sell_is_close(self):
        flat = avellaneda_quotes(market(), D("0"), Settings())
        long = avellaneda_quotes(market(), D("1"), Settings())
        self.assertLess(long[0].amount, flat[0].amount)
        self.assertFalse(long[0].close)
        self.assertTrue(long[1].close)

    def test_short_reduces_sell_size_and_buy_is_close(self):
        flat = avellaneda_quotes(market(), D("0"), Settings())
        short = avellaneda_quotes(market(), D("-1"), Settings())
        self.assertTrue(short[0].close)
        self.assertFalse(short[1].close)
        self.assertLess(short[1].amount, flat[1].amount)

    def test_close_does_not_flip_small_position(self):
        for pos in (D("0.01"), D("-0.01")):
            closes = [q for q in avellaneda_quotes(market(), pos, Settings()) if q.close]
            self.assertEqual(closes[0].amount, abs(pos))

    def test_cap_prevents_increasing_side(self):
        for pos in (D("2"), D("-2"), D("3")):
            quotes = avellaneda_quotes(market(), pos, Settings())
            self.assertTrue(all(q.close for q in quotes))

    def test_invalid_raw_bid_is_not_submitted(self):
        quotes = avellaneda_quotes(market(kappa=D("0.000000001")), D("0"), Settings())
        self.assertTrue(all(not q.buy for q in quotes))

    def test_round_down(self):
        self.assertEqual(floor_step(D("0.1234"), D("0.01")), D("0.12"))


class BudgetTests(unittest.TestCase):
    def intent(self, pair, notional, close=False):
        return Intent(pair, True, D(notional), D("1"), close)

    def test_shared_batch_cannot_reuse_collateral(self):
        intents = [self.intent("A", "30"), self.intent("B", "30")]
        selected = allocate(intents, {}, {}, D("60"), D("100"), Settings())
        self.assertEqual(len(selected), 1)

    def test_existing_positions_and_unacked_orders_count_against_budget(self):
        intents = [self.intent("A", "10"), self.intent("B", "10")]
        selected = allocate(intents, {"A": D("20")}, {"A": D("15")}, D("100"), D("100"), Settings())
        self.assertEqual([q.pair for q in selected], ["B"])

    def test_cap_applies_even_if_account_has_more_than_100(self):
        intents = [self.intent("A", "30"), self.intent("B", "30")]
        selected = allocate(intents, {"OLD": D("30")}, {}, D("1000"), D("1000"), Settings())
        self.assertEqual(len(selected), 1)

    def test_loss_and_reserve_reduce_opening_budget(self):
        self.assertEqual(allocate([self.intent("A", "5")], {}, {}, D("19"), D("19"), Settings()), [])

    def test_close_allowed_without_opening_collateral_and_prioritized(self):
        close = self.intent("A", "10", close=True)
        self.assertEqual(allocate([self.intent("B", "5"), close], {}, {}, D("0"), D("0"), Settings()), [close])


class UniverseTests(unittest.TestCase):
    def test_contract_multiplier_minimum_volume_and_allowlist(self):
        contracts = [dict(name=name, in_delisting=delisting, quanto_multiplier=mult, order_size_min=minimum)
                     for name, delisting, mult, minimum in
                     [("A_USDT", False, "0.1", "1"), ("B_USDT", False, "1", "2"),
                      ("C_USDT", True, "0.1", "1"), ("D_USDT", False, "0.1", "1"),
                      ("E_USD", False, "0.1", "1"), ("F_USDT", False, "NaN", "1"),
                      ("G_USDT", False, "0.1", "1"), ("MISSING_USDT", False, "0.1", "1")]]
        tickers = [dict(contract=name, last="10", volume_24h_quote=volume) for name, volume in
                   [("A_USDT", "2000000"), ("B_USDT", "9000000"), ("C_USDT", "9000000"),
                    ("D_USDT", "3000000"), ("F_USDT", "9000000"), ("G_USDT", "10")]]
        self.assertEqual(list(candidate_universe(contracts, tickers, Settings(), 2)), ["D-USDT", "A-USDT"])
        self.assertEqual(list(candidate_universe(contracts, tickers, Settings(), 2, ["A-USDT"])), ["A-USDT"])
        self.assertEqual(list(candidate_universe(contracts, tickers, Settings(), 1)), ["D-USDT"])


if __name__ == "__main__":
    unittest.main()
