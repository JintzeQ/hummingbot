import asyncio
import json
import math
import tempfile
import unittest
from dataclasses import replace
from decimal import Decimal as D
from pathlib import Path
from unittest.mock import AsyncMock

from hummingbot.strategy.gate_avellaneda.adaptive import AdaptiveSettings, entry_rejection, quote_plan
from hummingbot.strategy.gate_avellaneda.core import Settings
from hummingbot.strategy.gate_avellaneda.kline_volatility import (
    KlineObservation, KlineSettings, KlineVolatility, calibrated_half_spread, candle_estimate,
)
from hummingbot.strategy.gate_avellaneda.verification import QualityRecorder, TelemetrySettings, audit_report, replay_decision
from test.hummingbot.strategy.gate_avellaneda.test_core import market
from test.hummingbot.strategy.gate_avellaneda import test_adaptive_adapter as fixtures


def candles(now=90000, count=61, interval=60, amplitude=.001, scale=10):
    end = int(now // interval) * interval
    values = []
    for i in range(count):
        price = scale * math.exp(amplitude * (i % 2))
        values.append(dict(t=end - (count - i) * interval, o=str(price), h=str(price), l=str(price), c=str(price)))
    return values


def calibration(**changes):
    row = dict(enabled=True, mode='protect', ready=True, suggested_half_spread=D('.01'),
               weight=D('.25'), max_multiplier=D('1.5'), max_extra_bps=D(5))
    return dict(row, **changes)


class CandleTests(unittest.TestCase):
    def setUp(self):
        self.cfg = KlineSettings(min_returns=30, lookback_bars=60)
        self.model = KlineVolatility(self.cfg)

    def test_defaults_observe_and_configuration_bounds(self):
        self.assertEqual(KlineSettings().mode, 'observe')
        for changes in [dict(lookback_bars=120), dict(interval='spot'), dict(interval='1h'),
                        dict(mode='auto'), dict(tail_probability='NaN'), dict(weight='.51'),
                        dict(max_multiplier=3), dict(quote_horizon_seconds=0), dict(refresh_seconds=301),
                        dict(max_age_seconds=119), dict(fill_probability='.25')]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                KlineSettings(**changes)

    def test_closed_sorted_object_bars_and_unfinished_exclusion(self):
        rows = candles()
        expected = candle_estimate(rows, 90000, self.cfg)
        unfinished = dict(t=90000, o='NaN', h='NaN', l='NaN', c='NaN')
        self.assertEqual(candle_estimate(list(reversed(rows)) + [unfinished], 90020, self.cfg), expected)
        self.assertEqual(expected[1], 59)

    def test_rejects_spot_arrays_and_invalid_prices(self):
        bad = [[], [[90000, '10']], 'oops', candles(count=20), candles() * 40]
        for key, value in [('c','0'), ('c','NaN'), ('h','1'), ('l','100'), ('o','-1')]:
            rows = candles(); rows[-1][key] = value; bad.append(rows)
        for rows in bad:
            with self.subTest(rows=str(rows)[:60]):
                self.assertFalse(self.model.update('A-USDT', rows, 90000).ready)

    def test_rejects_duplicates_gaps_future_and_misaligned(self):
        rows = candles()
        gap = rows.copy(); gap.pop(-3)
        future = rows + [dict(rows[-1], t=90060)]
        misaligned = [dict(row, t=row['t'] + 1) for row in rows]
        for data in [rows + [rows[-1]], gap, future, misaligned]:
            self.assertFalse(self.model.update('A-USDT', data, 90000).ready)

    def test_daily_horizon_units_and_price_scale_invariance(self):
        first = self.model.update('A-USDT', candles(), 90000)
        scaled = self.model.update('B-USDT', candles(scale=1000), 90000)
        self.assertAlmostEqual(float(first.daily_volatility), float(scaled.daily_volatility), places=12)
        self.assertAlmostEqual(float(first.horizon_volatility / first.daily_volatility), math.sqrt(15 / 86400))
        self.assertAlmostEqual(float(first.daily_volatility), .001 * math.sqrt(60 / 59) * math.sqrt(1440), places=10)
        other = KlineVolatility(KlineSettings(interval='5m', min_returns=30, lookback_bars=60,
                                             max_age_seconds=600))
        second = other.update('A-USDT', candles(interval=300, amplitude=.001 * math.sqrt(5)), 90000)
        self.assertAlmostEqual(float(first.daily_volatility), float(second.daily_volatility), places=10)
        self.assertGreater(first.suggested_half_spread, first.horizon_volatility)

    def test_zero_volatility_is_valid_and_can_recover(self):
        first = self.model.update('A-USDT', candles(amplitude=0), 90000)
        self.assertTrue(first.ready)
        self.assertEqual(first.daily_volatility, 0)
        self.assertGreater(self.model.update('A-USDT', candles(now=90060), 90060).daily_volatility, 0)

    def test_smoothing_limits_jump_and_same_bar_cannot_accelerate(self):
        first = self.model.update('A-USDT', candles(), 90000)
        second = self.model.update('A-USDT', candles(now=90060, amplitude=.02), 90060)
        self.assertLessEqual(second.daily_volatility, first.daily_volatility * D('1.1'))
        self.assertGreater(second.raw_daily_volatility, second.daily_volatility)
        for now in (90061, 90062, 90070):
            value = self.model.update('A-USDT', candles(now=90060, amplitude=.02), now)
            self.assertEqual(value.daily_volatility, second.daily_volatility)

    def test_stale_gap_reseeds_and_backward_clock_is_rejected(self):
        self.model.update('A-USDT', candles(), 90000)
        second = self.model.update('A-USDT', candles(now=90420, amplitude=.02), 90420)
        self.assertEqual(second.daily_volatility, second.raw_daily_volatility)
        self.assertFalse(self.model.update('A-USDT', candles(), 90000).ready)
        self.assertFalse(self.model.update('A-USDT', candles(now=90060), 90430).ready)

    def test_fetch_freshness_does_not_refresh_old_closed_bar(self):
        self.model.update('A-USDT', candles(), 90000)
        self.assertTrue(self.model.snapshot('A-USDT', 90180)['ready'])
        self.assertFalse(self.model.snapshot('A-USDT', 90181)['ready'])
        self.assertFalse(self.model.update('A-USDT', candles(), 90181).ready)
        self.assertFalse(self.model.snapshot('A-USDT', 89999)['ready'])

    def test_failure_invalidates_and_restart_reacquires(self):
        self.model.update('A-USDT', candles(), 90000)
        self.model.fail('A-USDT', 90010, 'timeout')
        self.assertFalse(self.model.snapshot('A-USDT', 90010)['ready'])
        self.assertEqual(self.model.snapshot('A-USDT', 90010)['reason'], 'timeout')
        self.assertFalse(KlineVolatility(self.cfg).snapshot('A-USDT', 90010)['ready'])
        self.cfg.enabled = False
        self.assertIsNone(self.model.snapshot('A-USDT', 90010))


class CalibrationQuoteTests(unittest.TestCase):
    def test_blend_caps_and_never_narrows(self):
        base = D('.001')
        self.assertEqual(calibrated_half_spread(base, calibration()), D('.0015'))
        self.assertEqual(calibrated_half_spread(base, calibration(max_extra_bps=D(1))), D('.0011'))
        for changes in [dict(weight=0), dict(suggested_half_spread=0), dict(mode='observe'), dict(ready=False), dict(enabled=False)]:
            self.assertEqual(calibrated_half_spread(base, calibration(**changes)), base)
        self.assertEqual(calibrated_half_spread(base, calibration(suggested_half_spread=D('.002'))), D('.00125'))

    def test_invalid_calibration_rejected(self):
        for changes in [dict(weight='.8'), dict(max_extra_bps='NaN'), dict(max_multiplier='.5'), dict(suggested_half_spread=-1)]:
            with self.assertRaises(ValueError):
                calibrated_half_spread(D('.001'), calibration(**changes))

    def test_observe_identical_and_protect_changes_opening_only(self):
        args = (market(), D('.5'), 0, Settings(), AdaptiveSettings(), 100)
        normal = quote_plan(*args)
        observe = quote_plan(*args, calibration=calibration(mode='observe'))
        protect = quote_plan(*args, calibration=calibration())
        self.assertEqual(normal, observe)
        self.assertEqual([q for q in normal.intents if q.close], [q for q in protect.intents if q.close])
        self.assertEqual(normal.gamma, protect.gamma)
        self.assertEqual(normal.position_cap, protect.position_cap)
        self.assertEqual(normal.order_quote, protect.order_quote)
        for before, after in zip([q for q in normal.intents if not q.close], [q for q in protect.intents if not q.close]):
            self.assertLessEqual(after.price, before.price) if before.buy else self.assertGreaterEqual(after.price, before.price)
        self.assertTrue(all(q.price < market().ask if q.buy else q.price > market().bid for q in protect.intents))

    def test_unavailable_blocks_opening_preserves_close_both_directions(self):
        for position in (D('.5'), D('-.5')):
            args = (market(), position, 0, Settings(), AdaptiveSettings(), 100)
            normal = quote_plan(*args)
            controlled = quote_plan(*args, calibration=calibration(ready=False))
            self.assertEqual(controlled.entry_reason, 'K-line calibration unavailable')
            self.assertEqual([q for q in normal.intents if q.close], controlled.intents)
        self.assertEqual(entry_rejection(market(), Settings(), AdaptiveSettings(), 100,
                                        calibration=calibration(ready=False)), 'K-line calibration unavailable')

    def test_actual_shadow_and_legacy_records_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = QualityRecorder(TelemetrySettings(path=directory+'/q.jsonl'), True)
            args = (market(), D(0), 0, Settings(), AdaptiveSettings(), 100)
            observed = calibration(mode='observe')
            normal = quote_plan(*args, calibration=observed)
            shadow = quote_plan(*args, calibration=dict(observed, mode='protect'))
            recorder.decision(100, *args[:5], None, True, None, D(1), normal, calibration=observed, shadow=shadow)
            row = json.loads(recorder.path.read_text())
            self.assertTrue(replay_decision(row))
            self.assertEqual(audit_report([row])['kline_calibration']['shadow_checked'], 1)
            row['shadow']['gamma'] = '2'
            self.assertFalse(replay_decision(row))
            row.pop('shadow'); row.pop('calibration')
            self.assertTrue(replay_decision(row))


class KlineAdapterTests(unittest.IsolatedAsyncioTestCase):
    frame = fixtures.AdaptiveAdapterTests.frame
    ready = fixtures.AdaptiveAdapterTests.ready
    positions = fixtures.AdaptiveAdapterTests.positions
    clear_orders = fixtures.AdaptiveAdapterTests.clear_orders

    def setUp(self):
        fixtures.AdaptiveAdapterTests.setUp(self)
        self.temp = tempfile.TemporaryDirectory()
        self.config.telemetry.enabled = True
        self.bot.recorder.path = Path(self.temp.name)/'audit.jsonl'
        self.config.kline_volatility = KlineSettings(mode='protect')
        self.bot.kline_volatility = KlineVolatility(self.config.kline_volatility)

    async def asyncTearDown(self):
        await self.bot.on_stop()
        self.temp.cleanup()

    def seed(self, pair='A-USDT', spread=D('.01')):
        self.bot.kline_volatility.observations[pair] = KlineObservation(
            pair, True, '', 120, self.bot.current_timestamp, self.bot.current_timestamp,
            D('.1'), D('.1'), D('.001'), spread)

    async def test_selection_and_scoring_use_same_calibration(self):
        await self.ready()
        m = self.bot._snapshots()['A-USDT']
        self.assertEqual(self.bot._entry_check(m), 'K-line calibration unavailable')
        self.config.kline_volatility.mode = 'observe'
        baseline = self.bot._candidate_score(m)
        self.seed()
        self.assertIsNone(self.bot._entry_check(m))
        self.assertEqual(self.bot._candidate_score(m), baseline)
        self.config.kline_volatility.mode = 'protect'
        self.assertGreaterEqual(self.bot._candidate_score(m), baseline)
        # Existing quote-width constraints still reject a widened proposal.
        self.config.risk = replace(self.config.risk, max_quote_spread=D('.0015'))
        self.assertIsNotNone(self.bot._entry_check(m))

    def test_runtime_config_requires_pricing_and_audit(self):
        adapter = fixtures.adapter
        for kwargs in [dict(adaptive=dict(enabled=False), quality_control=dict(enabled=False)),
                       dict(telemetry=dict(enabled=False), quality_control=dict(enabled=False))]:
            with self.assertRaises(ValueError):
                adapter.GateAvellanedaPortfolioConfig(**kwargs)
        self.assertEqual(adapter.GateAvellanedaPortfolioConfig().kline_volatility.mode, 'observe')

    async def test_public_futures_requests_and_diagnostic_record(self):
        self.bot.current_timestamp = 90000
        self.connector._api_get = AsyncMock(return_value=candles(count=130))
        await self.bot._refresh_klines()
        self.assertEqual(self.connector._api_get.await_count, 3)
        for call in self.connector._api_get.await_args_list:
            self.assertEqual(call.kwargs['path_url'], 'futures/usdt/candlesticks')
            self.assertFalse(call.kwargs['is_auth_required'])
            self.assertEqual(call.kwargs['limit_id'], 'futures/usdt/candlesticks')
            self.assertEqual(set(call.kwargs['params']), {'contract','interval','limit','timezone'})
            self.assertIn(call.kwargs['params']['contract'], ['A_USDT','B_USDT','C_USDT'])
            self.assertEqual(call.kwargs['params']['limit'], 721)
        self.assertTrue(self.bot._calibration_snapshot('A-USDT')['ready'])
        self.assertEqual(len(self.bot.recorder.path.read_text().splitlines()), 3)

    async def test_request_failure_and_timeout_are_local(self):
        self.bot.current_timestamp = 90000
        self.seed()
        self.connector._api_get = AsyncMock(side_effect=IOError('public outage'))
        await self.bot._refresh_klines()
        self.assertFalse(self.bot._calibration_snapshot('A-USDT')['ready'])
        self.assertFalse(self.bot.halt_reason)
        self.config.kline_volatility.request_timeout_seconds = .01
        async def slow(**kwargs):
            await asyncio.Event().wait()
        self.connector._api_get = slow
        await self.bot._refresh_klines()
        self.assertFalse(self.bot._calibration_snapshot('A-USDT')['ready'])

    async def test_bounded_concurrency_and_stop_cancels_public_tasks(self):
        self.bot.contracts = {f'X{i}-USDT': {} for i in range(10)}
        active = 0
        peak = 0
        entered = asyncio.Event()
        async def blocked(**kwargs):
            nonlocal active, peak
            active += 1; peak = max(peak, active)
            if active == 4:
                entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                active -= 1
        self.connector._api_get = blocked
        self.bot._kline_tick(100)
        await asyncio.wait_for(entered.wait(), 1)
        self.assertEqual(peak, 4)
        self.bot._kline_tick(101)
        await self.bot.on_stop()
        self.assertTrue(self.bot.kline_task.cancelled())
        self.assertEqual(active, 0)

    async def test_private_refresh_and_close_continue_during_public_wait(self):
        self.config.kline_volatility.enabled = False
        await self.ready()
        await self.positions({'A-USDT': D('.5')})
        self.config.kline_volatility.enabled = True
        get = self.connector._api_get
        started = asyncio.Event()
        async def api(**kwargs):
            if kwargs['path_url'] == 'futures/usdt/candlesticks':
                started.set()
                await asyncio.Event().wait()
            return await get(**kwargs)
        self.connector._api_get = api
        self.bot._kline_tick(101)
        await asyncio.wait_for(started.wait(), 1)
        await asyncio.wait_for(self.bot._refresh(), 1)
        self.bot.on_tick()
        self.assertTrue(any(o[2] == 'A-USDT' and o[5]['position_action'] == fixtures.PositionAction.CLOSE for o in self.connector.sent))
        self.assertFalse(self.bot.halt_reason)

    async def test_observe_shadow_and_protect_reprice_cancel_only_openings(self):
        self.config.kline_volatility.mode = 'observe'
        await self.ready()
        self.seed(); self.seed('B-USDT')
        self.bot.on_tick()
        self.assertTrue(self.connector.sent)
        records = [json.loads(s) for s in self.bot.recorder.path.read_text().splitlines()]
        decisions = [r for r in records if r['kind'] == 'decision']
        self.assertTrue(any(r['shadow'] is not None for r in decisions))
        self.assertTrue(all(replay_decision(r) for r in decisions))
        submitted = [r for r in records if r['kind'] == 'submitted']
        self.assertTrue(all(r['calibration']['mode'] == 'observe' for r in submitted))
        self.config.kline_volatility.mode = 'protect'
        self.frame(102)
        self.bot.on_tick()
        self.assertTrue(self.connector.cancels)
        self.assertTrue(all(not self.bot.orders[oid].intent.close for oid in self.connector.cancels))
        self.assertIn('K-line=protect', self.bot.format_status())

    async def test_protect_outage_does_not_retire_pair_or_overwrite_five_u(self):
        self.config.kline_volatility.enabled = False
        await self.ready()
        self.config.kline_volatility.enabled = True
        for now in (102,103,104,105):
            self.frame(now)
            self.bot.portfolio.evaluate(self.bot._snapshots(), now, True, {}, set(), entry_check=self.bot._entry_check)
        self.assertEqual(self.bot.portfolio.slots['A-USDT'].state, 'active')
        self.assertEqual(self.config.risk.pair_loss_limit, D(5))
        self.assertTrue(self.bot.portfolio.stop_loss('A-USDT', D(-5), 105))
        self.assertTrue(self.bot.portfolio.slots['A-USDT'].force_market)
