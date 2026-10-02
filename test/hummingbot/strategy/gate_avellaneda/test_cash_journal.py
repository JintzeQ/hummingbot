import tempfile
import unittest
from decimal import Decimal as D
from pathlib import Path

from hummingbot.strategy.gate_avellaneda.cash_journal import CashJournal


def cash(rid, amount='-0.001', at=101, kind='fee', trade_id=''):
    return dict(id=rid, contract='A_USDT', type=kind, change=amount, time=at, trade_id=trade_id)


class CashJournalTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'cash.sqlite'
        self.journal = CashJournal(self.path, 'owner', 100)
        self.addCleanup(lambda: self.journal.close())

    def test_ten_thousand_is_checkpoint_not_terminal_limit(self):
        for page in range(10):
            interval = self.journal.page_range(102)
            self.assertEqual(interval[2], page * 1000)
            self.journal.ingest([cash(page * 1000 + i) for i in range(1000)], interval)
        self.journal.close()
        self.journal = CashJournal(self.path, 'owner', 100)
        self.assertEqual(self.journal.page_range(102)[2], 10000)
        self.assertTrue(self.journal.ingest([cash(10000)], self.journal.page_range(102)))
        self.assertEqual(self.journal.snapshot().cash_totals['fee'], D('-10.001'))
        self.assertEqual(self.journal.snapshot().records, {})

    def test_overlap_deduplicates_corrections_and_delayed_cash(self):
        self.journal.ingest([cash(1)], self.journal.page_range(102))
        self.journal.ingest([cash(1, '-0.002'), cash(2, '-0.004')], self.journal.page_range(104))
        self.assertEqual(self.journal.snapshot().cash_totals['fee'], D('-0.006'))
        self.assertEqual(self.journal.snapshot(['1']).cash_totals['fee'], D('-0.004'))
        self.assertEqual(set(self.journal.snapshot(bootstrap=True).records), {'1','2'})

    def test_historical_rescan_keeps_totals_and_durable_progress(self):
        self.journal.ingest([cash(1)], self.journal.page_range(500))
        self.journal.request_rescan()
        self.assertEqual(self.journal.page_range(500), [100, 500, 0])
        self.journal.ingest([cash(1)], self.journal.page_range(500), page_size=1)
        self.journal.request_rescan()
        self.assertEqual(self.journal.page_range(500)[2], 1)
        self.assertEqual(self.journal.snapshot().cash_totals['fee'], D('-0.001'))

    def test_delayed_trade_fee_owned_by_prior_cycle_after_reselection(self):
        self.journal.begin_cycle('A-USDT', 100)
        self.journal.own_trade('A-USDT', 'old', 100)
        self.journal.end_cycle('A-USDT', 100, 110)
        self.journal.begin_cycle('A-USDT', 200)
        self.journal.ingest([cash(1, '-4.999'), cash(2, '-0.002', at=201, trade_id='old'),
                             cash(3, '1', at=201, kind='pnl', trade_id='new')], self.journal.page_range(202))
        self.assertEqual(self.journal.pair_total('A-USDT', 100), D('-5.001'))
        self.assertEqual(self.journal.pair_total('A-USDT', 200), D(1))
        self.assertEqual(self.journal.closed_cycles(), [('A-USDT', 100, 110)])
        with self.assertRaises(ValueError):
            self.journal.own_trade('A-USDT', 'old', 200)

    def test_duplicate_or_invalid_page_rolls_back_totals_and_cursor(self):
        for rows in [[cash(1),cash(1)], [cash(1, 'NaN')], [cash(1, at=999)], [cash(None)]]:
            with self.subTest(rows=rows), self.assertRaises((RuntimeError, ValueError)):
                self.journal.ingest(rows, self.journal.page_range(102))
            self.assertEqual(self.journal.snapshot().cash_totals['fee'], 0)
            self.assertEqual(self.journal.page_range(102)[2], 0)

    def test_identity_and_session_cannot_change(self):
        for user, at in [('different', 100), ('owner', 101)]:
            with self.assertRaises(ValueError):
                CashJournal(self.path, user, at)

    def test_unknown_currency_flow_is_visible_to_account_risk(self):
        self.journal.ingest([cash(1, kind='point_fee')], self.journal.page_range(102))
        self.assertEqual(self.journal.snapshot().records['1']['kind'], 'point_fee')
