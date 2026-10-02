import asyncio
import tempfile
import types
import unittest
from dataclasses import replace
from decimal import Decimal as D
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from hummingbot.strategy.gate_avellaneda.adaptive import AdaptiveSettings, quote_plan
from hummingbot.strategy.gate_avellaneda.core import Intent, Settings
from hummingbot.strategy.gate_avellaneda.execution import DeadmanSettings, ExecutionSettings
from hummingbot.strategy.gate_avellaneda.quality_control import QualityControl, QualitySettings
from hummingbot.strategy.gate_avellaneda.verification import QualityRecorder, TelemetrySettings, replay_decision
from test.hummingbot.strategy.gate_avellaneda.support import (
    FakeConnector, OrderType, PositionAction, TradeType, contract, load_adapter, method_from_source, constants,
)
from test.hummingbot.strategy.gate_avellaneda.test_core import market

adapter = load_adapter()


def observation(model, oid, fid=None, net='-0.001', move='-0.002', **changes):
    fill = dict(pair='A-USDT', buy=True, regime='calm', close=False, market=False, anchor_mid=D(10),
                fee_estimate_usdt=D('.001'), order_id=str(oid), trade_id=str(fid if fid is not None else oid),
                amount=D('.5'), price=D(10), filled_at=100)
    fill.update(changes)
    model.observe(fill, dict(horizon=5, observed=True, fee_adjusted_estimate_usdt=D(net), post_fill_move_usdt=D(move)))


class QualityTests(unittest.TestCase):
    def setUp(self):
        self.model = QualityControl(QualitySettings(min_orders=10))

    def test_partial_fills_and_duplicates_are_one_order_sample(self):
        for i in range(40):
            observation(self.model, 'one', str(i))
            observation(self.model, 'one', str(i))
        row = self.model.estimate('A-USDT', True, 'calm', 110)
        self.assertEqual(row['orders'], 1)
        self.assertFalse(row['ready'])
        self.assertEqual(len(self.model.rows[('A-USDT', True, 'calm')]['one']['fills']), 40)

    def test_negative_evidence_blocks_only_qualified_side_and_regime(self):
        for i in range(10):
            observation(self.model, i)
        result = self.model.snapshot('A-USDT', 'calm', 110)
        self.assertTrue(result['True']['block'])
        self.assertEqual(D(result['True']['penalty_bps']), 4)
        self.assertFalse(result['False']['ready'])
        self.assertFalse(self.model.snapshot('A-USDT', 'stressed', 110)['True']['ready'])
        self.assertFalse(self.model.snapshot('B-USDT', 'calm', 110)['True']['ready'])

    def test_uncertain_evidence_shrinks_and_expired_samples_do_not_control(self):
        for i in range(10):
            observation(self.model, i, net='.002' if i % 2 else '-.001')
        result = self.model.snapshot('A-USDT', 'calm', 110)['True']
        self.assertTrue(result['ready'])
        self.assertFalse(result['block'])
        self.assertEqual(D(result['size_scale']), D('.5'))
        self.assertFalse(self.model.snapshot('A-USDT', 'calm', 4000)['True']['ready'])
        self.assertEqual(self.model.dump(4000), [])

    def test_unknown_fees_closes_and_taker_fills_do_not_train_maker_model(self):
        for i, changes in enumerate([dict(close=True), dict(market=True), dict(anchor_mid=None), dict(fee_estimate_usdt=None)]):
            observation(self.model, i, **changes)
        self.assertEqual(self.model.rows, {})

    def test_persistence_and_observe_ablation(self):
        for i in range(10):
            observation(self.model, i)
        other = QualityControl(QualitySettings(min_orders=10))
        other.restore(self.model.dump(110))
        self.assertEqual(other.snapshot('A-USDT', 'calm', 110), self.model.snapshot('A-USDT', 'calm', 110))
        other.settings.mode = 'observe'
        self.assertEqual(other.snapshot('A-USDT', 'calm', 110), {})
        self.assertTrue(other.estimate('A-USDT', True, 'calm', 110)['block'])

    def test_quality_never_blocks_or_widens_reduction(self):
        risk, adaptive = Settings(), AdaptiveSettings()
        normal = quote_plan(market(), D('.5'), 0, risk, adaptive, 100)
        controlled = quote_plan(market(), D('.5'), 0, risk, adaptive, 100,
                                quality={'True':dict(block=True, penalty_bps='20', size_scale='.5'),
                                         'False':dict(block=True, penalty_bps='20', size_scale='.5')}, exit_cost_bps=D(10))
        self.assertTrue(controlled.intents)
        self.assertTrue(all(q.close for q in controlled.intents))
        self.assertEqual([q for q in normal.intents if q.close], controlled.intents)

    def test_new_quality_decision_replays_exactly(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            recorder = QualityRecorder(TelemetrySettings(path=directory+'/quality.jsonl'), False)
            quality = {'True':dict(penalty_bps='2', size_scale='.5', block=False)}
            plan = quote_plan(market(), D(0), 0, Settings(), AdaptiveSettings(), 100, quality=quality, exit_cost_bps=D('.5'))
            recorder.decision(100, market(), D(0), 0, Settings(), AdaptiveSettings(), None, True, None, D(1), plan,
                              quality=quality, exit_cost_bps=D('.5'))
            row = json.loads(Path(recorder.path).read_text())
            self.assertTrue(replay_decision(row))

    def test_invalid_new_settings_rejected(self):
        for cls, values in [(QualitySettings,dict(confidence_z=float('inf'))),
                            (QualitySettings,dict(uncertain_size_scale=2)),
                            (ExecutionSettings,dict(max_quote_age_seconds=float('nan'))),
                            (DeadmanSettings,dict(renew_seconds=11))]:
            with self.assertRaises(ValueError):
                cls(**values)


class ExecutionAdapterTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        pairs = ['A-USDT','B-USDT']
        adapter.GateAvellanedaPortfolio._initial_contracts = {p:contract(p) for p in pairs}
        self.connector = FakeConnector(pairs)
        adapter.GateAvellanedaPortfolio._initial_tickers = {t['contract'].replace('_','-'):t for t in self.connector.tickers}
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.config = adapter.GateAvellanedaPortfolioConfig(kline_volatility=dict(enabled=False), dry_run=False, account_risk=dict(enabled=False),
            recovery=dict(enabled=False), telemetry=dict(path=self.directory.name+"/quality.jsonl"), micro=dict(enabled=False),
            adaptive=dict(require_correlation=False))
        self.bot = adapter.GateAvellanedaPortfolio({'gate_io_perpetual':self.connector}, self.config)
        self.addAsyncCleanup(self.bot.on_stop)
        self.connector._api_post = AsyncMock(return_value={'triggerTime':'130000'})

    async def ready(self):
        await self.bot._refresh()
        self.bot._sample()
        self.bot.portfolio.evaluate(self.bot._snapshots(), 100, True, {}, set(), entry_check=self.bot._entry_check)
        self.bot.next_account = self.bot.next_monitor = 999

    async def test_open_requires_acknowledged_contract_countdown(self):
        await self.ready()
        intent = Intent('A-USDT', True, D('.5'), D(10))
        self.bot._submit(intent)
        self.assertEqual(self.connector.sent, [])
        await self.bot._renew_heartbeat({'A-USDT'})
        self.bot._submit(intent)
        self.assertEqual(len(self.connector.sent), 1)
        self.assertFalse(self.bot._heartbeat_ready('B-USDT'))
        args = self.connector._api_post.call_args.kwargs
        self.assertEqual(args['data'], dict(timeout=30, contract='A_USDT'))
        self.assertTrue(args['is_auth_required'])

    async def test_bad_ack_and_request_failure_block_open_keep_market_close(self):
        await self.ready()
        for response in [{'triggerTime':'NaN'}, {'triggerTime':'101000'}, {'triggerTime':'999999999'}]:
            self.connector._api_post.return_value = response
            await self.bot._renew_heartbeat({'A-USDT'})
            self.assertFalse(self.bot._heartbeat_ready('A-USDT'))
        self.connector._api_post.side_effect = TimeoutError('exchange unreachable')
        await self.bot._renew_heartbeat({'A-USDT'})
        self.bot._submit(Intent('A-USDT', True, D('.5'), D(10)))
        self.assertEqual(self.connector.sent, [])
        self.bot._submit(Intent('A-USDT', False, D('.5'), D(10), close=True, market=True))
        self.assertEqual(self.connector.sent[-1][5]['position_action'], PositionAction.CLOSE)

    async def test_expired_countdown_cancels_open_and_new_slot_rearms(self):
        await self.ready()
        await self.bot._renew_heartbeat({'A-USDT'})
        self.bot._submit(Intent('A-USDT', True, D('.5'), D(10)))
        oid = next(iter(self.bot.orders))
        self.bot.current_timestamp = 121
        self.bot.account_at = 121
        self.connector._api_post.return_value = {'triggerTime':'151000'}
        self.bot._heartbeat_tick(121)
        self.assertIn(oid, self.connector.cancels)
        await self.bot.heartbeat_task
        self.assertTrue(self.bot._heartbeat_ready('B-USDT'))
        self.assertTrue(self.bot._heartbeat_ready('A-USDT'))

    async def test_dry_run_never_calls_countdown_and_stop_does_not_disable(self):
        await self.ready()
        self.config.dry_run = True
        self.bot._heartbeat_tick(100)
        self.connector._api_post.assert_not_awaited()
        self.config.dry_run = False
        await self.bot._renew_heartbeat({'A-USDT'})
        await self.bot.on_stop()
        self.assertEqual(self.connector._api_post.await_count, 1)
        self.assertEqual(self.connector._api_post.call_args.kwargs['data']['timeout'], 30)

    async def test_unchanged_quotes_retained_but_age_price_and_size_changes_replace(self):
        await self.ready()
        intent = Intent('A-USDT', True, D('.5'), D('9.99'))
        self.bot.orders['one'] = adapter.TrackedOrder(intent, intent.amount, 100)
        m = market('A-USDT')
        self.assertTrue(self.bot._keep_quotes('A-USDT', [intent], m, 115))
        self.assertFalse(self.bot._keep_quotes('A-USDT', [intent], m, 160))
        self.assertFalse(self.bot._keep_quotes('A-USDT', [replace(intent, price=D('9.98'))], m, 115))
        self.assertFalse(self.bot._keep_quotes('A-USDT', [replace(intent, amount=D('.4'))], m, 115))
        self.assertFalse(self.bot._keep_quotes('A-USDT', [], m, 115))
        self.bot.orders['one'].cancel_at = 114
        self.assertFalse(self.bot._keep_quotes('A-USDT', [intent], m, 115))

    async def test_quality_modes_and_cost_reserve_do_not_expand_risk(self):
        await self.ready()
        m = market('A-USDT')
        self.assertEqual(self.bot._exit_cost_bps(m), D('.5'))
        self.assertIsInstance(self.bot._candidate_score(m), float)
        self.config.quality_control.mode = 'observe'
        self.assertEqual(self.bot._exit_cost_bps(m), 0)
        self.assertEqual(self.bot._quality_snapshot(m), {})


    async def test_default_controls_work_together_through_tick_and_refresh(self):
        await self.ready()
        self.bot.on_tick()
        self.assertEqual(self.connector.sent, [])
        await self.bot.heartbeat_task
        self.bot.on_tick()
        self.assertEqual(len(self.connector.sent), 4)
        self.bot.current_timestamp = 115
        await self.bot._refresh()
        self.connector._api_post.return_value = {'triggerTime':'145000'}
        before = len(self.connector.cancels)
        self.bot.on_tick()
        await self.bot.heartbeat_task
        self.assertEqual(len(self.connector.cancels), before)
        self.assertEqual(len(self.connector.sent), 4)

    async def test_adverse_quality_cancels_open_before_regular_refresh(self):
        await self.ready()
        await self.bot._renew_heartbeat({'A-USDT','B-USDT'})
        self.bot.on_tick()
        if self.bot.heartbeat_task:
            await self.bot.heartbeat_task
        a_ids = [oid for oid, order in self.bot.orders.items() if order.intent.pair == 'A-USDT']
        self.assertEqual(len(a_ids),2)
        for i in range(30):
            observation(self.bot.quality, i)
        self.bot.current_timestamp = 101
        self.bot.on_tick()
        self.assertTrue(set(a_ids).issubset(self.connector.cancels))

    async def test_losing_inventory_soft_exit_and_hard_stop_priority(self):
        await self.ready()
        self.bot.positions = self.bot.expected_positions = {'A-USDT':D('.5')}
        self.bot.position_quote = {'A-USDT':D(5)}
        self.bot.position_marks = {'A-USDT':D(10)}
        self.bot.unrealised_pnl = {'A-USDT':D('-.1')}
        self.bot.inventory.opened_at['A-USDT'] = 100
        self.bot.current_timestamp = self.bot.account_at = 221
        m = market('A-USDT',observed_at=221)
        for i in range(30):
            observation(self.bot.quality,i)
        with patch.object(self.bot, '_snapshots', return_value={'A-USDT':m}):
            self.bot.on_tick()
            if self.bot.heartbeat_task:
                await self.bot.heartbeat_task
        slot = self.bot.portfolio.slots['A-USDT']
        self.assertEqual(slot.state,'retiring')
        self.assertIn('execution quality',slot.reason)
        self.assertFalse(slot.force_market)
        self.bot.unrealised_pnl['A-USDT'] = D(-5)
        self.bot.on_tick()
        self.assertTrue(slot.force_market)
        self.assertIn('A-USDT',self.bot.portfolio.excluded)

    async def test_candidate_evidence_changes_score_without_replacing_healthy_slot(self):
        await self.ready()
        m = market('A-USDT')
        score = self.bot._candidate_score(m)
        for i in range(30):
            observation(self.bot.quality,i,net='.001',move='.001')
        self.assertGreater(self.bot._candidate_score(m),score)
        before = set(self.bot.portfolio.slots)
        self.bot.portfolio.evaluate(self.bot._snapshots(),100,True,{},set(),
                                   entry_check=self.bot._entry_check,candidate_score=self.bot._candidate_score)
        self.assertEqual(set(self.bot.portfolio.slots),before)

    async def test_telemetry_callback_trains_only_observed_five_second_markout(self):
        await self.ready()
        self.bot.recorder.fill(100,'A-USDT','o','f',D('.5'),D('9.99'),True,False,False,100,
            market('A-USDT'),types.SimpleNamespace(percent=D('.0002'),percent_token=None,flat_fees=[]),regime='calm')
        self.bot.recorder.sample(101,{'A-USDT':market('A-USDT',observed_at=101)},2)
        self.assertEqual(self.bot.quality.rows,{})
        self.bot.recorder.sample(105,{'A-USDT':market('A-USDT',observed_at=105)},2)
        self.assertEqual(self.bot.quality.estimate('A-USDT',True,'calm',106)['orders'],1)


class NativeOrderSafetyTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_minimum_notional_guard_allows_only_valid_reduce_only_tail(self):
        namespace = dict(Decimal=D, OrderType=OrderType, s_decimal_0=D(0), asyncio=asyncio)
        method = method_from_source('hummingbot/connector/exchange_py_base.py','ExchangePyBase','_create_order',namespace)
        hook = method_from_source('hummingbot/connector/derivative/gate_io_perpetual/gate_io_perpetual_derivative.py',
                                 'GateIoPerpetualDerivative','_allow_small_reduce_only_order',dict(PositionAction=PositionAction))
        exchange = types.SimpleNamespace(_trading_rules={'A-USDT':types.SimpleNamespace(min_order_size=D('.001'), min_notional_size=D(1))},
            quantize_order_amount=lambda **kw:kw['amount'], _order_tracker=types.SimpleNamespace(active_orders={}),
            supported_order_types=lambda:[OrderType.MARKET], _place_order_and_process_update=AsyncMock(),
            _update_order_after_failure=Mock(), _on_order_failure=Mock())
        exchange.start_tracking_order=lambda **kw: exchange._order_tracker.active_orders.update({kw['order_id']:types.SimpleNamespace(**kw)})
        exchange._allow_small_reduce_only_order=lambda order,**kw:hook(exchange,order,**kw)
        for action, amount, expected in [(PositionAction.OPEN,D('.05'),0),(PositionAction.CLOSE,D('.05'),1),
                                         (PositionAction.CLOSE,D('.0001'),1)]:
            await method(exchange,TradeType.SELL,'tail','A-USDT',amount,OrderType.MARKET,D(10),position_action=action)
            self.assertEqual(exchange._place_order_and_process_update.await_count,expected)
        self.assertEqual(exchange._update_order_after_failure.call_count,2)

    async def test_ioc_partial_fill_keeps_exchange_id_and_integer_reduce_only_size(self):
        method=method_from_source('hummingbot/connector/derivative/gate_io_perpetual/gate_io_perpetual_derivative.py',
            'GateIoPerpetualDerivative','_place_order',dict(Decimal=D,OrderType=OrderType,PositionAction=PositionAction,CONSTANTS=constants))
        exchange=types.SimpleNamespace(exchange_symbol_associated_to_pair=AsyncMock(return_value='A_USDT'),
            _format_amount_to_size=lambda p,a:a/D('.001'), current_timestamp=100,
            _api_post=AsyncMock(return_value=dict(id=123,finish_as='ioc',size=-500,left=200)))
        result=await method(exchange,'tail','A-USDT',D('.5'),TradeType.SELL,OrderType.MARKET,D(10),position_action=PositionAction.CLOSE)
        self.assertEqual(result,('123',100))
        self.assertEqual(exchange._api_post.call_args.kwargs['data']['size'],-500)
        self.assertTrue(exchange._api_post.call_args.kwargs['data']['reduce_only'])
        with self.assertRaises(ValueError):
            await method(exchange,'bad','A-USDT',D('.0001'),TradeType.SELL,OrderType.MARKET,D(10),position_action=PositionAction.CLOSE)
        self.assertEqual(exchange._api_post.await_count,1)
