from __future__ import annotations

import math
import unittest
from datetime import datetime, timezone
from tempfile import TemporaryDirectory

import pandas as pd

from forex_auto_trader.forex_auto_trader.data import MarketQuote
from forex_auto_trader.forex_auto_trader.broker import (
    MT5ForexBroker,
    NegativePositionState,
    OrderResult,
    Position,
    _money_exit_reason,
    _mt5_comment,
    _negative_recovery_exit_reason,
    _normalize_order_stops,
    _order_send_with_comment_fallback,
    _mt5_retcode_message,
    _plain_symbol,
    _recovery_fallback_loss_exit_reason,
)
from forex_auto_trader.forex_auto_trader.config import ForexConfig
from forex_auto_trader.forex_auto_trader.data import synthetic_trend
from forex_auto_trader.forex_auto_trader.journal import TradeJournal, event_row, realized_profit_series
from forex_auto_trader.forex_auto_trader.main import (
    _apply_broker_stop_buffer,
    _entry_cost_guard_reason,
    _signal_exit_reason,
    run_once,
)
from forex_auto_trader.forex_auto_trader.risk import RiskManager
from forex_auto_trader.forex_auto_trader.strategy import ForexSignal, ForexTrendStrategy


class ForexConfigTests(unittest.TestCase):
    def test_rejects_otc_symbol(self):
        config = ForexConfig(symbols=("EURUSD-OTC",))

        with self.assertRaises(ValueError):
            config.validate()

    def test_mt5_requires_explicit_live_gate(self):
        config = ForexConfig(broker="mt5", live_trading_enabled=False)

        with self.assertRaises(ValueError):
            config.validate()

    def test_plain_forex_config_validates(self):
        config = ForexConfig(symbols=("EURUSD", "GBPUSD"), broker="paper")

        config.validate()

    def test_allows_subsecond_position_checks(self):
        config = ForexConfig(position_check_seconds=0.5)

        config.validate()

    def test_rejects_symbol_map_outside_configured_symbols(self):
        config = ForexConfig(symbols=("EURUSD",), symbol_map={"USDCAD": "USDCAD.a"})

        with self.assertRaises(ValueError):
            config.validate()

    def test_rejects_unknown_trader_profile(self):
        config = ForexConfig(trader_profile="unknown")

        with self.assertRaises(ValueError):
            config.validate()


class ForexStrategyTests(unittest.TestCase):
    def test_generates_buy_signal_for_uptrend(self):
        config = ForexConfig(min_score=5, min_score_gap=1, min_confidence=0.55, ai_enabled=False)
        signal = ForexTrendStrategy(config).generate("EURUSD", synthetic_trend("EURUSD", direction=1))

        self.assertEqual(signal.side, "buy")
        self.assertGreaterEqual(signal.score, config.min_score)
        self.assertIsNotNone(signal.stop_loss)
        self.assertIsNotNone(signal.take_profit)
        self.assertIn("pro", signal.reason)

    def test_generates_sell_signal_for_downtrend(self):
        config = ForexConfig(min_score=5, min_score_gap=1, min_confidence=0.55, ai_enabled=False)
        signal = ForexTrendStrategy(config).generate("EURUSD", synthetic_trend("EURUSD", direction=-1))

        self.assertEqual(signal.side, "sell")
        self.assertGreaterEqual(signal.score, config.min_score)
        self.assertIsNotNone(signal.stop_loss)
        self.assertIsNotNone(signal.take_profit)
        self.assertIn("pro", signal.reason)

    def test_professional_profile_reason_is_recorded(self):
        config = ForexConfig(
            min_score=5,
            min_score_gap=1,
            min_confidence=0.55,
            ai_enabled=False,
            professional_profiles_enabled=True,
            trader_profile="trend",
        )
        signal = ForexTrendStrategy(config).generate("EURUSD", synthetic_trend("EURUSD", direction=1))

        self.assertIn("profile", signal.reason)

    def test_ai_ensemble_generates_model_signal(self):
        config = ForexConfig(
            bars=360,
            min_confidence=0.50,
            min_score=5,
            min_score_gap=0,
            ai_enabled=True,
            ai_allow_rule_fallback=False,
            ai_min_train_rows=120,
            ai_forward_bars=3,
            ai_min_validation_accuracy=0.50,
            ai_min_validation_precision=0.45,
            ai_min_probability=0.52,
            ai_min_probability_gap=0.01,
            ai_min_move_atr=0.05,
            ai_min_move_pips=0.1,
            ai_training_iterations=160,
        )
        signal = ForexTrendStrategy(config).generate("EURUSD", synthetic_cycle("EURUSD", rows=360))

        self.assertEqual(signal.side, "buy")
        self.assertIn("AI ensemble buy", signal.reason)
        self.assertIsNotNone(signal.stop_loss)
        self.assertIsNotNone(signal.take_profit)

    def test_ai_uses_provisional_model_when_strict_validation_rejects(self):
        config = ForexConfig(
            bars=360,
            min_confidence=0.50,
            min_score=1,
            min_score_gap=0,
            ai_enabled=True,
            ai_allow_rule_fallback=False,
            ai_min_train_rows=120,
            ai_forward_bars=3,
            ai_min_validation_accuracy=0.99,
            ai_min_validation_precision=0.99,
            ai_min_probability=0.50,
            ai_min_probability_gap=0.0,
            ai_allow_provisional_models=True,
            ai_provisional_min_probability=0.50,
            ai_provisional_min_probability_gap=0.0,
            ai_provisional_min_validation_accuracy=0.0,
            ai_provisional_min_validation_precision=0.0,
            ai_min_move_atr=0.05,
            ai_min_move_pips=0.1,
            ai_training_iterations=160,
        )

        signal = ForexTrendStrategy(config).generate("EURUSD", synthetic_cycle("EURUSD", rows=360))

        self.assertIn(signal.side, {"buy", "sell"})
        self.assertIn("AI provisional", signal.reason)
        self.assertNotIn("AI model unavailable", signal.reason)


class ForexRiskAndJournalTests(unittest.TestCase):
    def test_journal_writes_csv_and_sqlite(self):
        with TemporaryDirectory() as directory:
            journal = TradeJournal(f"{directory}/trades.csv", f"{directory}/memory.sqlite3")
            journal.append(
                event_row(
                    event="exit",
                    broker="paper",
                    symbol="EURUSD",
                    side="buy",
                    realized_profit=12.5,
                    balance=1012.5,
                    reason="test",
                )
            )

            frame = journal.load()

            self.assertEqual(len(frame), 1)
            self.assertAlmostEqual(journal.today_profit(), 12.5)
            self.assertEqual(journal.edge_stats("EURUSD", "buy").wins, 1)

    def test_risk_allows_sized_trade_when_limits_clear(self):
        with TemporaryDirectory() as directory:
            config = ForexConfig(
                risk_per_trade_pct=0.01,
                max_lot=0.10,
                min_confidence=0.55,
                min_score=5,
                min_score_gap=1,
                ai_enabled=False,
            )
            journal = TradeJournal(f"{directory}/trades.csv", f"{directory}/memory.sqlite3")
            signal = ForexTrendStrategy(config).generate("EURUSD", synthetic_trend("EURUSD", direction=1))

            decision = RiskManager(config, journal).can_open(signal, balance=1000, open_positions=[])

            self.assertTrue(decision.allowed)
            self.assertGreaterEqual(decision.lots, config.min_lot)
            self.assertLessEqual(decision.lots, config.max_lot)

    def test_adaptive_memory_blocks_weak_underperforming_side(self):
        with TemporaryDirectory() as directory:
            config = ForexConfig(
                adaptive_min_trades=2,
                adaptive_target_win_rate=0.60,
                min_confidence=0.70,
                min_score=8,
                ai_enabled=False,
                professional_filters_enabled=False,
            )
            journal = TradeJournal(f"{directory}/trades.csv", f"{directory}/memory.sqlite3")
            for profit in (-10, 8):
                journal.append(
                    event_row(
                        event="exit",
                        broker="paper",
                        symbol="EURUSD",
                        side="buy",
                        realized_profit=profit,
                        balance=1000 + profit,
                        reason="test",
                    )
                )
            signal = ForexTrendStrategy(config).generate("EURUSD", synthetic_trend("EURUSD", direction=1))

            adjusted = RiskManager(config, journal).apply_adaptive_memory(signal)

            self.assertEqual(adjusted.side, "hold")
            self.assertIn("adaptive memory blocked", adjusted.reason)

    def test_adaptive_memory_can_use_full_journal(self):
        with TemporaryDirectory() as directory:
            journal = TradeJournal(f"{directory}/trades.csv", f"{directory}/memory.sqlite3")
            for profit in (-1, 2, 3):
                journal.append(
                    event_row(
                        event="exit",
                        broker="paper",
                        symbol="EURUSD",
                        side="buy",
                        realized_profit=profit,
                        balance=1000 + profit,
                        reason="test",
                    )
                )

            stats = journal.edge_stats("EURUSD", "buy", lookback=0)

            self.assertEqual(stats.trades, 3)
            self.assertEqual(stats.wins, 2)

    def test_latest_exit_returns_latest_symbol_side_result(self):
        with TemporaryDirectory() as directory:
            journal = TradeJournal(f"{directory}/trades.csv", f"{directory}/memory.sqlite3")
            journal.append(
                event_row(
                    event="exit",
                    broker="paper",
                    symbol="EURUSD",
                    side="buy",
                    realized_profit=-0.4,
                    created_utc="2026-01-01T00:00:00+00:00",
                )
            )
            journal.append(
                event_row(
                    event="exit",
                    broker="paper",
                    symbol="EURUSD",
                    side="buy",
                    realized_profit=0.2,
                    created_utc="2026-01-01T00:01:00+00:00",
                )
            )

            latest = journal.latest_exit("EURUSD", "buy")

            self.assertIsNotNone(latest)
            self.assertAlmostEqual(latest.profit, 0.2)

    def test_adaptive_loss_cooldown_blocks_recent_losing_side(self):
        with TemporaryDirectory() as directory:
            config = ForexConfig(
                adaptive_loss_cooldown_minutes=60,
                adaptive_loss_cooldown_min_confidence=0.95,
                adaptive_loss_cooldown_min_score=10,
            )
            journal = TradeJournal(f"{directory}/trades.csv", f"{directory}/memory.sqlite3")
            journal.append(
                event_row(
                    event="exit",
                    broker="paper",
                    symbol="EURUSD",
                    side="buy",
                    realized_profit=-0.5,
                )
            )
            signal = ForexSignal("EURUSD", "buy", 0.90, 8, 2, 1.1000, 1.0990, 1.1010, 0.0010, "test", "now")

            adjusted = RiskManager(config, journal).apply_adaptive_memory(signal)

            self.assertEqual(adjusted.side, "hold")
            self.assertIn("adaptive loss cooldown blocked", adjusted.reason)

    def test_journal_infers_profit_from_money_exit_reason(self):
        with TemporaryDirectory() as directory:
            journal = TradeJournal(f"{directory}/trades.csv", f"{directory}/memory.sqlite3")
            journal.append(
                event_row(
                    event="exit",
                    broker="mt5",
                    symbol="EURUSD",
                    side="sell",
                    realized_profit="",
                    reason="bot money loss exit -1.15",
                )
            )

            profit = realized_profit_series(journal.load())

            self.assertAlmostEqual(float(profit.iloc[0]), -1.15)

    def test_mt5_retcode_10027_explains_algo_trading_disabled(self):
        message = _mt5_retcode_message(10027)

        self.assertIn("Algo Trading", message)
        self.assertIn("10027", message)

    def test_plain_symbol_strips_broker_suffix(self):
        symbol = _plain_symbol("EURUSD.a", ("EURUSD", "USDCAD"), {})

        self.assertEqual(symbol, "EURUSD")

    def test_mt5_broker_resolves_symbol_suffix(self):
        broker = MT5ForexBroker(ForexConfig(symbols=("USDCAD",), broker="mt5", live_trading_enabled=True))
        broker.mt5 = FakeMt5()

        resolved = broker._resolve_symbol("USDCAD")

        self.assertEqual(resolved, "USDCAD.a")
        self.assertIn("USDCAD.a", broker.mt5.selected_symbols)

    def test_signal_exit_closes_on_strong_reversal(self):
        config = ForexConfig(exit_reversal_confidence=0.70, exit_reversal_min_score=6)
        position = Position("1", "EURUSD", "buy", 0.01, 1.1000, 1.0950, 1.1100, 0.0010, "now", "paper")
        signal = ForexSignal("EURUSD", "sell", 0.80, 7, 2, 1.0992, None, None, 0.0010, "test", "now")

        reason = _signal_exit_reason(config, position, 1.0992, signal)

        self.assertIn("reversal", reason)

    def test_signal_exit_closes_on_profit_lock(self):
        config = ForexConfig(exit_profit_rr=1.0)
        position = Position("1", "EURUSD", "buy", 0.01, 1.1000, 1.0980, 1.1060, 0.0010, "now", "paper")
        signal = ForexSignal("EURUSD", "buy", 0.80, 7, 2, 1.1025, None, None, 0.0010, "test", "now")

        reason = _signal_exit_reason(config, position, 1.1025, signal)

        self.assertIn("profit lock", reason)

    def test_signal_exit_can_disable_early_atr_loss(self):
        config = ForexConfig(exit_early_loss_enabled=False, exit_loss_amount=1.0)
        position = Position(
            "1",
            "EURUSD",
            "buy",
            0.01,
            1.1000,
            1.0980,
            1.1060,
            0.0001,
            "now",
            "paper",
            profit=-0.30,
        )
        signal = ForexSignal("EURUSD", "buy", 0.80, 7, 2, 1.0996, None, None, 0.0001, "test", "now")

        reason = _signal_exit_reason(config, position, 1.0996, signal)

        self.assertEqual(reason, "")

    def test_money_exit_closes_when_profit_amount_reached(self):
        config = ForexConfig(exit_profit_amount=1.0)
        position = Position("1", "EURUSD", "buy", 0.01, 1.1000, 1.0980, 1.1060, 0.0010, "now", "paper", profit=1.25)

        reason = _money_exit_reason(config, position)

        self.assertIn("money profit", reason)

    def test_negative_recovery_exit_waits_then_closes_on_small_profit(self):
        config = ForexConfig(exit_recovery_after_seconds=60, exit_recovery_profit_amount=0.01)
        tracker = {"1": 100.0}
        position = Position("1", "EURUSD", "buy", 0.01, 1.1000, 1.0980, 1.1060, 0.0010, "now", "paper", profit=0.02)

        reason = _negative_recovery_exit_reason(config, position, tracker, now_monotonic=161.0)

        self.assertIn("recovery profit", reason)

    def test_recovery_fallback_loss_closes_before_broker_stop(self):
        config = ForexConfig(exit_recovery_fallback_loss_amount=0.50)
        position = Position("1", "EURUSD", "buy", 0.01, 1.1000, 1.0980, 1.1060, 0.0010, "now", "paper", profit=-0.52)

        reason = _recovery_fallback_loss_exit_reason(config, position)

        self.assertIn("before broker stop", reason)

    def test_smart_negative_exit_closes_stalled_loss(self):
        config = ForexConfig(
            exit_negative_grace_seconds=60,
            exit_negative_min_loss_amount=0.10,
            exit_negative_min_improvement_amount=0.05,
        )
        tracker = {}
        position = Position("1", "EURUSD", "buy", 0.01, 1.1000, 1.0980, 1.1060, 0.0010, "now", "paper", profit=-0.20)

        first = _negative_recovery_exit_reason(config, position, tracker, now_monotonic=100.0)
        reason = _negative_recovery_exit_reason(config, position, tracker, now_monotonic=161.0)

        self.assertEqual(first, "")
        self.assertIn("smart negative exit", reason)

    def test_smart_negative_exit_waits_when_loss_is_recovering(self):
        config = ForexConfig(
            exit_negative_grace_seconds=60,
            exit_negative_min_loss_amount=0.10,
            exit_negative_min_improvement_amount=0.05,
        )
        tracker = {"1": NegativePositionState(started_at=100.0, worst_profit=-0.40)}
        position = Position("1", "EURUSD", "buy", 0.01, 1.1000, 1.0980, 1.1060, 0.0010, "now", "paper", profit=-0.20)

        reason = _negative_recovery_exit_reason(config, position, tracker, now_monotonic=161.0)

        self.assertEqual(reason, "")

    def test_entry_cost_guard_blocks_spread_too_close_to_profit_target(self):
        config = ForexConfig(exit_profit_amount=0.50, max_entry_cost_to_profit_ratio=0.40, slippage_guard_enabled=False)
        signal = ForexSignal("EURUSD", "buy", 0.90, 8, 2, 1.1000, 1.0990, 1.1010, 0.0010, "test", "now")

        reason = _entry_cost_guard_reason(config, FakeQuoteBroker("EURUSD", 1.1000, 1.1005), signal, lots=0.05)

        self.assertIn("entry cost guard", reason)

    def test_entry_cost_guard_blocks_absolute_starting_cost(self):
        config = ForexConfig(max_entry_cost_amount=0.07, slippage_guard_enabled=False)
        signal = ForexSignal("EURUSD", "buy", 0.90, 8, 2, 1.1000, 1.0990, 1.1010, 0.0010, "test", "now")

        reason = _entry_cost_guard_reason(config, FakeQuoteBroker("EURUSD", 1.1000, 1.1008), signal, lots=0.01)

        self.assertIn("max entry cost", reason)

    def test_slippage_guard_blocks_signal_price_drift(self):
        config = ForexConfig(max_signal_price_drift_pips=1.0)
        signal = ForexSignal("EURUSD", "buy", 0.90, 8, 2, 1.1000, 1.0990, 1.1010, 0.0010, "test", "now")

        reason = _entry_cost_guard_reason(config, FakeQuoteBroker("EURUSD", 1.1000, 1.1003), signal, lots=0.01)

        self.assertIn("slippage guard", reason)

    def test_mt5_stop_normalization_respects_symbol_minimum_distance(self):
        config = ForexConfig(mt5_stop_buffer_points=5)
        signal = ForexSignal("EURUSD", "buy", 0.90, 8, 2, 1.1000, 1.0999, 1.1001, 0.0010, "test", "now")
        info = FakeSymbolInfo(point=0.00001, digits=5, trade_stops_level=50)

        adjusted = _normalize_order_stops(config, signal, entry_price=1.1000, symbol_info=info)

        self.assertLessEqual(adjusted.stop_loss, 1.09945)
        self.assertGreaterEqual(adjusted.take_profit, 1.10055)
        self.assertIn("MT5 stop distance normalized", adjusted.reason)

    def test_broker_stop_buffer_widens_stop_beyond_bot_loss(self):
        config = ForexConfig(exit_loss_amount=0.50, broker_stop_loss_buffer_amount=0.20)
        signal = ForexSignal("EURUSD", "buy", 0.90, 8, 2, 1.1000, 1.0996, 1.1010, 0.0010, "test", "now")

        adjusted = _apply_broker_stop_buffer(config, signal, lots=0.01)

        self.assertLess(adjusted.stop_loss, signal.stop_loss)
        self.assertIn("broker stop buffered", adjusted.reason)

    def test_run_once_can_open_multiple_trades_per_scan(self):
        with TemporaryDirectory() as directory:
            config = ForexConfig(
                symbols=("EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD"),
                max_open_positions=5,
                max_new_trades_per_scan=5,
                min_score=5,
                min_score_gap=1,
                min_confidence=0.55,
                ai_enabled=False,
                slippage_guard_enabled=False,
                professional_filters_enabled=False,
                log_path=f"{directory}/trades.csv",
                db_path=f"{directory}/memory.sqlite3",
            )
            journal = TradeJournal(config.log_file, config.db_file)
            broker = FakeMultiOrderBroker(config)

            run_once(config, journal, broker, ForexTrendStrategy(config), RiskManager(config, journal))

            self.assertEqual(len(broker.positions), 5)

    def test_mt5_comment_is_short_and_safe(self):
        comment = _mt5_comment('bot reversal exit: sell signal confidence 0.80, score 7/10')

        self.assertLessEqual(len(comment), 31)
        self.assertNotIn(":", comment)
        self.assertNotIn("/", comment)

    def test_order_send_retries_without_invalid_comment(self):
        mt5 = FakeOrderSendMt5()
        request = {"symbol": "EURUSD", "comment": "bad"}

        result = _order_send_with_comment_fallback(mt5, request)

        self.assertIsNotNone(result)
        self.assertEqual(len(mt5.requests), 2)
        self.assertNotIn("comment", mt5.requests[-1])


class FakeSymbol:
    def __init__(self, name, visible=True, trade_mode=1):
        self.name = name
        self.visible = visible
        self.trade_mode = trade_mode


class FakeSymbolInfo:
    def __init__(self, point=0.00001, digits=5, trade_stops_level=0, trade_freeze_level=0):
        self.point = point
        self.digits = digits
        self.trade_stops_level = trade_stops_level
        self.trade_freeze_level = trade_freeze_level


class FakeMt5:
    SYMBOL_TRADE_MODE_DISABLED = 0

    def __init__(self):
        self.selected_symbols = []

    def symbol_info(self, name):
        return None

    def symbols_get(self, pattern):
        if pattern == "USDCAD*":
            return [FakeSymbol("USDCAD.a"), FakeSymbol("USDCAD-disabled", trade_mode=0)]
        return []

    def symbol_select(self, name, visible):
        self.selected_symbols.append(name)
        return visible


class FakeOrderResult:
    retcode = 10009
    order = 123


class FakeOrderSendMt5:
    def __init__(self):
        self.requests = []

    def order_send(self, request):
        self.requests.append(dict(request))
        if "comment" in request:
            return None
        return FakeOrderResult()

    def last_error(self):
        return (-2, 'Invalid "comment" argument')


class FakeQuoteBroker:
    def __init__(self, symbol, bid, ask):
        self._quote = MarketQuote(symbol, bid, ask, datetime.now(timezone.utc))

    def quote(self, symbol):
        return self._quote


class FakeMultiOrderBroker:
    def __init__(self, config):
        self.config = config
        self.positions = []
        self.order_count = 0

    def connect(self):
        return None

    def account_balance(self):
        return 10000.0

    def manage_positions(self):
        return []

    def get_candles(self, symbol):
        return synthetic_trend(symbol, direction=1)

    def quote(self, symbol):
        price = 159.10 if symbol.endswith("JPY") else 1.1000
        spread = 0.002 if symbol.endswith("JPY") else 0.00002
        return MarketQuote(symbol, price, price + spread, datetime.now(timezone.utc))

    def open_positions(self):
        return list(self.positions)

    def place_order(self, signal, lots):
        self.order_count += 1
        self.positions.append(
            Position(
                str(self.order_count),
                signal.symbol,
                signal.side,
                lots,
                signal.price,
                signal.stop_loss or 0,
                signal.take_profit or 0,
                signal.atr,
                "now",
                "paper",
            )
        )
        return OrderResult(True, str(self.order_count), "opened")


def synthetic_cycle(symbol: str, rows: int = 360) -> pd.DataFrame:
    base = 150.0 if symbol.endswith("JPY") else 1.1000
    amplitude = 0.18 if symbol.endswith("JPY") else 0.0018
    wobble = 0.025 if symbol.endswith("JPY") else 0.00025
    records = []
    for index in range(rows):
        close = base + amplitude * math.sin(index / 7.0) + wobble * math.sin(index / 2.0)
        open_price = close - wobble * 0.32 * math.cos(index / 3.0)
        high = max(open_price, close) + wobble
        low = min(open_price, close) - wobble
        records.append(
            {
                "time": pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(minutes=5 * index),
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": 1000 + 100 * math.sin(index / 9.0),
            }
        )
    return pd.DataFrame(records)


if __name__ == "__main__":
    unittest.main()
