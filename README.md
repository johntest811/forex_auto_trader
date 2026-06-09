# Forex Auto Trader

This is a separate Forex-only project. It does not place fixed-expiry option trades and it does not depend on IQ Option.

Live trading is supported through MetaTrader 5 only when `FOREX_BROKER=mt5` and `FOREX_LIVE_TRADING_ENABLED=true`. Paper mode is the default.

## Start

```powershell
cd C:\Users\Ezra\Pictures\TradingCode
py -3.13 -m venv forex_auto_trader\.venv
forex_auto_trader\.venv\Scripts\python.exe -m pip install --upgrade pip
forex_auto_trader\.venv\Scripts\python.exe -m pip install -r forex_auto_trader\requirements.txt
forex_auto_trader\.venv\Scripts\python.exe forex_auto_trader\run.py doctor --env forex_auto_trader\.env
forex_auto_trader\.venv\Scripts\python.exe forex_auto_trader\run.py once --env forex_auto_trader\.env
```

## Dashboard

Start the local control panel:

```powershell
forex_auto_trader\.venv\Scripts\python.exe forex_auto_trader\run.py dashboard --env forex_auto_trader\.env --open
```

The dashboard shows account status, open trades, journal progress, win rate, recent signals/orders, and editable risk/strategy settings. Use the **Start** button to run the bot loop, **Stop** to pause it, and **Run Once** for a single decision cycle.

## Live MT5

1. Install MetaTrader 5 and log in to a real Forex/CFD broker account.
2. Keep `FOREX_BROKER=paper` until the journal shows stable paper performance.
3. To enable live trading, set:

```dotenv
FOREX_BROKER=mt5
FOREX_LIVE_TRADING_ENABLED=true
```

Then install the optional MT5 dependency inside the project venv:

```powershell
forex_auto_trader\.venv\Scripts\python.exe -m pip install -r forex_auto_trader\requirements-mt5.txt
```

The bot uses market orders, ATR-based stop loss, risk-based lot sizing, take-profit, and bot-managed trailing exits. It journals every signal and order to CSV and SQLite.

## AI Signal Engine

The trade bot now uses a local machine-learning ensemble before any order is considered. On each scan it:

- builds technical features from recent candles,
- trains buy/sell classifiers over several forward horizons,
- rejects models that fail validation accuracy or precision gates,
- can use the best provisional AI model with stronger live thresholds when all strict validation gates reject,
- requires a minimum live probability and buy-vs-sell probability gap,
- can optionally fall back to the deterministic strategy, but live Pepperstone mode is configured AI-only.

The model feature set includes technical indicators, volatility regime, trend efficiency, wick pressure, session timing, London/New York overlap, and pair context such as USD, JPY, commodity-currency, safe-haven, and cross-pair flags.

Useful controls in `.env`:

```dotenv
FOREX_AI_ENABLED=true
FOREX_AI_ALLOW_RULE_FALLBACK=false
FOREX_AI_MIN_TRAIN_ROWS=160
FOREX_AI_FORWARD_BARS=6
FOREX_AI_MIN_VALIDATION_ACCURACY=0.54
FOREX_AI_MIN_VALIDATION_PRECISION=0.54
FOREX_AI_MIN_PROBABILITY=0.68
FOREX_AI_MIN_PROBABILITY_GAP=0.06
FOREX_AI_ALLOW_PROVISIONAL_MODELS=true
FOREX_AI_PROVISIONAL_MIN_PROBABILITY=0.78
FOREX_AI_PROVISIONAL_MIN_PROBABILITY_GAP=0.25
FOREX_AI_PROVISIONAL_MIN_VALIDATION_ACCURACY=0.45
FOREX_AI_PROVISIONAL_MIN_VALIDATION_PRECISION=0.35
```

No AI model can guarantee profit or a fixed win rate. Treat the validation values as a live filter, not a promise.

## Slippage And Exits

The bot accounts for execution friction before it opens a trade:

```dotenv
FOREX_SLIPPAGE_GUARD_ENABLED=true
FOREX_MAX_SIGNAL_PRICE_DRIFT_PIPS=1.0
FOREX_EXPECTED_SLIPPAGE_PIPS=0.25
FOREX_MAX_SLIPPAGE_COST_AMOUNT=0.05
FOREX_MAX_SLIPPAGE_TO_PROFIT_RATIO=0.25
MT5_ORDER_DEVIATION_POINTS=10
MT5_STOP_BUFFER_POINTS=5
```

It blocks entries when the live bid/ask has drifted too far from the signal price or when spread plus expected slippage is too large relative to the reward/profit target. MT5 orders also use a configurable deviation and the bot widens SL/TP to satisfy broker stop-distance rules before sending the order.

For positions that go negative, the bot now uses a patient recovery window. It can close at a small positive recovery after the trade has spent time negative, while the stalled-negative exit waits longer so it does not cut a recoverable trade too quickly:

```dotenv
FOREX_EXIT_RECOVERY_ENABLED=true
FOREX_EXIT_RECOVERY_AFTER_SECONDS=300
FOREX_EXIT_RECOVERY_PROFIT_AMOUNT=0.01
FOREX_EXIT_NEGATIVE_SMART_ENABLED=true
FOREX_EXIT_NEGATIVE_GRACE_SECONDS=900
FOREX_EXIT_NEGATIVE_MIN_LOSS_AMOUNT=0.25
FOREX_EXIT_NEGATIVE_MIN_IMPROVEMENT_AMOUNT=0.05
```

## Pepperstone Symbols

Some broker accounts use suffixes such as `EURUSD.a` instead of plain `EURUSD`. The bot now tries to resolve those automatically. If a symbol still says unavailable, set the exact broker symbol in `forex_auto_trader\.env`:

```dotenv
FOREX_SYMBOL_MAP=EURUSD=EURUSD.a,USDCAD=USDCAD.a
```

`maximum positions reached for EURUSD` is not a crash. It means the bot already has the configured maximum open EURUSD positions, so it will now try the next qualified market in the same cycle.

No strategy can guarantee profit. Use small paper/live sizing and check the journal before increasing risk.
