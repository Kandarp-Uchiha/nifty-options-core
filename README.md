# nifty-options-core

[![ci](https://github.com/Kandarp-Uchiha/nifty-options-core/actions/workflows/ci.yml/badge.svg)](https://github.com/Kandarp-Uchiha/nifty-options-core/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.12-blue) ![uv](https://img.shields.io/badge/packaging-uv-purple) ![license](https://img.shields.io/badge/license-MIT-green)

The shared library behind three NIFTY index-options projects. It covers data ingestion, Black-Scholes
implied-vol and vol-surface features, an Indian F&O cost model, backtest helpers, and broker abstractions,
with the **same classes used in backtest, paper and live** (guardrail G9: only the event source differs).

| Project | What it does with the core |
|---|---|
| [nifty-iron-condor-ml](https://github.com/Kandarp-Uchiha/nifty-iron-condor-ml) | Regime-gated, XGBoost-filtered weekly iron condor; event-driven backtester; automated paper trading |
| [nifty-calendar-spread-live](https://github.com/Kandarp-Uchiha/nifty-calendar-spread-live) | Calendar-spread bot deployed live on Kotak Neo, plus a survivorship-bias post-mortem |
| [quant-research-lab](https://github.com/Kandarp-Uchiha/quant-research-lab) | Options / ETF / equity research with negative results documented |

## What's inside

| Package | Modules | Highlights |
|---|---|---|
| `nifty_core.core` | `events`, `expiries` | Order / leg / spread event model; NIFTY weekly-expiry calendar (handles the **Thu → Tue switch on 2025-09-01**) |
| `nifty_core.data` | `storage`, `sources.breeze`, `sources.nselib_fno`, `sources.yfinance_index`, `sources.yfinance_cross_asset` | Rate-limited, resumable ingestion of 1-minute options bars (ICICI Breeze), NSE bhavcopy/PCR, index and cross-asset dailies into a partitioned Parquet store |
| `nifty_core.research` | `probes`, `iv`, `vol_surface`, `trade_features`, `feature_lab`, `option_utils` | Black-Scholes pricing and bisection IV inversion; ATM IV, put/call skew, term slope; per-trade feature panels; VIX regime classifier |
| `nifty_core.exec` | `broker`, `backtest_broker`, `dry_run_broker` | Broker ABC; simulated fills with brokerage, STT, exchange charges, GST, stamp duty and slippage |
| `nifty_core.backtest` | `bars`, `metrics` | Minute-bar loading per contract; Sharpe, profit factor, drawdown, Calmar |
| `nifty_core.risk` | `manager` | Pre-trade risk limits, daily-loss stop, max concurrent positions |
| `nifty_core.notify` | `telegram` | Telegram alerts |
| `nifty_core.strategy` | `base` | Strategy ABC that every concrete strategy implements |

## Install (as a dependency)

```toml
# pyproject.toml of a consuming project
[project]
dependencies = ["nifty-options-core"]

[tool.uv.sources]
nifty-options-core = { git = "https://github.com/Kandarp-Uchiha/nifty-options-core", tag = "v1.0.0" }
```

`uv sync` installs it; `uv.lock` pins the exact commit. For local co-development, temporarily point the source
at `{ path = "../nifty-options-core", editable = true }`.

## Project root, data and config

Because the package lives inside a virtualenv, it never derives paths from `__file__`
([`paths.py`](src/nifty_core/paths.py)). The project root is:

1. `$NIFTY_HOME`, if set; otherwise
2. the nearest ancestor of the current directory containing `pyproject.toml`; otherwise
3. the current directory.

```
<project root>/
├── config/secrets.env                       # broker / data-vendor credentials (never committed)
└── data/
    ├── raw/breeze/options_1minute/<SYMBOL_EXPIRY_STRIKE_RIGHT>/<YYYYMMDD>.parquet
    ├── raw/yfinance/index_daily/<INDEX>/...
    └── _meta/                               # caches and session tokens (git-ignored)
```

Market data is **not** distributed (it is broker-licensed). Fetch it with your own credentials via the backfill
scripts in the consuming projects.

## Development

```bash
uv sync
uv run pytest -q
```

## Disclaimer

Educational and research code. Not investment advice. Nothing here is a recommendation to trade. Live trading
in India requires broker algo registration under SEBI's framework; this library does not provide that.

MIT licensed.
