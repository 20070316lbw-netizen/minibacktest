from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd


@dataclass(frozen=True, slots=True)
class Result:
    """Summary statistics for a single backtest run.

    Field names mirror the columns produced by `backtesting.py`'s
    `Backtest.run()` output (see the `[%]` / `[$]` suffixes in the
    original), but are renamed to snake_case and given concrete types
    instead of raw strings.
    """

    # --- Timing ---------------------------------------------------
    start: datetime
    end: datetime
    duration: timedelta
    exposure_time_pct: float          # Exposure Time [%]

    # --- Equity -----------------------------------------------------
    equity_final: float               # Equity Final [$]
    equity_peak: float                # Equity Peak [$]

    # --- Returns ------------------------------------------------
    return_pct: float                 # Return [%]
    buy_and_hold_return_pct: float     # Buy & Hold Return [%]
    return_ann_pct: float             # Return (Ann.) [%]
    volatility_ann_pct: float         # Volatility (Ann.) [%]
    cagr_pct: float                   # CAGR [%]

    # --- Risk-adjusted ratios --------------------------------------
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    alpha_pct: float                  # Alpha [%]
    beta: float

    # --- Drawdown ---------------------------------------------------
    max_drawdown_pct: float           # Max. Drawdown [%]
    avg_drawdown_pct: float           # Avg. Drawdown [%]
    max_drawdown_duration: timedelta  # Max. Drawdown Duration
    avg_drawdown_duration: timedelta  # Avg. Drawdown Duration

    # --- Trades ------------------------------------------------------
    n_trades: int                     # # Trades
    win_rate_pct: float               # Win Rate [%]
    best_trade_pct: float             # Best Trade [%]
    worst_trade_pct: float            # Worst Trade [%]
    avg_trade_pct: float              # Avg. Trade [%]
    max_trade_duration: timedelta     # Max. Trade Duration
    avg_trade_duration: timedelta     # Avg. Trade Duration
    profit_factor: float
    expectancy_pct: float             # Expectancy [%]
    sqn: float
    kelly_criterion: float

    # --- Non-scalar extras (underscore-prefixed, as in the original) ---
    strategy: str = field(repr=False)             # _strategy
    equity_curve: pd.DataFrame = field(repr=False)  # _equity_curve
    trades: pd.DataFrame = field(repr=False)        # _trades