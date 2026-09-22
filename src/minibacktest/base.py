from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd


@dataclass(frozen=True, slots=True)
class Result:
    """单次回测的汇总统计结果。

    只保留"净值曲线级别"的指标: 只要有一条逐日净值序列(不管是单标的
    事件驱动策略跑出来的, 还是截面多资产向量化策略跑出来的)就能算,
    跟回测范式无关
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

    # --- Non-scalar extras (underscore-prefixed, as in the original) ---
    strategy: str = field(repr=False)               # _strategy
    equity_curve: pd.DataFrame = field(repr=False)  # _equity_curve

    # --- Trading costs ------------------------------------------------
    commission_bps: float = 0.0       # 单边佣金费率 [bps], 按调仓换手名义金额计
    slippage_bps: float = 0.0         # 单边滑点费率 [bps], 计法同佣金
    turnover_ann_pct: float = 0.0     # 年化换手率 [%]
    total_cost_pct: float = 0.0       # 交易成本对期末净值的总拖累 [%] (相对期初资金)
