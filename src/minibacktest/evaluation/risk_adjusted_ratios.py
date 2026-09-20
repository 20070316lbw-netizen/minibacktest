""""""
from __future__ import annotations

import pandas as pd
import numpy as np


# Sharpe ratio
def sharpe_ratio(nav: pd.Series, periods_per_year: float) -> float:
    """用净值序列算年化夏普比率(无风险利率按 0 处理, 要扣无风险利率的话
    在传入 nav 前自己处理, 这里不做)。

    Args:
        nav: 逐日净值(见 engine.BacktestResult.nav)。
        periods_per_year: nav 一年有多少个观测点(日频数据通常是 252)。

    Returns:
        年化夏普比率; 有效收益率样本少于 2 个或标准差为 0 时返回 NaN。

    Example:
        >>> nav = pd.Series([1.0, 1.02, 1.01, 1.05, 1.03],
        ...                 index=pd.date_range("2024-01-01", periods=5))
        >>> round(sharpe_ratio(nav, periods_per_year=252), 4)
        4.5161
    """
    ret = nav.pct_change().dropna()
    if len(ret) < 2 or ret.std() == 0:
        return float("nan")
    return float(ret.mean() / ret.std() * np.sqrt(periods_per_year))


