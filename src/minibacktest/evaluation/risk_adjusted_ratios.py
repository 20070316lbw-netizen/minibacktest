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


# Sortino ratio
def sortino_ratio(nav: pd.Series, periods_per_year: float) -> float:
    """用净值序列算年化索提诺比率(无风险利率按 0 处理)。

    跟夏普比率的区别: 分母只用下行波动率(只对负收益求标准差), 不惩罚
    向上的波动。

    Args:
        nav: 逐日净值。
        periods_per_year: nav 一年有多少个观测点(日频通常是 252)。

    Returns:
        年化索提诺比率; 有效收益率样本少于 2 个、没有负收益、或负收益
        标准差为 0 时返回 NaN。

    Example:
        >>> nav = pd.Series([1.0, 1.02, 0.99, 1.05, 1.03],
        ...                 index=pd.date_range("2024-01-01", periods=5))
        >>> round(sortino_ratio(nav, periods_per_year=252), 4)
        14.5647
    """
    ret = nav.pct_change().dropna()
    downside = ret[ret < 0]
    if len(ret) < 2 or len(downside) == 0 or downside.std() == 0:
        return float("nan")
    return float(ret.mean() / downside.std() * np.sqrt(periods_per_year))


# Calmar ratio
def calmar_ratio(nav: pd.Series, periods_per_year: float) -> float:
    """卡玛比率 = 年化收益 / |最大回撤|, 衡量收益相对于最坏情况的性价比。

    Args:
        nav: 逐日净值。
        periods_per_year: nav 一年有多少个观测点(日频通常是 252)。

    Returns:
        卡玛比率; 没有收益样本或最大回撤为 0(比如全程只涨不跌)时返回 NaN。

    Example:
        >>> nav = pd.Series([1.0, 1.2, 0.9, 1.1, 1.3],
        ...                 index=pd.date_range("2024-01-01", periods=5))
        >>> round(calmar_ratio(nav, periods_per_year=252), 4)
        84.7295
    """
    from minibacktest.evaluation.drawdown import max_drawdown

    ret = nav.pct_change().dropna()
    if len(ret) == 0:
        return float("nan")
    ann_return = (1 + ret.mean()) ** periods_per_year - 1
    mdd = max_drawdown(nav)
    if mdd == 0:
        return float("nan")
    return float(ann_return / abs(mdd))


