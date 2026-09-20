"""净值曲线的构造: 从每日组合收益率算净值, 以及等权买入持有基准。"""

from __future__ import annotations

import pandas as pd


def build_nav(daily_returns: pd.Series, initial_capital: float = 1.0) -> pd.Series:
    """把逐日收益率序列累乘成净值曲线。

    Args:
        daily_returns: 逐日组合收益率(比如 engine.py 里权重和个股收益的
            点积), index 是日期。
        initial_capital: 期初资金, 默认 1.0(方便直接把净值当成倍数看)。

    Returns:
        逐日净值 pd.Series, 与 daily_returns 同索引。

    Example:
        >>> r = pd.Series([0.0, 0.1, -0.05])
        >>> build_nav(r, initial_capital=100.0).round(2).tolist()
        [100.0, 110.0, 104.5]
    """
    return initial_capital * (1.0 + daily_returns).cumprod()


def buy_and_hold_nav(price: pd.DataFrame, initial_capital: float = 1.0) -> pd.Series:
    """等权买入持有基准净值, 用作 Buy & Hold Return / Alpha / Beta 的对照组。

    简化处理: 用"每天对当天所有有值标的取平均收益率"来近似等权基准, 相当于
    每天都按等权重新平衡一次(不是严格意义上"期初买入、之后权重自然漂移"
    的买入持有, 但足够作为一个不带择时/选股能力的朴素对照)。

    Args:
        price: 收盘价宽表, index 是日期, columns 是 ticker。
        initial_capital: 期初资金, 默认 1.0。

    Returns:
        逐日净值 pd.Series, 与 price 同索引。
    """
    ret = price.pct_change()
    portfolio_ret = ret.mean(axis=1).fillna(0.0)
    return initial_capital * (1.0 + portfolio_ret).cumprod()
