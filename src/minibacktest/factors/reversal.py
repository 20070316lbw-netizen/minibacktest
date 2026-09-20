"""短期反转因子: 短期涨得越多的票, 越可能均值回归, 所以用负的短期动量当分数。"""

from __future__ import annotations

import pandas as pd

from minibacktest.factors.momentum import momentum
from minibacktest.factors.registry import register


@register("reversal")
def reversal(price: pd.DataFrame, *, window: int) -> pd.Series:
    """短期反转因子: 过去 window 个交易日涨跌幅取反, 押注短期超涨会回落、
    超跌会反弹(跟 momentum 用的是同一套计算, 方向相反, 通常配一个更短的
    window, 比如 5~10 个交易日, 才能跟动量因子的"半年动量"体现出区别)。

    Args:
        price: 收盘价宽表(建议用 adj_close), index 是日期, columns 是 ticker。
        window: 回看窗口(交易日数), 短期反转通常用比动量因子短得多的窗口。

    Returns:
        pd.Series, [date, ticker] MultiIndex, 值是过去 window 日涨跌幅的
        相反数。

    Example:
        >>> dates = pd.bdate_range("2024-01-01", periods=5)
        >>> price = pd.DataFrame({"A": [1.0, 1.1, 1.2, 1.3, 1.4]}, index=dates)
        >>> reversal(price, window=2).dropna()  # doctest: +NORMALIZE_WHITESPACE
        date        ticker
        2024-01-04  A        -0.200000
        2024-01-05  A        -0.181818
        Name: reversal, dtype: float64
    """
    return (-momentum(price, window=window)).rename("reversal")
