"""动量因子: 过去 window 个交易日的涨跌幅。"""

from __future__ import annotations

import pandas as pd

from minibacktest.factors.registry import register


@register("momentum")
def momentum(price: pd.DataFrame, *, window: int) -> pd.Series:
    """动量因子: 过去 window 个交易日的涨跌幅, 用 t-1 相对 t-1-window 计算
    (不用当天收盘价, 避免用到"今天才知道"的信息)。

    Args:
        price: 收盘价宽表(建议用 adj_close, 排除分红/拆股干扰), index 是
            日期, columns 是 ticker。
        window: 回看窗口(交易日数), 比如 126 约等于半年动量。

    Returns:
        pd.Series, [date, ticker] MultiIndex, 值是过去 window 日涨跌幅;
        窗口内数据不足的地方是 NaN。

    Example:
        >>> import numpy as np
        >>> dates = pd.bdate_range("2024-01-01", periods=5)
        >>> price = pd.DataFrame({"A": [1.0, 1.1, 1.2, 1.3, 1.4]}, index=dates)
        >>> momentum(price, window=2).dropna()  # doctest: +NORMALIZE_WHITESPACE
        date        ticker
        2024-01-04  A         0.200000
        2024-01-05  A         0.181818
        Name: momentum, dtype: float64
    """
    mom = price.shift(1) / price.shift(1 + window) - 1.0
    s = mom.stack()
    s.index = s.index.set_names(["date", "ticker"])
    return s.rename("momentum") # type: ignore
