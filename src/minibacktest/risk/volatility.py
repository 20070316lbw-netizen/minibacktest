"""已实现波动率: 拿来给分层函数当"切层依据", 不进 factor 打分链路。"""

from __future__ import annotations

import pandas as pd


def realized_volatility(price: pd.DataFrame, *, window: int) -> pd.Series:
    """逐日收益率的滚动标准差, 堆成 [date, ticker] MultiIndex 长表, 形状
    跟 factors/ 里因子函数的输出一致, 可以直接喂给 vol_neutral_* 系列函数
    的 vol 参数(也可以喂给 zscore_by_date, 如果哪天想把它也当一个正经
    因子来试试看)。

    Args:
        price: 收盘价宽表, index 是交易日历, columns 是 ticker。
        window: 滚动窗口(交易日数), 比如 21 约等于一个月。

    Returns:
        pd.Series, [date, ticker] MultiIndex, 值是过去 window 天日收益率
        的标准差; 窗口内数据不足(含新上市不久的票)的地方是 NaN。

    Example:
        >>> import numpy as np
        >>> dates = pd.bdate_range("2024-01-01", periods=5)
        >>> price = pd.DataFrame({"A": [1, 1, 1, 1, 1], "B": [1, 2, 1, 2, 1]}, index=dates)
        >>> vol = realized_volatility(price, window=2)
        >>> bool(round(vol.xs("B", level="ticker").iloc[-1], 4) > 0)
        True
    """

    daily_return = price.pct_change()
    vol = daily_return.rolling(window).std()    # <-- 现在是宽表, index=日期, columns=ticker(单层), 值=波动率

    out = vol.stack()                           # <-- # 现在是长表: index=[date, ticker](双层), 每行一个值 → 必然是 Series
    out.index = out.index.set_names(["date", "ticker"])
    return out.rename(f"volatility_{window}") # type: ignore
