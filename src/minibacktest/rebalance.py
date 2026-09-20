from __future__ import annotations

import pandas as pd


def rebalance_dates(dates: pd.DatetimeIndex, freq: int) -> pd.DatetimeIndex:
    """从全量交易日历里, 每隔 freq 个交易日取一个调仓日, 从最早的一天开始。

    Args:
        dates: 全量交易日历, 不要求预先排序或去重。
        freq: 调仓间隔, 按交易日数算(比如 21)。

    Returns:
        调仓日组成的 DatetimeIndex, 升序, 第一个元素是 dates 里最早的一天。

    Raises:
        ValueError: freq 不是正整数。

    Example:
        >>> dates = pd.date_range("2024-01-01", periods=7, freq="D")
        >>> list(rebalance_dates(dates, freq=3))
        [Timestamp('2024-01-01 00:00:00'), Timestamp('2024-01-04 00:00:00'), Timestamp('2024-01-07 00:00:00')]
    """
    if freq <= 0:
        raise ValueError("freq 必须是正整数")
    d = pd.DatetimeIndex(dates).sort_values().unique()
    return d[::freq]


def rebalance_block(dates: pd.DatetimeIndex, freq: int) -> pd.Series:
    """把全量交易日历里的每一天, 映射到它所属的调仓日(区间起点)。

    非调仓日属于"最近一次已经发生的调仓日"所在区间, 一直到下一个调仓日
    (不含)为止; 调仓日自己属于自己开启的新区间。

    Args:
        dates: 全量交易日历, 不要求预先排序或去重。
        freq: 调仓间隔, 见 rebalance_dates。

    Returns:
        与去重排序后的 dates 等长的 pd.Series, 索引是日期, 值是该日所属
        调仓日(pd.Timestamp), 可以直接拿来 groupby 做区间内的向量化计算
        (比如 engine.py 里按区间算相对收益)。

    Example:
        >>> dates = pd.date_range("2024-01-01", periods=7, freq="D")
        >>> rebalance_block(dates, freq=3)
        2024-01-01   2024-01-01
        2024-01-02   2024-01-01
        2024-01-03   2024-01-01
        2024-01-04   2024-01-04
        2024-01-05   2024-01-04
        2024-01-06   2024-01-04
        2024-01-07   2024-01-07
        Freq: D, Name: rebalance_date, dtype: datetime64[us]
    """
    d = pd.DatetimeIndex(dates).sort_values().unique()
    rb = rebalance_dates(d, freq)
    idx = rb.searchsorted(d, side="right") - 1
    return pd.Series(rb[idx], index=d, name="rebalance_date")