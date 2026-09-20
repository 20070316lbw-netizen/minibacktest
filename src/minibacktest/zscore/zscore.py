'''
标准化分数: 对每一天（每个截面），把每一列因子转换成"距离当天均值几个标准差"
'''

from __future__ import annotations

import pandas as pd

def zscore_by_date(
        factors: pd.DataFrame,
        *,
        date_level: str = "date"
) -> pd.DataFrame:
    '''
    对每一天（每个截面），把每一列因子转换成"距离当天均值几个标准差",
    把因子面板逐列做横截面标准化(按日期分组, 均值 0、标准差 1):
        z = (原始值 - 当天该因子的均值) / 当天该因子的标准差

    Args:
        factors: [date, ticker] MultiIndex 因子面板, 一列一个因子, 可能有 NaN(某天某票某因子缺失)。
        date_level: 索引里日期所在的 level 名, 默认 "date"

    Returns:
        与 factors 同形状的 DataFrame, 每列在每个日期截面内标准化;
        某天某因子只有一只股票有值(标准差为 0 或 NaN)时该处为 NaN

    Example:
        >>> dates = pd.date_range("2024-01-01", periods=1, freq="D")
        >>> idx = pd.MultiIndex.from_product([dates, ["A", "B", "C"]], names=["date", "ticker"])
        >>> factors = pd.DataFrame({"f1": [1, 2, 3]}, index=idx)
        >>> zscore_by_date(factors)  # doctest: +NORMALIZE_WHITESPACE
                            f1
        date       ticker
        2024-01-01 A      -1.0
                   B       0.0
                   C       1.0
    '''
    grouped = factors.groupby(level=date_level)

    return (factors - grouped.transform("mean") / grouped.transform("std"))