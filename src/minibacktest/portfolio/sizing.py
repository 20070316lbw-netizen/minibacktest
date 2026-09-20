"""仓位构造

目前设计的是一个
    按分数把每个调仓日的横截面分成 n_quantiles 组, 最高组等权做多、
    最低组等权做空, 中间组权重为 0, 多空两腿各自内部等权且总敞口相等
    (多头权重和为 +1, 空头权重和为 -1)
"""

from __future__ import annotations

import pandas as pd


def quantile_long_short(
        *,
        score       : pd.Series,
        n_quantiles : int,
        date_level  : str = "date"
) -> pd.Series:
    """依赖 _weight() 函数, 按分数把每个调仓日的横截面分成 n_quantiles 组,
    最高组等权做多, 最低组等权做空, 中间组权重为 0, 
    多空两腿各自内部等权且总敞口相等
    (多头权重和为 +1, 空头权重和为 -1)

    Args:
        score: 调仓日的横截面分数(比如 expected_returns.ic_scaled_alpha
            或者 combine.combine_scores 的输出), [date, ticker] MultiIndex,
            只需要包含调仓日。
        n_quantiles: 分组数, 比如 5(五分位, 最高最低各 20% 做多空)。
        date_level: 索引里日期所在的 level 名, 默认 "date"。

    Returns:
        组合权重, 与 score(去掉 NaN 后)同索引的 pd.Series, 中间组为 0、
        多头组为 +1/多头组内股票数、空头组为 -1/空头组内股票数。某天
        有效股票数不够分 n_quantiles 组时, 该天没有输出(不产生 0 行)。

    Raises:
        ValueError: n_quantiles 小于 2(至少要有多空两组)。

    Example:
        >>> d = pd.Timestamp("2024-01-01")
        >>> s = pd.Series({"A": 5.0, "B": 3.0, "C": 1.0, "D": -1.0, "E": -3.0, "F": -5.0})
        >>> s.index = pd.MultiIndex.from_product([[d], s.index], names=["date", "ticker"])
        >>> quantile_long_short(s, n_quantiles=3)
        date        ticker
        2024-01-01  A         0.5
                    B         0.5
                    C         0.0
                    D         0.0
                    E        -0.5
                    F        -0.5
        dtype: float64  
    """
    def _weight(s: pd.Series) -> pd.Series:
        """根据因子值,把一批股票分成多个分位数(quantile)组,然后构造一个"多空对冲"(long-short)的权重方案
        
        Args: 
            score: 调仓日的横截面分数(比如 expected_returns.ic_scaled_alpha
                或者 combine.combine_scores 的输出), [date, ticker] MultiIndex,
                只需要包含调仓日。
            
        Returns:
            组合权重
        
        """

        # 若股票数量比分组数量少则返回空 Series
        if len(s) < n_quantiles:
            return pd.Series(dtype=float)

        # 把股票按照因子值大小分成 n_quantiles 组, labels=False 让每组用数字编号(0, 1, 2, 3, 4)表示,数字越大表示因子值越大。duplicates="drop" 是为了防止出现重复的分位数边界导致报错(比如很多股票因子值相同时)
        bucket = pd.qcut(s, n_quantiles, labels=False, duplicates="drop")
        top, bottom = bucket.max(), bucket.min()

        # 所有股票权重先初始化为 0
        w = pd.Series(0.0, index = s.index)

        # 标记哪些股票属于最高 / 最低分位组
        long_mask   = bucket == top
        short_mask  = bucket == bottom

        # 给高低两组股票分配相等权重
        # 最高分位组分配正权重, 最低的为负权重; 中间分位为 0 
        w[long_mask] = 1.0 / long_mask.sum()
        w[short_mask] = 1.0 / short_mask.sum()

        return w
    

    if n_quantiles < 2:
        raise ValueError("n_quantiles 至少为2, 才可以分出多头与空头")

    score = score.dropna()

    return score.groupby(level=date_level, group_keys=False).apply(_weight)
