"""按打分分组算前瞻收益, 用来检验截面因子是否单调有效(对应 README 流程图
里最后一步"五分位单调性检验")。
"""

from __future__ import annotations

import pandas as pd

from minibacktest.rebalance import rebalance_dates


def quantile_forward_returns(
    score: pd.Series,
    price: pd.DataFrame,
    *,
    freq: int,
    n_quantiles: int,
    date_level: str = "date",
) -> pd.Series:
    """把每个调仓日的打分按 n_quantiles 分组, 算每组到下一个调仓日的平均前瞻收益。

    分组方式跟 portfolio.sizing.quantile_long_short 内部用的一致(同一天
    横截面 pd.qcut), 这样两边的分位数编号才是可比的。

    Args:
        score: 调仓日的横截面打分, [date, ticker] MultiIndex, 只需要包含
            调仓日(跟 quantile_long_short 的输入一样)。
        price: 收盘价宽表, index 是完整交易日历, columns 是 ticker。
        freq: 调仓间隔(交易日数), 需要和生成 score 时用的一致。
        n_quantiles: 分组数, 跟 quantile_long_short 用的保持一致才有可比性。
        date_level: 索引里日期所在的 level 名, 默认 "date"。

    Returns:
        pd.Series, 索引是分位数编号(0 是打分最低组, n_quantiles - 1 是
        最高组), 值是该组在所有调仓日上的平均前瞻收益。理想情况下应该
        单调递增(打分越高、后续涨得越多)。

    Example:
        >>> import numpy as np
        >>> dates = pd.bdate_range("2024-01-01", periods=40)
        >>> tickers = ["A", "B", "C", "D"]
        >>> price = pd.DataFrame(1.0, index=dates, columns=tickers)
        >>> price["D"] = np.linspace(1, 2, 40)  # D 一路上涨
        >>> idx = pd.MultiIndex.from_product([[dates[0]], tickers], names=["date", "ticker"])
        >>> score = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)
        >>> quantile_forward_returns(score, price, freq=20, n_quantiles=2)  # doctest: +SKIP
        bucket
        0    0.0
        1    0.5
        Name: forward_return, dtype: float64
    """
    rb_dates = rebalance_dates(price.index, freq) # type: ignore
    rb_price = price.reindex(rb_dates)
    forward_return = (rb_price.shift(-1) / rb_price - 1.0).stack()
    forward_return.index = forward_return.index.set_names([date_level, "ticker"])

    def _bucket(s: pd.Series) -> pd.Series:
        if len(s) < n_quantiles:
            return pd.Series(dtype=float)
        return pd.qcut(s, n_quantiles, labels=False, duplicates="drop")

    bucket = score.dropna().groupby(level=date_level, group_keys=False).apply(_bucket)

    aligned = pd.concat(
        [bucket.rename("bucket"), forward_return.rename("forward_return")], # type: ignore
        axis=1,
        join="inner",
    ).dropna()

    return aligned.groupby("bucket")["forward_return"].mean()


def vol_neutral_quantile_forward_returns(
    score: pd.Series,
    vol: pd.Series,
    price: pd.DataFrame,
    *,
    freq: int,
    n_vol_groups: int,
    n_quantiles: int,
    date_level: str = "date",
) -> pd.Series:
    """跟 quantile_forward_returns 做的事一样(按 score 分组算前瞻收益),
    但先按 vol 分层、层内再按 score 切 n_quantiles 组, 再把同一个 score
    分位编号在所有层上的前瞻收益汇总求平均 —— 对应
    portfolio.sizing.vol_neutral_quantile_long_short 的分组方式, 用来
    检验"波动率中性化"之后, score 对前瞻收益的单调预测力是不是变干净了
    (对照组是 quantile_forward_returns 那张图里的 U 形)。

    Args:
        score: 调仓日的横截面分数, [date, ticker] MultiIndex。
        vol: 调仓日的横截面波动率(切层依据), 索引结构跟 score 一致。
        price: 收盘价宽表, index 是完整交易日历, columns 是 ticker。
        freq: 调仓间隔(交易日数), 需要和生成 score 时用的一致。
        n_vol_groups: 波动率分几层, 需要和 vol_neutral_quantile_long_short
            用的保持一致才有可比性。
        n_quantiles: 每层内部按 score 分几组, 同上需要保持一致。
        date_level: 索引里日期所在的 level 名, 默认 "date"。

    Returns:
        pd.Series, 索引是分位数编号(0 是打分最低组, n_quantiles - 1 是
        最高组), 值是该组在所有层、所有调仓日上的平均前瞻收益。
    """
    rb_dates = rebalance_dates(price.index, freq)  # type: ignore
    rb_price = price.reindex(rb_dates)
    forward_return = (rb_price.shift(-1) / rb_price - 1.0).stack()
    forward_return.index = forward_return.index.set_names([date_level, "ticker"])

    df = pd.concat(
        [score.rename("score"), vol.rename("vol")], axis=1, join="inner"
    ).dropna()

    def _bucket(day: pd.DataFrame) -> pd.Series:
        if len(day) < n_vol_groups * n_quantiles:
            return pd.Series(dtype=float)

        vol_bucket = pd.qcut(day["vol"], n_vol_groups, labels=False, duplicates="drop")
        out = pd.Series(index=day.index, dtype=float)

        for layer_idx in day.groupby(vol_bucket).groups.values():
            layer_score = day.loc[layer_idx, "score"]
            if len(layer_score) < n_quantiles:
                continue
            out.loc[layer_score.index] = pd.qcut(
                layer_score, n_quantiles, labels=False, duplicates="drop"
            )
        return out

    # 原因见 portfolio.sizing.vol_neutral_quantile_long_short 里的同款注释:
    # DataFrameGroupBy.apply 在只有一个分组时会把返回的同长 Series 误判成
    # "一行数据", 索引乱掉, 手动循环+concat 规避。
    pieces = [b for _, day in df.groupby(level=date_level) if not (b := _bucket(day)).empty]
    bucket = pd.concat(pieces).rename("bucket") if pieces else pd.Series(dtype=float, name="bucket")

    aligned = pd.concat(
        [bucket, forward_return.rename("forward_return")], axis=1, join="inner"
    ).dropna()

    return aligned.groupby("bucket")["forward_return"].mean()
