"""仓位构造

目前设计的是一个
    按分数把每个调仓日的横截面分成 n_quantiles 组, 最高组等权做多、
    最低组等权做空, 中间组权重为 0, 多空两腿各自内部等权且总敞口相等
    (多头权重和为 +1, 空头权重和为 -1)
"""

from __future__ import annotations

import pandas as pd

from minibacktest.risk.volatility import realized_volatility


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
        >>> quantile_long_short(score=s, n_quantiles=3)
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
        w[short_mask] = -1.0 / short_mask.sum()

        return w
    

    if n_quantiles < 2:
        raise ValueError("n_quantiles 至少为2, 才可以分出多头与空头")

    score = score.dropna()

    return score.groupby(level=date_level, group_keys=False).apply(_weight)


def vol_neutral_quantile_long_short(
        *,
        score        : pd.Series,
        vol          : pd.Series,
        n_vol_groups : int,
        n_quantiles  : int,
        date_level   : str = "date",
) -> pd.Series:
    """跟 quantile_long_short 做的事一样(最高组做多、最低组做空), 只是
    多加一步"波动率分层": 每个调仓日先按 vol 把股票切成 n_vol_groups 层,
    再在每一层内部单独按 score 切 n_quantiles 组、选最高最低两组, 最后把
    所有层选出来的多头合并等权(权重和 +1), 空头同理(权重和 -1)。

    这是为了中性化"打分排在最极端的股票, 往往也是波动率最高的那批"这个
    跟 score 具体是什么因子无关的结构性混淆(参考 evaluation.quantile 的
    单调性检验图里那个 U 形 —— 前瞻收益两端偏高, 跟波动率分层高度吻合,
    而不是 score 真的有"两端都灵"的预测力)。vol 可以是
    risk.volatility.realized_volatility 的输出, 也可以是别的风险指标;
    这个函数从头到尾不关心 score 是哪个因子算出来的, 换因子只需要换
    score 这个参数, 这块分层逻辑不用动。

    Args:
        score: 调仓日的横截面分数, [date, ticker] MultiIndex。
        vol: 调仓日的横截面波动率(切层依据), 索引结构跟 score 一致;
            两者取交集(inner join)后再算, 所以不要求完全同索引。
        n_vol_groups: 波动率分几层。
        n_quantiles: 每层内部按 score 分几组, 至少要 2(才有多空两组)。
        date_level: 索引里日期所在的 level 名, 默认 "date"。

    Returns:
        组合权重, [date, ticker] MultiIndex(score/vol 取交集去 NaN 之后);
        多头股票权重为 +1/多头总数, 空头为 -1/空头总数, 不产生 0 行。

    Raises:
        ValueError: n_quantiles 小于 2。

    Example:
        >>> d = pd.Timestamp("2024-01-01")
        >>> tickers = ["A", "B", "C", "D", "E", "F", "G", "H"]
        >>> idx = pd.MultiIndex.from_product([[d], tickers], names=["date", "ticker"])
        >>> score = pd.Series([9, 7, 5, 3, 20, 16, 12, 4], index=idx)
        >>> vol = pd.Series([1, 1, 1, 1, 9, 9, 9, 9], index=idx)  # 低波动/高波动两层
        >>> w = vol_neutral_quantile_long_short(score=score, vol=vol, n_vol_groups=2, n_quantiles=2)
        >>> sorted(w[w > 0].index.get_level_values("ticker"))  # 每层各出前一半多头
        ['A', 'B', 'E', 'F']
        >>> sorted(w[w < 0].index.get_level_values("ticker"))  # 每层各出后一半空头
        ['C', 'D', 'G', 'H']
    """
    if n_quantiles < 2:
        raise ValueError("n_quantiles 至少为2, 才可以分出多头与空头")

    df = pd.concat(
        [score.rename("score"), vol.rename("vol")], axis=1, join="inner"
    ).dropna()

    def _pick(day: pd.DataFrame) -> pd.Series:
        if len(day) < n_vol_groups * n_quantiles:
            return pd.Series(dtype=float)

        vol_bucket = pd.qcut(day["vol"], n_vol_groups, labels=False, duplicates="drop")

        long_mask = pd.Series(False, index=day.index)
        short_mask = pd.Series(False, index=day.index)

        for layer_idx in day.groupby(vol_bucket).groups.values():
            layer_score = day.loc[layer_idx, "score"]
            if len(layer_score) < n_quantiles:
                continue
            bucket = pd.qcut(layer_score, n_quantiles, labels=False, duplicates="drop")
            long_mask.loc[layer_score.index[bucket == bucket.max()]] = True
            short_mask.loc[layer_score.index[bucket == bucket.min()]] = True

        w = pd.Series(0.0, index=day.index)
        if long_mask.sum():
            w[long_mask] = 1.0 / long_mask.sum()
        if short_mask.sum():
            w[short_mask] = -1.0 / short_mask.sum()
        return w

    # 不用 DataFrameGroupBy.apply: 当只有一个调仓日(一个分组)时, pandas
    # 会把"每组返回一个跟组内行数一样长的 Series"这种情况, 误判成"每组
    # 返回一行", 把 Series 自己的索引拍成了列, 索引全乱掉。手动循环+concat
    # 规避这个歧义, 行为不随分组数量变化。
    pieces = [w for _, day in df.groupby(level=date_level) if not (w := _pick(day)).empty]
    if not pieces:
        return pd.Series(dtype=float)
    weight = pd.concat(pieces)
    return weight[weight != 0.0]


def make_quantile_sizer(*, n_quantiles: int):
    """返回一个 (score, price) -> weight 的"仓位构造函数", 内部就是包了一层
    quantile_long_short, 可以直接传给 Backtester(sizing_fn=...)。

    Backtester 只认"给一份打分、给一份价格, 吐出一份权重"这个统一接口,
    不关心内部具体怎么分组 —— 这个函数和下面的
    make_vol_neutral_sizer 都是这个接口的其中一种实现, 以后想加新的分组
    方式(比如按行业中性化), 照这个模式再写一个 make_xxx_sizer 就行,
    不用改 Backtester 一行代码, 也不用在它的构造函数里加新的开关参数。

    price 参数在这里用不上(quantile_long_short 只需要 score), 保留它
    只是为了跟其他 sizer 保持同一个函数签名, 这样 Backtester 才能
    不管三七二十一统一调用 `sizing_fn(score, price)`。

    Args:
        n_quantiles: 分组数, 透传给 quantile_long_short。

    Returns:
        Callable[[pd.Series, pd.DataFrame], pd.Series]。
    """
    def _sizer(score: pd.Series, price: pd.DataFrame) -> pd.Series:
        return quantile_long_short(score=score, n_quantiles=n_quantiles)

    return _sizer


def make_vol_neutral_sizer(*, n_vol_groups: int, n_quantiles: int, vol_window: int = 21):
    """跟 make_quantile_sizer 是同一种"仓位构造函数"接口, 内部包的是
    vol_neutral_quantile_long_short —— 先按 vol_window 天已实现波动率把
    每个调仓日分成 n_vol_groups 层, 层内再按 score 分 n_quantiles 组选
    多空。

    这里才真正用到 price 参数: 波动率是从 price 现场算出来的(见
    risk.volatility.realized_volatility), 不像 score 那样由 Backtester
    提前算好传进来。

    Args:
        n_vol_groups: 波动率分几层。
        n_quantiles: 每层内部按 score 分几组。
        vol_window: 算已实现波动率用的滚动窗口(交易日数), 默认 21。

    Returns:
        Callable[[pd.Series, pd.DataFrame], pd.Series]。
    """
    def _sizer(score: pd.Series, price: pd.DataFrame) -> pd.Series:
        vol = realized_volatility(price, window=vol_window)
        vol = vol[vol.index.isin(score.index)]
        return vol_neutral_quantile_long_short(
            score=score, vol=vol, n_vol_groups=n_vol_groups, n_quantiles=n_quantiles
        )

    return _sizer
