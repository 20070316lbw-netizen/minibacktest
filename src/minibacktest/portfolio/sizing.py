"""仓位构造

目前设计的是一个
    按分数把每个调仓日的横截面分成 n_quantiles 组, 最高组等权做多、
    最低组等权做空, 中间组权重为 0, 多空两腿各自内部等权且总敞口相等
    (多头权重和为 +1, 空头权重和为 -1)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from minibacktest.risk.volatility import realized_volatility


def _check_buffer(buffer: float, n_quantiles: int) -> None:
    """缓冲带宽度校验: 多头出场线 (1 - 1/n - buffer) 必须高于空头出场线
    (1/n + buffer), 即 buffer < 0.5 - 1/n; buffer=0 总是合法。"""
    max_buffer = 0.5 - 1.0 / n_quantiles
    if buffer < 0 or (buffer > 0 and buffer >= max_buffer):
        raise ValueError(
            f"buffer 必须在 [0, {max_buffer:.4f}) 之间, 否则多空两条出场线会交叉"
        )


def _exit_quantiles(buffer: float, n_quantiles: int) -> tuple[float, float]:
    """多头/空头留任阈值对应的截面分位点。用跟 qcut 内部相同的
    np.linspace 算分位, 避免 1 - 1/3 这类写法跟 linspace 差最后一位浮点,
    导致 buffer=0 时边界判断跟 qcut 不一致。"""
    edges = np.linspace(0, 1, n_quantiles + 1)
    return edges[-2] - buffer, edges[1] + buffer


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


def buffered_quantile_long_short(
        *,
        score       : pd.Series,
        n_quantiles : int,
        buffer      : float,
        date_level  : str = "date",
) -> pd.Series:
    """带缓冲带的分位数多空: 进场和出场用两条不同的线, 用来压换手。

    - 进场: 跟 quantile_long_short 一字不差, 当期 qcut 最高组做多、最低组做空
    - 留任: 上一期已经在多头里的票, 只要本期分数仍高于截面的
      (1 - 1/n - buffer) 分位点就继续持有; 空头对称, 分数不高于
      (1/n + buffer) 分位点就继续持有
    - 本期多头 = 进场集合 ∪ 留任集合, 组内等权归一到 +1, 空头归一到 -1

    所以每条腿的只数是浮动的, 且只会 >= 当期最高/最低组的只数; 缓冲区
    (比如 60%~80%)只对上期就持有的老票生效, 新票必须够到最高组才能进。

    留任阈值用的是分数的截面分位点(Series.quantile, 线性插值), 跟 qcut
    内部切边界的方法相同, 所以 buffer=0 时留任集合一定落在进场集合里,
    结果与 quantile_long_short 逐元素相等(test_sizing 里有回归测试)。

    因为本期持仓依赖上期持仓, 这个函数是按调仓日顺序循环带状态的, 所以
    score 必须一次性给全所有调仓日(Backtester 本来就是这么调用 sizing_fn 的)。

    Args:
        score: 调仓日的横截面分数, [date, ticker] MultiIndex。
        n_quantiles: 分组数, 至少 2。
        buffer: 缓冲带宽度(截面分位数单位), 0 表示不设缓冲带。必须小于
            0.5 - 1/n_quantiles, 否则多头出场线会低于空头出场线(五分位
            下要求 buffer < 0.3)。
        date_level: 索引里日期所在的 level 名, 默认 "date"。

    Returns:
        组合权重, 格式与 quantile_long_short 相同: 与 score(去掉 NaN 后)
        同索引, 中间的票为 0。某天有效股票数不够分组时该天没有输出, 并且
        持仓状态清空(视为当天空仓, 下一期所有票都按新票处理)。

    Raises:
        ValueError: n_quantiles 小于 2, 或 buffer 超出允许范围。
    """
    if n_quantiles < 2:
        raise ValueError("n_quantiles 至少为2, 才可以分出多头与空头")
    _check_buffer(buffer, n_quantiles)
    long_exit_q, short_exit_q = _exit_quantiles(buffer, n_quantiles)

    score = score.dropna()
    empty = pd.Index([])
    prev_long, prev_short = empty, empty
    pieces = []

    for _, day in score.groupby(level=date_level, sort=True):
        s = day.droplevel(date_level)
        if len(s) < n_quantiles:
            prev_long, prev_short = empty, empty
            continue

        bucket = pd.qcut(s, n_quantiles, labels=False, duplicates="drop")
        long_entry = (bucket == bucket.max()).to_numpy()
        short_entry = (bucket == bucket.min()).to_numpy()

        keep_long = s.index.isin(prev_long) & (s > s.quantile(long_exit_q)).to_numpy()
        keep_short = s.index.isin(prev_short) & (s <= s.quantile(short_exit_q)).to_numpy()

        # 正常情况下(buffer 合法)留任集合不会跟对面腿的进场集合重叠, 这里
        # 再兜底一次, 防止大量同分导致 qcut 丢边界时出现一只票两边都在
        long_mask = long_entry | (keep_long & ~short_entry)
        short_mask = short_entry | (keep_short & ~long_entry)

        # 赋值顺序跟 quantile_long_short 一致(先多后空), 极端情况下
        # 全部同分、最高组 == 最低组时, 行为也跟旧函数一样
        w = pd.Series(0.0, index=day.index)
        w[long_mask] = 1.0 / long_mask.sum()
        w[short_mask] = -1.0 / short_mask.sum()
        pieces.append(w)

        prev_long = s.index[w.to_numpy() > 0]
        prev_short = s.index[w.to_numpy() < 0]

    if not pieces:
        return pd.Series(dtype=float)
    return pd.concat(pieces)


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


def buffered_vol_neutral_quantile_long_short(
        *,
        score        : pd.Series,
        vol          : pd.Series,
        n_vol_groups : int,
        n_quantiles  : int,
        buffer       : float,
        date_level   : str = "date",
) -> pd.Series:
    """vol_neutral_quantile_long_short 的缓冲带版本: 波动率分层不变,
    每一层内部用 buffered_quantile_long_short 同样的"进场/留任两条线"。

    - 进场: 跟 vol_neutral_quantile_long_short 一字不差, 层内 qcut
      最高组做多、最低组做空
    - 留任: 上期在多头里的票, 只要在**本期所在的那一层**里, 分数仍高于
      层内 (1 - 1/n - buffer) 分位点就继续持有; 空头对称
    - 所有层的多头合并后等权归一到 +1, 空头归一到 -1

    注意留任看的是本期的层, 不是上期的层: 一只票这个月波动率升高、换到
    了高波动层, 就要跟高波动层的票比分数。这样保证每一期的多空两腿在
    各波动率层里的分布依然是对称的, 这正是做波动率中性化的初衷; 如果
    按上期的层判断, 缓冲带留下的老票会慢慢把波动率暴露带偏。

    本期某一层票数不够 n_quantiles 时, 该层跳过(跟原函数一样), 层内的
    老票也不留任。buffer=0 时结果与 vol_neutral_quantile_long_short
    逐元素相等。

    Args:
        score: 调仓日的横截面分数, [date, ticker] MultiIndex。
        vol: 调仓日的横截面波动率, 索引结构跟 score 一致, 两者取交集。
        n_vol_groups: 波动率分几层。
        n_quantiles: 每层内部按 score 分几组, 至少 2。
        buffer: 缓冲带宽度, 约束同 buffered_quantile_long_short
            (buffer < 0.5 - 1/n_quantiles)。
        date_level: 索引里日期所在的 level 名, 默认 "date"。

    Returns:
        组合权重, 格式与 vol_neutral_quantile_long_short 相同(不产生 0 行)。
        某天有效股票数不够分层分组时该天没有输出, 持仓状态清空。

    Raises:
        ValueError: n_quantiles 小于 2, 或 buffer 超出允许范围。
    """
    if n_quantiles < 2:
        raise ValueError("n_quantiles 至少为2, 才可以分出多头与空头")
    _check_buffer(buffer, n_quantiles)
    long_exit_q, short_exit_q = _exit_quantiles(buffer, n_quantiles)

    df = pd.concat(
        [score.rename("score"), vol.rename("vol")], axis=1, join="inner"
    ).dropna()

    empty = pd.Index([])
    prev_long, prev_short = empty, empty
    pieces = []

    for _, day in df.groupby(level=date_level, sort=True):
        if len(day) < n_vol_groups * n_quantiles:
            prev_long, prev_short = empty, empty
            continue

        tickers = day.index.droplevel(date_level)
        held_long = tickers.isin(prev_long)
        held_short = tickers.isin(prev_short)

        vol_bucket = pd.qcut(day["vol"], n_vol_groups, labels=False, duplicates="drop")
        long_mask = np.zeros(len(day), dtype=bool)
        short_mask = np.zeros(len(day), dtype=bool)

        for layer in np.unique(vol_bucket.to_numpy()):
            pos = np.flatnonzero(vol_bucket.to_numpy() == layer)
            layer_score = day["score"].iloc[pos]
            if len(layer_score) < n_quantiles:
                continue

            bucket = pd.qcut(layer_score, n_quantiles, labels=False, duplicates="drop")
            long_entry = (bucket == bucket.max()).to_numpy()
            short_entry = (bucket == bucket.min()).to_numpy()
            keep_long = held_long[pos] & (layer_score > layer_score.quantile(long_exit_q)).to_numpy()
            keep_short = held_short[pos] & (layer_score <= layer_score.quantile(short_exit_q)).to_numpy()

            long_mask[pos] |= long_entry | (keep_long & ~short_entry)
            short_mask[pos] |= short_entry | (keep_short & ~long_entry)

        w = pd.Series(0.0, index=day.index)
        if long_mask.sum():
            w[long_mask] = 1.0 / long_mask.sum()
        if short_mask.sum():
            w[short_mask] = -1.0 / short_mask.sum()
        pieces.append(w)

        prev_long = tickers[w.to_numpy() > 0]
        prev_short = tickers[w.to_numpy() < 0]

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


def make_buffered_quantile_sizer(*, n_quantiles: int, buffer: float):
    """跟 make_quantile_sizer 同一种接口, 内部包的是
    buffered_quantile_long_short(带缓冲带的分位数多空), 用法:

        Backtester(..., sizing_fn=make_buffered_quantile_sizer(n_quantiles=5, buffer=0.1))

    buffer=0 时行为与 make_quantile_sizer 完全相同。price 参数用不上,
    只是为了保持统一签名。

    Args:
        n_quantiles: 分组数, 透传给 buffered_quantile_long_short。
        buffer: 缓冲带宽度, 透传给 buffered_quantile_long_short。

    Returns:
        Callable[[pd.Series, pd.DataFrame], pd.Series]。
    """
    # 构造时就校验一次参数, 不用等到跑完因子才报错
    if n_quantiles >= 2:
        _check_buffer(buffer, n_quantiles)

    def _sizer(score: pd.Series, price: pd.DataFrame) -> pd.Series:
        return buffered_quantile_long_short(
            score=score, n_quantiles=n_quantiles, buffer=buffer
        )

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


def make_buffered_vol_neutral_sizer(
    *, n_vol_groups: int, n_quantiles: int, buffer: float, vol_window: int = 21
):
    """跟 make_vol_neutral_sizer 同一种接口, 内部包的是
    buffered_vol_neutral_quantile_long_short(波动率分层 + 层内缓冲带):

        Backtester(..., sizing_fn=make_buffered_vol_neutral_sizer(
            n_vol_groups=5, n_quantiles=5, buffer=0.1))

    波动率的算法跟 make_vol_neutral_sizer 完全一样, buffer=0 时两者结果
    相同。

    Args:
        n_vol_groups: 波动率分几层。
        n_quantiles: 每层内部按 score 分几组。
        buffer: 缓冲带宽度, 约束同 buffered_quantile_long_short。
        vol_window: 算已实现波动率用的滚动窗口(交易日数), 默认 21。

    Returns:
        Callable[[pd.Series, pd.DataFrame], pd.Series]。
    """
    if n_quantiles >= 2:
        _check_buffer(buffer, n_quantiles)

    def _sizer(score: pd.Series, price: pd.DataFrame) -> pd.Series:
        vol = realized_volatility(price, window=vol_window)
        vol = vol[vol.index.isin(score.index)]
        return buffered_vol_neutral_quantile_long_short(
            score=score, vol=vol, n_vol_groups=n_vol_groups,
            n_quantiles=n_quantiles, buffer=buffer,
        )

    return _sizer
