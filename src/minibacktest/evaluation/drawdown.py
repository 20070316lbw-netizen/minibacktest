"""回撤相关指标: 最大回撤、回撤序列、平均回撤深度、回撤持续时间。"""

from __future__ import annotations

import pandas as pd


def max_drawdown(nav: pd.Series) -> float:
    """算最大回撤(<= 0 的数, 比如 -0.23 代表最大回撤 23%)。

    Args:
        nav: 逐日净值。

    Returns:
        最大回撤, <= 0 的 float。

    Example:
        >>> nav = pd.Series([1.0, 1.2, 0.9, 1.1])
        >>> max_drawdown(nav)
        -0.25
    """
    running_max = nav.cummax()
    drawdown = nav / running_max - 1.0
    return float(drawdown.min())


def drawdown_series(nav: pd.Series) -> pd.Series:
    """逐日回撤序列(<= 0), 用于画回撤曲线, 也是 avg_drawdown / 持续时间的基础。

    Args:
        nav: 逐日净值。

    Returns:
        与 nav 同索引的回撤序列, 新高当天为 0。
    """
    running_max = nav.cummax()
    return nav / running_max - 1.0


def avg_drawdown(nav: pd.Series) -> float:
    """算平均回撤深度: 把回撤序列切成一段段"跌破新高到重新创出新高"的
    区间, 对每段区间里的最低点取平均。

    Args:
        nav: 逐日净值。

    Returns:
        平均回撤深度, <= 0 的 float; 全程都在创新高(没发生过回撤)时返回 0.0。

    Example:
        >>> nav = pd.Series([1.0, 1.2, 0.9, 1.1, 1.3, 1.0])
        >>> round(avg_drawdown(nav), 4)
        -0.225
    """
    dd = drawdown_series(nav)
    in_dd = dd < 0
    if not in_dd.any():
        return 0.0
    episode = (~in_dd).cumsum()
    troughs = dd[in_dd].groupby(episode[in_dd]).min()
    return float(troughs.mean())


def drawdown_durations(nav: pd.Series) -> tuple[pd.Timedelta, pd.Timedelta]:
    """算最大 / 平均回撤持续时间(从上一次净值新高, 到重新创出新高为止)。

    做法: 对每一天算"距离上一个新高过了多久", 只在处于回撤中的那些天里看
    这个时长, 按"属于哪一次新高之后"分组, 组内的最大值就是那一次回撤从
    发生到恢复(或数据结束)经过的总时长。

    Args:
        nav: 逐日净值。

    Returns:
        (max_duration, avg_duration); 全程都在创新高时都返回 pd.Timedelta(0)。

    Example:
        >>> nav = pd.Series(
        ...     [1.0, 1.2, 0.9, 1.1, 1.3],
        ...     index=pd.date_range("2024-01-01", periods=5),
        ... )
        >>> max_dur, avg_dur = drawdown_durations(nav)
        >>> max_dur
        Timedelta('2 days 00:00:00')
    """
    running_max = nav.cummax()
    is_peak = nav >= running_max
    if is_peak.all():
        zero = pd.Timedelta(0)
        return zero, zero

    peak_date = nav.index.to_series().where(is_peak).ffill()
    since_peak = nav.index.to_series() - peak_date

    in_dd = ~is_peak
    per_episode_max = since_peak[in_dd].groupby(peak_date[in_dd]).max()
    return per_episode_max.max(), per_episode_max.mean()