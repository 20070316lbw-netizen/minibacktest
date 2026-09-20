"""收益率相关指标: 总收益、年化收益、CAGR、年化波动率、Alpha/Beta"""

from __future__ import annotations

import numpy as np
import pandas as pd


def total_return(nav: pd.Series) -> float:
    """总收益率 = 期末净值 / 期初净值 - 1。

    Example:
        >>> nav = pd.Series([1.0, 1.1, 1.21])
        >>> round(total_return(nav), 4)
        0.21
    """
    return float(nav.iloc[-1] / nav.iloc[0] - 1.0)


def annualized_return(nav: pd.Series, periods_per_year: float) -> float:
    """年化收益率, 按日频收益率的均值复利折算(不是简单总收益/年数)。

    Args:
        nav: 逐日净值。
        periods_per_year: nav 一年有多少个观测点(日频通常是 252)。

    Returns:
        年化收益率; 没有有效收益率样本时返回 NaN。
    """
    ret = nav.pct_change().dropna()
    if len(ret) == 0:
        return float("nan")
    return float((1 + ret.mean()) ** periods_per_year - 1)


def cagr(nav: pd.Series) -> float:
    """复合年增长率, 按实际经过的自然日折算(365.25 天/年)。

    Args:
        nav: 逐日净值, index 必须是日期。

    Returns:
        CAGR; 经过的自然日数 <= 0 时返回 NaN。
    """
    n_days = (nav.index[-1] - nav.index[0]).days
    if n_days <= 0:
        return float("nan")
    years = n_days / 365.25
    return float((nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1)


def annualized_volatility(nav: pd.Series, periods_per_year: float) -> float:
    """年化波动率 = 日收益标准差 * sqrt(年化周期数)。

    Args:
        nav: 逐日净值。
        periods_per_year: nav 一年有多少个观测点(日频通常是 252)。

    Returns:
        年化波动率; 有效收益率样本少于 2 个时返回 NaN。
    """
    ret = nav.pct_change().dropna()
    if len(ret) < 2:
        return float("nan")
    return float(ret.std() * np.sqrt(periods_per_year))


def alpha_beta(
    strategy_nav: pd.Series,
    benchmark_nav: pd.Series,
    periods_per_year: float,
) -> tuple[float, float]:
    """用简单线性回归(策略日收益 ~ 基准日收益)算 Alpha / Beta。

    Beta 是回归斜率(策略相对基准的系统性风险敞口), Alpha 是剔除 Beta
    后的日均超额收益, 按 periods_per_year 复利年化成百分比。

    Args:
        strategy_nav: 策略逐日净值。
        benchmark_nav: 基准(买入持有)逐日净值, 索引需要能跟 strategy_nav
            对齐(取交集)。
        periods_per_year: 年化周期数, 日频通常 252。

    Returns:
        (alpha_pct, beta); 两者共同的有效观测点少于 2 个、或基准收益方差
        为 0 时都返回 NaN。
    """
    strat_ret = strategy_nav.pct_change().dropna()
    bench_ret = benchmark_nav.pct_change().dropna()
    aligned = pd.concat(
        [strat_ret.rename("strategy"), bench_ret.rename("benchmark")],
        axis=1,
        join="inner",
    ).dropna()
    if len(aligned) < 2:
        return float("nan"), float("nan")

    x = aligned["benchmark"].to_numpy()
    y = aligned["strategy"].to_numpy()
    var_x = x.var(ddof=1)
    if var_x == 0:
        return float("nan"), float("nan")

    beta = float(np.cov(x, y, ddof=1)[0, 1] / var_x)
    alpha_daily = float(y.mean() - beta * x.mean())
    alpha_pct = (1 + alpha_daily) ** periods_per_year - 1
    return float(alpha_pct * 100), beta
