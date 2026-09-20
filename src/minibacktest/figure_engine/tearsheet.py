"""把各张单图拼成一张总览图。"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from minibacktest.figure_engine.drawdown import plot_drawdown
from minibacktest.figure_engine.equity import plot_equity_curve
from minibacktest.figure_engine.heatmap import plot_monthly_heatmap
from minibacktest.figure_engine.quantile import plot_quantile_returns
from minibacktest.figure_engine.risk_adjusted_ratios import plot_rolling_sharpe


def plot_tearsheet(
    equity_curve: pd.DataFrame,
    *,
    quantile_returns: pd.Series | None = None,
    rolling_window: int = 126,
    periods_per_year: float = 252,
) -> plt.Figure:
    """把净值对比、回撤、滚动 Sharpe、月度热力图(、分位数单调性检验)拼成一张总览图。

    Args:
        equity_curve: base.Result.equity_curve。
        quantile_returns: evaluation.quantile.quantile_forward_returns 的输出,
            不传就不画分位数那张图(比如只关心净值表现, 不关心因子本身)。
        rolling_window: 滚动 Sharpe 的窗口长度。
        periods_per_year: 年化用的每年观测点数。

    Returns:
        画完的 Figure。
    """
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(3, 2)
    nav = equity_curve["nav"]

    plot_equity_curve(equity_curve, ax=fig.add_subplot(gs[0, :]))
    plot_drawdown(nav, ax=fig.add_subplot(gs[1, 0]))
    plot_rolling_sharpe(
        nav,
        window=rolling_window,
        periods_per_year=periods_per_year,
        ax=fig.add_subplot(gs[1, 1]),
    )
    plot_monthly_heatmap(nav, ax=fig.add_subplot(gs[2, 0]))

    if quantile_returns is not None:
        plot_quantile_returns(quantile_returns, ax=fig.add_subplot(gs[2, 1]))

    fig.tight_layout()
    return fig
