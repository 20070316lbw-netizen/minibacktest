"""策略净值与基准净值对比图。"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_equity_curve(equity_curve: pd.DataFrame, *, ax: plt.Axes | None = None) -> plt.Axes:
    """画策略净值与基准净值对比。

    Args:
        equity_curve: base.Result.equity_curve, 至少要有 nav / benchmark_nav 两列。
        ax: 画在哪个 Axes 上, 默认新建一张图。

    Returns:
        画完的 Axes。
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    ax.plot(equity_curve.index, equity_curve["nav"], label="Strategy", linewidth=1.5)
    ax.plot(
        equity_curve.index,
        equity_curve["benchmark_nav"],
        label="Benchmark (Equal-Weight Buy & Hold)",
        linewidth=1.0,
        alpha=0.7,
    )
    ax.set_title("Equity Curve")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    return ax
