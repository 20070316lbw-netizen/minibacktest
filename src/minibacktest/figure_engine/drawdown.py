"""回撤曲线图。"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from minibacktest.evaluation.drawdown import drawdown_series, max_drawdown


def plot_drawdown(nav: pd.Series, *, ax: plt.Axes | None = None) -> plt.Axes:
    """画回撤曲线, 并标出最大回撤发生的时点(净值最低点)。

    Args:
        nav: 逐日净值(比如 equity_curve["nav"])。
        ax: 画在哪个 Axes 上, 默认新建一张图。

    Returns:
        画完的 Axes。
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3))

    dd = drawdown_series(nav) * 100
    ax.fill_between(dd.index, dd.to_numpy(), 0, color="firebrick", alpha=0.4)
    ax.plot(dd.index, dd.to_numpy(), color="firebrick", linewidth=0.8)
    ax.axvline(dd.idxmin(), color="black", linestyle="--", linewidth=0.8, alpha=0.6)

    ax.set_title(f"回撤曲线(最大回撤 {max_drawdown(nav) * 100:.1f}%)")
    ax.set_ylabel("回撤 [%]")
    ax.grid(alpha=0.3)
    return ax
