"""月度收益热力图。"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_monthly_heatmap(nav: pd.Series, *, ax: plt.Axes | None = None) -> plt.Axes:
    """画月度收益热力图, 行是年份, 列是月份。

    Args:
        nav: 逐日净值。
        ax: 画在哪个 Axes 上, 默认新建一张图。

    Returns:
        画完的 Axes。
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    monthly_nav = nav.resample("ME").last()
    monthly_return = monthly_nav.pct_change()
    monthly_return.iloc[0] = monthly_nav.iloc[0] / nav.iloc[0] - 1.0

    table = monthly_return.to_frame("ret")
    table["year"] = table.index.year
    table["month"] = table.index.month
    pivot = (table.pivot(index="year", columns="month", values="ret") * 100).reindex(
        columns=range(1, 13)
    )

    im = ax.imshow(pivot.to_numpy(), cmap="RdYlGn", aspect="auto", vmin=-10, vmax=10)
    ax.set_xticks(range(12), [f"{m}月" for m in range(1, 13)])
    ax.set_yticks(range(len(pivot.index)), pivot.index)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.to_numpy()[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=8)

    ax.set_title("月度收益热力图 [%]")
    plt.colorbar(im, ax=ax, fraction=0.03)
    return ax
