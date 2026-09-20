"""分位数单调性检验图, 对应 evaluation.quantile.quantile_forward_returns 的数据。"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_quantile_returns(
    quantile_returns: pd.Series, *, ax: plt.Axes | None = None
) -> plt.Axes:
    """画分位数平均前瞻收益柱状图(五分位单调性检验)。

    Args:
        quantile_returns: evaluation.quantile.quantile_forward_returns 的输出,
            索引是分位数编号(0 是最低组), 值是平均前瞻收益。
        ax: 画在哪个 Axes 上, 默认新建一张图。

    Returns:
        画完的 Axes。
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    q = quantile_returns.sort_index()
    colors = ["firebrick" if v < 0 else "seagreen" for v in q.to_numpy()]
    ax.bar([f"Q{int(i) + 1}" for i in q.index], q.to_numpy() * 100, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("分位数平均前瞻收益(单调性检验)")
    ax.set_ylabel("平均前瞻收益 [%]")
    ax.grid(alpha=0.3, axis="y")
    return ax
