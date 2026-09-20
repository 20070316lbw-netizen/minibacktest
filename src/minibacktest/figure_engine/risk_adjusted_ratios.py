"""滚动风险调整收益图(目前只有滚动 Sharpe)。"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_rolling_sharpe(
    nav: pd.Series,
    *,
    window: int = 126,
    periods_per_year: float = 252,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """画滚动 Sharpe, 固定窗口, 样本不足的开头自然留空(NaN 不画)。

    Args:
        nav: 逐日净值。
        window: 滚动窗口长度(交易日数), 默认 126(约半年)。
        periods_per_year: 年化用的每年观测点数, 日频默认 252。
        ax: 画在哪个 Axes 上, 默认新建一张图。

    Returns:
        画完的 Axes。
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3))

    ret = nav.pct_change()
    rolling_sharpe = (
        ret.rolling(window).mean() / ret.rolling(window).std() * periods_per_year**0.5
    )

    ax.plot(rolling_sharpe.index, rolling_sharpe.to_numpy(), linewidth=1.0)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_title(f"滚动 Sharpe(窗口 {window} 个交易日)")
    ax.grid(alpha=0.3)
    return ax
