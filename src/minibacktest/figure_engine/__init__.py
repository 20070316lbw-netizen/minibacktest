"""回测完成之后的结果产出图表, 用 matplotlib 画:

    plot_equity_curve      策略净值与基准净值对比, 两条线共用一个起点(equity.py)
    plot_drawdown           回撤曲线, 标出最大回撤发生的时点(drawdown.py)
    plot_monthly_heatmap    月度收益热力图, 行是年份, 列是月份(heatmap.py)
    plot_rolling_sharpe     滚动 Sharpe, 固定窗口, 样本不足留空(risk_adjusted_ratios.py)
    plot_quantile_returns   分位数平均前瞻收益柱状图(五分位单调性检验),
                             数据来自 evaluation.quantile.quantile_forward_returns(quantile.py)
    plot_tearsheet          把以上几张拼成一张总览图(tearsheet.py)

原来这里还有一张"单笔交易净盈亏分布", 是从 backtesting.py / pyfolio 那类事件驱动
策略的报表抄来的 —— 这个项目是截面多空组合, 只有逐日权重矩阵, 没有"一笔交易"这个
概念, 所以换成了分位数单调性检验, 这是判断截面因子有没有效更直接的图。
"""

from __future__ import annotations

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# matplotlib 默认字体(DejaVu Sans)不认中文, 所有汉字都会变成方块; 这里按
# 系统里能找到的第一个中文字体设置, 找不到就静默跳过(退化成方块, 不报错)。
# 放在 __init__.py 里是因为要在导入任何 figure_engine 子模块之前生效。
_CJK_FONT_CANDIDATES = ["Heiti SC", "PingFang SC", "Arial Unicode MS", "Songti SC", "SimHei"]
_available = {f.name for f in fm.fontManager.ttflist}
_cjk_font = next((f for f in _CJK_FONT_CANDIDATES if f in _available), None)
if _cjk_font:
    plt.rcParams["font.sans-serif"] = [_cjk_font]
plt.rcParams["axes.unicode_minus"] = False

from minibacktest.figure_engine.drawdown import plot_drawdown
from minibacktest.figure_engine.equity import plot_equity_curve
from minibacktest.figure_engine.heatmap import plot_monthly_heatmap
from minibacktest.figure_engine.quantile import plot_quantile_returns
from minibacktest.figure_engine.risk_adjusted_ratios import plot_rolling_sharpe
from minibacktest.figure_engine.tearsheet import plot_tearsheet

__all__ = [
    "plot_drawdown",
    "plot_equity_curve",
    "plot_monthly_heatmap",
    "plot_quantile_returns",
    "plot_rolling_sharpe",
    "plot_tearsheet",
]
