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

import matplotlib.pyplot as plt

# 图表文字全部用英文(标题/坐标轴/图例), 不依赖任何中文字体——原来按系统里
# 能找到的中文字体名字("Heiti SC"/"PingFang SC"...)做检测+回退, 换一台没装
# 这些字体的机器(比如这次这个云端沙箱)就会静默退化成方块字, 治标不治本。
# axes.unicode_minus 这行跟中文无关, 单独修的是负号在某些字体下画不出来的
# 老问题, 留着无害。
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
