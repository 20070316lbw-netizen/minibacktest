"""用 rich 把 base.Result 打印成分组表格, 替代 print(result) 那行挤在一起、
终端里自动换行换得很乱的 dataclass repr。
"""

from __future__ import annotations

import datetime as dt
import math

import pandas as pd
from rich.console import Console
from rich.table import Table

from minibacktest.base import Result

# 分组跟 base.py 里的字段分区(Timing / Equity / Returns / 风险调整后收益 /
# Drawdown)保持一致, 字段名 -> 中文标签。
_SECTIONS: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "时间",
        [
            ("start", "起始日期"),
            ("end", "截止日期"),
            ("duration", "总时长"),
            ("exposure_time_pct", "持仓时间占比 [%]"),
        ],
    ),
    (
        "收益与资金曲线",
        [
            ("equity_final", "期末净值 [$]"),
            ("equity_peak", "净值峰值 [$]"),
            ("return_pct", "总收益率 [%]"),
            ("buy_and_hold_return_pct", "买入持有收益率 [%]"),
            ("return_ann_pct", "年化收益率 [%]"),
            ("volatility_ann_pct", "年化波动率 [%]"),
            ("cagr_pct", "复合年增长率 [%]"),
        ],
    ),
    (
        "风险调整后收益",
        [
            ("sharpe_ratio", "Sharpe"),
            ("sortino_ratio", "Sortino"),
            ("calmar_ratio", "Calmar"),
            ("alpha_pct", "Alpha [%]"),
            ("beta", "Beta"),
        ],
    ),
    (
        "回撤",
        [
            ("max_drawdown_pct", "最大回撤 [%]"),
            ("avg_drawdown_pct", "平均回撤 [%]"),
            ("max_drawdown_duration", "最大回撤持续时间"),
            ("avg_drawdown_duration", "平均回撤持续时间"),
        ],
    ),
    (
        "交易成本",
        [
            ("commission_bps", "单边佣金 [bps]"),
            ("slippage_bps", "单边滑点 [bps]"),
            ("turnover_ann_pct", "年化换手率 [%]"),
            ("total_cost_pct", "费用总拖累 [%]"),
        ],
    ),
]


def _format_value(value: object) -> str:
    """数字保留两位小数、加千分位; NaN 显示成 "-"; 日期只留年月日(价格数据
    本来就没有日内时间); 时长折算成天数(小数), 不逐字段打印 days/时分秒。
    """
    if isinstance(value, float):
        if math.isnan(value):
            return "-"
        return f"{value:,.2f}"
    if isinstance(value, pd.Timedelta):
        return f"{value.total_seconds() / 86400:,.1f} 天"
    if isinstance(value, dt.datetime):
        return value.strftime("%Y-%m-%d")
    return str(value)


def print_result(result: Result, *, console: Console | None = None) -> None:
    """把 Result 打印成分组表格, 每组一张小表, 字段名换成中文标签。

    Args:
        result: engine.run_backtest 的输出。
        console: 复用一个已有的 rich Console(比如已经配置过主题), 默认新建一个。
    """
    console = console or Console()

    for i, (section_title, fields) in enumerate(_SECTIONS):
        if i > 0:
            console.print()  # 每组之间空一行, 不然标题跟上一组最后一行贴在一起
        table = Table(title=section_title, show_header=False, box=None, padding=(0, 2, 0, 0))
        table.add_column(style="bold cyan", no_wrap=True)
        table.add_column()
        for field, label in fields:
            table.add_row(label, _format_value(getattr(result, field)))
        console.print(table)
