from __future__ import annotations

import io
from datetime import UTC, datetime

import pandas as pd
from rich.console import Console

from minibacktest.base import Result
from minibacktest.report import _format_value, print_result


def test_format_value():
    assert _format_value(1234567.891) == "1,234,567.89"
    assert _format_value(float("nan")) == "-"
    assert _format_value(pd.Timedelta(days=5, hours=12)) == "5.5 天"
    assert _format_value(datetime(2024, 1, 15, 10, 30, tzinfo=UTC)) == "2024-01-15"
    assert _format_value("some_text") == "some_text"
    assert _format_value(42) == "42"


def test_print_result(sample_result: Result):
    buf = io.StringIO()
    console = Console(file=buf, width=120)
    print_result(sample_result, console=console)
    output = buf.getvalue()

    # 验证关键组标题与字段在终端输出中正常渲染
    assert "时间" in output
    assert "收益与资金曲线" in output
    assert "风险调整后收益" in output
    assert "回撤" in output
    assert "Sharpe" in output
    assert "最大回撤" in output
