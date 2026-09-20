from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # 强制无头模式, 适合 CI 环境
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from minibacktest.figure_engine import (
    plot_drawdown,
    plot_equity_curve,
    plot_monthly_heatmap,
    plot_quantile_returns,
    plot_rolling_sharpe,
    plot_tearsheet,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_plot_equity_curve(sample_equity_curve: pd.DataFrame):
    ax = plot_equity_curve(sample_equity_curve)
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == "净值曲线"

    # 测试传入现有 ax
    _fig, custom_ax = plt.subplots()
    returned_ax = plot_equity_curve(sample_equity_curve, ax=custom_ax)
    assert returned_ax is custom_ax


def test_plot_drawdown(sample_nav: pd.Series):
    ax = plot_drawdown(sample_nav)
    assert isinstance(ax, plt.Axes)
    assert "回撤曲线" in ax.get_title()


def test_plot_monthly_heatmap():
    # 构造跨越几个月的净值序列
    dates = pd.date_range("2023-01-01", "2024-03-31", freq="B")
    np.random.seed(42)
    daily_ret = np.random.normal(0.0005, 0.01, len(dates))
    nav = pd.Series(100.0 * np.cumprod(1 + daily_ret), index=dates)

    ax = plot_monthly_heatmap(nav)
    assert isinstance(ax, plt.Axes)
    assert "月度收益热力图" in ax.get_title()


def test_plot_rolling_sharpe(sample_nav: pd.Series):
    ax = plot_rolling_sharpe(sample_nav, window=5)
    assert isinstance(ax, plt.Axes)
    assert "滚动 Sharpe" in ax.get_title()


def test_plot_quantile_returns():
    q_ret = pd.Series([-0.05, 0.01, 0.08], index=[0, 1, 2])
    ax = plot_quantile_returns(q_ret)
    assert isinstance(ax, plt.Axes)
    assert "分位数平均前瞻收益" in ax.get_title()


def test_plot_tearsheet(sample_equity_curve: pd.DataFrame):
    # 1. 不带 quantile_returns
    fig1 = plot_tearsheet(sample_equity_curve)
    assert isinstance(fig1, plt.Figure)

    # 2. 带 quantile_returns
    q_ret = pd.Series([-0.02, 0.03, 0.05], index=[0, 1, 2])
    fig2 = plot_tearsheet(sample_equity_curve, quantile_returns=q_ret)
    assert isinstance(fig2, plt.Figure)
