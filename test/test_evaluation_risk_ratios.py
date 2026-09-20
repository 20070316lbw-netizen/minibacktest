from __future__ import annotations

import math

import numpy as np
import pandas as pd

from minibacktest.evaluation.risk_adjusted_ratios import (
    calmar_ratio,
    sharpe_ratio,
    sortino_ratio,
)


def test_sharpe_ratio_normal():
    dates = pd.date_range("2024-01-01", periods=5)
    nav = pd.Series([1.0, 1.02, 1.01, 1.05, 1.03], index=dates)
    ret = nav.pct_change().dropna()
    expected = (ret.mean() / ret.std()) * np.sqrt(252)

    sr = sharpe_ratio(nav, periods_per_year=252)
    assert math.isclose(sr, expected, rel_tol=1e-5)


def test_sharpe_ratio_edge_cases():
    # 样本不足 (< 2)
    nav_single = pd.Series([1.0, 1.02])
    assert math.isnan(sharpe_ratio(nav_single, periods_per_year=252))

    # 标准差为 0 (价格完全不波动)
    nav_flat = pd.Series([100.0, 100.0, 100.0])
    assert math.isnan(sharpe_ratio(nav_flat, periods_per_year=252))


def test_sortino_ratio_normal():
    dates = pd.date_range("2024-01-01", periods=5)
    nav = pd.Series([1.0, 1.02, 0.99, 1.05, 1.03], index=dates)
    ret = nav.pct_change().dropna()
    downside = ret[ret < 0]
    expected = (ret.mean() / downside.std()) * np.sqrt(252)

    sr = sortino_ratio(nav, periods_per_year=252)
    assert math.isclose(sr, expected, rel_tol=1e-5)


def test_sortino_ratio_no_downside_returns_nan():
    # 只有正收益，没有下行波动
    dates = pd.date_range("2024-01-01", periods=5)
    nav = pd.Series([1.0, 1.01, 1.03, 1.06, 1.10], index=dates)
    assert math.isnan(sortino_ratio(nav, periods_per_year=252))


def test_calmar_ratio_normal():
    dates = pd.date_range("2024-01-01", periods=5)
    nav = pd.Series([1.0, 1.2, 0.9, 1.1, 1.3], index=dates)
    ret = nav.pct_change().dropna()
    ann_return = (1 + ret.mean()) ** 252 - 1
    mdd = -0.25  # 0.9 / 1.2 - 1 = -0.25
    expected = ann_return / abs(mdd)

    cr = calmar_ratio(nav, periods_per_year=252)
    assert math.isclose(cr, expected, rel_tol=1e-5)


def test_calmar_ratio_zero_drawdown_returns_nan():
    # 只有上涨，最大回撤为 0
    dates = pd.date_range("2024-01-01", periods=4)
    nav = pd.Series([1.0, 1.1, 1.2, 1.3], index=dates)
    assert math.isnan(calmar_ratio(nav, periods_per_year=252))
