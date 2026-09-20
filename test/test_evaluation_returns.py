from __future__ import annotations

import math

import numpy as np
import pandas as pd

from minibacktest.evaluation.returns import (
    alpha_beta,
    annualized_return,
    annualized_volatility,
    cagr,
    total_return,
)


def test_total_return():
    nav = pd.Series([100.0, 110.0, 120.0, 150.0])
    assert math.isclose(total_return(nav), 0.5)

    nav_down = pd.Series([100.0, 80.0])
    assert math.isclose(total_return(nav_down), -0.2)


def test_annualized_return_normal():
    # 252 交易日，每日收益恒定 0.001
    r = 0.001
    nav = pd.Series(100.0 * (1 + r) ** np.arange(253))
    ann_ret = annualized_return(nav, periods_per_year=252)
    expected = (1 + r) ** 252 - 1
    assert math.isclose(ann_ret, expected, rel_tol=1e-5)


def test_annualized_return_empty_or_single():
    nav = pd.Series([100.0])
    assert math.isnan(annualized_return(nav, periods_per_year=252))


def test_cagr_normal():
    dates = pd.date_range("2020-01-01", "2022-01-01")  # 731 天 (约 2 年)
    # 净值从 100 翻倍到 200
    nav = pd.Series(np.linspace(100, 200, len(dates)), index=dates)
    res = cagr(nav)
    years = (dates[-1] - dates[0]).days / 365.25
    expected = (200.0 / 100.0) ** (1 / years) - 1
    assert math.isclose(res, expected, rel_tol=1e-5)


def test_cagr_invalid_duration():
    dates = pd.date_range("2024-01-01", periods=1)
    nav = pd.Series([100.0], index=dates)
    assert math.isnan(cagr(nav))


def test_annualized_volatility_normal():
    # 收益率交替 +0.01 和 -0.01
    rets = np.array([0.01, -0.01, 0.01, -0.01])
    nav = pd.Series(100.0 * np.cumprod(1 + np.insert(rets, 0, 0.0)))
    vol = annualized_volatility(nav, periods_per_year=252)
    expected = pd.Series(rets).std() * np.sqrt(252)
    assert math.isclose(vol, expected, rel_tol=1e-5)


def test_annualized_volatility_few_samples():
    nav = pd.Series([100.0, 101.0])  # 只有 1 个收益率样本
    assert math.isnan(annualized_volatility(nav, periods_per_year=252))


def test_alpha_beta_normal():
    dates = pd.date_range("2024-01-01", periods=100)
    np.random.seed(123)
    bench_ret = np.random.normal(0.0005, 0.01, 100)
    # 策略严格是 1.5 * 基准 + 0.0002 alpha
    strat_ret = 1.5 * bench_ret + 0.0002

    strat_nav = pd.Series(100.0 * np.cumprod(1 + strat_ret), index=dates)
    bench_nav = pd.Series(100.0 * np.cumprod(1 + bench_ret), index=dates)

    alpha_pct, beta = alpha_beta(strat_nav, bench_nav, periods_per_year=252)
    assert math.isclose(beta, 1.5, rel_tol=1e-3)
    expected_alpha_pct = ((1 + 0.0002) ** 252 - 1) * 100
    assert math.isclose(alpha_pct, expected_alpha_pct, rel_tol=1e-2)


def test_alpha_beta_insufficient_samples_or_zero_variance():
    dates = pd.date_range("2024-01-01", periods=2)
    strat_nav = pd.Series([100.0, 101.0], index=dates)
    bench_nav = pd.Series([100.0, 101.0], index=dates)
    # 仅 1 个收益率样本 (< 2)
    a, b = alpha_beta(strat_nav, bench_nav, periods_per_year=252)
    assert math.isnan(a) and math.isnan(b)

    # 基准方差为 0
    dates5 = pd.date_range("2024-01-01", periods=5)
    strat_nav5 = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0], index=dates5)
    bench_nav5 = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0], index=dates5)  # 无波动
    a2, b2 = alpha_beta(strat_nav5, bench_nav5, periods_per_year=252)
    assert math.isnan(a2) and math.isnan(b2)
