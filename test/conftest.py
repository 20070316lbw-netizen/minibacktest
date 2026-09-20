from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from minibacktest.base import Result


@pytest.fixture
def sample_dates() -> pd.DatetimeIndex:
    return pd.bdate_range("2024-01-01", periods=60)


@pytest.fixture
def sample_tickers() -> list[str]:
    return ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]


@pytest.fixture
def sample_price(sample_dates: pd.DatetimeIndex, sample_tickers: list[str]) -> pd.DataFrame:
    np.random.seed(42)
    n_days = len(sample_dates)
    n_tickers = len(sample_tickers)
    daily_returns = np.random.normal(0.0005, 0.015, size=(n_days, n_tickers))
    daily_returns[0, :] = 0.0
    base_prices = np.array([150.0, 300.0, 400.0, 120.0, 130.0])
    price_matrix = base_prices * np.cumprod(1.0 + daily_returns, axis=0)
    return pd.DataFrame(price_matrix, index=sample_dates, columns=sample_tickers)


@pytest.fixture
def sample_factor_panel(sample_dates: pd.DatetimeIndex, sample_tickers: list[str]) -> pd.DataFrame:
    np.random.seed(123)
    idx = pd.MultiIndex.from_product([sample_dates, sample_tickers], names=["date", "ticker"])
    f1 = np.random.normal(0, 1, size=len(idx))
    f2 = np.random.normal(2, 0.5, size=len(idx))
    return pd.DataFrame({"f1": f1, "f2": f2}, index=idx)


@pytest.fixture
def sample_score(sample_dates: pd.DatetimeIndex, sample_tickers: list[str]) -> pd.Series:
    np.random.seed(99)
    rb_dates = sample_dates[::21]
    idx = pd.MultiIndex.from_product([rb_dates, sample_tickers], names=["date", "ticker"])
    scores = np.random.normal(0, 1, size=len(idx))
    return pd.Series(scores, index=idx, name="score")


@pytest.fixture
def sample_nav(sample_dates: pd.DatetimeIndex) -> pd.Series:
    values = [
        100.0, 102.0, 105.0, 103.0, 101.0, 98.0, 99.0, 103.0, 106.0, 108.0,
        107.0, 105.0, 104.0, 106.0, 109.0, 112.0, 110.0, 108.0, 111.0, 115.0,
    ]
    dates = sample_dates[:len(values)]
    return pd.Series(values, index=dates, name="nav")


@pytest.fixture
def sample_equity_curve(sample_dates: pd.DatetimeIndex) -> pd.DataFrame:
    np.random.seed(7)
    n = 60
    r_strat = np.random.normal(0.001, 0.01, n)
    r_bench = np.random.normal(0.0005, 0.01, n)
    nav_strat = 100.0 * np.cumprod(1 + r_strat)
    nav_bench = 100.0 * np.cumprod(1 + r_bench)
    return pd.DataFrame({"nav": nav_strat, "benchmark_nav": nav_bench}, index=sample_dates)


@pytest.fixture
def sample_result(sample_equity_curve: pd.DataFrame) -> Result:
    dates = sample_equity_curve.index
    return Result(
        start=dates[0].to_pydatetime(),
        end=dates[-1].to_pydatetime(),
        duration=dates[-1] - dates[0],
        exposure_time_pct=95.0,
        equity_final=115.0,
        equity_peak=120.0,
        return_pct=15.0,
        buy_and_hold_return_pct=10.0,
        return_ann_pct=25.0,
        volatility_ann_pct=15.0,
        cagr_pct=22.0,
        sharpe_ratio=1.5,
        sortino_ratio=2.0,
        calmar_ratio=1.8,
        alpha_pct=5.0,
        beta=0.85,
        max_drawdown_pct=-8.5,
        avg_drawdown_pct=-2.5,
        max_drawdown_duration=timedelta(days=15),
        avg_drawdown_duration=timedelta(days=5),
        strategy="test_strategy",
        equity_curve=sample_equity_curve,
    )
