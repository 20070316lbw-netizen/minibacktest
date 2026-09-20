from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from minibacktest.portfolio.sizing import quantile_long_short


def test_quantile_long_short_invalid_n_quantiles():
    d = pd.Timestamp("2024-01-01")
    s = pd.Series([1.0, 2.0], index=pd.MultiIndex.from_product([[d], ["A", "B"]]))
    with pytest.raises(ValueError, match="n_quantiles 至少为2"):
        quantile_long_short(score=s, n_quantiles=1)


def test_quantile_long_short_basic():
    d = pd.Timestamp("2024-01-01")
    tickers = ["A", "B", "C", "D", "E", "F"]
    scores = [5.0, 3.0, 1.0, -1.0, -3.0, -5.0]
    idx = pd.MultiIndex.from_product([[d], tickers], names=["date", "ticker"])
    s = pd.Series(scores, index=idx)

    w = quantile_long_short(score=s, n_quantiles=3)

    # 3分位:
    # 最高组: A, B (权重各自 1/2 = 0.5)
    # 中间组: C, D (权重 0.0)
    # 最低组: E, F (权重各自 -1/2 = -0.5)
    assert w.loc[(d, "A")] == 0.5
    assert w.loc[(d, "B")] == 0.5
    assert w.loc[(d, "C")] == 0.0
    assert w.loc[(d, "D")] == 0.0
    assert w.loc[(d, "E")] == -0.5
    assert w.loc[(d, "F")] == -0.5

    # 验证多头和为 1, 空头和为 -1, 总权重和为 0 (市场中性)
    assert np.isclose(w[w > 0].sum(), 1.0)
    assert np.isclose(w[w < 0].sum(), -1.0)
    assert np.isclose(w.sum(), 0.0)


def test_quantile_long_short_insufficient_stocks():
    d = pd.Timestamp("2024-01-01")
    tickers = ["A", "B"]
    idx = pd.MultiIndex.from_product([[d], tickers], names=["date", "ticker"])
    s = pd.Series([1.0, 2.0], index=idx)

    # 只有2只股票，但要求分5组 -> 当天不产生权重输出
    w = quantile_long_short(score=s, n_quantiles=5)
    assert len(w) == 0


def test_quantile_long_short_multiple_dates():
    dates = pd.date_range("2024-01-01", periods=2, freq="D")
    tickers = ["A", "B", "C", "D"]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    s = pd.Series([10.0, 5.0, -5.0, -10.0, -10.0, -5.0, 5.0, 10.0], index=idx)

    w = quantile_long_short(score=s, n_quantiles=2)

    for d in dates:
        day_w = w.xs(d, level="date")
        assert np.isclose(day_w[day_w > 0].sum(), 1.0)
        assert np.isclose(day_w[day_w < 0].sum(), -1.0)


def test_quantile_long_short_handles_nans():
    d = pd.Timestamp("2024-01-01")
    tickers = ["A", "B", "C", "D", "E"]
    idx = pd.MultiIndex.from_product([[d], tickers], names=["date", "ticker"])
    # C 是 NaN，有效股票 4 只
    s = pd.Series([4.0, 3.0, np.nan, 2.0, 1.0], index=idx)

    w = quantile_long_short(score=s, n_quantiles=2)
    # NaN 应该被剔除，不出现在输出中或被 drop
    assert (d, "C") not in w.index
    assert len(w) == 4
    assert np.isclose(w[w > 0].sum(), 1.0)
    assert np.isclose(w[w < 0].sum(), -1.0)
