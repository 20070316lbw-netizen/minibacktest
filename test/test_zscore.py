from __future__ import annotations

import numpy as np
import pandas as pd

from minibacktest.zscore.zscore import zscore_by_date


def test_zscore_by_date_basic():
    dates = pd.date_range("2024-01-01", periods=2, freq="D")
    tickers = ["A", "B", "C"]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    factors = pd.DataFrame(
        {"f1": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0]},
        index=idx,
    )
    z = zscore_by_date(factors)

    # 每个截面均值为 0，标准差为 1
    for d in dates:
        xs = z.xs(d, level="date")["f1"]
        np.testing.assert_allclose(xs.mean(), 0.0, atol=1e-7)
        np.testing.assert_allclose(xs.std(), 1.0, atol=1e-7)

    # 验证具体数值: [1, 2, 3] 均值 2, 标准差 1 -> [-1, 0, 1]
    expected_day1 = np.array([-1.0, 0.0, 1.0])
    np.testing.assert_allclose(z.xs(dates[0], level="date")["f1"].to_numpy(), expected_day1)


def test_zscore_by_date_multiple_columns():
    dates = pd.date_range("2024-01-01", periods=1, freq="D")
    tickers = ["A", "B", "C", "D"]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    factors = pd.DataFrame(
        {
            "f1": [1.0, 2.0, 3.0, 4.0],
            "f2": [10.0, 30.0, 20.0, 40.0],
        },
        index=idx,
    )
    z = zscore_by_date(factors)
    assert set(z.columns) == {"f1", "f2"}
    assert z.shape == factors.shape
    np.testing.assert_allclose(z["f1"].mean(), 0.0, atol=1e-7)
    np.testing.assert_allclose(z["f1"].std(), 1.0, atol=1e-7)
    np.testing.assert_allclose(z["f2"].mean(), 0.0, atol=1e-7)
    np.testing.assert_allclose(z["f2"].std(), 1.0, atol=1e-7)


def test_zscore_by_date_with_nans():
    dates = pd.date_range("2024-01-01", periods=1, freq="D")
    tickers = ["A", "B", "C", "D"]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    # C 是 NaN，仅 A, B, D 计算 zscore
    factors = pd.DataFrame({"f1": [1.0, 2.0, np.nan, 3.0]}, index=idx)
    z = zscore_by_date(factors)

    # 保持 NaN 位置
    assert pd.isna(z.loc[(dates[0], "C"), "f1"])
    # 非 NaN 处标准差为 1, 均值为 0
    valid = z["f1"].dropna()
    assert len(valid) == 3
    np.testing.assert_allclose(valid.mean(), 0.0, atol=1e-7)
    np.testing.assert_allclose(valid.std(), 1.0, atol=1e-7)


def test_zscore_by_date_single_ticker_returns_nan():
    dates = pd.date_range("2024-01-01", periods=1, freq="D")
    tickers = ["A"]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    factors = pd.DataFrame({"f1": [5.0]}, index=idx)
    z = zscore_by_date(factors)
    # 单样本时 std 为 NaN，计算结果为 NaN
    assert pd.isna(z.iloc[0, 0])


def test_zscore_by_date_zero_variance_returns_nan():
    dates = pd.date_range("2024-01-01", periods=1, freq="D")
    tickers = ["A", "B", "C"]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    factors = pd.DataFrame({"f1": [5.0, 5.0, 5.0]}, index=idx)
    z = zscore_by_date(factors)
    # 标准差为 0 时，0 / 0 返回 NaN
    assert z["f1"].isna().all()


def test_zscore_by_date_custom_date_level():
    dates = pd.date_range("2024-01-01", periods=1, freq="D")
    tickers = ["A", "B", "C"]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["trade_date", "symbol"])
    factors = pd.DataFrame({"f1": [1.0, 2.0, 3.0]}, index=idx)
    z = zscore_by_date(factors, date_level="trade_date")
    np.testing.assert_allclose(z["f1"].mean(), 0.0, atol=1e-7)
