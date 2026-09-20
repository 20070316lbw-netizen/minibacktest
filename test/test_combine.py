from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from minibacktest.signal.combine import combine_scores


def test_combine_scores_equal_weights():
    dates = pd.date_range("2024-01-01", periods=1, freq="D")
    tickers = ["A", "B", "C"]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    z = pd.DataFrame(
        {
            "f1": [-1.0, 0.0, 1.0],
            "f2": [1.0, 2.0, 3.0],
        },
        index=idx,
    )
    score = combine_scores(z)
    # 等权平均: A: (-1+1)/2 = 0, B: (0+2)/2 = 1, C: (1+3)/2 = 2
    expected = pd.Series([0.0, 1.0, 2.0], index=idx)
    pd.testing.assert_series_equal(score, expected, check_names=False)


def test_combine_scores_custom_weights():
    dates = pd.date_range("2024-01-01", periods=1, freq="D")
    tickers = ["A", "B"]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    z = pd.DataFrame(
        {
            "f1": [1.0, 2.0],
            "f2": [3.0, 4.0],
        },
        index=idx,
    )
    # 权重 0.7 和 0.3
    weights = {"f1": 0.7, "f2": 0.3}
    score = combine_scores(z, weights=weights)
    # A: 1*0.7 + 3*0.3 = 1.6
    # B: 2*0.7 + 4*0.3 = 2.6
    expected = pd.Series([1.6, 2.6], index=idx)
    pd.testing.assert_series_equal(score, expected, check_names=False)


def test_combine_scores_missing_factor_in_z_raises_key_error():
    dates = pd.date_range("2024-01-01", periods=1, freq="D")
    idx = pd.MultiIndex.from_product([dates, ["A"]], names=["date", "ticker"])
    z = pd.DataFrame({"f1": [1.0]}, index=idx)
    with pytest.raises(KeyError, match="weights 里面有 标准化后没有的因子"):
        combine_scores(z, weights={"f1": 0.5, "unknown_factor": 0.5})


def test_combine_scores_unspecified_factor_weight_zero():
    dates = pd.date_range("2024-01-01", periods=1, freq="D")
    idx = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["date", "ticker"])
    z = pd.DataFrame(
        {
            "f1": [1.0, 2.0],
            "f2": [10.0, 20.0],
        },
        index=idx,
    )
    # 仅指定 f1，f2 权重为 0
    score = combine_scores(z, weights={"f1": 1.0})
    expected = pd.Series([1.0, 2.0], index=idx)
    pd.testing.assert_series_equal(score, expected, check_names=False)


def test_combine_scores_partial_nans():
    dates = pd.date_range("2024-01-01", periods=1, freq="D")
    idx = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["date", "ticker"])
    # B 的 f2 为 NaN
    z = pd.DataFrame(
        {
            "f1": [1.0, 2.0],
            "f2": [3.0, np.nan],
        },
        index=idx,
    )
    score = combine_scores(z, weights={"f1": 2.0, "f2": 2.0})
    # A: (1*2 + 3*2) / 4 = 2.0
    # B: (2*2) / 2 = 2.0 (只用有值的 f1，分母是 |2|=2)
    expected = pd.Series([2.0, 2.0], index=idx)
    pd.testing.assert_series_equal(score, expected, check_names=False)


def test_combine_scores_all_nans_row_returns_nan():
    dates = pd.date_range("2024-01-01", periods=1, freq="D")
    idx = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["date", "ticker"])
    z = pd.DataFrame(
        {
            "f1": [np.nan, 2.0],
            "f2": [np.nan, 4.0],
        },
        index=idx,
    )
    score = combine_scores(z)
    assert pd.isna(score.loc[(dates[0], "A")])
    assert score.loc[(dates[0], "B")] == 3.0
