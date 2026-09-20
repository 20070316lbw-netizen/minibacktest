from __future__ import annotations

import pandas as pd
import pytest

from minibacktest.rebalance import rebalance_block, rebalance_dates


def test_rebalance_dates_basic():
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    rb = rebalance_dates(dates, freq=3)
    expected = pd.DatetimeIndex([
        "2024-01-01",
        "2024-01-04",
        "2024-01-07",
        "2024-01-10",
    ])
    pd.testing.assert_index_equal(rb, expected)


def test_rebalance_dates_unsorted_and_duplicates():
    dates = pd.DatetimeIndex([
        "2024-01-05",
        "2024-01-01",
        "2024-01-03",
        "2024-01-01",
        "2024-01-05",
        "2024-01-02",
        "2024-01-04",
    ])
    rb = rebalance_dates(dates, freq=2)
    expected = pd.DatetimeIndex(["2024-01-01", "2024-01-03", "2024-01-05"])
    pd.testing.assert_index_equal(rb, expected)


def test_rebalance_dates_invalid_freq():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    with pytest.raises(ValueError, match="freq 必须是正整数"):
        rebalance_dates(dates, freq=0)
    with pytest.raises(ValueError, match="freq 必须是正整数"):
        rebalance_dates(dates, freq=-2)


def test_rebalance_dates_single_date():
    dates = pd.date_range("2024-01-01", periods=1, freq="D")
    rb = rebalance_dates(dates, freq=5)
    assert len(rb) == 1
    assert rb[0] == pd.Timestamp("2024-01-01")


def test_rebalance_block_basic():
    dates = pd.date_range("2024-01-01", periods=7, freq="D")
    block = rebalance_block(dates, freq=3)
    assert isinstance(block, pd.Series)
    assert block.name == "rebalance_date"
    assert len(block) == 7

    # 2024-01-01 到 01-03 映射到 01-01
    assert block.loc["2024-01-01"] == pd.Timestamp("2024-01-01")
    assert block.loc["2024-01-02"] == pd.Timestamp("2024-01-01")
    assert block.loc["2024-01-03"] == pd.Timestamp("2024-01-01")

    # 2024-01-04 到 01-06 映射到 01-04
    assert block.loc["2024-01-04"] == pd.Timestamp("2024-01-04")
    assert block.loc["2024-01-05"] == pd.Timestamp("2024-01-04")
    assert block.loc["2024-01-06"] == pd.Timestamp("2024-01-04")

    # 2024-01-07 映射到 01-07
    assert block.loc["2024-01-07"] == pd.Timestamp("2024-01-07")


def test_rebalance_block_freq_larger_than_dates():
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    block = rebalance_block(dates, freq=10)
    # 所有日期都应该映射到第一个日期
    assert (block == pd.Timestamp("2024-01-01")).all()


def test_rebalance_block_freq_1():
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    block = rebalance_block(dates, freq=1)
    # 每天都是自己的调仓日
    for d in dates:
        assert block.loc[d] == d
