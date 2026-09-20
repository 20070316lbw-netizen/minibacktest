from __future__ import annotations

import math

import pandas as pd

from minibacktest.evaluation.drawdown import (
    avg_drawdown,
    drawdown_durations,
    drawdown_series,
    max_drawdown,
)


def test_max_drawdown_basic():
    # 峰值 1.2, 谷底 0.9 -> 回撤 = 0.9 / 1.2 - 1 = -0.25
    nav = pd.Series([1.0, 1.2, 0.9, 1.1])
    mdd = max_drawdown(nav)
    assert math.isclose(mdd, -0.25)


def test_drawdown_series():
    nav = pd.Series([100.0, 120.0, 90.0, 120.0, 150.0, 120.0])
    # cummax: 100, 120, 120, 120, 150, 150
    # dd: 0, 0, -0.25, 0, 0, -0.20
    dd = drawdown_series(nav)
    expected = pd.Series([0.0, 0.0, -0.25, 0.0, 0.0, -0.20])
    pd.testing.assert_series_equal(dd, expected, check_names=False)


def test_avg_drawdown():
    # 两个回撤周期:
    # 1. 峰值 1.2, 最低 0.9 -> -0.25
    # 2. 峰值 1.3, 最低 1.0 -> -0.230769
    nav = pd.Series([1.0, 1.2, 0.9, 1.1, 1.3, 1.0])
    avg_dd = avg_drawdown(nav)
    expected = (-0.25 + (1.0 / 1.3 - 1.0)) / 2.0
    assert math.isclose(avg_dd, expected, rel_tol=1e-5)


def test_drawdown_durations_basic():
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    # 01-01: 100
    # 01-02: 120 (新高)
    # 01-03: 90  (回撤中，距 01-02 1 天)
    # 01-04: 110 (回撤中，距 01-02 2 天)
    # 01-05: 130 (创新高，回撤结束，该 episode 最长持续 2 天)
    # 01-06: 125 (新回撤，距 01-05 1 天，未恢复)
    nav = pd.Series([100.0, 120.0, 90.0, 110.0, 130.0, 125.0], index=dates)
    max_dur, avg_dur = drawdown_durations(nav)
    assert max_dur == pd.Timedelta(days=2)
    # 两个 episode: 2 天 和 1 天 -> 均值 1.5 天
    assert avg_dur == pd.Timedelta(days=1.5)


def test_drawdown_when_monotonically_increasing():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    nav = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0], index=dates)

    assert max_drawdown(nav) == 0.0
    assert avg_drawdown(nav) == 0.0
    max_dur, avg_dur = drawdown_durations(nav)
    assert max_dur == pd.Timedelta(0)
    assert avg_dur == pd.Timedelta(0)
