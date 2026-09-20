from __future__ import annotations

import numpy as np
import pandas as pd

from minibacktest.evaluation.equity import build_nav, buy_and_hold_nav


def test_build_nav_basic():
    r = pd.Series([0.0, 0.1, -0.05])
    nav = build_nav(r, initial_capital=100.0)
    # 100 * 1 = 100
    # 100 * 1.1 = 110
    # 110 * 0.95 = 104.5
    expected = pd.Series([100.0, 110.0, 104.5])
    pd.testing.assert_series_equal(nav.round(2), expected)


def test_build_nav_default_initial_capital():
    r = pd.Series([0.0, 0.05, 0.05])
    nav = build_nav(r)
    expected = pd.Series([1.0, 1.05, 1.05 * 1.05])
    pd.testing.assert_series_equal(nav, expected)


def test_buy_and_hold_nav_basic():
    dates = pd.date_range("2024-01-01", periods=3)
    # A 涨 10% 然后涨 10%
    # B 跌 10% 然后平
    # Day 0: 初始，收益为 0
    # Day 1: A 收益 +0.1, B 收益 -0.1 -> 平均 0.0
    # Day 2: A 收益 +0.1, B 收益 0.0  -> 平均 +0.05
    price = pd.DataFrame(
        {
            "A": [100.0, 110.0, 121.0],
            "B": [100.0, 90.0, 90.0],
        },
        index=dates,
    )

    nav = buy_and_hold_nav(price, initial_capital=1000.0)
    assert len(nav) == 3
    assert np.isclose(nav.iloc[0], 1000.0)
    assert np.isclose(nav.iloc[1], 1000.0 * (1 + 0.0))
    assert np.isclose(nav.iloc[2], 1000.0 * (1 + 0.0) * (1 + 0.05))
