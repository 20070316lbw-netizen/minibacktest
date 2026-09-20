from __future__ import annotations

import math

import pandas as pd

from minibacktest.evaluation.chores import exposure_time_pct


def test_exposure_time_pct_normal():
    # 4 天，前 3 天有持仓，最后 1 天空仓
    w_hold = pd.DataFrame({
        "A": [0.5, 0.0, 0.2, 0.0],
        "B": [0.0, -0.5, 0.0, 0.0],
    })
    # 3/4 = 75%
    pct = exposure_time_pct(w_hold)
    assert math.isclose(pct, 75.0)


def test_exposure_time_pct_all_held():
    w_hold = pd.DataFrame({"A": [0.5, 0.5], "B": [-0.5, -0.5]})
    assert math.isclose(exposure_time_pct(w_hold), 100.0)


def test_exposure_time_pct_all_cash():
    w_hold = pd.DataFrame({"A": [0.0, 0.0], "B": [0.0, 0.0]})
    assert math.isclose(exposure_time_pct(w_hold), 0.0)


def test_exposure_time_pct_empty():
    w_hold = pd.DataFrame()
    assert math.isnan(exposure_time_pct(w_hold))
