from __future__ import annotations

import numpy as np
import pandas as pd

from minibacktest.base import Result
from minibacktest.engine import run_backtest


def test_run_backtest_end_to_end(sample_price: pd.DataFrame):
    # 构造调仓日目标权重
    dates = sample_price.index
    rb_dates = dates[::20]
    tickers = sample_price.columns

    # 简单多空权重: 第一只做多 1.0, 第二只做空 -1.0
    records = []
    for d in rb_dates:
        records.append({"date": d, "ticker": tickers[0], "weight": 1.0})
        records.append({"date": d, "ticker": tickers[1], "weight": -1.0})

    target_w_df = pd.DataFrame(records).set_index(["date", "ticker"])
    target_weight = target_w_df["weight"]

    res = run_backtest(
        price=sample_price,
        target_weight=target_weight,
        freq=20,
        initial_capital=100_000.0,
    )

    assert isinstance(res, Result)
    assert res.start == dates[0].to_pydatetime()
    assert res.end == dates[-1].to_pydatetime()
    assert res.duration == dates[-1] - dates[0]
    assert 0 <= res.exposure_time_pct <= 100
    assert res.equity_final > 0
    assert res.equity_peak >= res.equity_final or np.isclose(res.equity_peak, res.equity_final)
    assert res.strategy == "quantile_long_short"

    # 验证净值曲线
    assert isinstance(res.equity_curve, pd.DataFrame)
    assert "nav" in res.equity_curve.columns
    assert "benchmark_nav" in res.equity_curve.columns
    assert len(res.equity_curve) == len(sample_price)
    assert np.isclose(res.equity_curve["nav"].iloc[0], 100_000.0)


def test_run_backtest_shift_1_no_lookahead():
    # 测试今天收盘前发生的价格变化不会提前影响今天的持仓决策
    dates = pd.bdate_range("2024-01-01", periods=5)
    tickers = ["A", "B"]
    price = pd.DataFrame({"A": [10.0, 11.0, 12.0, 13.0, 14.0], "B": [20.0, 20.0, 20.0, 20.0, 20.0]}, index=dates)

    # 仅在 Day 0 给出目标权重
    idx = pd.MultiIndex.from_product([[dates[0]], tickers], names=["date", "ticker"])
    target_weight = pd.Series([1.0, 0.0], index=idx)

    res = run_backtest(price=price, target_weight=target_weight, freq=5, initial_capital=100.0)
    nav = res.equity_curve["nav"]

    # 第 0 天因为 shift(1)，持仓尚未生效 (w_hold = 0)，净值仍为初始值 100
    assert np.isclose(nav.iloc[0], 100.0)
    # 第 1 天持仓生效 (持有 A: 100% 仓位)，A 价格从 10 涨到 11 (+10%)，净值变为 110
    assert np.isclose(nav.iloc[1], 110.0)
