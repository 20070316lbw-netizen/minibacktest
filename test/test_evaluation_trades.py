from __future__ import annotations

import math

import pandas as pd

from minibacktest.evaluation.trades import extract_trades, trade_stats


def test_extract_trades_empty():
    w_hold = pd.DataFrame({"A": [0.0, 0.0], "B": [0.0, 0.0]})
    price = pd.DataFrame({"A": [10.0, 11.0], "B": [20.0, 21.0]})
    trades = extract_trades(w_hold, price)
    assert trades.empty
    assert list(trades.columns) == [
        "ticker", "entry_date", "exit_date", "direction", "duration", "return_pct"
    ]


def test_extract_trades_long_and_short():
    dates = pd.date_range("2024-01-01", periods=4)
    # A 多头持有 2 天 (01-01 到 01-02), 价格 100 -> 110 (+10%)
    # B 空头持有 2 天 (01-02 到 01-03), 价格 50 -> 45 (做空收益: -1 * (45/50 - 1) * 100 = +10%)
    w_hold = pd.DataFrame(
        {
            "A": [0.5, 0.5, 0.0, 0.0],
            "B": [0.0, -0.5, -0.5, 0.0],
        },
        index=dates,
    )
    price = pd.DataFrame(
        {
            "A": [100.0, 110.0, 120.0, 130.0],
            "B": [50.0, 50.0, 45.0, 40.0],
        },
        index=dates,
    )

    trades = extract_trades(w_hold, price)
    assert len(trades) == 2

    trade_a = trades[trades["ticker"] == "A"].iloc[0]
    assert trade_a["direction"] == 1
    assert trade_a["entry_date"] == dates[0]
    assert trade_a["exit_date"] == dates[1]
    assert trade_a["duration"] == pd.Timedelta(days=1)
    assert math.isclose(trade_a["return_pct"], 10.0)

    trade_b = trades[trades["ticker"] == "B"].iloc[0]
    assert trade_b["direction"] == -1
    assert trade_b["entry_date"] == dates[1]
    assert trade_b["exit_date"] == dates[2]
    assert math.isclose(trade_b["return_pct"], 10.0)


def test_trade_stats_empty():
    trades = pd.DataFrame()
    stats = trade_stats(trades)
    assert stats["n_trades"] == 0
    assert math.isnan(stats["win_rate_pct"])
    assert stats["max_trade_duration"] == pd.Timedelta(0)


def test_trade_stats_normal():
    dates = pd.date_range("2024-01-01", periods=5)
    trades = pd.DataFrame([
        {
            "ticker": "A",
            "entry_date": dates[0],
            "exit_date": dates[1],
            "direction": 1,
            "duration": pd.Timedelta(days=1),
            "return_pct": 10.0,  # 赢
        },
        {
            "ticker": "B",
            "entry_date": dates[1],
            "exit_date": dates[2],
            "direction": 1,
            "duration": pd.Timedelta(days=1),
            "return_pct": -5.0,  # 输
        },
    ])

    stats = trade_stats(trades)
    assert stats["n_trades"] == 2
    assert math.isclose(stats["win_rate_pct"], 50.0)
    assert math.isclose(stats["best_trade_pct"], 10.0)
    assert math.isclose(stats["worst_trade_pct"], -5.0)
    assert math.isclose(stats["avg_trade_pct"], 2.5)
    # profit factor = 10 / 5 = 2.0
    assert math.isclose(stats["profit_factor"], 2.0)
    assert not math.isnan(stats["sqn"])
    assert not math.isnan(stats["kelly_criterion"])
