from __future__ import annotations

import numpy as np
import pandas as pd

from minibacktest.evaluation.quantile import quantile_forward_returns


def test_quantile_forward_returns_basic():
    dates = pd.bdate_range("2024-01-01", periods=40)
    tickers = ["A", "B", "C", "D"]
    # 构造确定性价格：A 平稳, B 跌, C 微涨, D 暴涨
    price = pd.DataFrame(100.0, index=dates, columns=tickers)
    price["A"] = 100.0
    price["B"] = np.linspace(100, 50, 40)
    price["C"] = np.linspace(100, 120, 40)
    price["D"] = np.linspace(100, 200, 40)

    # 调仓日: 0, 20
    # 在第 0 天给出分数: B 最低, A 次之, C 再次之, D 最高
    idx0 = pd.MultiIndex.from_product([[dates[0]], tickers], names=["date", "ticker"])
    score0 = pd.Series([2.0, 1.0, 3.0, 4.0], index=idx0)

    idx20 = pd.MultiIndex.from_product([[dates[20]], tickers], names=["date", "ticker"])
    score20 = pd.Series([2.0, 1.0, 3.0, 4.0], index=idx20)

    score = pd.concat([score0, score20])

    q_ret = quantile_forward_returns(score, price, freq=20, n_quantiles=2)

    assert isinstance(q_ret, pd.Series)
    assert len(q_ret) == 2
    # 分组 0 (打分较低组: A, B) 的前瞻收益应低于分组 1 (打分较高组: C, D)
    assert q_ret.loc[1] > q_ret.loc[0]


def test_quantile_forward_returns_insufficient_tickers():
    dates = pd.bdate_range("2024-01-01", periods=20)
    tickers = ["A", "B"]
    price = pd.DataFrame(100.0, index=dates, columns=tickers)

    idx = pd.MultiIndex.from_product([[dates[0]], tickers], names=["date", "ticker"])
    score = pd.Series([1.0, 2.0], index=idx)

    # 只有2只股票无法切成5组，返回空 Series
    q_ret = quantile_forward_returns(score, price, freq=10, n_quantiles=5)
    assert len(q_ret) == 0
