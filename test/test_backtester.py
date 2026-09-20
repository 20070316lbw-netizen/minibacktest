from __future__ import annotations

from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from minibacktest.backtester import Backtester
from minibacktest.base import Result


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_backtester_init():
    bt = Backtester(
        tickers=["AAPL", "MSFT"],
        start="2024-01-01",
        factor_specs=[("momentum", {"window": 10})],
        freq=10,
        n_quantiles=2,
    )
    assert bt.tickers == ["AAPL", "MSFT"]
    assert bt.start == "2024-01-01"
    assert bt.freq == 10
    assert bt.n_quantiles == 2
    assert bt.result is None


def test_backtester_plot_before_run_raises_error():
    bt = Backtester(
        tickers=["AAPL"],
        start="2024-01-01",
        factor_specs=[("momentum", {"window": 5})],
    )
    with pytest.raises(RuntimeError, match="请先调用 run"):
        bt.plot()


def test_backtester_run_and_plot(sample_price: pd.DataFrame):
    bt = Backtester(
        tickers=list(sample_price.columns),
        start="2024-01-01",
        factor_specs=[
            ("momentum", {"window": 5}),
            ("reversal", {"window": 5}),
        ],
        factor_weights={"momentum": 0.6, "reversal": 0.4},
        freq=10,
        n_quantiles=2,
    )

    # 隔离数据库与网络: 直接 mock _load_price 返回合成数据
    with patch.object(bt, "_load_price", return_value=sample_price):
        res = bt.run(refresh_data=False)

    assert isinstance(res, Result)
    assert bt.price is sample_price
    assert isinstance(bt.score, pd.Series)
    assert isinstance(bt.weight, pd.Series)
    assert bt.result is res

    # 运行后绘图
    fig = bt.plot()
    assert isinstance(fig, plt.Figure)
    assert isinstance(bt.quantile_returns, pd.Series)


def test_backtester_pull_and_store_mock():
    bt = Backtester(
        tickers=["AAPL"],
        start="2024-01-01",
        factor_specs=[("momentum", {"window": 5})],
        db_path="mock.db",
    )

    fake_prices = pd.DataFrame([{"ticker": "AAPL", "close": 150.0}])
    with (
        patch("minibacktest.backtester.get_prices", return_value=fake_prices) as mock_get_prices,
        patch("minibacktest.backtester.liudb.init_schema") as mock_init_schema,
        patch("minibacktest.backtester.liudb.save_prices") as mock_save_prices,
    ):
        bt._pull_and_store()
        mock_get_prices.assert_called_once_with(["AAPL"], start="2024-01-01")
        mock_init_schema.assert_called_once_with("mock.db")
        mock_save_prices.assert_called_once_with(fake_prices, "mock.db")
