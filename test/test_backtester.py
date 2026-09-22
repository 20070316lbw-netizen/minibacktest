from __future__ import annotations

from unittest.mock import patch

import liudb
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


def test_backtester_load_price_returns_adjusted_close_for_requested_tickers(tmp_path):
    """_load_price() 应该只按 tickers/start 查回复权后的 close, 不是整表读出来。"""
    db_path = str(tmp_path / "test_sp500.db")
    liudb.init_schema(db_path)
    liudb.save_prices(
        pd.DataFrame(
            [
                {
                    "date": "2024-01-02",
                    "ticker": "AAPL",
                    "open": 180.0,
                    "high": 185.0,
                    "low": 179.0,
                    "close": 182.0,
                    "adj_close": 181.5,
                    "volume": 5_000_000.0,
                },
                {
                    "date": "2024-01-03",
                    "ticker": "AAPL",
                    "open": 182.0,
                    "high": 186.0,
                    "low": 181.0,
                    "close": 184.0,
                    "adj_close": 183.5,
                    "volume": 6_000_000.0,
                },
                # 不在 Backtester.tickers 候选池里, 不应该出现在结果里
                {
                    "date": "2024-01-02",
                    "ticker": "MSFT",
                    "open": 370.0,
                    "high": 375.0,
                    "low": 368.0,
                    "close": 372.0,
                    "adj_close": 372.0,
                    "volume": 3_000_000.0,
                },
            ]
        ),
        path=db_path,
    )

    bt = Backtester(
        tickers=["AAPL"],
        start="2024-01-02",
        factor_specs=[("momentum", {"window": 1})],
        db_path=db_path,
    )
    price = bt._load_price()

    assert price.columns.tolist() == ["AAPL"]
    assert price.loc["2024-01-02", "AAPL"] == 181.5  # 复权收盘价, 不是未复权的 182.0
    assert price.loc["2024-01-03", "AAPL"] == 183.5


def test_backtester_load_price_respects_end(tmp_path):
    """传了 end 之后, _load_price() 不该把 end 之后的行也读回来。"""
    db_path = str(tmp_path / "test_sp500.db")
    liudb.init_schema(db_path)
    liudb.save_prices(
        pd.DataFrame(
            [
                {
                    "date": "2024-01-02",
                    "ticker": "AAPL",
                    "open": 180.0,
                    "high": 185.0,
                    "low": 179.0,
                    "close": 182.0,
                    "adj_close": 181.5,
                    "volume": 5_000_000.0,
                },
                {
                    "date": "2024-01-03",
                    "ticker": "AAPL",
                    "open": 182.0,
                    "high": 186.0,
                    "low": 181.0,
                    "close": 184.0,
                    "adj_close": 183.5,
                    "volume": 6_000_000.0,
                },
            ]
        ),
        path=db_path,
    )

    bt = Backtester(
        tickers=["AAPL"],
        start="2024-01-02",
        end="2024-01-02",
        factor_specs=[("momentum", {"window": 1})],
        db_path=db_path,
    )
    price = bt._load_price()

    assert price.index.astype(str).tolist() == ["2024-01-02"]


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
        mock_get_prices.assert_called_once_with(["AAPL"], start="2024-01-01", end=None)
        mock_init_schema.assert_called_once_with("mock.db")
        mock_save_prices.assert_called_once_with(fake_prices, "mock.db")


def test_backtester_pull_and_store_passes_end():
    bt = Backtester(
        tickers=["AAPL"],
        start="2024-01-01",
        end="2024-06-30",
        factor_specs=[("momentum", {"window": 5})],
        db_path="mock.db",
    )

    fake_prices = pd.DataFrame([{"ticker": "AAPL", "close": 150.0}])
    with (
        patch("minibacktest.backtester.get_prices", return_value=fake_prices) as mock_get_prices,
        patch("minibacktest.backtester.liudb.init_schema"),
        patch("minibacktest.backtester.liudb.save_prices"),
    ):
        bt._pull_and_store()
        mock_get_prices.assert_called_once_with(["AAPL"], start="2024-01-01", end="2024-06-30")
