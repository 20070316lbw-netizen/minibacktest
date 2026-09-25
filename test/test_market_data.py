from __future__ import annotations

import liudb
import pandas as pd
import pytest

from minibacktest.backtester import Backtester
from minibacktest.factors import FactorSpecError, compile_spec, get, required_fields
from minibacktest.market_data import load_factor_fields


def _make_db(tmp_path) -> str:
    path = str(tmp_path / "prices.db")
    dates = pd.bdate_range("2024-01-02", periods=6)
    rows = []
    for i, date in enumerate(dates):
        for ticker, base, change in (("A", 100.0, 2.0), ("B", 200.0, -1.0)):
            close = base + i * change
            rows.append({
                "date": date,
                "ticker": ticker,
                "open": close - change,
                "high": close + 2,
                "low": close - 3,
                "close": close,
                "adj_close": close * 0.8,
                "volume": 1000 + i * 100 + (ticker == "B") * 500,
            })
    liudb.save_prices(pd.DataFrame(rows), path=path)
    return path


def test_factor_fields_use_same_adjustment_as_close(tmp_path):
    path = _make_db(tmp_path)
    bt = Backtester(tickers=["A"], start="2024-01-02", end="2024-01-04",
                    factor_specs=[("kbar_mid", {})], db_path=path)
    close = bt._load_price()
    fields = load_factor_fields(db_path=path, tickers=["A"], start=bt.start,
                                end=bt.end, adjusted_close=close)

    assert close.columns.tolist() == ["A"]
    assert close.index.astype(str).tolist() == ["2024-01-02", "2024-01-03", "2024-01-04"]
    assert close.iloc[0, 0] == pytest.approx(80.0)
    assert fields["open"].iloc[0, 0] == pytest.approx(78.4)
    assert fields["high"].iloc[0, 0] == pytest.approx(81.6)
    assert fields["low"].iloc[0, 0] == pytest.approx(77.6)
    assert fields["volume"].iloc[0, 0] == 1000
    assert all(values.index.equals(close.index) and values.columns.equals(close.columns)
               for values in fields.values())


def test_backtester_calculates_ohlc_factor_from_database(tmp_path):
    path = _make_db(tmp_path)
    bt = Backtester(tickers=["A", "B"], start="2024-01-02", freq=1, n_quantiles=2,
                    factor_specs=[("kbar_mid", {})], db_path=path)
    bt.run(refresh_data=False)

    assert set(bt.factor_fields) == {"open", "high", "low", "volume"}
    day = pd.Timestamp("2024-01-03")
    # 调仓日只使用前一日的 K 线。
    assert bt.score.loc[(day, "A")] > bt.score.loc[(day, "B")]
    assert bt.result is not None


def test_bad_ohlc_and_zero_volume_are_excluded_from_related_factors(tmp_path):
    path = str(tmp_path / "bad_rows.db")
    liudb.save_prices(pd.DataFrame([
        {"date": "2024-01-02", "ticker": "A", "open": 10, "high": 12, "low": 9,
         "close": 11, "adj_close": 8.8, "volume": 0},
        {"date": "2024-01-03", "ticker": "A", "open": 10, "high": 10, "low": 11,
         "close": 10, "adj_close": 8, "volume": 100},
    ]), path=path)
    bt = Backtester(tickers=["A"], start="2024-01-02", factor_specs=[("kbar_mid", {})], db_path=path)
    close = bt._load_price()
    fields = load_factor_fields(db_path=path, tickers=bt.tickers, start=bt.start,
                                end=bt.end, adjusted_close=close)

    assert pd.isna(fields["volume"].iloc[0, 0])
    assert pd.isna(fields["high"].iloc[1, 0])
    assert pd.isna(fields["open"].iloc[1, 0])
    assert close.iloc[1, 0] == 8  # 收盘价仍可用于仅依赖 close 的因子


def test_kbar_upper_shadow_uses_previous_day_and_required_fields():
    dates = pd.bdate_range("2024-01-02", periods=3)
    close = pd.DataFrame({"A": [11.0, 12.0, 99.0]}, index=dates)
    fields = {
        "open": pd.DataFrame({"A": [10.0, 11.0, 88.0]}, index=dates),
        "high": pd.DataFrame({"A": [13.0, 14.0, 100.0]}, index=dates),
    }
    assert required_fields("kbar_upper_shadow") == {"open", "high"}
    values = get("kbar_upper_shadow")(close, fields=fields)
    assert values.loc[(dates[1], "A")] == pytest.approx((13 - 11) / 10)
    assert values.loc[(dates[2], "A")] == pytest.approx((14 - 12) / 11)
    with pytest.raises(FactorSpecError, match=r"缺少行情字段 \['high'\]"):
        get("kbar_upper_shadow")(close, fields={"open": fields["open"]})


def test_new_rolling_ops_are_trailing_and_validate_window():
    dates = pd.bdate_range("2024-01-02", periods=5)
    close = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=dates)
    volume = pd.DataFrame({"A": [2.0, 4.0, 6.0, 8.0, 10.0]}, index=dates)
    summed = compile_spec({"steps": [{"id": "s", "op": "rolling_sum", "input": {"ref": "price"},
                                        "window": {"const": 3}}], "output": "s"}, factor_name="sum")
    corr_spec = {"steps": [{"id": "r", "op": "rolling_corr", "a": {"ref": "close"},
                            "b": {"ref": "volume"}, "window": {"const": 3}}], "output": "r"}
    corr = compile_spec(corr_spec, factor_name="corr")

    assert pd.isna(summed(close).loc[(dates[1], "A")])
    assert summed(close).loc[(dates[2], "A")] == pytest.approx(6.0)
    assert corr(close, fields={"volume": volume}).loc[(dates[2], "A")] == pytest.approx(1.0)
    assert pd.isna(corr(close, fields={"volume": volume}).loc[(dates[1], "A")])
    corr_spec["steps"][0]["window"] = {"const": 0}
    with pytest.raises(FactorSpecError, match="滚动窗口"):
        compile_spec(corr_spec, factor_name="bad_corr")
