"""读取因子所需的日线字段，并统一价格字段的复权口径。"""

from __future__ import annotations

import liudb
import pandas as pd


def load_factor_fields(
    *,
    db_path: str,
    tickers: list[str],
    start: str,
    end: str | None,
    adjusted_close: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """返回与复权收盘价对齐的 open/high/low 和原始 volume 宽表。

    liudb.Query 的逻辑 close 是 adj_close，但 open/high/low 仍为原始价格。
    因此这里读取物理表的 raw close，用 adj_close/raw close 的逐日比例调整
    OHLC。这个比例只能用于价格字段；volume 保留数据源的原始成交量。
    """
    if not tickers:
        raise ValueError("tickers 不能为空")

    placeholders = ", ".join("?" for _ in tickers)
    sql = (
        "SELECT date, ticker, open, high, low, close AS raw_close, volume "
        f"FROM prices WHERE ticker IN ({placeholders}) AND date >= ?"
    )
    params: list[str] = [*tickers, start]
    if end is not None:
        sql += " AND date <= ?"
        params.append(end)
    sql += " ORDER BY date, ticker"

    with liudb.get_duckdb(path=db_path, read_only=True) as connection:
        long = connection.execute(sql, params).df()
    if long.empty:
        raise ValueError("查询区间内没有因子行情数据")

    long["date"] = pd.to_datetime(long["date"])
    wide = long.set_index(["date", "ticker"])
    raw = {
        field: wide[field].unstack("ticker").reindex_like(adjusted_close)
        for field in ("open", "high", "low", "raw_close", "volume")
    }
    raw_close = raw["raw_close"]
    ratio = adjusted_close.div(raw_close.where(raw_close > 0))
    valid_ohlc = (
        raw["low"].le(raw["high"])
        & raw["open"].ge(raw["low"])
        & raw["open"].le(raw["high"])
        & raw_close.ge(raw["low"])
        & raw_close.le(raw["high"])
    )
    result = {
        field: raw[field].where(valid_ohlc).mul(ratio)
        for field in ("open", "high", "low")
    }
    result["volume"] = raw["volume"].where(raw["volume"] > 0)
    return result
