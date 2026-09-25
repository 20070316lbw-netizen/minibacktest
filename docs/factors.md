# 行情字段与首批因子

## 本地数据核查

2026-09-26 只读检查项目内 `data/sp500.db` 的 `prices` 表：

| 项目 | 结果 |
| --- | --- |
| 日期范围 | 2001-01-02 ～ 2026-09-21 |
| 股票数、日线记录数 | 503、2,857,188 |
| 开、高、低、原始收盘、复权收盘、成交量 | 每列 2,857,188 个非空值 |
| 非正价格 | 0 条 |
| 零成交量 | 6,444 条 |
| 开高低收关系异常 | 1 条（HUBB，2021-05-05） |
| VWAP | 未存储，也不由 OHLC 推算 |

不同股票的上市日期不同；例如 HONA 从 2026-06-15 才有记录。以上“非空”仅指数据库已有的行，不代表每只股票覆盖整个区间。`src/minibacktest/config.py` 的标的名单是当前标普 500 成分股，并非历史逐日成分股；早期回测有幸存者偏差。

`liudb.Query(columns=["close"])` 返回物理表的 `adj_close`；`open/high/low` 原样查询时仍是未复权价格。多字段因子读取物理 `close`，按每行 `adj_close / close` 调整开高低，再与复权收盘价对齐。成交量使用数据库的 `volume`，没有额外复权。零成交量视为缺失；开高低收关系异常的记录不参与 K 线因子计算。这个做法让同一天的 OHLC 保持一致，但遇到拆股时，未经额外处理的成交量序列仍可能跳变。

## 新增因子

下列因子在调仓日 `t` 只使用 `t-1` 及更早的行情。窗口参数都表示交易日数；滚动统计需要完整窗口。

| 名称 | 原始值（`C/O/H/L/V` 分别为复权收盘/开盘/最高/最低价和数据库成交量） |
| --- | --- |
| `kbar_mid` | `(C-O)/O` |
| `kbar_range` | `(H-L)/O` |
| `kbar_upper_shadow` | `(H-max(O,C))/O` |
| `kbar_lower_shadow` | `(min(O,C)-L)/O` |
| `close_location` | `(C-L)/(H-L+1e-12)` |
| `realized_volatility(window)` | 日收益的滚动标准差 |
| `relative_volume(window)` | `V/滚动均值(V)` |
| `return_volume_corr(window)` | 日收益与成交量变化率的滚动相关系数 |
| `ohlc_channel_position(window)` | `(C-滚动最低价L)/(滚动最高价H-滚动最低价L+1e-12)` |

前四个 K 线因子的表达式分别与 qlib 的 KMID、KLEN、KUP、KLOW 对应，但这里统一延迟一天，并使用本项目的复权口径。其余因子是基础量价特征，不宣称与 qlib Alpha158 的同名列逐值一致。

示例：

```python
bt = Backtester(
    tickers=tickers,
    start="2020-01-01",
    factor_specs=[
        ("kbar_mid", {}),
        ("relative_volume", {"window": 20}),
        ("return_volume_corr", {"window": 20}),
    ],
    db_path="data/sp500.db",
)
result = bt.run(refresh_data=False)
```

当前组合器默认把每一列标准化后等权合成。直接将高度相关的因子全部放入 `factor_specs`，会增加这些因子所属类别的隐含权重；本批因子先用于计算和研究，组合权重需要单独评估。
