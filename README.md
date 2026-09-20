# minibacktest

个人的截面多因子回测引擎: 拉数据 -> 因子标准化 -> 合成打分 -> 分位数多空组合 -> 向量化回测出净值曲线。

不是事件驱动(不逐笔模拟买卖), 而是矩阵化的向量化回测, 所以 `Result` 只包含净值
曲线级别的指标(收益、波动、Sharpe/Sortino/Calmar、回撤等), 没有 Win Rate、
Profit Factor 这类假设"一笔笔独立交易"存在的字段。

## 安装

```bash
uv sync
```

## 流程

```
[原始量价/财务数据]
       │  矩阵化整理
[T × N 对齐矩阵 (Close, Tradable Mask)]
       │  纵向滚动算子 + 横向截面标准化 (zscore.py)
[综合多因子矩阵 F]
       │  加权合成 (signal/combine.py)
[打分 Score]
       │  截面分位数切分 (portfolio/sizing.py)
[目标权重矩阵 W]
       │  铺到每个交易日 + shift(1) 消除未来函数 (engine.py)
[实际生效持仓矩阵 W_hold]
       │  与日收益点乘, 逐日累乘
[净值曲线]
       │
[Result: 收益 / 夏普 / 回撤 / 五分位单调性检验]
```

## 用法

```python
from minibacktest.zscore.zscore import zscore_by_date
from minibacktest.signal.combine import combine_scores
from minibacktest.portfolio.sizing import quantile_long_short
from minibacktest.engine import run_backtest
from minibacktest import figure_engine as fe

z = zscore_by_date(factors)                                # 因子横截面标准化
score = combine_scores(z)                                  # 多因子合成打分
weight = quantile_long_short(score=score, n_quantiles=5)   # 调仓日目标权重
result = run_backtest(price=price, target_weight=weight, freq=21)

fig = fe.plot_tearsheet(result.equity_curve)                # 净值/回撤/滚动Sharpe/月度热力图
```
