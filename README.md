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

### 1. 一键全流程回测 (推荐)

使用 `Backtester` 统一调度数据读取、多因子加权合成、分位数多空组合、向量化回测及图表生成：

```python
from minibacktest.backtester import Backtester
from minibacktest.config import tickers
from minibacktest.report import print_result

# 配置并初始化回测
bt = Backtester(
    tickers=tickers,
    start="2021-01-01",
    factor_specs=[
        ("momentum", {"window": 126}),  # 半年动量
        ("reversal", {"window": 5}),     # 一周短期反转
    ],
    factor_weights={"momentum": 0.7, "reversal": 0.3},
    freq=21,        # 调仓间隔 (21 个交易日，即月度调仓)
    n_quantiles=5,  # 五分位数多空对冲
)

# 运行回测 (若需从网络重新拉取行情并存入 liudb 可设 refresh_data=True)
result = bt.run(refresh_data=False)

# 终端打印格式化指标表 (收益/夏普/最大回撤/Alpha/Beta 等)
print_result(result)

# 生成 Tearsheet 总览图 (净值曲线/回撤/滚动 Sharpe/月度热力图/五分位单调性检验)
fig = bt.plot()
fig.savefig("outputs/backtester_tearsheet.png", dpi=120)
```

### 2. 底层模块分步调用

若已有整理好的价格宽表与因子面板，也可直接调用底层核心算子完成定制回测：

```python
from minibacktest import figure_engine as fe
from minibacktest.engine import run_backtest
from minibacktest.portfolio.sizing import quantile_long_short
from minibacktest.signal.combine import combine_scores
from minibacktest.zscore.zscore import zscore_by_date

# 1. 因子横截面标准化 (每个截面均值 0、标准差 1)
z = zscore_by_date(factors)

# 2. 多因子加权合成打分
score = combine_scores(z, weights={"momentum": 0.7, "reversal": 0.3})

# 3. 截面分位数多空分配权重 (多头和 +1, 空头和 -1, 中间为 0)
weight = quantile_long_short(score=score, n_quantiles=5)

# 4. 向量化回测 (shift(1) 消除未来函数, 计算净值与绩效)
result = run_backtest(price=price, target_weight=weight, freq=21)

# 5. 绘制 Tearsheet 图表
fig = fe.plot_tearsheet(result.equity_curve)
```
