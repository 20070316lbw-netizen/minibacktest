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

### 交易成本

`engine.run_backtest`(以及 `Backtester`) 支持按 `commission_bps`(单边佣金)
和 `slippage_bps`(单边滑点)算交易成本, 两者单位都是 bp(1bp = 0.01%),
默认都是 0(不计费用, 向后兼容)。算法: 每天的换手 = 当天持仓权重相对前
一天的变动量之和(绝对值), 只有真正发生调仓的那天才会产生非零换手; 当天
组合收益率里减去 `换手 × (commission_bps + slippage_bps) / 10000`。

`Result` 里对应新增了 `commission_bps` / `slippage_bps`(记录本次跑的费率)、
`turnover_ann_pct`(年化换手率)、`total_cost_pct`(费用对期末净值的总拖累,
= 未扣费净值与扣费净值之差占期初资金的比例); `equity_curve` 也多了一列
`gross_nav`(未扣费净值), 方便跟扣费后的 `nav` 对比。

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
    commission_bps=5,   # 单边佣金, 5bp = 0.05%
    slippage_bps=5,     # 单边滑点, 5bp = 0.05%
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

# 4. 向量化回测 (shift(1) 消除未来函数, 计算净值/绩效/换手与交易成本拖累)
result = run_backtest(
    price=price, target_weight=weight, freq=21,
    commission_bps=5, slippage_bps=5,  # 不传则默认为 0, 即不计费用
)

# 5. 绘制 Tearsheet 图表
fig = fe.plot_tearsheet(result.equity_curve)
```
