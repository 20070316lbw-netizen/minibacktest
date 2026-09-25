# minibacktest

[![CI](https://github.com/20070316lbw-netizen/minibacktest/actions/workflows/ci.yml/badge.svg)](https://github.com/20070316lbw-netizen/minibacktest/actions/workflows/ci.yml)

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
## 因子注册表

`factors/` 目录下的因子不再靠"新建 .py 文件 + `@register` 装饰器 + 在
`factors/__init__.py` 里 `import ... as _` 触发副作用注册"这套机制——那套东西
改一个因子要动三个地方，也不直观。现在 `factors/registry.py` 直接扫这个目录下
所有 `*.yaml`，把每个文件编译成一个 `price -> [date, ticker] MultiIndex Series`
的函数；`get(name)`/`available()` 每次调用都会重新扫描 + 重新编译，改一个 YAML
立刻生效，不用管 Python 的 import 缓存，也不用重启进程。

一个因子的 YAML 是一个很小的、纯数据的计算图，不是可执行代码：

```yaml
# factors/momentum.yaml
name: momentum
params: [window]
steps:
  - id: lag_amount
    op: add
    a: {const: 1}
    b: {param: window}
  - id: near
    op: shift
    input: {ref: price}
    by: {const: 1}
  - id: far
    op: shift
    input: {ref: price}
    by: {ref: lag_amount}
  - id: ratio
    op: divide
    a: {ref: near}
    b: {ref: far}
  - id: result
    op: subtract
    a: {ref: ratio}
    b: {const: 1.0}
output: result
```

每一步有 `id`（后面的步骤靠 `{ref: 这个 id}` 引用它）和 `op`（支持四则运算、
`shift`、滚动统计、逐元素最大/最小值、滚动相关、`cross_section_rank`），操作数只有
三种写法：`{const: 数字}`、`{param: 参数名}`（从调用时的 `**kwargs` 里取）、
`{ref: 前面某一步的 id, 或者内置行情字段}`。`price`/`close` 都指向复权收盘价；
`open`/`high`/`low` 是同口径复权价格，`volume` 是数据库中的成交量。
`output` 指明哪一步是最终结果。`momentum`/`reversal` 两个因子都已经
迁移成这种格式（等价的原始 Python 写法留在各自 YAML 文件的注释里，方便对照）。

新增一个因子：在这个目录下新建一个 `<名字>.yaml`，参照 `momentum.yaml` 的格式
写，不用碰任何 `.py` 文件，也不用改 `registry.py` 或者 `Backtester`。

多字段因子由 `Backtester` 按需读取行情字段；直接调用 `get(name)` 时，除
`price` 外的字段要通过 `fields={"open": ..., "high": ..., ...}` 传入，且宽表
日期与标的必须和 `price` 完全对齐。首批新增了 9 个 K 线、量价和波动因子，
具体口径、数据核查结果及用法见 [因子与行情字段说明](docs/factors.md)。

## 图表文字用英文

`figure_engine/` 出的图（净值曲线、回撤、月度热力图、滚动 Sharpe、分位数检验）
标题/坐标轴/图例都是英文。之前是按系统里能找到的中文字体名字（"Heiti SC"/
"PingFang SC"/...）做检测，找到就用，找不到就静默退化成方块字——换一台没装
这些字体的机器（比如在云端沙箱里跑）就会打回原样。直接用英文没有这个问题，
不依赖任何字体，换机器也不会再出现方块字。

## 未来打算：让 Agent 拼装因子

这套 YAML 因子机制除了解决"注册表不好改"这个问题，也是为了给
`../quant-assistant` 那边的 AI Agent（跑在 DeepSeek Harness 上，通过 MCP 调用
本地工具）铺路——目标是让 Agent（也包括人自己）能自己"发明"新因子并跑回测，
而不是只能调用几个写死的现成因子。

设计上已经定下来的几条：

- **不用字符串表达式 + AST 解析器，也不需要沙箱跑任意 Python**。因子只能从
  一个受限的"零件"词表里拼（价格列引用、常数、参数、加减乘除、挪 N 天），
  跟"是谁在拼"没关系——这套词表本身就没有 `eval`/`import`/文件/网络这类
  危险原语，人自己直接手写 YAML 也是走同一条路径，不存在"因为要允许 Agent
  用所以人被多绕一层"的问题。
- **"沙箱"要防的是资源失控，不是恶意代码**。调研过 E2B/Daytona/Docker/
  Firecracker/gVisor 这类东西，结论是它们解决的是"多租户、陌生人代码"的
  问题，用在"本机、自己电脑，就是不想因子跑飞把内存吃满或者卡死"这个场景上
  是杠杆用错方向——真正需要的只是子进程 + `resource.setrlimit`（内存/CPU
  上限）+ `subprocess.run(timeout=...)`，本地就能做，不用引入任何外部依赖。
  这一层这次还没写，等真的接 Agent 工具的时候再加。
- **Agent 拼因子的方式是一步步调用工具，不是让它直接写 YAML 文本**——"新建
  因子叫 xxx"、"选 CLOSE"、"挪 N 天"、"除"、"减常数"这样一步步调，每一步
  落地成 YAML 里的一个 `step`，跟它一步步调 MCP 工具这件事本身是同构的
  （`quant-assistant` 那边 `quant-tool-policy` 的白名单机制是同一个防御
  思路，可以直接复用）。参数（比如 `window`）要保留成"参数"而不是写死的
  常量，这样同一个因子模板才能像 `momentum`/`reversal` 一样被复用不同参数，
  而不是换一个数字就要新建一个因子。

`quant-assistant` 那边已经接上了（2026-09-23，见该仓库 README「Agent 拼装因子
工具」一节）：新建草稿 / 加一步 / 撤销 / 指定输出 / 保存 / 试算 / 回测都包成了
MCP 工具，试算和回测跑在带 CPU/内存上限 + 墙钟超时的子进程里。为此这边的
`factors/registry.py` 补了几样东西：

- `validate_spec(spec, factor_name=..., partial=False)`：纯静态校验(不碰数据),
  `_compile` 一开始就调它, 所以 op 写错、引用了后面才出现的 id、用了没声明的
  参数这类错误在扫描注册表时就报, 不再拖到真正计算的时候。`partial=True` 给
  "还没拼完的草稿"用(允许没有 steps/output)。
- 防未来函数: `shift` 的位移必须是非负整数, 常数在静态校验时拦, 参数算出来的
  负数在运行时拦。
- `MINIBACKTEST_EXTRA_FACTOR_DIRS` 环境变量(或 `load_specs(extra_dirs=...)`)
  把本仓库之外的因子目录挂进注册表; 跟内置因子同名直接报错, 不会覆盖。
- 公开 `compile_spec` / `load_specs` / `factor_dirs` / `ALLOWED_OPS` / `OP_FIELDS`,
  外部调用方不用再碰下划线开头的私有函数。

词表支持滚动均值、标准差、最大、最小、求和以及同日横截面排名：普通滚动运算使用 `input: {ref: ...}`
与 `window: {const: 整数}` 或 `{param: 参数名}`，窗口限 1~512 个交易日，
需要完整窗口才产出值；`cross_section_rank` 只需 `input`，每天对非空标的做
升序百分位排名，空值和无穷值不参与排名。滚动计算包含当日及过去的数据，不读取未来行。
`rolling_corr` 使用 `a`、`b` 两个宽表引用和 `window`；`elementwise_max`/
`elementwise_min` 使用 `a`、`b` 两个操作数。`quant-assistant` 草稿工具对新增
字段和运算的支持需要另行同步。
