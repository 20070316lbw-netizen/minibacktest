"""用 Backtester 跑一次两因子组合的回测: 动量 + 短期反转, 演示 factor_specs
怎么加第二个因子——对比只有动量因子那一版(见 git 历史), 用来确认多因子
合成这条路径(注册表 -> 横向拼因子面板 -> 标准化 -> 加权合成)是不是真的
能打通。
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from minibacktest.backtester import Backtester
from minibacktest.config import tickers
from minibacktest.report import print_result

common_kwargs = dict(
    tickers=tickers,
    start="2010-01-01",
    factor_specs=[
        ("momentum", {"window": 126}),  # 半年动量
        ("reversal", {"window": 5}),  # 一周短期反转
    ],
    factor_weights={"momentum": 0.7, "reversal": 0.3},
    freq=21,  # 月度调仓
    n_quantiles=5,
    db_path="data/sp500.db", 
)

# 无费率(原来的行为) vs 有费率, 对比看手续费/滑点拖累有多大
bt = Backtester(**common_kwargs)
bt_cost = Backtester(**common_kwargs, commission_bps=5, slippage_bps=5)

result = bt.run(refresh_data=False)
logger.info("===== 不计手续费/滑点 =====")
print_result(result)
print(bt.score.groupby("date").size().tail())  # type: ignore # 顺手看一眼每个调仓日进了多少只票

result_cost = bt_cost.run(refresh_data=False)
logger.info("===== 单边佣金 5bp + 单边滑点 5bp =====")
print_result(result_cost)

logger.info(
    f"费用拖累: 年化换手率 {result_cost.turnover_ann_pct:.1f}%, "
    f"期末净值总拖累 {result_cost.total_cost_pct:.2f}%, "
    f"期末净值 {result.equity_final:,.0f} -> {result_cost.equity_final:,.0f}"
)

fig = bt_cost.plot()
out_dir = Path("outputs")
out_dir.mkdir(exist_ok=True)
out_path = out_dir / "backtester_demo_tearsheet_2010.png"
fig.savefig(out_path, dpi=120)
logger.info(f"图存到 {out_path.resolve()}")
