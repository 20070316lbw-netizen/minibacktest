"""对新入口: 使用波动率构造仓位的设计进行测试"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from minibacktest.backtester import Backtester
from minibacktest.config import tickers
from minibacktest.portfolio.sizing import make_vol_neutral_sizer
from minibacktest.report import print_result

common_kwargs = {
    "tickers": tickers,
    "start": "2010-01-01",

    "factor_specs": [
        ("momentum", {"window": 126}),
        ("reversal", {"window": 5}),
        ("channel_position_rank", {"window": 5}),
        ("ma_deviation_rank", {"window": 5}),
    ],

    "factor_weights": {"momentum": 0.5, "reversal": 0.2, "channel_position_rank": 0.2, "ma_deviation_rank": 0.1},
    "freq": 21,  # 月度调仓
    "n_quantiles": 5,
    "db_path": "data/sp500.db",
}

bt = Backtester(**common_kwargs)
bt_cost = Backtester(
    **common_kwargs, 
    commission_bps=5, 
    slippage_bps=5, 
    sizing_fn=make_vol_neutral_sizer(n_vol_groups=5, n_quantiles=5, vol_window=21),
    )

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
out_path = out_dir / "backtester_demo_tearsheet_vol_2010.png"
fig.savefig(out_path, dpi=120)
logger.info(f"图存到 {out_path.resolve()}")
