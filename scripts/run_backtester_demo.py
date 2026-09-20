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

bt = Backtester(
    tickers=tickers,
    start="2021-01-01",
    factor_specs=[
        ("momentum", {"window": 126}),  # 半年动量
        ("reversal", {"window": 5}),  # 一周短期反转
    ],
    factor_weights={"momentum": 0.7, "reversal": 0.3},
    freq=21,  # 月度调仓
    n_quantiles=5,
)

result = bt.run(refresh_data=False)
print_result(result)
print(bt.score.groupby("date").size().tail())  # type: ignore # 顺手看一眼每个调仓日进了多少只票

fig = bt.plot()
out_dir = Path("outputs")
out_dir.mkdir(exist_ok=True)
out_path = out_dir / "backtester_demo_tearsheet.png"
fig.savefig(out_path, dpi=120)
logger.info(f"图存到 {out_path.resolve()}")
