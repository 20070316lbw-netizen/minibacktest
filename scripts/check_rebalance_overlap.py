"""检查相邻调仓日多头/空头集合的留任比例, 验证高换手率是不是来自因子
组合本身(而不是 engine.py 的 bug)。

跟 run_backtester_demo.py 用一样的 factor_specs/freq/n_quantiles, 但不算
完整回测, 只看 bt.weight(调仓日目标权重)本身, 跑起来更快。
"""

from __future__ import annotations

import itertools

from minibacktest.backtester import Backtester
from minibacktest.config import tickers

bt = Backtester(
    tickers=tickers,
    start="2021-01-01",
    factor_specs=[
        ("momentum", {"window": 126}),
        ("reversal", {"window": 5}),
    ],
    factor_weights={"momentum": 0.7, "reversal": 0.3},
    freq=21,
    n_quantiles=5,
)

bt.run(refresh_data=False)

weight = bt.weight  # [date, ticker] MultiIndex, +1/n 多头, -1/n 空头, 0 已被 dropna 剔除

by_date = weight.groupby(level="date")
dates = sorted(by_date.groups.keys())

print(f"共 {len(dates)} 个调仓日, {dates[0].date()} ~ {dates[-1].date()}")
print(f"{'调仓日':<12}{'多头留任%':>10}{'空头留任%':>10}{'多头数':>8}{'空头数':>8}")

long_overlaps: list[float] = []
short_overlaps: list[float] = []

for prev_d, curr_d in itertools.pairwise(dates):
    prev_w = by_date.get_group(prev_d)
    curr_w = by_date.get_group(curr_d)

    prev_long = set(prev_w[prev_w > 0].index.get_level_values("ticker"))
    curr_long = set(curr_w[curr_w > 0].index.get_level_values("ticker"))
    prev_short = set(prev_w[prev_w < 0].index.get_level_values("ticker"))
    curr_short = set(curr_w[curr_w < 0].index.get_level_values("ticker"))

    long_overlap = len(prev_long & curr_long) / len(prev_long) * 100 if prev_long else float("nan")
    short_overlap = len(prev_short & curr_short) / len(prev_short) * 100 if prev_short else float("nan")
    long_overlaps.append(long_overlap)
    short_overlaps.append(short_overlap)

    print(
        f"{curr_d.date()!s:<12}{long_overlap:>10.1f}{short_overlap:>10.1f}"
        f"{len(curr_long):>8}{len(curr_short):>8}"
    )

import statistics

avg_long = statistics.fmean(long_overlaps)
avg_short = statistics.fmean(short_overlaps)
print(f"\n平均多头留任率: {avg_long:.1f}%  平均空头留任率: {avg_short:.1f}%")
print("(留任率越低, 说明分位组构成换得越勤, 换手率越高)")
