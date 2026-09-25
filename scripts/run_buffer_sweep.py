"""缓冲带扫参: momentum(126) + channel_position_rank(5) 各 0.5 权重,
5 层波动率中性构仓和普通分位构仓各扫一遍不同 buffer(buffer=0 即原版)。

同一份 score 只算一次, 各场景只换仓位构造; 每个场景分别跑毛(无费用)和
净(单边佣金 5bp + 滑点 5bp)两遍。分位前瞻收益只取决于 score, 跟 buffer
无关, 所以单独打印一次(普通分组 vs 波动率中性化分组)。

在 minibacktest 仓库根目录运行:
    uv run python scripts/run_buffer_sweep.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from loguru import logger

from minibacktest.backtester import Backtester
from minibacktest.config import tickers
from minibacktest.engine import run_backtest
from minibacktest.evaluation.quantile import (
    quantile_forward_returns,
    vol_neutral_quantile_forward_returns,
)
from minibacktest.portfolio.sizing import (
    buffered_quantile_long_short,
    buffered_vol_neutral_quantile_long_short,
)
from minibacktest.risk.volatility import realized_volatility

REPO = Path(__file__).resolve().parents[1]
FREQ = 21
N_Q = 5
N_VOL = 5
VOL_WINDOW = 21
BUFFERS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]


def leg_sizes(weight: pd.Series) -> tuple[float, float]:
    by_date = weight.groupby(level="date")
    return by_date.apply(lambda w: (w > 0).sum()).mean(), by_date.apply(lambda w: (w < 0).sum()).mean()


def print_quantiles(title: str, qret: pd.Series) -> None:
    cells = " | ".join(f"Q{int(q) + 1}: {v * 100:+.3f}%" for q, v in qret.items())
    print(f"{title:<14} {cells}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=REPO / "data/sp500.db")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default=None)
    args = parser.parse_args()
    logger.remove()  # 关掉 DEBUG 日志, 只留表格

    bt = Backtester(
        tickers=tickers,
        start=args.start,
        end=args.end,
        factor_specs=[("momentum", {"window": 126}), ("channel_position_rank", {"window": 5})],
        factor_weights={"momentum": 0.5, "channel_position_rank": 0.5},
        freq=FREQ,
        n_quantiles=N_Q,
        db_path=str(args.db),
    )
    bt.run(refresh_data=False)
    price, score = bt.price, bt.score

    vol = realized_volatility(price, window=VOL_WINDOW)
    vol = vol[vol.index.isin(score.index)]

    print(f"区间 {price.index[0].date()} ~ {price.index[-1].date()}, {price.shape[1]} 只标的, "
          f"调仓间隔 {FREQ} 日, {N_Q} 分位")
    print("\n分位前瞻收益(每期平均, 只取决于 score, 跟 buffer 无关):")
    print_quantiles("普通分组", quantile_forward_returns(score, price, freq=FREQ, n_quantiles=N_Q))
    print_quantiles(
        f"{N_VOL}层波动率中性",
        vol_neutral_quantile_forward_returns(
            score, vol, price, freq=FREQ, n_vol_groups=N_VOL, n_quantiles=N_Q
        ),
    )

    scenarios: list[tuple[str, pd.Series]] = []
    for b in BUFFERS:
        w = buffered_vol_neutral_quantile_long_short(
            score=score, vol=vol, n_vol_groups=N_VOL, n_quantiles=N_Q, buffer=b
        )
        scenarios.append((f"波动率中性 buffer={b:.2f}", w))
    for b in BUFFERS:
        w = buffered_quantile_long_short(score=score, n_quantiles=N_Q, buffer=b)
        scenarios.append((f"普通 buffer={b:.2f}", w))

    print("\n| 仓位构造 | 毛年化 | 净年化 | 毛Sharpe | 净Sharpe | 净最大回撤 | 年化换手 | 平均多/空只数 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    rows = []
    for label, weight in scenarios:
        gross = run_backtest(price, weight, freq=FREQ)
        net = run_backtest(price, weight, freq=FREQ, commission_bps=5, slippage_bps=5)
        n_long, n_short = leg_sizes(weight)
        rows.append({
            "sizing": label,
            "gross_return_ann_pct": gross.return_ann_pct,
            "net_return_ann_pct": net.return_ann_pct,
            "gross_sharpe": gross.sharpe_ratio,
            "net_sharpe": net.sharpe_ratio,
            "net_max_drawdown_pct": net.max_drawdown_pct,
            "turnover_ann_pct": net.turnover_ann_pct,
            "avg_long": n_long,
            "avg_short": n_short,
        })
        print(
            f"| {label} | {gross.return_ann_pct:.2f}% | {net.return_ann_pct:.2f}% | "
            f"{gross.sharpe_ratio:.3f} | {net.sharpe_ratio:.3f} | {net.max_drawdown_pct:.2f}% | "
            f"{net.turnover_ann_pct:.1f}% | {n_long:.0f} / {n_short:.0f} |"
        )

    out = REPO / "outputs/buffer_sweep"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    print(f"\n汇总已存: {out / 'summary.csv'}")


if __name__ == "__main__":
    main()
