"""用真实本地行情完整比较普通分位构仓与波动率分桶构仓。

实验因子定义在 src/minibacktest/factors/*.yaml。本脚本只读行情库，不拉取网络
数据；同一个因子的两种仓位构造使用相同的有效股票与调仓日，再分别交给
run_backtest 计算净值，并各生成一张含分位前瞻收益的 tearsheet。

在 minibacktest 仓库根目录运行：
    uv run python scripts/run_extended_factors_full.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import liudb
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from minibacktest.backtester import Backtester
from minibacktest.engine import run_backtest
from minibacktest.evaluation.quantile import (
    quantile_forward_returns,
    vol_neutral_quantile_forward_returns,
)
from minibacktest.figure_engine.tearsheet import plot_tearsheet
from minibacktest.portfolio.sizing import (
    quantile_long_short,
    vol_neutral_quantile_long_short,
)
from minibacktest.risk.volatility import realized_volatility

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DB = REPO.parent / "quant-assistant/data/sp500.db"
DEFAULT_OUTPUT = REPO / "outputs/extended_factors_vol_buckets"
SCENARIOS = [
    ("momentum", {"window": 126}),
    ("ma_deviation_rank", {"window": 126}),
    ("channel_position_rank", {"window": 126}),
    ("vol_adjusted_momentum", {"window": 126, "vol_window": 20}),
]


def available_tickers(db: Path, start: str, end: str) -> list[str]:
    query = liudb.Query(columns=["close"], tickers=None, start=start, end=end)
    prices = liudb.loader(request=query, path=str(db))
    tickers = sorted(prices.index.get_level_values("ticker").unique().tolist())
    if not tickers:
        raise ValueError(f"数据库在 {start} ~ {end} 没有行情")
    return tickers


def paired_inputs(score: pd.Series, price: pd.DataFrame, vol_window: int) -> tuple[pd.Series, pd.Series]:
    # 分桶依据也延迟一天；t 日调仓只用 t-1 收盘时已知的收益波动率。
    vol = realized_volatility(price.shift(1), window=vol_window)
    aligned = pd.concat([score.rename("score"), vol.rename("vol")], axis=1, join="inner")
    aligned = aligned.replace([float("inf"), float("-inf")], float("nan")).dropna()
    if aligned.empty:
        raise ValueError("因子分数与历史波动率没有共同的有效样本")
    return aligned["score"], aligned["vol"]


def draw_comparison(
    results: dict[str, dict[str, object]], output_dir: Path, *, start: str, end: str, n_tickers: int
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for ax, (name, methods) in zip(axes.flat, results.items(), strict=True):
        for method, result in methods.items():
            nav = result.equity_curve["nav"] / 100_000
            ax.plot(nav.index, nav, label=method, linewidth=1.2)
        ax.set_title(name)
        ax.set_ylabel("NAV / initial capital")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(f"Standard quantiles vs volatility buckets | {n_tickers} stocks | {start}–{end}")
    fig.tight_layout()
    path = output_dir / "comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2026-09-21")
    parser.add_argument("--vol-window", type=int, default=21)
    parser.add_argument("--vol-groups", type=int, default=2)
    parser.add_argument("--quantiles", type=int, default=5)
    args = parser.parse_args()
    if args.vol_window < 2 or args.vol_groups < 1 or args.quantiles < 2:
        parser.error("vol-window 至少 2，vol-groups 至少 1，quantiles 至少 2")
    db = args.db.expanduser().resolve()
    if not db.is_file():
        parser.error(f"找不到数据库：{db}")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tickers = available_tickers(db, args.start, args.end)
    if len(tickers) < args.vol_groups * args.quantiles:
        parser.error("股票数不足以在每个波动率层内再分出指定的分位组")

    print(f"数据库：{db}")
    print(f"区间：{args.start} ~ {args.end}；标的：{len(tickers)}；调仓间隔：21 日")
    print(f"仓位：普通 {args.quantiles} 分位 vs {args.vol_groups} 个波动率层 × 层内 {args.quantiles} 分位")
    print("成本：单边佣金 5bp + 滑点 5bp；波动率和因子都只用前一日及更早的数据。")
    print(f"\n| 因子 | 仓位构造 | 年化收益 | Sharpe | 最大回撤 | 年化换手 | Q1/Q{args.quantiles} 前瞻收益 |")
    print("|---|---|---:|---:|---:|---:|---:|")

    all_results: dict[str, dict[str, object]] = {}
    summary_rows = []
    for name, params in SCENARIOS:
        bt = Backtester(
            tickers=tickers,
            start=args.start,
            end=args.end,
            factor_specs=[(name, params)],
            freq=21,
            n_quantiles=args.quantiles,
            db_path=str(db),
            commission_bps=5,
            slippage_bps=5,
        )
        bt.run(refresh_data=False)
        price = bt.price
        score, vol = paired_inputs(bt.score, price, args.vol_window)
        all_results[name] = {}

        for method in ("standard", "vol_bucket"):
            if method == "standard":
                weight = quantile_long_short(score=score, n_quantiles=args.quantiles)
                qret = quantile_forward_returns(score, price, freq=21, n_quantiles=args.quantiles)
            else:
                weight = vol_neutral_quantile_long_short(
                    score=score, vol=vol, n_vol_groups=args.vol_groups, n_quantiles=args.quantiles
                )
                qret = vol_neutral_quantile_forward_returns(
                    score, vol, price, freq=21, n_vol_groups=args.vol_groups, n_quantiles=args.quantiles
                )
            if weight.empty:
                raise ValueError(f"{name}/{method} 没有产生仓位，请减少分组数")
            result = run_backtest(
                price, weight, freq=21, commission_bps=5, slippage_bps=5
            )
            all_results[name][method] = result

            fig = plot_tearsheet(result.equity_curve, quantile_returns=qret)
            fig.suptitle(f"{name} | {method}", fontsize=14)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            chart = output_dir / f"{name}_{method}.png"
            fig.savefig(chart, dpi=150)
            plt.close(fig)

            q1 = qret.get(0, float("nan")) * 100
            q5 = qret.get(args.quantiles - 1, float("nan")) * 100
            summary_rows.append({
                "factor": name,
                "sizing": method,
                "return_ann_pct": result.return_ann_pct,
                "sharpe_ratio": result.sharpe_ratio,
                "max_drawdown_pct": result.max_drawdown_pct,
                "turnover_ann_pct": result.turnover_ann_pct,
                "q1_forward_return_pct": q1,
                "top_quantile_forward_return_pct": q5,
                "chart": chart.name,
            })
            print(
                f"| {name} | {method} | {result.return_ann_pct:.2f}% | {result.sharpe_ratio:.3f} | "
                f"{result.max_drawdown_pct:.2f}% | {result.turnover_ann_pct:.1f}% | {q1:.2f}% / {q5:.2f}% |"
            )

    pd.DataFrame(summary_rows).to_csv(output_dir / "summary.csv", index=False)
    comparison = draw_comparison(
        all_results, output_dir, start=args.start, end=args.end, n_tickers=len(tickers)
    )
    print(f"\n总览图：{comparison}")
    print(f"各因子的 8 张完整 tearsheet：{output_dir}")


if __name__ == "__main__":
    main()
