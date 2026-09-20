"""回测结果的组装与调仓日定权重, 期间买入持有"""

from __future__ import annotations

import pandas as pd

from minibacktest.base import Result
from minibacktest.evaluation.chores import exposure_time_pct
from minibacktest.evaluation.drawdown import (
    avg_drawdown,
    drawdown_durations,
    max_drawdown,
)
from minibacktest.evaluation.equity import build_nav, buy_and_hold_nav
from minibacktest.evaluation.returns import (
    alpha_beta,
    annualized_return,
    annualized_volatility,
    cagr,
    total_return,
)
from minibacktest.evaluation.risk_adjusted_ratios import (
    calmar_ratio,
    sharpe_ratio,
    sortino_ratio,
)
from minibacktest.rebalance import rebalance_block

PERIODS_PER_YEAR = 252


def run_backtest(
    price: pd.DataFrame,
    target_weight: pd.Series,
    *,
    freq: int,
    initial_capital: float = 100_000.0,
    periods_per_year: float = PERIODS_PER_YEAR,
) -> Result:
    """跑一次完整的截面多空回测: 调仓日定权重, 期间买入持有, 到下个调仓日
    再换仓, 最后把逐日净值、回撤、风险调整收益、交易层面统计组装成 Result。

    Args:
        price: 收盘价宽表(比如 close 或 adj_close 透视后), index 是完整
            交易日历, columns 是 ticker, 需要覆盖 target_weight 里出现的
            所有 ticker。
        target_weight: 调仓日的目标权重, [date, ticker] MultiIndex(比如
            portfolio.sizing.quantile_long_short 的输出), 只需要包含调仓
            日那些天的数据, 非调仓日不用给。
        freq: 调仓间隔(交易日数), 必须和生成 target_weight 时用的一致,
            这里只用它把权重从调仓日铺开到每个交易日(见 rebalance.py)。
        initial_capital: 起始资金, 默认 100000。
        periods_per_year: 年化用的每年观测点数, 日频默认 252。

    Returns:
        Result, 组装好的完整回测统计结果。
    """
    price = price.sort_index()
    dates = price.index

    # 1. 把调仓日权重铺开成每日权重: 每天用"最近一次已发生的调仓日"的权重,
    #    (某个调仓日在 target_weight 里没有输出的话, 视为当天空仓 0)
    w_target = (
        target_weight.unstack("ticker")
        .reindex(columns=price.columns, fill_value=0.0)
    )
    block = rebalance_block(dates, freq)
    w_daily = (
        w_target.reindex(block.to_numpy())
        .set_axis(block.index, axis=0)
        .reindex(dates)
        .fillna(0.0)
    )

    # 2. shift(1) 消除未来函数: 今天生效的持仓, 是昨天收盘时就定好的权重
    w_hold = w_daily.shift(1).fillna(0.0)

    # 3. 逐日组合收益率 = 权重 · 当日个股收益率(向量化, 不显式算每笔换手/费用)
    asset_returns = price.pct_change().fillna(0.0)
    daily_returns = (w_hold * asset_returns).sum(axis=1)

    # 4. 净值曲线: 策略 vs. 等权买入持有基准
    nav = build_nav(daily_returns, initial_capital=initial_capital)
    benchmark_nav = buy_and_hold_nav(price, initial_capital=initial_capital)

    # 5. 回撤持续时间 / Alpha·Beta(依赖净值曲线, 单独算一次)
    max_dd_duration, avg_dd_duration = drawdown_durations(nav)
    alpha_pct, beta = alpha_beta(nav, benchmark_nav, periods_per_year)

    return Result(
        start=dates[0].to_pydatetime(),
        end=dates[-1].to_pydatetime(),
        duration=dates[-1] - dates[0],
        exposure_time_pct=exposure_time_pct(w_hold),
        equity_final=float(nav.iloc[-1]),
        equity_peak=float(nav.max()),
        return_pct=total_return(nav) * 100,
        buy_and_hold_return_pct=total_return(benchmark_nav) * 100,
        return_ann_pct=annualized_return(nav, periods_per_year) * 100,
        volatility_ann_pct=annualized_volatility(nav, periods_per_year) * 100,
        cagr_pct=cagr(nav) * 100,
        sharpe_ratio=sharpe_ratio(nav, periods_per_year),
        sortino_ratio=sortino_ratio(nav, periods_per_year),
        calmar_ratio=calmar_ratio(nav, periods_per_year),
        alpha_pct=alpha_pct,
        beta=beta,
        max_drawdown_pct=max_drawdown(nav) * 100,
        avg_drawdown_pct=avg_drawdown(nav) * 100,
        max_drawdown_duration=max_dd_duration,
        avg_drawdown_duration=avg_dd_duration,
        strategy="quantile_long_short",
        equity_curve=pd.DataFrame({"nav": nav, "benchmark_nav": benchmark_nav}),
    )
