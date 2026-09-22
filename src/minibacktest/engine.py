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
    commission_bps: float = 0.0,
    slippage_bps: float = 0.0,
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
        commission_bps: 单边佣金费率(基点, 1bp = 0.01%), 按每天持仓权重相
            对前一天的变动量(换手)计, 默认 0(不计佣金)。
        slippage_bps: 单边滑点费率(基点), 计法与 commission_bps 相同, 两者
            会直接相加, 默认 0(不计滑点)。

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

    # 3. 逐日组合毛收益率 = 权重 · 当日个股收益率(向量化, 未扣费用前)
    asset_returns = price.pct_change().fillna(0.0)
    gross_daily_returns = (w_hold * asset_returns).sum(axis=1)

    # 3.5 交易成本: 换手 = 每天持仓权重相对前一天的变动量之和(绝对值), 换仓
    #     当天(w_hold 相对昨天发生变化的那天)才会产生非零换手, 期间买入
    #     持有不换手; 佣金 + 滑点按换手名义金额的固定费率计, 直接从当天
    #     组合收益率里扣掉(向量化近似, 不逐笔模拟委托/成交)。
    turnover = w_hold.diff().abs().sum(axis=1).fillna(0.0)
    cost_rate = (commission_bps + slippage_bps) / 10_000.0
    daily_cost = turnover * cost_rate
    daily_returns = gross_daily_returns - daily_cost

    # 4. 净值曲线: 策略(扣费后) vs. 策略(未扣费, 仅用于估算费用拖累) vs.
    #    等权买入持有基准
    nav = build_nav(daily_returns, initial_capital=initial_capital)
    gross_nav = build_nav(gross_daily_returns, initial_capital=initial_capital)
    benchmark_nav = buy_and_hold_nav(price, initial_capital=initial_capital)

    # 5. 回撤持续时间 / Alpha·Beta(依赖净值曲线, 单独算一次)
    max_dd_duration, avg_dd_duration = drawdown_durations(nav)
    alpha_pct, beta = alpha_beta(nav, benchmark_nav, periods_per_year)

    # 6. 换手 / 费用拖累汇总: 年化换手率按日均换手折算, 费用拖累用"未扣费
    #    净值 - 扣费净值"占期初资金的比例来近似(不是简单加总每日费用率,
    #    避免忽略复利效应)。
    turnover_ann_pct = float(turnover.mean() * periods_per_year * 100)
    total_cost_pct = float((gross_nav.iloc[-1] - nav.iloc[-1]) / initial_capital * 100)

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
        equity_curve=pd.DataFrame(
            {"nav": nav, "benchmark_nav": benchmark_nav, "gross_nav": gross_nav}
        ),
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        turnover_ann_pct=turnover_ann_pct,
        total_cost_pct=total_cost_pct,
    )
