"""交易层面的统计: 把逐日持仓权重矩阵切成一笔笔"trade", 算胜率/盈亏比等指标。

这里的"trade"跟单标的择时策略(比如 backtesting.py)里的"trade"含义不同:
本项目是截面多空组合, 没有单一的买入/卖出时点。这里把"trade"定义为:
    某只股票的持仓权重从 0 变为非 0, 一直持有到权重变回 0 为止的一段连续
    区间; 区间收益率按该股票在持仓期间的价格涨跌幅计算(做空方向取反)。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def extract_trades(w_hold: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    """把持仓权重矩阵切成一笔笔 trade。

    Args:
        w_hold: 逐日生效持仓权重矩阵(已经 shift(1) 消除未来函数), index 是
            日期, columns 是 ticker, 值是权重(0 表示当天不持有)。
        price: 收盘价矩阵, 与 w_hold 同 index/columns。

    Returns:
        DataFrame, 每行一笔 trade, 列为
        [ticker, entry_date, exit_date, direction, duration, return_pct]。
        direction 是 1(多)或 -1(空); return_pct 是按方向调整过的价格
        涨跌幅(百分比)。没有任何持仓时返回空的同结构 DataFrame。
    """
    columns = ["ticker", "entry_date", "exit_date", "direction", "duration", "return_pct"]
    held = w_hold.fillna(0.0) != 0.0
    if not held.to_numpy().any():
        return pd.DataFrame(columns=columns)

    records: list[dict[str, object]] = []
    for ticker in w_hold.columns:
        col_held = held[ticker]
        if not col_held.any():
            continue

        col_w = w_hold[ticker]
        col_price = price[ticker]

        # 用"是否持仓"发生变化的地方, 把时间轴切成一段段连续区间
        change = col_held.ne(col_held.shift(fill_value=False))
        block_id = change.cumsum()

        for idx in col_held[col_held].groupby(block_id[col_held]).groups.values():
            entry_date, exit_date = idx[0], idx[-1]
            entry_price = col_price.loc[entry_date]
            exit_price = col_price.loc[exit_date]
            if pd.isna(entry_price) or pd.isna(exit_price) or entry_price == 0:
                continue

            direction = 1 if col_w.loc[entry_date] > 0 else -1
            return_pct = direction * (exit_price / entry_price - 1.0) * 100

            records.append(
                {
                    "ticker": ticker,
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "direction": direction,
                    "duration": exit_date - entry_date,
                    "return_pct": float(return_pct),
                }
            )

    return pd.DataFrame(records, columns=columns)


def trade_stats(trades: pd.DataFrame) -> dict[str, object]:
    """从 extract_trades 的结果汇总胜率/盈亏比/SQN/凯利仓位等交易层面指标。

    Args:
        trades: extract_trades 的输出。

    Returns:
        字段跟 base.Result 里交易层面的字段一一对应的 dict, 交易数为 0
        时数量类字段为 0、比率类字段为 NaN。
    """
    if trades.empty:
        return {
            "n_trades": 0,
            "win_rate_pct": float("nan"),
            "best_trade_pct": float("nan"),
            "worst_trade_pct": float("nan"),
            "avg_trade_pct": float("nan"),
            "max_trade_duration": pd.Timedelta(0),
            "avg_trade_duration": pd.Timedelta(0),
            "profit_factor": float("nan"),
            "expectancy_pct": float("nan"),
            "sqn": float("nan"),
            "kelly_criterion": float("nan"),
        }

    r = trades["return_pct"]
    wins, losses = r[r > 0], r[r < 0]
    win_rate = len(wins) / len(r)


    gross_profit, gross_loss = wins.sum(), losses.abs().sum()
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("nan")

    avg_win = wins.mean() if len(wins) else 0.0
    avg_loss = losses.abs().mean() if len(losses) else 0.0
    payoff_ratio = avg_win / avg_loss if avg_loss > 0 else float("nan")
    kelly = (
        win_rate - (1 - win_rate) / payoff_ratio
        if payoff_ratio and not np.isnan(payoff_ratio) and payoff_ratio > 0
        else float("nan")
    )

    sqn = (
        float(np.sqrt(len(r)) * r.mean() / r.std())
        if len(r) > 1 and r.std() > 0
        else float("nan")
    )

    return {
        "n_trades": len(r),
        "win_rate_pct": float(win_rate * 100),
        "best_trade_pct": float(r.max()),
        "worst_trade_pct": float(r.min()),
        "avg_trade_pct": float(r.mean()),
        "max_trade_duration": trades["duration"].max(),
        "avg_trade_duration": trades["duration"].mean(),
        "profit_factor": float(profit_factor),
        "expectancy_pct": float(r.mean()),
        "sqn": sqn,
        "kelly_criterion": float(kelly),
    }
