"""杂项指标: 不好单独归到 equity / returns / drawdown / trades 的小工具。"""

from __future__ import annotations

import pandas as pd


def exposure_time_pct(w_hold: pd.DataFrame) -> float:
    """持仓时间占比: 有任意非零权重的交易日, 占总交易日的比例(百分比)。

    Args:
        w_hold: 逐日生效持仓权重矩阵, index 是日期, columns 是 ticker。

    Returns:
        0~100 的 float; w_hold 为空时返回 NaN。
    """
    if w_hold.empty:
        return float("nan")
    has_position = w_hold.fillna(0.0).abs().sum(axis=1) > 0
    return float(has_position.mean() * 100)
