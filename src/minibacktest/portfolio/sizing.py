"""仓位构造

目前设计的是一个
    按分数把每个调仓日的横截面分成 n_quantiles 组, 最高组等权做多、
    最低组等权做空, 中间组权重为 0, 多空两腿各自内部等权且总敞口相等
    (多头权重和为 +1, 空头权重和为 -1)



"""

from __future__ import annotations

import pandas as pd

def _weight(s: pd.Series) -> pd.Series:
    ...


def quantile_long_short() -> pd.Series:
    ...