""""""

from __future__ import annotations

import pandas as pd
import numpy as np

def max_drawdown(nav: pd.Series) -> float:
    """算最大回撤(<= 0 的数, 比如 -0.23 代表最大回撤 23%)。

    Args:
        nav: 逐日净值。

    Returns:
        最大回撤, <= 0 的 float。

    Example:
        >>> nav = pd.Series([1.0, 1.2, 0.9, 1.1])
        >>> max_drawdown(nav)
        -0.25
    """
    running_max = nav.cummax()
    drawdown = nav / running_max - 1.0
    return float(drawdown.min())