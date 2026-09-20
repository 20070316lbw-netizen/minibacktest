"""对所有已经标准化的因子进行加权复合变成一个总分"""

from __future__ import annotations

import pandas as pd

def combine_scores(
    z: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> pd.Series:
    """把标准化后的多个因子列合成一个综合分数, 逐行按权重加权平均, 每行只用它实际有值

    核心: 加权平均, 用分数除以权重获取平均下来某只股票的水平. 加权平均对

    Args:
        z: 标准化后的因子面板(见 zscore_by_date), [date, ticker] MultiIndex,
            一列一个因子
        weights: 每个因子的合成权重, 键是因子名(z 的列名)、值是权重;
            为 None 时等权(z 的每一列权重相等)。权重不要求归一化到 1,
            内部按每行实际参与因子的权重绝对值之和归一化。weights 里
            未出现的因子权重记为 0(即不参与合成)

    Returns:
        综合分数, pd.Series, 索引跟 z 一样是 [date, ticker]; 某行所有
        因子都缺失(或权重全为 0)时为 NaN

    Raises:
        KeyError: weights 里出现了 z 没有的因子名

    Example:
        >>> dates = pd.date_range("2024-01-01", periods=1, freq="D")
        >>> idx = pd.MultiIndex.from_product([dates, ["A", "B", "C"]], names=["date", "ticker"])
        >>> z = pd.DataFrame({"f1": [-1.0, 0.0, 1.0], "f2": [-0.71, float("nan"), 0.71]}, index=idx)
        >>> combine_scores(z)  # B 的 f2 缺失, 只用 f1
        date        ticker
        2024-01-01  A        -0.855
                    B         0.000
                    C         0.855
        dtype: float64
    """

    # 如果没给权重就等权
    if weights is None:
        w = pd.Series(1.0, index = z.columns)

    else:
        # 对数据进行校验, 对齐权重到 z 的列
        missing = set(weights) - set(z.columns)

        if missing:
            raise KeyError(f"weights 里面有 标准化后没有的因子 -- {missing}")

        w = pd.Series(weights, index = list(weights)).reindex(z.columns).fillna(0.0)

    present = z.notna()                                 # 获取一个和 z 同形状的布尔矩阵, 标记每个因子是否有值
    numer = z.fillna(0.0).mul(w, axis=1).sum(axis=1)    # 把每行内的 NaN 当成 0 , 乘权重后求和
    denom = present.mul(w.abs(), axis=1).sum(axis=1)    # 把每行实际有值的因子的权重绝对值求和

    return (numer / denom).where(denom > 0)