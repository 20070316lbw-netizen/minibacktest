from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from minibacktest.factors import available, get, register
from minibacktest.factors.momentum import momentum
from minibacktest.factors.reversal import reversal


def test_factor_registry_basic():
    # 验证已内置的因子
    all_factors = available()
    assert "momentum" in all_factors
    assert "reversal" in all_factors

    # 成功获取
    fn = get("momentum")
    assert callable(fn)


def test_factor_registry_get_unknown():
    with pytest.raises(KeyError, match="没有注册名为 'nonexistent' 的因子"):
        get("nonexistent")


def test_factor_registry_duplicate():
    # 注册一个临时因子
    @register("test_temp_factor")
    def _dummy(price, **kwargs):
        return pd.Series(dtype=float)

    assert "test_temp_factor" in available()

    # 重复注册报错
    with pytest.raises(ValueError, match="已经被注册过"):
        @register("test_temp_factor")
        def _dummy2(price, **kwargs):
            return pd.Series(dtype=float)


def test_momentum_factor_calculation():
    dates = pd.bdate_range("2024-01-01", periods=5)
    price = pd.DataFrame({"A": [1.0, 1.1, 1.2, 1.3, 1.4]}, index=dates)

    # window=2
    # 日期 0 (01-01): NaN
    # 日期 1 (01-02): NaN
    # 日期 2 (01-03): NaN (shift(1) 是 01-02 的 1.1, shift(1+2)=shift(3) 是 NaN)
    # 日期 3 (01-04): shift(1) 是 01-03 (1.2), shift(3) 是 01-01 (1.0) -> 1.2/1.0 - 1 = 0.20
    # 日期 4 (01-05): shift(1) 是 01-04 (1.3), shift(3) 是 01-02 (1.1) -> 1.3/1.1 - 1 = 0.181818...
    mom = momentum(price, window=2)
    assert isinstance(mom, pd.Series)
    assert mom.name == "momentum"
    assert mom.index.names == ["date", "ticker"]

    valid = mom.dropna()
    assert len(valid) == 2
    assert np.isclose(valid.loc[(dates[3], "A")], 0.20)
    assert np.isclose(valid.loc[(dates[4], "A")], 1.3 / 1.1 - 1.0)


def test_momentum_no_lookahead():
    # 证明当天价格变动不影响当天的 momentum 输出 (t 时刻只能看到 t-1)
    dates = pd.bdate_range("2024-01-01", periods=5)
    price1 = pd.DataFrame({"A": [1.0, 1.1, 1.2, 1.3, 1.4]}, index=dates)
    price2 = pd.DataFrame({"A": [1.0, 1.1, 1.2, 1.3, 999.0]}, index=dates)  # 最后一天价格暴涨

    mom1 = momentum(price1, window=2)
    mom2 = momentum(price2, window=2)

    # 在最后一天，mom 应该完全相同，因为最后一天价格还没收盘定格
    assert np.isclose(mom1.loc[(dates[4], "A")], mom2.loc[(dates[4], "A")])


def test_reversal_factor_is_negative_momentum():
    dates = pd.bdate_range("2024-01-01", periods=6)
    price = pd.DataFrame(
        {
            "A": [10.0, 11.0, 12.0, 11.5, 13.0, 14.0],
            "B": [20.0, 19.0, 18.0, 18.5, 17.0, 16.0],
        },
        index=dates,
    )

    mom = momentum(price, window=2)
    rev = reversal(price, window=2)

    assert isinstance(rev, pd.Series)
    assert rev.name == "reversal"

    # 对齐后比对取负值
    pd.testing.assert_series_equal(rev, -mom, check_names=False)
