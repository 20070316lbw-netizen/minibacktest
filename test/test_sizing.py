from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from minibacktest.portfolio.sizing import (
    buffered_quantile_long_short,
    buffered_vol_neutral_quantile_long_short,
    make_buffered_quantile_sizer,
    make_buffered_vol_neutral_sizer,
    make_vol_neutral_sizer,
    quantile_long_short,
    vol_neutral_quantile_long_short,
)


def test_quantile_long_short_invalid_n_quantiles():
    d = pd.Timestamp("2024-01-01")
    s = pd.Series([1.0, 2.0], index=pd.MultiIndex.from_product([[d], ["A", "B"]]))
    with pytest.raises(ValueError, match="n_quantiles 至少为2"):
        quantile_long_short(score=s, n_quantiles=1)


def test_quantile_long_short_basic():
    d = pd.Timestamp("2024-01-01")
    tickers = ["A", "B", "C", "D", "E", "F"]
    scores = [5.0, 3.0, 1.0, -1.0, -3.0, -5.0]
    idx = pd.MultiIndex.from_product([[d], tickers], names=["date", "ticker"])
    s = pd.Series(scores, index=idx)

    w = quantile_long_short(score=s, n_quantiles=3)

    # 3分位:
    # 最高组: A, B (权重各自 1/2 = 0.5)
    # 中间组: C, D (权重 0.0)
    # 最低组: E, F (权重各自 -1/2 = -0.5)
    assert w.loc[(d, "A")] == 0.5
    assert w.loc[(d, "B")] == 0.5
    assert w.loc[(d, "C")] == 0.0
    assert w.loc[(d, "D")] == 0.0
    assert w.loc[(d, "E")] == -0.5
    assert w.loc[(d, "F")] == -0.5

    # 验证多头和为 1, 空头和为 -1, 总权重和为 0 (市场中性)
    assert np.isclose(w[w > 0].sum(), 1.0)
    assert np.isclose(w[w < 0].sum(), -1.0)
    assert np.isclose(w.sum(), 0.0)


def test_quantile_long_short_insufficient_stocks():
    d = pd.Timestamp("2024-01-01")
    tickers = ["A", "B"]
    idx = pd.MultiIndex.from_product([[d], tickers], names=["date", "ticker"])
    s = pd.Series([1.0, 2.0], index=idx)

    # 只有2只股票，但要求分5组 -> 当天不产生权重输出
    w = quantile_long_short(score=s, n_quantiles=5)
    assert len(w) == 0


def test_quantile_long_short_multiple_dates():
    dates = pd.date_range("2024-01-01", periods=2, freq="D")
    tickers = ["A", "B", "C", "D"]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    s = pd.Series([10.0, 5.0, -5.0, -10.0, -10.0, -5.0, 5.0, 10.0], index=idx)

    w = quantile_long_short(score=s, n_quantiles=2)

    for d in dates:
        day_w = w.xs(d, level="date")
        assert np.isclose(day_w[day_w > 0].sum(), 1.0)
        assert np.isclose(day_w[day_w < 0].sum(), -1.0)


def test_quantile_long_short_handles_nans():
    d = pd.Timestamp("2024-01-01")
    tickers = ["A", "B", "C", "D", "E"]
    idx = pd.MultiIndex.from_product([[d], tickers], names=["date", "ticker"])
    # C 是 NaN，有效股票 4 只
    s = pd.Series([4.0, 3.0, np.nan, 2.0, 1.0], index=idx)

    w = quantile_long_short(score=s, n_quantiles=2)
    # NaN 应该被剔除，不出现在输出中或被 drop
    assert (d, "C") not in w.index
    assert len(w) == 4
    assert np.isclose(w[w > 0].sum(), 1.0)
    assert np.isclose(w[w < 0].sum(), -1.0)


# ---------------------------------------------------------------------------
# buffered_quantile_long_short
# ---------------------------------------------------------------------------


def _two_day_score(day1: list[float], day2: list[float]) -> pd.Series:
    """10 只票 A~J 两个调仓日的分数, 方便手算的小例子。"""
    dates = pd.to_datetime(["2024-01-01", "2024-02-01"])
    tickers = list("ABCDEFGHIJ")
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    return pd.Series(day1 + day2, index=idx, dtype=float)


def _random_panel(seed: int, n_dates: int = 24, n_tickers: int = 60) -> pd.Series:
    """随机面板: 分数有自相关(模拟真实因子), 带 NaN(模拟没上市/缺数据),
    还有一部分四舍五入制造同分, 用来压边界情况。"""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_dates, freq="MS")
    tickers = [f"T{i:03d}" for i in range(n_tickers)]
    x = np.zeros((n_dates, n_tickers))
    x[0] = rng.normal(size=n_tickers)
    for t in range(1, n_dates):
        x[t] = 0.7 * x[t - 1] + 0.3 * rng.normal(size=n_tickers)
    x[:, :10] = np.round(x[:, :10], 1)  # 前 10 只制造同分
    x[rng.random(x.shape) < 0.05] = np.nan  # 随机缺失
    x[: n_dates // 3, -8:] = np.nan  # 最后 8 只"晚上市"
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    return pd.Series(x.ravel(), index=idx)


@pytest.mark.parametrize("n_quantiles", [2, 3, 5, 10])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_buffered_zero_buffer_equals_plain(n_quantiles, seed):
    # 最重要的回归测试: buffer=0 必须跟旧函数逐元素相等
    s = _random_panel(seed)
    plain = quantile_long_short(score=s, n_quantiles=n_quantiles)
    buffered = buffered_quantile_long_short(score=s, n_quantiles=n_quantiles, buffer=0.0)
    pd.testing.assert_series_equal(buffered, plain, check_names=False)


def test_buffered_keeps_holding_inside_buffer():
    # 第一期: A,B 进多头(五分位, 10 只票每组 2 只), I,J 进空头
    # 第二期: B 掉到第 3 名(分数 8), 在 60% 分位点(6.4)之上 -> 留任
    #         D 分数 7 也在缓冲区里, 但上期不在多头 -> 不能进
    day1 = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]  # A..J
    day2 = [10, 8, 9, 7, 6, 5, 4, 3, 2, 1]  # B 和 C 互换
    s = _two_day_score(day1, day2)
    d2 = pd.Timestamp("2024-02-01")

    w = buffered_quantile_long_short(score=s, n_quantiles=5, buffer=0.2)
    w2 = w.xs(d2, level="date")

    assert sorted(w2[w2 > 0].index) == ["A", "B", "C"]
    assert np.allclose(w2[w2 > 0], 1 / 3)
    assert w2["D"] == 0.0
    assert sorted(w2[w2 < 0].index) == ["I", "J"]

    # 对照: 不设缓冲带时 B 会被卖掉
    p2 = quantile_long_short(score=s, n_quantiles=5).xs(d2, level="date")
    assert sorted(p2[p2 > 0].index) == ["A", "C"]


def test_buffered_exits_when_breaking_threshold():
    # 第二期 B 跌到 5 分, 低于 60% 分位点 6.4 -> 卖出
    day1 = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    day2 = [10, 5, 9, 8, 7, 6, 4, 3, 2, 1]
    s = _two_day_score(day1, day2)
    w2 = buffered_quantile_long_short(score=s, n_quantiles=5, buffer=0.2).xs(
        pd.Timestamp("2024-02-01"), level="date"
    )
    assert sorted(w2[w2 > 0].index) == ["A", "C"]
    assert w2["B"] == 0.0


def test_buffered_short_leg_symmetric():
    # 空头对称: I 从倒数第 2 升到倒数第 3(分数 3), 不高于 40% 分位点 4.6 -> 留任
    day1 = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    day2 = [10, 9, 8, 7, 6, 5, 4, 2, 3, 1]  # H 和 I 互换
    s = _two_day_score(day1, day2)
    w2 = buffered_quantile_long_short(score=s, n_quantiles=5, buffer=0.2).xs(
        pd.Timestamp("2024-02-01"), level="date"
    )
    assert sorted(w2[w2 < 0].index) == ["H", "I", "J"]
    assert np.allclose(w2[w2 < 0], -1 / 3)


@pytest.mark.parametrize("buffer", [0.05, 0.1, 0.2, 0.29])
def test_buffered_invariants_on_random_panel(buffer):
    s = _random_panel(seed=42)
    w = buffered_quantile_long_short(score=s, n_quantiles=5, buffer=buffer)
    plain = quantile_long_short(score=s, n_quantiles=5)

    for d, day in w.groupby(level="date"):
        # 两条腿各自归一, 市场中性
        assert np.isclose(day[day > 0].sum(), 1.0)
        assert np.isclose(day[day < 0].sum(), -1.0)
        # 只数只会 >= 不带缓冲带的版本
        p = plain.xs(d, level="date")
        assert (day > 0).sum() >= (p > 0).sum()
        assert (day < 0).sum() >= (p < 0).sum()


def test_buffered_reduces_turnover_on_persistent_signal():
    s = _random_panel(seed=7)

    def total_turnover(w: pd.Series) -> float:
        wide = w.unstack("ticker").fillna(0.0)
        return wide.diff().abs().sum(axis=1).iloc[1:].sum()

    plain = quantile_long_short(score=s, n_quantiles=5)
    buffered = buffered_quantile_long_short(score=s, n_quantiles=5, buffer=0.15)
    assert total_turnover(buffered) < total_turnover(plain)


@pytest.mark.parametrize("buffer", [-0.1, 0.3, 0.5])
def test_buffered_invalid_buffer(buffer):
    s = _random_panel(seed=0)
    with pytest.raises(ValueError, match="buffer 必须在"):
        buffered_quantile_long_short(score=s, n_quantiles=5, buffer=buffer)
    with pytest.raises(ValueError, match="buffer 必须在"):
        make_buffered_quantile_sizer(n_quantiles=5, buffer=buffer)


def test_make_buffered_sizer_matches_function():
    s = _random_panel(seed=3)
    sizer = make_buffered_quantile_sizer(n_quantiles=5, buffer=0.1)
    expected = buffered_quantile_long_short(score=s, n_quantiles=5, buffer=0.1)
    pd.testing.assert_series_equal(sizer(s, pd.DataFrame()), expected)


# ---------------------------------------------------------------------------
# buffered_vol_neutral_quantile_long_short
# ---------------------------------------------------------------------------


def _random_vol_panel(seed: int, n_dates: int = 24, n_tickers: int = 60) -> pd.Series:
    """跟 _random_panel 同样的索引, 波动率有自相关(偶尔换层), 带少量 NaN。"""
    rng = np.random.default_rng(seed + 1000)
    dates = pd.date_range("2020-01-01", periods=n_dates, freq="MS")
    tickers = [f"T{i:03d}" for i in range(n_tickers)]
    base = rng.lognormal(mean=0.0, sigma=0.5, size=n_tickers)
    v = base * np.exp(0.2 * rng.normal(size=(n_dates, n_tickers)))
    v[rng.random(v.shape) < 0.03] = np.nan
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    return pd.Series(v.ravel(), index=idx)


@pytest.mark.parametrize(("n_vol_groups", "n_quantiles"), [(2, 2), (2, 5), (3, 3), (5, 2), (5, 5)])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_buffered_vol_zero_buffer_equals_plain(n_vol_groups, n_quantiles, seed):
    # 回归测试: buffer=0 必须跟 vol_neutral_quantile_long_short 逐元素相等
    s, v = _random_panel(seed), _random_vol_panel(seed)
    plain = vol_neutral_quantile_long_short(
        score=s, vol=v, n_vol_groups=n_vol_groups, n_quantiles=n_quantiles
    )
    buffered = buffered_vol_neutral_quantile_long_short(
        score=s, vol=v, n_vol_groups=n_vol_groups, n_quantiles=n_quantiles, buffer=0.0
    )
    pd.testing.assert_series_equal(buffered, plain, check_names=False)


def _two_layer_panel(score2: dict[str, float], vol2: dict[str, float]) -> tuple[pd.Series, pd.Series]:
    """20 只票两期: 第一期 A~J 低波动(分数 10..1), K~T 高波动(分数 100..91);
    第二期只改动 score2 / vol2 里给出的票。"""
    dates = pd.to_datetime(["2024-01-01", "2024-02-01"])
    tickers = list("ABCDEFGHIJKLMNOPQRST")
    s1 = dict(zip(tickers, [*range(10, 0, -1), *range(100, 90, -1)], strict=True))
    v1 = {t: (1.0 if i < 10 else 9.0) for i, t in enumerate(tickers)}
    s2, v2 = {**s1, **score2}, {**v1, **vol2}
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    score = pd.Series([s1[t] for t in tickers] + [s2[t] for t in tickers], index=idx, dtype=float)
    vol = pd.Series([v1[t] for t in tickers] + [v2[t] for t in tickers], index=idx, dtype=float)
    return score, vol


def _legs(w: pd.Series, date: str) -> tuple[list[str], list[str]]:
    day = w.xs(pd.Timestamp(date), level="date")
    return sorted(day[day > 0].index), sorted(day[day < 0].index)


def test_buffered_vol_keeps_holding_within_layer():
    # 第一期多头: 低波动层 A,B + 高波动层 K,L
    # 第二期低波动层里 B 和 C 互换(B=8), 仍在层内 60% 分位点(6.4)之上 -> 留任
    score, vol = _two_layer_panel({"B": 8, "C": 9}, {})
    w = buffered_vol_neutral_quantile_long_short(
        score=score, vol=vol, n_vol_groups=2, n_quantiles=5, buffer=0.2
    )
    long_, short = _legs(w, "2024-02-01")
    assert long_ == ["A", "B", "C", "K", "L"]
    assert np.allclose(w.xs(pd.Timestamp("2024-02-01"), level="date")[long_], 1 / 5)
    assert short == ["I", "J", "S", "T"]

    plain = vol_neutral_quantile_long_short(score=score, vol=vol, n_vol_groups=2, n_quantiles=5)
    assert _legs(plain, "2024-02-01")[0] == ["A", "C", "K", "L"]


def test_buffered_vol_judges_by_current_layer():
    # B 第二期波动率升高换到高波动层, T 降到低波动层。B 的分数 8 在高波动层
    # 里垫底 -> 不但不能留在多头, 还按本期所在层进了空头; T 同理反过来
    score, vol = _two_layer_panel({"B": 8}, {"B": 9.0, "T": 1.0})
    w = buffered_vol_neutral_quantile_long_short(
        score=score, vol=vol, n_vol_groups=2, n_quantiles=5, buffer=0.2
    )
    long_, short = _legs(w, "2024-02-01")
    assert long_ == ["A", "K", "L", "T"]
    assert short == ["B", "I", "J", "S"]


@pytest.mark.parametrize("buffer", [0.05, 0.1, 0.2, 0.29])
def test_buffered_vol_invariants_on_random_panel(buffer):
    s, v = _random_panel(seed=42), _random_vol_panel(seed=42)
    w = buffered_vol_neutral_quantile_long_short(
        score=s, vol=v, n_vol_groups=3, n_quantiles=5, buffer=buffer
    )
    plain = vol_neutral_quantile_long_short(score=s, vol=v, n_vol_groups=3, n_quantiles=5)
    for d, day in w.groupby(level="date"):
        assert np.isclose(day[day > 0].sum(), 1.0)
        assert np.isclose(day[day < 0].sum(), -1.0)
        p = plain.xs(d, level="date")
        assert (day > 0).sum() >= (p > 0).sum()
        assert (day < 0).sum() >= (p < 0).sum()


@pytest.mark.parametrize("buffer", [-0.1, 0.3, 0.5])
def test_buffered_vol_invalid_buffer(buffer):
    s, v = _random_panel(seed=0), _random_vol_panel(seed=0)
    with pytest.raises(ValueError, match="buffer 必须在"):
        buffered_vol_neutral_quantile_long_short(
            score=s, vol=v, n_vol_groups=2, n_quantiles=5, buffer=buffer
        )
    with pytest.raises(ValueError, match="buffer 必须在"):
        make_buffered_vol_neutral_sizer(n_vol_groups=2, n_quantiles=5, buffer=buffer)


def test_make_buffered_vol_sizer_zero_buffer_matches_plain_sizer():
    # 用真实的 (score, price) 接口走一遍: 波动率在 sizer 内部现场算
    rng = np.random.default_rng(5)
    days = pd.bdate_range("2020-01-01", periods=160)
    tickers = [f"T{i:02d}" for i in range(40)]
    price = pd.DataFrame(
        100 * np.exp(np.cumsum(0.02 * rng.normal(size=(len(days), len(tickers))), axis=0)),
        index=days, columns=tickers,
    )
    rb = days[40::21]
    idx = pd.MultiIndex.from_product([rb, tickers], names=["date", "ticker"])
    score = pd.Series(rng.normal(size=len(idx)), index=idx)

    plain = make_vol_neutral_sizer(n_vol_groups=2, n_quantiles=5)(score, price)
    buffered0 = make_buffered_vol_neutral_sizer(n_vol_groups=2, n_quantiles=5, buffer=0.0)(score, price)
    pd.testing.assert_series_equal(buffered0, plain, check_names=False)

    buffered = make_buffered_vol_neutral_sizer(n_vol_groups=2, n_quantiles=5, buffer=0.15)(score, price)
    assert not buffered.empty
