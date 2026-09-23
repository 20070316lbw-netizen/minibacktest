from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from minibacktest.factors import (
    EXTRA_DIRS_ENV_VAR,
    available,
    compile_spec,
    get,
    load_specs,
    validate_spec,
)
from minibacktest.factors.registry import FactorSpecError, _compile


def test_factor_registry_basic():
    # 验证已注册的内置因子(现在是 factors/*.yaml, 不是装饰器)
    all_factors = available()
    assert "momentum" in all_factors
    assert "reversal" in all_factors

    # 成功获取
    fn = get("momentum")
    assert callable(fn)


def test_factor_registry_get_unknown():
    with pytest.raises(KeyError, match="没有注册名为 'nonexistent' 的因子"):
        get("nonexistent")


def test_factor_registry_duplicate_name_across_files(tmp_path, monkeypatch):
    # 两个 YAML 文件写了同一个 name, 扫描阶段就该报错(不是悄悄让后一个覆盖前一个)。
    import minibacktest.factors.registry as registry_mod

    (tmp_path / "a.yaml").write_text(
        "name: dup\nparams: []\nsteps:\n  - id: r\n    op: add\n    a: {const: 1}\n    b: {const: 1}\noutput: r\n",
        encoding="utf-8",
    )
    (tmp_path / "b.yaml").write_text(
        "name: dup\nparams: []\nsteps:\n  - id: r\n    op: add\n    a: {const: 2}\n    b: {const: 2}\noutput: r\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(registry_mod, "_FACTORS_DIR", tmp_path)

    with pytest.raises(FactorSpecError, match="重复出现"):
        registry_mod.available()


def test_factor_spec_missing_output_raises():
    spec = {"params": [], "steps": [{"id": "r", "op": "add", "a": {"const": 1}, "b": {"const": 1}}]}
    with pytest.raises(FactorSpecError, match="缺 output"):
        _compile(spec, factor_name="broken")


def test_factor_spec_unknown_op_raises():
    spec = {
        "params": [],
        "steps": [{"id": "r", "op": "not_a_real_op", "a": {"const": 1}, "b": {"const": 1}}],
        "output": "r",
    }
    # 静态校验: 编译(也就是扫描注册表)时就报, 不拖到真正算的时候。
    with pytest.raises(FactorSpecError, match="不在白名单里"):
        _compile(spec, factor_name="broken")


def test_factor_spec_unknown_param_raises():
    spec = {
        "params": ["window"],
        "steps": [{"id": "r", "op": "add", "a": {"const": 1}, "b": {"param": "window"}}],
        "output": "r",
    }
    fn = _compile(spec, factor_name="broken")
    dates = pd.bdate_range("2024-01-01", periods=3)
    price = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=dates)
    with pytest.raises(FactorSpecError, match="不认识参数"):
        fn(price, not_declared=1)


def test_momentum_factor_calculation():
    dates = pd.bdate_range("2024-01-01", periods=5)
    price = pd.DataFrame({"A": [1.0, 1.1, 1.2, 1.3, 1.4]}, index=dates)

    # window=2
    # 日期 3 (01-04): shift(1) 是 01-03 (1.2), shift(3) 是 01-01 (1.0) -> 1.2/1.0 - 1 = 0.20
    # 日期 4 (01-05): shift(1) 是 01-04 (1.3), shift(3) 是 01-02 (1.1) -> 1.3/1.1 - 1 = 0.181818...
    momentum = get("momentum")
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

    momentum = get("momentum")
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

    momentum = get("momentum")
    reversal = get("reversal")
    mom = momentum(price, window=2)
    rev = reversal(price, window=2)

    assert isinstance(rev, pd.Series)
    assert rev.name == "reversal"

    # 对齐后比对取负值
    pd.testing.assert_series_equal(rev, -mom, check_names=False)


# ---- 静态校验 / 防未来函数 / 额外因子目录 ----

def _price() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=6)
    return pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}, index=dates)


def test_negative_const_shift_rejected_at_compile():
    spec = {"steps": [{"id": "f", "op": "shift", "input": {"ref": "price"}, "by": {"const": -1}}], "output": "f"}
    with pytest.raises(FactorSpecError, match="未来函数"):
        compile_spec(spec, factor_name="peek")


def test_negative_param_shift_rejected_at_runtime():
    spec = {
        "params": ["n"],
        "steps": [{"id": "f", "op": "shift", "input": {"ref": "price"}, "by": {"param": "n"}}],
        "output": "f",
    }
    fn = compile_spec(spec, factor_name="peek")
    assert fn(_price(), n=1).notna().any()
    with pytest.raises(FactorSpecError, match="未来函数"):
        fn(_price(), n=-2)


def test_non_integer_shift_rejected():
    spec = {"steps": [{"id": "f", "op": "shift", "input": {"ref": "price"}, "by": {"const": 1.5}}], "output": "f"}
    with pytest.raises(FactorSpecError, match="必须是整数"):
        compile_spec(spec, factor_name="frac")


def test_ref_to_later_step_rejected():
    spec = {
        "steps": [
            {"id": "a", "op": "add", "a": {"ref": "b"}, "b": {"const": 1}},
            {"id": "b", "op": "add", "a": {"ref": "price"}, "b": {"const": 1}},
        ],
        "output": "b",
    }
    with pytest.raises(FactorSpecError, match="不存在"):
        validate_spec(spec, factor_name="loop")


def test_undeclared_param_and_extra_field_rejected():
    with pytest.raises(FactorSpecError, match="没在 params 里声明"):
        validate_spec(
            {"steps": [{"id": "a", "op": "add", "a": {"ref": "price"}, "b": {"param": "w"}}], "output": "a"},
            factor_name="x",
        )
    with pytest.raises(FactorSpecError, match="不认识的字段"):
        validate_spec(
            {"steps": [{"id": "a", "op": "add", "a": {"ref": "price"}, "b": {"const": 1}, "c": 2}], "output": "a"},
            factor_name="x",
        )


def test_shift_input_must_be_ref_and_const_must_be_number():
    with pytest.raises(FactorSpecError, match="input 必须是"):
        validate_spec(
            {"steps": [{"id": "a", "op": "shift", "input": {"const": 1}, "by": {"const": 1}}], "output": "a"},
            factor_name="x",
        )
    with pytest.raises(FactorSpecError, match="const 必须是数字"):
        validate_spec(
            {"steps": [{"id": "a", "op": "add", "a": {"ref": "price"}, "b": {"const": "1"}}], "output": "a"},
            factor_name="x",
        )


def test_validate_spec_partial_allows_unfinished_draft():
    validate_spec({"params": ["w"]}, factor_name="draft", partial=True)
    validate_spec(
        {"params": [], "steps": [{"id": "a", "op": "add", "a": {"ref": "price"}, "b": {"const": 1}}]},
        factor_name="draft",
        partial=True,
    )
    with pytest.raises(FactorSpecError, match="缺 output"):
        validate_spec({"params": [], "steps": [{"id": "a", "op": "add", "a": {"ref": "price"}, "b": {"const": 1}}]},
                      factor_name="draft")


def test_extra_factor_dirs_env(tmp_path, monkeypatch):
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "double.yaml").write_text(
        "name: double\nsteps:\n  - id: r\n    op: multiply\n    a: {ref: price}\n    b: {const: 2}\noutput: r\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(EXTRA_DIRS_ENV_VAR, f"{lib}{os.pathsep}{tmp_path / 'missing'}")
    assert "double" in available()
    assert "momentum" in available()
    out = get("double")(_price())
    assert out.iloc[0] == 2.0
    path, spec = load_specs()["double"]
    assert path == lib / "double.yaml"
    assert spec["output"] == "r"


def test_extra_dir_cannot_shadow_builtin(tmp_path, monkeypatch):
    (tmp_path / "m.yaml").write_text(
        "name: momentum\nsteps:\n  - id: r\n    op: add\n    a: {ref: price}\n    b: {const: 0}\noutput: r\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(EXTRA_DIRS_ENV_VAR, str(tmp_path))
    with pytest.raises(FactorSpecError, match="重复出现"):
        available()
