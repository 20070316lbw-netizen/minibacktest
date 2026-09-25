"""因子注册表的汇总入口。

新增因子: 在这个目录下新建一个 <名字>.yaml (格式参考 momentum.yaml),
不用碰任何 .py 文件, 也不用改这个文件或者 Backtester —— registry.get()/
available() 每次调用都会重新扫一遍这个目录, 改完 YAML 立刻生效。

本仓库之外的因子库(比如 quant-assistant 里 Agent 拼出来的因子)通过环境变量
MINIBACKTEST_EXTRA_FACTOR_DIRS 挂进来, 见 registry 模块的说明。
"""

from __future__ import annotations

from minibacktest.factors.registry import (
    ALLOWED_OPS,
    BUILTIN_FIELDS,
    BUILTIN_REF,
    EXTRA_DIRS_ENV_VAR,
    OP_FIELDS,
    FactorSpecError,
    available,
    compile_spec,
    factor_dirs,
    get,
    load_specs,
    required_fields,
    validate_spec,
)

__all__ = [
    "ALLOWED_OPS",
    "BUILTIN_FIELDS",
    "BUILTIN_REF",
    "EXTRA_DIRS_ENV_VAR",
    "OP_FIELDS",
    "FactorSpecError",
    "available",
    "compile_spec",
    "factor_dirs",
    "get",
    "load_specs",
    "required_fields",
    "validate_spec",
]
