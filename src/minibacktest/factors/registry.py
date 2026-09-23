"""因子注册表: 扫 factors/ 目录下所有 *.yaml, 把每个因子的名字和"能算出
[date, ticker] MultiIndex Series 的函数"对应起来, 让 Backtester 按名字
动态取因子。

新增一个因子只需要在这个目录下新建一个 <名字>.yaml (格式参考
momentum.yaml), 不用碰任何 .py 文件, 也不用改这个文件或者 Backtester。
get()/available() 每次调用都会重新扫一遍目录并重新编译, 改完 YAML 立刻
生效——不会像原来"新建 .py + 装饰器 + 在 __init__.py 里 import 触发副作用"
那套机制一样卡在 Python 的模块导入缓存里, 也不需要重启进程。

YAML 因子定义的形状(一个小的、数据驱动的计算图, 不是可执行代码):

    name: momentum          # 因子名, 省略则用文件名(不带 .yaml)
    params: [window]        # 这个因子接受哪些参数名(对应 get(name)(price, **params) 的 kwargs)
    steps:
      - id: lag_amount       # 每一步给自己起个 id, 后面的步骤可以用 {ref: lag_amount} 引用它
        op: add
        a: {const: 1}
        b: {param: window}
      - id: near
        op: shift
        input: {ref: price}  # "price" 是内置的引用, 指向传进来的整张价格宽表
        by: {const: 1}
      - id: far
        op: shift
        input: {ref: price}
        by: {ref: lag_amount}
      - id: ratio
        op: divide
        a: {ref: near}
        b: {ref: far}
      - id: result
        op: subtract
        a: {ref: ratio}
        b: {const: 1.0}
    output: result           # 哪一步的结果是这个因子最终的值

操作数只有三种写法: {const: 数字}、{param: 参数名}(从调用时传的 kwargs 里取)、
{ref: 某个 step 的 id 或 "price"}。op 只认 _ALLOWED_OPS 里列的几种, 都是
纯数值运算, 没有任何能读文件/读网络/执行任意代码的路径——不管这个 YAML
是人手写的还是 agent 拼出来的, 风险都是同一个量级。

额外的因子目录: 环境变量 MINIBACKTEST_EXTRA_FACTOR_DIRS(多个目录用 os.pathsep
分隔, macOS/Linux 上是 ":")指向的目录也会被一起扫描, 用来放不属于本仓库的
因子库(比如 ../quant-assistant 里 Agent 拼出来并保存的因子)。跟内置因子同名
会直接报"重复出现", 不会悄悄覆盖内置因子。

防未来函数: shift 的位移必须是非负整数。shift(-n) 等于"拿 n 天之后的价格",
不管 YAML 是谁写的都一律拒绝(静态校验时看常数, 运行时再看参数/引用算出来的值)。
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

FactorFn = Callable[..., pd.Series]

_FACTORS_DIR = Path(__file__).parent

# op -> 需要的操作数字段名; "shift" 是唯一的一元-with-参数运算(input + by),
# 其余都是二元运算(a + b)。新增 op 时同时更新这里和 _apply_op。
_BINARY_OPS = frozenset({"add", "subtract", "multiply", "divide"})
ALLOWED_OPS = _BINARY_OPS | {"shift"}
_ALLOWED_OPS = ALLOWED_OPS  # 旧名字, 保留给已有调用方

# 每种 op 需要哪些操作数字段(按顺序)。
OP_FIELDS: dict[str, tuple[str, str]] = {
    **{op: ("a", "b") for op in sorted(_BINARY_OPS)},
    "shift": ("input", "by"),
}

# 内置引用: 整张价格宽表。step id 不能跟它撞名。
BUILTIN_REF = "price"

EXTRA_DIRS_ENV_VAR = "MINIBACKTEST_EXTRA_FACTOR_DIRS"


class FactorSpecError(ValueError):
    """YAML 因子定义写错了(缺字段、引用不存在的 id、用了没在白名单里的 op 之类)。"""


def _resolve_operand(operand: Any, env: dict[str, Any], params: Mapping[str, Any], *, factor_name: str) -> Any:
    """把一个操作数字典(YAML 里的 {const:...}/{param:...}/{ref:...})解析成实际值。

    Args:
        operand: 恰好一个键的字典, 键是 "const"/"param"/"ref" 之一。
        env: 到目前为止已经算出来的 step 结果, 外加内置的 "price"。
        params: 调用 fn() 时传进来的 kwargs(op 是 "param" 时从这里取值)。
        factor_name: 只用来在报错信息里指出是哪个因子写错了。

    Returns:
        const 直接返回字面量; param 返回 params 里对应的值; ref 返回 env
        里对应 step(或 "price")算出来的结果。

    Raises:
        FactorSpecError: operand 形状不对、引用了不存在的 id、用了没在
            params 里声明的参数名、或者 kind 不是 const/param/ref 之一。
    """
    if not isinstance(operand, dict) or len(operand) != 1:
        raise FactorSpecError(f"因子 {factor_name!r} 里有个操作数写法不对(应该是恰好一个键的字典): {operand!r}")
    (kind, value), = operand.items()
    if kind == "const":
        return value
    if kind == "param":
        if value not in params:
            raise FactorSpecError(f"因子 {factor_name!r} 需要参数 {value!r}, 调用时没有传")
        return params[value]
    if kind == "ref":
        if value not in env:
            raise FactorSpecError(f"因子 {factor_name!r} 引用了不存在的 id {value!r}")
        return env[value]
    raise FactorSpecError(f"因子 {factor_name!r} 用了不认识的操作数类型 {kind!r}(只认 const/param/ref)")


def _shift_amount(by: Any, *, factor_name: str) -> int:
    """把 shift 的位移转成非负整数; 负数(会用到未来价格)或非整数直接报错。"""
    # numpy 的整数/浮点标量也要认(参数可能是从 DataFrame 里取出来的), 所以不用
    # isinstance(int) 判断, 而是看能不能无损转成 int; DataFrame 之类转不了的直接报错。
    try:
        n = int(by)
        exact = not isinstance(by, bool) and n == by
    except (TypeError, ValueError):
        exact = False
    if not exact:
        raise FactorSpecError(f"因子 {factor_name!r} 的 shift 位移必须是整数, 拿到的是 {by!r}")
    if n < 0:
        raise FactorSpecError(
            f"因子 {factor_name!r} 的 shift 位移是 {n}: 负数位移会用到未来的价格(未来函数), 不允许"
        )
    return n


def _check_operand(
    operand: Any, *, known_ids: set[str], declared_params: list[str], factor_name: str, where: str
) -> None:
    """静态检查一个操作数: 形状对不对、param 有没有声明、ref 指向的 id 是否已经出现过。"""
    if not isinstance(operand, dict) or len(operand) != 1:
        raise FactorSpecError(
            f"因子 {factor_name!r} {where} 的操作数写法不对(应该是恰好一个键的字典): {operand!r}"
        )
    (kind, value), = operand.items()
    if kind == "const":
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise FactorSpecError(f"因子 {factor_name!r} {where} 的 const 必须是数字, 拿到的是 {value!r}")
    elif kind == "param":
        if value not in declared_params:
            raise FactorSpecError(
                f"因子 {factor_name!r} {where} 用了没在 params 里声明的参数 {value!r}(已声明: {declared_params})"
            )
    elif kind == "ref":
        if value not in known_ids:
            raise FactorSpecError(
                f"因子 {factor_name!r} {where} 引用了不存在(或还没算出来)的 id {value!r}, "
                f"只能引用 {BUILTIN_REF!r} 或排在前面的步骤: {sorted(known_ids)}"
            )
    else:
        raise FactorSpecError(
            f"因子 {factor_name!r} {where} 用了不认识的操作数类型 {kind!r}(只认 const/param/ref)"
        )


def validate_spec(spec: Any, *, factor_name: str, partial: bool = False) -> None:
    """只看结构、不碰数据的静态校验, 编译时(也就是扫描注册表时)就把写错的 YAML 拦下来。

    检查: 顶层是字典; params 是不重复的字符串列表; 每一步有唯一 id(不跟
    "price" 撞名)、op 在白名单里、字段齐全且没有多余字段; 操作数形状正确,
    param 已声明, ref 只能指向 "price" 或前面的步骤(所以不可能成环);
    shift 的 input 必须是 ref, 常数位移必须是非负整数; output 指向存在的步骤。

    Args:
        spec: yaml.safe_load 出来的顶层对象。
        factor_name: 用于报错信息。
        partial: True 时允许 steps 为空、允许没有 output——给"一步步拼因子"
            的草稿用, 每加一步都能立刻校验, 不用等整个因子拼完。

    Raises:
        FactorSpecError: 任何一条不满足。
    """
    if not isinstance(spec, dict):
        raise FactorSpecError(f"因子 {factor_name!r} 的 YAML 顶层必须是一个字典, 拿到的是 {type(spec).__name__}")

    declared_params = spec.get("params") or []
    if not isinstance(declared_params, list) or not all(isinstance(p, str) and p for p in declared_params):
        raise FactorSpecError(f"因子 {factor_name!r} 的 params 必须是参数名字符串列表: {declared_params!r}")
    if len(set(declared_params)) != len(declared_params):
        raise FactorSpecError(f"因子 {factor_name!r} 的 params 里有重复的参数名: {declared_params!r}")

    steps = spec.get("steps")
    if steps is None and partial:
        steps = []
    if not isinstance(steps, list) or (not steps and not partial):
        raise FactorSpecError(f"因子 {factor_name!r} 缺 steps(或者 steps 不是一个列表)")

    known_ids = {BUILTIN_REF}
    for i, step in enumerate(steps):
        if not isinstance(step, dict) or "id" not in step:
            raise FactorSpecError(f"因子 {factor_name!r} 有一步没写 id: {step!r}")
        step_id = step["id"]
        if not isinstance(step_id, str) or not step_id:
            raise FactorSpecError(f"因子 {factor_name!r} 第 {i + 1} 步的 id 必须是非空字符串: {step_id!r}")
        if step_id in known_ids:
            raise FactorSpecError(f"因子 {factor_name!r} 里 id {step_id!r} 重复了(或者跟内置的 'price' 撞名)")
        op = step.get("op")
        if op not in ALLOWED_OPS:
            raise FactorSpecError(f"因子 {factor_name!r} 用了不在白名单里的 op {op!r}, 只认 {sorted(ALLOWED_OPS)}")
        fields = OP_FIELDS[op]
        missing = [f for f in fields if f not in step]
        if missing:
            raise FactorSpecError(f"因子 {factor_name!r} 的 {op} 步骤 {step_id!r} 缺字段 {missing}")
        extra = set(step) - {"id", "op", *fields}
        if extra:
            raise FactorSpecError(f"因子 {factor_name!r} 的步骤 {step_id!r} 有不认识的字段 {sorted(extra)}")
        for field in fields:
            _check_operand(
                step[field],
                known_ids=known_ids,
                declared_params=declared_params,
                factor_name=factor_name,
                where=f"步骤 {step_id!r} 的 {field}",
            )
        if op == "shift":
            if "ref" not in step["input"]:
                raise FactorSpecError(f"因子 {factor_name!r} 的 shift 步骤 {step_id!r} 的 input 必须是 {{ref: ...}}")
            if "const" in step["by"]:
                _shift_amount(step["by"]["const"], factor_name=factor_name)
        known_ids.add(step_id)

    output_id = spec.get("output")
    if output_id is None and partial:
        return
    if not output_id:
        raise FactorSpecError(f"因子 {factor_name!r} 缺 output(要指明哪一步是最终结果)")
    if output_id not in known_ids:
        raise FactorSpecError(f"因子 {factor_name!r} 的 output {output_id!r} 找不到对应的 step id")


def _apply_op(step: dict[str, Any], env: dict[str, Any], params: Mapping[str, Any], *, factor_name: str) -> Any:
    """执行 steps 列表里的一步(先解析出 a/b 或 input/by, 再套对应的运算)。

    Args:
        step: 一个 step 字典, 至少要有 "op", 二元运算再要 "a"/"b",
            "shift" 再要 "input"/"by"。
        env: 到目前为止已经算出来的 step 结果, 外加内置的 "price"。
        params: 调用 fn() 时传进来的 kwargs, 原样透传给 _resolve_operand。
        factor_name: 只用来在报错信息里指出是哪个因子写错了。

    Returns:
        这一步算出来的结果(跟 price 同形状的 DataFrame), 之后会被存进 env。

    Raises:
        FactorSpecError: op 不在 _ALLOWED_OPS 白名单里, 或者缺了这个 op
            需要的字段(a/b, 或者 shift 的 input/by)。
    """
    op = step.get("op")
    if op not in _ALLOWED_OPS:
        raise FactorSpecError(f"因子 {factor_name!r} 用了不在白名单里的 op {op!r}, 只认 {sorted(_ALLOWED_OPS)}")

    if op == "shift":
        if "input" not in step or "by" not in step:
            raise FactorSpecError(f"因子 {factor_name!r} 的 shift 步骤缺 input 或 by: {step!r}")
        source = _resolve_operand(step["input"], env, params, factor_name=factor_name)
        by = _resolve_operand(step["by"], env, params, factor_name=factor_name)
        return source.shift(_shift_amount(by, factor_name=factor_name))

    if "a" not in step or "b" not in step:
        raise FactorSpecError(f"因子 {factor_name!r} 的 {op} 步骤缺 a 或 b: {step!r}")
    a = _resolve_operand(step["a"], env, params, factor_name=factor_name)
    b = _resolve_operand(step["b"], env, params, factor_name=factor_name)
    if op == "add":
        return a + b
    if op == "subtract":
        return a - b
    if op == "multiply":
        return a * b
    return a / b  # op == "divide"


def _compile(spec: dict[str, Any], *, factor_name: str) -> FactorFn:
    """把一份解析好的 YAML(dict)编译成可调用的因子函数。

    先做 validate_spec 的完整静态校验, 不会真的去跑任何运算——运算是在
    返回的 fn() 被调用时才按 steps 顺序执行的。

    Args:
        spec: yaml.safe_load 出来的顶层字典, 要有 "steps" 和 "output",
            "params" 可选(省略等于没有参数)。
        factor_name: 这个因子的名字(用于报错信息, 也是返回结果 Series
            的 name)。

    Returns:
        一个 FactorFn: 输入价格宽表(index 日期, columns ticker) + 因子
        参数(kwargs), 输出 [date, ticker] MultiIndex 的 Series。

    Raises:
        FactorSpecError: validate_spec 不通过。fn() 被真正调用时还可能
            再抛(传了没声明的参数、参数算出来的 shift 位移是负数等)。
    """
    validate_spec(spec, factor_name=factor_name)

    declared_params = list(spec.get("params") or [])
    steps = spec["steps"]
    output_id = spec["output"]

    def fn(price: pd.DataFrame, **params: Any) -> pd.Series:
        """按 steps 顺序依次算出每一步, 最后把 output 那一步 stack 成长表。

        Args:
            price: 价格宽表, index 是日期, columns 是 ticker。
            **params: 这个因子声明的参数(必须是 declared_params 的子集)。

        Returns:
            [date, ticker] MultiIndex 的 Series, name 是 factor_name。

        Raises:
            FactorSpecError: 传了没声明的参数、某个 step 没写 id、id 跟
                别的 step(或内置的 "price")重复、或者 output 指向的 id
                根本没算出来过。
        """
        extra = set(params) - set(declared_params)
        if extra:
            raise FactorSpecError(f"因子 {factor_name!r} 不认识参数 {sorted(extra)}, 只声明了 {declared_params}")

        env: dict[str, Any] = {"price": price}
        for step in steps:
            if not isinstance(step, dict) or "id" not in step:
                raise FactorSpecError(f"因子 {factor_name!r} 有一步没写 id: {step!r}")
            step_id = step["id"]
            if step_id == "price" or step_id in env:
                raise FactorSpecError(f"因子 {factor_name!r} 里 id {step_id!r} 重复了(或者跟内置的 'price' 撞名)")
            env[step_id] = _apply_op(step, env, params, factor_name=factor_name)

        if output_id not in env:
            raise FactorSpecError(f"因子 {factor_name!r} 的 output {output_id!r} 找不到对应的 step id")

        result = env[output_id]
        s = result.stack()
        s.index = s.index.set_names(["date", "ticker"])
        return s.rename(factor_name)  # type: ignore[return-value]

    return fn


def factor_dirs(extra_dirs: list[str | Path] | None = None) -> list[Path]:
    """要扫描的因子目录: 本包自带的 factors/, 加上环境变量
    MINIBACKTEST_EXTRA_FACTOR_DIRS 里列出的目录, 再加上显式传入的 extra_dirs
    (不存在的目录跳过——比如因子库还没保存过任何因子、目录还没建出来)。"""
    dirs = [_FACTORS_DIR]
    parts: list[str | Path] = [*os.environ.get(EXTRA_DIRS_ENV_VAR, "").split(os.pathsep), *(extra_dirs or [])]
    for part in parts:
        if not str(part).strip():
            continue
        path = Path(part).expanduser()
        if path.is_dir() and all(path.resolve() != d.resolve() for d in dirs):
            dirs.append(path)
    return dirs


def load_specs(extra_dirs: list[str | Path] | None = None) -> dict[str, tuple[Path, dict[str, Any]]]:
    """扫一遍所有因子目录下的 *.yaml, 返回 {因子名: (文件路径, 解析出来的 spec)}。

    只解析、不编译; 给需要"看因子长什么样"的调用方(比如把因子定义展示给
    Agent)用。每次调用都重新读盘。extra_dirs 见 factor_dirs。

    Raises:
        FactorSpecError: 两个 YAML 文件(不管在不在同一个目录)用了同一个因子名。
    """
    found: dict[str, tuple[Path, dict[str, Any]]] = {}
    for directory in factor_dirs(extra_dirs):
        for path in sorted(directory.glob("*.yaml")):
            spec = yaml.safe_load(path.read_text(encoding="utf-8"))
            name = (spec or {}).get("name") if isinstance(spec, dict) else None
            name = name or path.stem
            if name in found:
                raise FactorSpecError(
                    f"因子名 {name!r} 在 {path} 里重复出现了(已经被 {found[name][0]} 用过)"
                )
            found[name] = (path, spec)
    return found


def compile_spec(spec: Any, *, factor_name: str) -> FactorFn:
    """把一份 spec(dict)编译成因子函数, 不需要先写成文件、也不进注册表。

    跟注册表里的因子走完全同一条编译路径(同样的静态校验和 op 白名单),
    给"草稿还没保存, 先试算一下"这种场景用。
    """
    return _compile(spec, factor_name=factor_name)


def _load_all() -> dict[str, FactorFn]:
    """扫一遍所有因子目录(见 factor_dirs)下的 *.yaml, 编译成 {因子名: 因子函数}。

    每次调用都会重新读盘、重新编译(不做缓存), 所以改完 YAML 立刻生效,
    但也意味着 get()/available() 每次调用都有一次目录扫描的开销。

    Raises:
        FactorSpecError: 某个 YAML 编译失败, 或者两个 YAML 文件用了同一
            个因子名。
    """
    return {name: _compile(spec, factor_name=name) for name, (_, spec) in load_specs().items()}


def get(name: str) -> FactorFn:
    """按名字取出一个因子函数: 输入价格宽表(index 日期, columns ticker) +
    这个因子的参数(kwargs), 输出 [date, ticker] MultiIndex 的 Series。

    每次调用都会重新扫一遍 factors/ 目录下的 YAML 并重新编译, 改完文件
    立刻生效。

    Raises:
        KeyError: 没有这个名字的因子。
        FactorSpecError: 某个 YAML 文件写得不对(不影响其他因子, 但这个
            名字本身取不出来; 扫描阶段就会报, 不会拖到真正算的时候才炸)。
    """
    registry = _load_all()
    if name not in registry:
        raise KeyError(f"没有注册名为 {name!r} 的因子, 已注册的有: {available()}")
    return registry[name]


def available() -> list[str]:
    """列出当前已注册的所有因子名字, 按字母排序。"""
    return sorted(_load_all())
