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
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

FactorFn = Callable[..., pd.Series]

_FACTORS_DIR = Path(__file__).parent

# op -> 需要的操作数字段名; "shift" 是唯一的一元-with-参数运算(input + by),
# 其余都是二元运算(a + b)。新增 op 时同时更新这里和 _apply_op。
_BINARY_OPS = {"add", "subtract", "multiply", "divide"}
_ALLOWED_OPS = _BINARY_OPS | {"shift"}


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
        return source.shift(int(by))

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

    只做结构校验(steps/output 存不存在、id 有没有重复), 不会真的去跑
    任何运算——运算是在返回的 fn() 被调用时才按 steps 顺序执行的。

    Args:
        spec: yaml.safe_load 出来的顶层字典, 要有 "steps" 和 "output",
            "params" 可选(省略等于没有参数)。
        factor_name: 这个因子的名字(用于报错信息, 也是返回结果 Series
            的 name)。

    Returns:
        一个 FactorFn: 输入价格宽表(index 日期, columns ticker) + 因子
        参数(kwargs), 输出 [date, ticker] MultiIndex 的 Series。

    Raises:
        FactorSpecError: spec 顶层不是字典、缺 steps/output、或者 steps
            不是列表。fn() 被真正调用时还可能再抛(见 fn 内部)。
    """
    if not isinstance(spec, dict):
        raise FactorSpecError(f"因子 {factor_name!r} 的 YAML 顶层必须是一个字典, 拿到的是 {type(spec).__name__}")

    declared_params = list(spec.get("params") or [])
    steps = spec.get("steps")
    output_id = spec.get("output")
    if not steps or not isinstance(steps, list):
        raise FactorSpecError(f"因子 {factor_name!r} 缺 steps(或者 steps 不是一个列表)")
    if not output_id:
        raise FactorSpecError(f"因子 {factor_name!r} 缺 output(要指明哪一步是最终结果)")

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


def _load_all() -> dict[str, FactorFn]:
    """扫一遍 factors/ 目录下所有 *.yaml, 编译成 {因子名: 因子函数} 的字典。

    每次调用都会重新读盘、重新编译(不做缓存), 所以改完 YAML 立刻生效,
    但也意味着 get()/available() 每次调用都有一次目录扫描的开销。

    Returns:
        因子名到编译好的 FactorFn 的映射。

    Raises:
        FactorSpecError: 某个 YAML 编译失败, 或者两个 YAML 文件用了同一
            个因子名。
    """
    registry: dict[str, FactorFn] = {}
    for path in sorted(_FACTORS_DIR.glob("*.yaml")):
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
        name = (spec or {}).get("name") or path.stem
        if name in registry:
            raise FactorSpecError(f"因子名 {name!r} 在 {path.name} 里重复出现了(已经被别的 YAML 文件用过)")
        registry[name] = _compile(spec, factor_name=name)
    return registry


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
