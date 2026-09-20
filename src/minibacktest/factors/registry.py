"""因子注册表: 把因子名字和计算函数对应起来, 让 Backtester 按名字动态取因子,
不用每加一个新因子就去改 Backtester 的代码。

新增一个因子只需要: 在 factors/ 下新建一个文件, 写一个签名统一的函数
(输入 price, 输出 [date, ticker] MultiIndex 的 Series), 用 @register("名字")
装饰它, 再在 factors/__init__.py 里 import 一下这个文件(触发装饰器执行)。
不用改这个文件, 也不用改 Backtester。
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

FactorFn = Callable[..., pd.Series]

_REGISTRY: dict[str, FactorFn] = {}


def register(name: str) -> Callable[[FactorFn], FactorFn]:
    """装饰器: 把被装饰的函数注册进因子表, 以后按 name 检索。

    Args:
        name: 因子名字, 用作 Backtester(factor_specs=[(name, params), ...])
            里的 key, 也会作为 combine_scores 合成时的列名。

    Returns:
        原样返回被装饰的函数(不做任何包装), 只是顺手注册一下。

    Raises:
        ValueError: name 已经被注册过(避免不小心重名, 后一个悄悄覆盖前一个)。

    Example:
        >>> @register("demo_factor_for_doctest")
        ... def my_factor(price, *, window):
        ...     pass
        >>> get("demo_factor_for_doctest") is my_factor
        True
    """

    def decorator(fn: FactorFn) -> FactorFn:
        if name in _REGISTRY:
            raise ValueError(f"因子名 {name!r} 已经被注册过, 换一个名字")
        _REGISTRY[name] = fn
        return fn

    return decorator


def get(name: str) -> FactorFn:
    """按名字取出已注册的因子函数。

    Raises:
        KeyError: 没有这个名字的因子, 通常是忘了在 factors/__init__.py 里
            import 对应模块, 导致装饰器没被执行到。
    """
    if name not in _REGISTRY:
        raise KeyError(f"没有注册名为 {name!r} 的因子, 已注册的有: {available()}")
    return _REGISTRY[name]


def available() -> list[str]:
    """列出当前已注册的所有因子名字, 按字母排序。"""
    return sorted(_REGISTRY)
