"""因子注册表的汇总入口。

新增因子: 在这个目录下新建一个 <名字>.yaml (格式参考 momentum.yaml),
不用碰任何 .py 文件, 也不用改这个文件或者 Backtester —— registry.get()/
available() 每次调用都会重新扫一遍这个目录, 改完 YAML 立刻生效。
"""

from __future__ import annotations

from minibacktest.factors.registry import FactorSpecError, available, get

__all__ = ["FactorSpecError", "available", "get"]
