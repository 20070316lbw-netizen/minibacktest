"""因子注册表 + 具体因子实现的汇总入口。

import 这个包会触发下面每个因子模块顶部的 @register 装饰器, 把所有因子
登记进 registry, 所以只要 `import minibacktest.factors`(Backtester 内部
就是这么用的), 就能通过 registry.get(name) 按名字拿到任意因子函数。

新增因子: 在这个目录下新建一个文件(参考 momentum.py), 写函数 + 用
@register("你的因子名") 装饰, 然后在下面加一行 import, 不用改
registry.py 或者 backtester.py。
"""

from __future__ import annotations

from minibacktest.factors import momentum as _momentum  # noqa: F401  触发注册
from minibacktest.factors import reversal as _reversal  # noqa: F401  触发注册
from minibacktest.factors.registry import available, get, register

__all__ = ["available", "get", "register"]
