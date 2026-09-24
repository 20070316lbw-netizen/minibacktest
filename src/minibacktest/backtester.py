"""把整条回测流水线(拉数据 -> 存 liudb -> 读回 -> 多因子合成 -> 分位数多空
-> run_backtest -> 出图)包一层"壳", 方便整体调用。

Backtester 本身不实现任何计算逻辑 —— 因子在 factors/ 里按名字注册, 标准化
用 zscore.zscore_by_date, 合成用 signal.combine.combine_scores, 分组用
sizing_fn(默认 portfolio.sizing.make_quantile_sizer, 可换成
portfolio.sizing.make_vol_neutral_sizer 等其他仓位构造策略), 回测用
engine.run_backtest, 出图用 figure_engine.plot_tearsheet。这个类只负责把
调用顺序和参数传递这件"胶水"的事包起来 —— 新增因子/评估指标/图表/仓位
构造策略都直接加在对应模块里, 不用改这个类。
"""

from __future__ import annotations

from collections.abc import Callable

import liudb
import pandas as pd
from loguru import logger
from sources import get_prices

from minibacktest import factors as factor_registry
from minibacktest.base import Result
from minibacktest.engine import run_backtest
from minibacktest.evaluation.quantile import quantile_forward_returns
from minibacktest.figure_engine import plot_tearsheet
from minibacktest.portfolio.sizing import make_quantile_sizer
from minibacktest.rebalance import rebalance_dates
from minibacktest.signal.combine import combine_scores
from minibacktest.zscore.zscore import zscore_by_date


class Backtester:
    def __init__(
        self,
        tickers: list[str],
        start: str,
        end: str | None = None,
        *,
        factor_specs: list[tuple[str, dict]],
        factor_weights: dict[str, float] | None = None,
        freq: int = 21,
        n_quantiles: int = 5,
        initial_capital: float = 100_000.0,
        periods_per_year: float = 252,
        db_path: str = "sp500.db",
        commission_bps: float = 0.0,
        slippage_bps: float = 0.0,
        sizing_fn: Callable[[pd.Series, pd.DataFrame], pd.Series] | None = None,
    ) -> None:
        self.tickers = tickers  # 候选池
        self.start = start  # 行情窗口起始日期(拉取和回测读取共用)
        self.end = end  # 行情窗口结束日期, 不传则不限(拉取到最新, 读取到最新)
        self.factor_specs = factor_specs  # [(因子名, {参数}), ...], 因子名对应 factors 注册表里的 key
        self.factor_weights = factor_weights  # 各因子合成权重, None 就是等权(combine_scores 的默认行为)
        self.freq = freq  # 调仓间隔(交易日数)
        self.n_quantiles = n_quantiles  # 分位数分组数(同时也是 plot() 里单调性检验图默认用的分组数)
        self.initial_capital = initial_capital
        self.periods_per_year = periods_per_year
        self.db_path = db_path
        self.commission_bps = commission_bps  # 单边佣金(bps), 按换手计, 默认 0
        self.slippage_bps = slippage_bps  # 单边滑点(bps), 计法同佣金, 默认 0

        # 仓位构造策略: (score, price) -> weight 的可插拔函数。不传就用
        # portfolio.sizing.make_quantile_sizer(n_quantiles) 包出来的默认
        # 行为(纯分位数多空), 跟以前完全一样。想用波动率中性化, 传
        # portfolio.sizing.make_vol_neutral_sizer(...) 的返回值进来即可,
        # Backtester 本身不需要为每种分组方式单独加开关参数。
        self.sizing_fn = sizing_fn or make_quantile_sizer(n_quantiles=n_quantiles)

        self.price: pd.DataFrame | None = None  # run() 之后: adj_close 宽表
        self.score: pd.Series | None = None  # run() 之后: 调仓日打分(多因子合成后)
        self.weight: pd.Series | None = None  # run() 之后: 调仓日目标权重
        self.result: Result | None = None  # run() 之后: base.Result
        self.quantile_returns: pd.Series | None = None  # plot() 之后: 分位数单调性检验数据

    def _pull_and_store(self) -> None:
        """拉价格数据存进 liudb(按主键覆盖, 重复跑不会产生重复数据)。"""
        logger.info(f"拉取 {len(self.tickers)} 只标的的价格数据, 区间 {self.start} ~ {self.end or '最新'}")
        prices = get_prices(self.tickers, start=self.start, end=self.end)
        liudb.init_schema(self.db_path)
        liudb.save_prices(prices, self.db_path)
        logger.info(f"存入 {len(prices)} 条价格记录到 {self.db_path}")

    def _load_price(self) -> pd.DataFrame:
        """从 liudb 按 tickers/start/end 查回复权后的 close 长表, 透视成宽表。

        走 liudb.Query/loader 的注册表查询路径: 过滤条件下推到 SQL 里,
        不会像 liudb.load_prices() 那样把整张 prices 表读出来; 取到的
        "close" 已经是复权后的价格(liudb 的 prices 注册表只登记复权口径)。
        """
        query = liudb.Query(
            columns=["close"], tickers=self.tickers, start=self.start, end=self.end
        )
        long = liudb.loader(request=query, path=self.db_path)
        wide = long["close"].unstack("ticker")
        return wide.sort_index()

    def _build_score(self) -> tuple[pd.Series, pd.Series]:
        """按 factor_specs 依次算出每个因子, 横向拼成一张因子面板(一列一个
        因子), 标准化后加权合成一个打分, 再按打分分位数算出多空目标权重。

        Returns:
            (score, weight): 调仓日的合成打分, 和对应的多空目标权重。
        """
        rb_dates = rebalance_dates(self.price.index, self.freq) # type: ignore

        columns = []
        for name, params in self.factor_specs:
            fn = factor_registry.get(name)
            s = fn(self.price, **params).rename(name)
            s = s[s.index.get_level_values("date").isin(rb_dates)]
            columns.append(s)

        factor_df = pd.concat(columns, axis=1)  # 横向拼: 按 (date, ticker) 对齐, 缺的地方是 NaN
        z = zscore_by_date(factor_df)
        score = combine_scores(z, weights=self.factor_weights)
        weight = self.sizing_fn(score, self.price)
        return score, weight

    def run(self, *, refresh_data: bool = True) -> Result:
        """跑一次完整回测: (可选)拉数据 -> 存 -> 读 -> 因子合成 -> 分位数多空
        -> 向量化回测, 返回 base.Result。

        Args:
            refresh_data: 是否重新拉一次行情存进 liudb, 默认 True; 数据没变
                的话可以传 False 跳过网络请求, 直接用 liudb 里已有的数据。

        Returns:
            base.Result。同时把中间结果(price/score/weight/result)存进
            对应的实例属性, 方便事后检查或者喂给 plot()。
        """
        if refresh_data:
            self._pull_and_store()
        self.price = self._load_price()
        self.score, self.weight = self._build_score()
        self.result = run_backtest(
            price=self.price,
            target_weight=self.weight,
            freq=self.freq,
            initial_capital=self.initial_capital,
            periods_per_year=self.periods_per_year,
            commission_bps=self.commission_bps,
            slippage_bps=self.slippage_bps,
        )
        return self.result

    def plot(self):
        """run() 跑完之后调用, 出一张 tearsheet 总览图(净值/回撤/滚动 Sharpe
        /月度热力图/分位数单调性检验)。

        Returns:
            matplotlib Figure。

        Raises:
            RuntimeError: 还没调用过 run()。
        """
        if self.result is None:
            raise RuntimeError("请先调用 run(), 拿到 result 之后才能画图")

        self.quantile_returns = quantile_forward_returns(
            self.score, self.price, freq=self.freq, n_quantiles=self.n_quantiles # type: ignore
        )
        return plot_tearsheet(
            self.result.equity_curve,
            quantile_returns=self.quantile_returns,
            periods_per_year=self.periods_per_year,
        )
