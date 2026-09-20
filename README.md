## 个人极简暂时的回测引擎

虽然是暂时的回测引擎. 但是也将包含:
```md
# 基本信息
Start                       回测数据的起始日期                                                         
End                         回测数据的截止日期                                                          
Duration                    回测总时长              
Exposure Time [%]           持仓时间占总回测时间的比例   

# 收益与资金曲线
Equity Final [$]            回测结束时的账户净值             
Equity Peak [$]             回测期间账户净值的最高点                    
Return [%]                  总收益率                         
Buy & Hold Return [%]       如果全程只是买入并持有标的（不做任何交易）的收益率，用来做对比基准            
Return (Ann.) [%]           年化收益率              
Volatility (Ann.) [%]       年化波动率（收益的标准差）          
CAGR [%]                    复合年增长率          

# 风险调整后收益
Sharpe Ratio                夏普比率：单位风险（总波动）换来的超额收益，越高越好         
Sortino Ratio               索提诺比率            
Calmar Ratio                卡玛比率：年化收益 ÷ 最大回撤，衡量收益相对于最坏情况的性价比   

# 与基线相关
Alpha [%]                   超额收益，策略跑赢基准的部分（剔除 Beta 后）                       
Beta                        策略相对于基准的系统性风险敞口
Max. Drawdown [%]           最大回撤：从最高点到最低点的最大跌幅                 
Avg. Drawdown [%]           平均回撤幅度                  
Max. Drawdown Duration      最大回撤持续的时间（从高点跌到低点又恢复的最长时间） 
Avg. Drawdown Duration      平均每次回撤持续的时间

# 交易层面                              
Win Rate [%]                胜率，盈利交易占总交易的比例            
Best Trade [%]              单笔最好交易的收益率            
Worst Trade [%]             单笔最差交易的收益率          
Avg. Trade [%]              平均每笔交易的收益率         
Max. Trade Duration         单笔交易持仓的最长时长
Avg. Trade Duration         单笔交易持仓的平均时长   
Profit Factor               盈利因子：总盈利 ÷ 总亏损（绝对值），大于1才算盈利策略             
Expectancy [%]              期望值：平均每笔交易能赚多少           
SQN                         System Quality Number（系统质量指数），综合衡量交易系统稳定性和盈利能力的指标   
Kelly Criterion             凯利公式建议仓位：根据胜率和盈亏比算出的理论最优下注比例
```

看着是不是很眼熟...其实这些就是`backtesting.py`（Python 回测库）跑完 `Backtest.run()` 后返回的 `stats` 结果字段，


附上常见的完整流程架构图
```md
[原始量价/财务数据] 
       │
       ▼ (矩阵化整理)
[T × N 对齐矩阵 (Close, Open, Tradable Mask)]
       │
       ▼ (纵向滚动算子 + 横向截面去极值/标准化/中性化)
[综合多因子矩阵 F]
       │
       ▼ (截面 Rank 切片 + 归一化)
[目标权重矩阵 W]
       │
       ▼ (扣除涨跌停受限 + shift(1) 消除未来函数)
[实际生效持仓矩阵 W_hold]
       │
       ├──────────────┐
       ▼ (与 R 点乘)    ▼ (差分计算绝对变化)
  [每日毛收益]     [每日换手率 × 费率]
       │              │
       └──────┬───────┘
              ▼
        [每日净收益 Net PnL]
              │
              ▼
   [累计净值 + 夏普/回撤 + 五分位单调性检验]
```