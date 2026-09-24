"""波动率相关的辅助指标, 用来做"波动率分层中性化"(见 portfolio.sizing 里
的 vol_neutral_quantile_long_short), 不是选股因子, 不参与 zscore/
combine_scores 那条打分链路 —— 只是给分层函数提供"按什么切层"的依据。
"""
