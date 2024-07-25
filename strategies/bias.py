from alphalib.contrib.strategy import f1_conf_types, f1_strategy

conf = f1_strategy.F1StrategyConfig(
    strategy_name='Bias',
    hold_period=24,
    long_factors=[f1_conf_types.F1FactorConfig('Bias', True, 96, 0, 1)],
    short_factors=[f1_conf_types.F1FactorConfig('Bias', True, 96, 0, 1)],
    enable_seal999=False,
    filter_before_params=[
        f1_conf_types.F1FilterParams('df1', 'KlineCnt_fl_1000', 'value', 'gte', 1000 - 1e-7, False, False),
        f1_conf_types.F1FilterParams('df2', 'KlineCnt_fl_1000', 'value', 'gte', 1000 - 1e-7, False, False),
        f1_conf_types.F1FilterParams('df1', 'ClosePctChangeMax_fl_24', 'value', 'lte', 0.2, False, False),
        f1_conf_types.F1FilterParams('df2', 'ClosePctChangeMax_fl_24', 'value', 'lte', 0.2, False, False),
        f1_conf_types.F1FilterParams('df1', 'QuoteVolumeSum_fl_24', 'rank', 'lte', 60, False, False),
        f1_conf_types.F1FilterParams('df2', 'QuoteVolumeSum_fl_24', 'rank', 'lte', 60, False, False),
    ],
    long_weight=1,
    short_weight=1,
    long_coin_num=1,
    short_coin_num=1,
)

strategy = f1_strategy.F1Strategy(conf)
