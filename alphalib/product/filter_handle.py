from alphalib.common_utils.f1_utils import FilterAfter, RankAscending, do_filter, filter_generate


def f1_before_filter(df1, df2):
    """
    F1前置过滤自定义部分
    :return:
    """
    # 1.白名单处理
    # df1 = df1[df1['symbol'].isin(['BTCUSDT', 'ETHUSDT'])]
    # 2.黑名单处理
    # df1 = df1[~df1['symbol'].isin(['BTCUSDT', 'ETHUSDT'])]
    # 3.过滤F1配置
    filter_before_params = [
        ['df1', 'ClosePctChangeMax_fl_24', 'value', 'lte', 0.2, RankAscending.FALSE, FilterAfter.FALSE],
        ['df2', 'ClosePctChangeMax_fl_24', 'value', 'lte', 0.2, RankAscending.FALSE, FilterAfter.FALSE],
        ['df1', 'QuoteVolumeSum_fl_24', 'rank', 'lte', 60, RankAscending.FALSE, FilterAfter.FALSE],
        ['df2', 'QuoteVolumeSum_fl_24', 'rank', 'lte', 60, RankAscending.FALSE, FilterAfter.FALSE],
    ]
    filter_before_exec = [filter_generate(param=param) for param in filter_before_params]
    # 4.是否开启并行过滤
    # filter_before_exec, tag = parallel_filter_handle(filter_before_exec)
    return do_filter(df1, df2, filter_before_exec)


def default_handler(df1, df2):
    return df1, df2


def filter_before(df1, df2):
    # """
    filter_factor = 'Bias_fl_100'
    # 破下轨不做多
    df1 = df1[~(df1[filter_factor] == -1)]
    # 破上轨不做空
    df2 = df2[~(df2[filter_factor] == 1)]
    # """

    df1, df2 = filter_fundingrate(df1, df2)

    return df1, df2


def filter_fundingrate(df1, df2):
    df2 = df2[df2['fundingRate'] > -0.025]

    feature = ['费率min_fl_24'][0]

    df2[feature + '升序'] = df2.groupby('candle_begin_time')[feature].apply(
        lambda x: x.rank(pct=False, ascending=True, method='first'))
    df2 = df2[(df2[feature + '升序'] >= 2) | (df2[feature] >= -0.01)]

    feature = ['费率max_fl_24'][0]
    df1[feature + '降序'] = df1.groupby('candle_begin_time')[feature].apply(
        lambda x: x.rank(pct=False, ascending=False, method='first'))
    df1 = df1[(df1[feature + '降序'] >= 2) | (df1[feature] <= 0.01)]

    return df1, df2
