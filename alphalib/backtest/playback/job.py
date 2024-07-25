import os
import re
import sys
from datetime import timedelta
from typing import Dict, List, Tuple, TypedDict, Union
from warnings import simplefilter

import numpy as np
import pandas as pd
from joblib import load
from loguru import logger as log

from alphalib.backtest.playback.function import cal_factor_by_cross, cal_factor_by_vertical, \
    neutral_strategy_playback, np_gen_selected, plot_log_double, plot_output, rtn_data_path, \
    plot_log_weekly
from alphalib.backtest.playback.timer import timer
from alphalib.backtest.utils import reader, tools

pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 1000)  # 最多显示数据的行数
# setcopywarning
pd.set_option('mode.chained_assignment', None)
# UserWarning
# warnings.filterwarnings("ignore")
# FutureWarning
simplefilter(action='ignore', category=FutureWarning)


class OtherConfig(TypedDict):
    log_level: str
    cal_factor_type: str
    hourly_details: bool
    filter_before_exec: List[str]
    filter_after_exec: List[str]
    start_date: str
    end_date: str
    factor_long_list: List[Tuple]
    factor_short_list: List[Tuple]
    trade_type: str
    compound_name: str
    quit_symbol_filter_hour: np.int64
    select_offsets: List[List[int]]
    white_list: List[List[str]]
    black_list: List[List[str]]


def extract_filter_factors_from_exec(exec_str: str) -> List[str]:
    return re.findall(r"\['(.+?)'\]", exec_str)


def get_factor_cls_and_cols(
    oth_cfg: OtherConfig
) -> Tuple[List[str], List[str], List[str], Dict[str, List[str]]]:
    all_factor_list = oth_cfg['factor_long_list'] + oth_cfg['factor_short_list']
    factor_list = tools.convert_to_feature(all_factor_list)
    factor_class_list = tools.convert_to_cls(all_factor_list)

    filter_list = [
        item
        for exec_str in oth_cfg['filter_before_exec'] + oth_cfg['filter_after_exec']
        for item in extract_filter_factors_from_exec(exec_str)
    ]
    filter_list = list(dict.fromkeys(filter_list))
    filter_cols_dic = tools.convert_to_filter_cls_quick(filter_list)

    return factor_list, filter_list, factor_class_list, filter_cols_dic


def preprocess_df_for_playback(
    play_cfg: np.ndarray, oth_cfg: OtherConfig,
    preload_df: pd.DataFrame = None
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], Dict[str, int], pd.Timestamp, pd.Timestamp]:
    feature_list, filter_list, factor_class_list, filter_cols_dic = get_factor_cls_and_cols(oth_cfg)

    if preload_df is None:
        with timer('读取数据'):
            preload_df = reader.readhour_quick(
                oth_cfg['trade_type'],
                factor_class_list,
                filter_cols_dic=filter_cols_dic,
                feature_list=feature_list
            )

    start_date = pd.to_datetime(oth_cfg['start_date'])
    end_date = pd.to_datetime(oth_cfg['end_date'])
    if preload_df['candle_begin_time'].max() < pd.to_datetime(end_date):
        data_modify_time = preload_df['candle_begin_time'].max() - timedelta(hours=1)
        log.warning(f'本地数据最新日期小于设定回测结束时间,请检查。本次回测结束时间将被改为:{data_modify_time}')
        end_date = data_modify_time
    if preload_df['candle_begin_time'].min() > (
        pd.to_datetime(start_date) - timedelta(hours=int(play_cfg['hold_hour_num'][0]))
    ):
        data_modify_time = preload_df['candle_begin_time'].min() + timedelta(hours=int(play_cfg['hold_hour_num'][0]))
        log.warning(f'本地数据最早日期大于设定回测开始时间,请检查。本次回测开始时间将被改为:{data_modify_time}')
        start_date = data_modify_time

    # 筛选日期范围
    date_range_mask_start = preload_df['candle_begin_time'] >= pd.to_datetime(start_date) - timedelta(
        hours=int(play_cfg['hold_hour_num'][0]))
    date_range_mask_end = preload_df['candle_begin_time'] <= pd.to_datetime(end_date)
    df = preload_df[date_range_mask_start & date_range_mask_end]

    all_symbol_list = sorted(list(set(df['symbol'].unique())))
    mapping_symbol_to_int = {v: k for k, v in enumerate(all_symbol_list)}
    df['symbol'] = df['symbol'].replace(mapping_symbol_to_int)
    symbols_data = df[['candle_begin_time', 'symbol', 'close', 'avg_price', 'funding_rate_raw']]

    # 删除某些行数据
    df = df[df['volume'] > 0]  # 该周期不交易的币种
    # 最后几行数据,下个周期_avg_price为空
    df = df.dropna(subset=['下个周期_avg_price'])
    # ===数据预处理
    df = df[['candle_begin_time', 'close', 'symbol'] + feature_list + filter_list]
    df = df.set_index(['candle_begin_time', 'symbol']).sort_index()
    df = df.replace([np.inf, -np.inf], np.nan)
    # 因子空值都用中位数填充, 如果填0可能后面rank排序在第一或者最后
    df[feature_list] = df[feature_list].apply(lambda x: x.fillna(x.median()))
    df = df.reset_index()

    # 提前排除退市币种
    # 首先获取可能退市的币种
    max_time = df['candle_begin_time'].max()
    quit_df = df.groupby('symbol')['candle_begin_time'].max()
    quit_symbols = quit_df[quit_df < max_time].index
    # 退市币种的处理，实盘提前N小时加入黑名单
    quit_time_thresholds = quit_df.loc[quit_symbols] - pd.Timedelta(hours=oth_cfg['quit_symbol_filter_hour'] + 1)
    symbol_to_threshold = quit_time_thresholds.to_dict()
    ths_time = df['symbol'].map(symbol_to_threshold)
    # 使用掩码来筛选数据
    quit_mask = df['symbol'].isin(quit_symbols)
    final_quit_mask = quit_mask & (df['candle_begin_time'] <= ths_time)
    df = df[~quit_mask | final_quit_mask]

    return df, symbols_data, all_symbol_list, mapping_symbol_to_int, start_date, end_date


def get_swap_df(df_path):
    if not hasattr(get_swap_df, "swap_dict"):
        get_swap_df.swap_dict = {}
        log.debug("初始化 swap_dict")
    if df_path not in get_swap_df.swap_dict:
        log.debug(f"加载数据 {df_path}")
        get_swap_df.swap_dict[df_path] = load(df_path, mmap_mode='c')
    return get_swap_df.swap_dict[df_path]


def process_playback_job(
    df: Union[pd.DataFrame, str],
    symbols_data: Union[pd.DataFrame, str],
    all_symbol_list: List[str], mapping_symbol_to_int: Dict[str, int],
    start_date: pd.Timestamp, end_date: pd.Timestamp,
    play_cfg: np.ndarray, oth_cfg: OtherConfig,
    save_playback_path=None,
    save_playback_fa_path=None,
    save_playback_fi_path=None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    log.remove()
    log.add(sys.stdout, level=oth_cfg['log_level'])

    feature_list, filter_list, factor_class_list, filter_cols_dic = get_factor_cls_and_cols(oth_cfg)
    filter_class_list = list(filter_cols_dic)

    if isinstance(df, str):
        df: pd.DataFrame = get_swap_df(df)
        df = df[['candle_begin_time', 'close', 'symbol'] + feature_list + filter_list]
    if isinstance(symbols_data, str):
        symbols_data: pd.DataFrame = get_swap_df(symbols_data)
    # ===计算因子
    if oth_cfg['cal_factor_type'] == 'cross':
        # 横截面
        df = cal_factor_by_cross(df, oth_cfg['factor_long_list'], oth_cfg['factor_short_list'])
    elif oth_cfg['cal_factor_type'] == 'vertical':
        # 纵截面
        df = cal_factor_by_vertical(df, oth_cfg['factor_long_list'], oth_cfg['factor_short_list'])
    else:
        raise ValueError('cal_factor_type set error!')
    log.info('因子计算完成')

    # numpy 选币
    base_index = pd.date_range(start=start_date - timedelta(hours=int(play_cfg['hold_hour_num'][0])), end=end_date,
                               freq='1H').tolist()
    select_coin_long, select_coin_short = np_gen_selected(
        df, base_index, oth_cfg['filter_before_exec'], oth_cfg['filter_after_exec'], play_cfg,
        oth_cfg['select_offsets'], oth_cfg['white_list'], oth_cfg['black_list'], mapping_symbol_to_int)
    log.info('选币完成')

    res, curve, account_df, display_df, order_df = neutral_strategy_playback(
        play_cfg, start_date, end_date, symbols_data, all_symbol_list,
        mapping_symbol_to_int,
        {v: k for k, v in mapping_symbol_to_int.items()},
        select_coin_long, select_coin_short,
        compound_name=oth_cfg['compound_name'],
        hourly_details=oth_cfg['hourly_details'])
    if save_playback_path is not None:
        if not os.path.exists(save_playback_path):
            os.makedirs(save_playback_path)
        # 回放数据保存
        save_path = os.path.join(save_playback_path, '净值持仓数据.csv')
        res.to_csv(save_path, encoding='gbk')
        curve.to_csv(save_path, encoding='gbk', mode='a')
        save_path = os.path.join(save_playback_path, '虚拟账户数据.csv')
        account_df.to_csv(save_path, encoding='gbk')
        save_path = os.path.join(save_playback_path, '持仓面板数据.pkl')
        display_df.to_pickle(save_path)
        save_path = os.path.join(save_playback_path, '下单面板数据.pkl')
        order_df.to_pickle(save_path)
    log.info(f'\n{res.to_markdown()}')

    if save_playback_fa_path is not None:
        assert save_playback_fi_path is not None
        if not os.path.exists(save_playback_fa_path):
            os.makedirs(save_playback_fa_path)
        if not os.path.exists(save_playback_fi_path):
            os.makedirs(save_playback_fi_path)
        res_fa = res.copy()
        # 将因子参数添加到res中，保存格式
        res_fa['因子名'] = oth_cfg['factor_long_list'][0][0]
        res_fa['因子TF'] = oth_cfg['factor_long_list'][0][1]
        res_fa['因子参数'] = oth_cfg['factor_long_list'][0][2]
        res_fa['因子差分'] = oth_cfg['factor_long_list'][0][3]

        for i in range(len(filter_list)):
            res_fa[f'过滤因子_{i + 1}'] = filter_list[i].split('_fl_')[0]
            res_fa[f'过滤因子_参数_{i + 1}'] = filter_list[i].split('_fl_')[1]

        if not os.path.exists(os.path.join(save_playback_fa_path, f'{factor_class_list}.csv')):
            res_fa.to_csv(os.path.join(save_playback_fa_path, f'{factor_class_list}.csv'))
        else:
            res_fa.to_csv(os.path.join(save_playback_fa_path, f'{factor_class_list}.csv'), header=False, mode='a')

        if not filter_class_list:
            filter_class_list.append(None)
        if not os.path.exists(os.path.join(save_playback_fi_path, f'{filter_class_list}.csv')):
            res_fa.to_csv(os.path.join(save_playback_fi_path, f'{filter_class_list}.csv'))
        else:
            res_fa.to_csv(os.path.join(save_playback_fi_path, f'{filter_class_list}.csv'), header=False, mode='a')
    return res, curve


def playback_start(playCfg: np.ndarray, othCfg: OtherConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df, symbols_data, all_symbol_list, mapping_symbol_to_int, start_date, end_date = preprocess_df_for_playback(
        play_cfg=playCfg, oth_cfg=othCfg)
    log.info('数据处理完成')

    save_playback_path = os.path.join(rtn_data_path, othCfg['compound_name'])
    res, curve = process_playback_job(
        df, symbols_data, all_symbol_list, mapping_symbol_to_int, start_date, end_date,
        playCfg, othCfg,
        save_playback_path=save_playback_path
    )

    # plotly 作图
    plot_output(curve, res, save_playback_path, save_html=True)
    # 船队作图整合
    # plot_log_double(curve)
    # 每周收益
    plot_log_weekly(curve)
    return res, curve
