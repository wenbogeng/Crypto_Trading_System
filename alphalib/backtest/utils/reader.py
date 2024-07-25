import os
from glob import glob
from typing import Dict, List, Tuple

import pandas as pd
from joblib import Parallel, delayed

from alphalib.config_backtest import data_path


def get_factors_path(trade_type, factor_class_list):
    result = dict()

    for path in glob(os.path.join(data_path, trade_type, '*', 'coin_alpha_factor_*.pkl')):
        symbol = path.split(os.sep)[-2]
        class_name = path.split(os.sep)[-1].replace('.pkl', '').replace('coin_alpha_factor_', '')
        if class_name in factor_class_list:
            if symbol not in result.keys():
                result[symbol] = []
            result[symbol].append(path)

    return result


def get_factors_path_quick(trade_type: str, factor_class_list: List[str], feature_list: List[str]) -> Tuple[
    Dict[str, List[List[str]]], Dict[str, List[str]]
]:
    symbol_list = [os.path.basename(x) for x in glob(os.path.join(data_path, trade_type, '*'))]
    result = {symbol: [] for symbol in symbol_list}
    select_cols_dic = {}
    for class_name in factor_class_list:
        select_cols_dic[class_name] = [x for x in feature_list if x.split('_bh')[0] == class_name]
        for symbol in symbol_list:
            path = os.path.join(data_path, trade_type, f'{symbol}', f'coin_alpha_factor_{class_name}.pkl')
            result[symbol].append([path, class_name])
    return result, select_cols_dic


def readhour_quick(trade_type: str, factor_class_list: List[str], filter_cols_dic: Dict[str, List[str]] = {},
                   njobs: int = 16, feature_list: List[str] = []) -> pd.DataFrame:
    def _read(trade_type: str, symbol: str, path_list: List[List[str]], filter_cols_dic: Dict[str, List[str]],
              select_cols_dic: Dict[str, List[str]]) -> pd.DataFrame:
        df_list = []
        df = pd.read_feather(os.path.join(data_path, trade_type, symbol, 'coin_alpha_head.pkl'))
        df_list.append(df)
        # 读因子文件
        for it in path_list:
            path, class_name = it
            select_cols = select_cols_dic[class_name] if select_cols_dic[class_name] else None
            try:
                df_ = pd.read_feather(path, columns=select_cols)
                df_list.append(df_)
            except FileNotFoundError as e:
                pass
                # print(f'{path}: {e}')

        # 读过滤文件
        for filter_name, select_cols in filter_cols_dic.items():
            filter_path = os.path.join(data_path, trade_type, symbol, f'coin_alpha_filter_{filter_name}.pkl')
            select_cols = select_cols if select_cols else None
            try:
                df_ = pd.read_feather(filter_path, columns=select_cols)
                df_list.append(df_)
            except FileNotFoundError as e:
                pass
        df = pd.concat(df_list, axis=1)
        df.sort_values(by=['candle_begin_time', ], inplace=True)
        df.reset_index(drop=True, inplace=True)
        # 删除前N行
        # df.drop(df.index[:999], inplace=True)
        df = df.iloc[999:]
        # 处理极端情况
        if df.empty:
            return
        df.reset_index(drop=True, inplace=True)
        return df

    factor_paths, select_cols_dic = get_factors_path_quick(trade_type, factor_class_list, feature_list=feature_list)
    all_list = Parallel(n_jobs=njobs)(
        delayed(_read)(trade_type, symbol, path_list, filter_cols_dic, select_cols_dic)
        for symbol, path_list in factor_paths.items()
    )

    all_df = pd.concat(all_list, ignore_index=True)
    all_df.sort_values(by=['candle_begin_time', 'symbol'], inplace=True)
    all_df.reset_index(drop=True, inplace=True)

    return all_df


def readhour(trade_type, factor_class_list, filter_class_list=[], njobs=16):
    def _read(trade_type, symbol, path_list, filter_class_list):
        df = pd.read_feather(os.path.join(data_path, trade_type, symbol, 'coin_alpha_head.pkl'))

        # 读因子文件
        feature_list = []
        for path in path_list:
            df_ = pd.read_feather(path)
            for f in df_.columns:
                df[f] = df_[f]
                feature_list.append(f)
        # 读过滤文件
        filter_list = []
        for filter_name in filter_class_list:
            filter_path = os.path.join(data_path, trade_type, symbol, f'coin_alpha_filter_{filter_name}.pkl')
            df_ = pd.read_feather(filter_path)
            filter_columns = list(set(df_.columns))
            for f in filter_columns:
                df[f] = df_[f]
                filter_list.append(f)

        df.sort_values(by=['candle_begin_time', ], inplace=True)
        df.reset_index(drop=True, inplace=True)
        # 删除前N行
        # df.drop(df.index[:999], inplace=True)
        df = df.iloc[999:]
        # 处理极端情况
        if df.empty:
            return
        df.reset_index(drop=True, inplace=True)
        return df

    factor_paths = get_factors_path(trade_type, factor_class_list)

    all_list = Parallel(n_jobs=njobs)(
        delayed(_read)(trade_type, symbol, path_list, filter_class_list)
        for symbol, path_list in factor_paths.items()
    )

    all_df = pd.concat(all_list, ignore_index=True)
    all_df.sort_values(by=['candle_begin_time', 'symbol'], inplace=True)
    all_df.reset_index(drop=True, inplace=True)

    return all_df
