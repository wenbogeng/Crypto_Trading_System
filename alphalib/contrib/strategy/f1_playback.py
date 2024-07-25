import dataclasses
import glob
import os
from typing import Callable, Dict, Iterable, Literal, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from alphalib.backtest.playback import function
from alphalib.backtest.playback import timer
from alphalib.contrib.strategy import base_strategy
from alphalib.contrib.strategy import playback_core

CUR_FILE_PATH = os.path.abspath(os.path.dirname(__file__))  # 返回当前文件路径
ROOT_PATH = os.path.abspath(os.path.join(CUR_FILE_PATH, '..', '..', '..'))  # 返回根目录文件夹


@dataclasses.dataclass
class F1PlaybackConfig:
    compound_name: str = 'COMPOUND_NAME'
    pickle_path: str = os.path.join(ROOT_PATH, 'data', 'pickle_data')
    output_path: str = os.path.join(ROOT_PATH, 'data', 'output')
    min_qty_path: str = os.path.join(ROOT_PATH, 'data', 'market', '最小下单量.csv')  # 最小下单量路径
    trade_type: Literal['swap', 'spot'] = 'swap'
    # head_columns: list = dataclasses.field(default_factory=lambda: [
    #     'candle_begin_time', 'symbol', 'open', 'high', 'low', 'close',
    #     'avg_price', '下个周期_avg_price', 'volume', 'funding_rate_raw',
    # ])
    n_jobs: int = max(os.cpu_count() - 2, 1)
    start_date: str = '2022-01-01'
    end_date: str = '2024-01-01'
    c_rate: float = 6 / 10000
    initial_trade_usdt: float = 10000
    leverage: float = 1.0
    enable_funding_rate: bool = True
    min_margin_ratio: float = 0.01


def parallel_process(func: Callable, iterable: Iterable, n_jobs: int) -> list:
    # 确保所有输入都是元组
    processed_iterable = (item if isinstance(item, tuple) else (item,) for item in iterable)

    if n_jobs == 1:
        # 单线程顺序执行
        results = [func(*item) for item in processed_iterable]
    else:
        # 使用joblib的Parallel进行多线程执行
        results = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(func)(*item) for item in processed_iterable)

    return results


def read_single_df(pkl_file_path: str) -> pd.DataFrame:
    # print(pkl_file_path)
    df = pd.read_feather(pkl_file_path)
    df.sort_values(by='candle_begin_time', inplace=True)
    df.drop_duplicates(subset=['candle_begin_time'], inplace=True, keep='last')
    # df['下个周期_avg_price'] = df['avg_price'].shift(-1)
    df.reset_index(drop=True, inplace=True)
    return df


# 接下来是要补全的函数，现在包括转换键的逻辑
def read_all_df_for_backtest(conf: F1PlaybackConfig) -> Dict[str, pd.DataFrame]:
    # 获取所有的pkl文件
    pkl_list = sorted(glob.glob(os.path.join(conf.pickle_path, f'{conf.trade_type}', '*USDT.pkl')))
    # 并行读取
    results = parallel_process(read_single_df, pkl_list, conf.n_jobs)
    # 将结果组装成字典
    all_df = {
        os.path.basename(pkl_file).replace('-USDT.pkl', 'USDT'): result
        for pkl_file, result in zip(pkl_list, results)
    }
    return all_df


def get_swap_df(df_path: str):
    if not hasattr(get_swap_df, "swap_dict"):
        get_swap_df.swap_dict = {}
        logger.debug("初始化 swap_dict")
    if df_path not in get_swap_df.swap_dict:
        logger.debug(f"加载数据 {df_path}")
        get_swap_df.swap_dict[df_path] = joblib.load(df_path, mmap_mode='c')
    return get_swap_df.swap_dict[df_path]


def run_playback_lite(
        stg: base_strategy.BaseStrategy,
        conf: F1PlaybackConfig,
        all_df: Union[Dict[str, pd.DataFrame], str] = None,
        alloc_ratio: pd.Series = None,
) -> Tuple[playback_core.AccountData, playback_core.SymbolRecord, pd.DataFrame, np.ndarray]:
    # 读取min_qty文件并转为dict格式
    min_qty_df = pd.read_csv(conf.min_qty_path, encoding='gbk')
    min_qty_df['合约'] = min_qty_df['合约'].str.replace('-', '')
    min_qty_df['最小下单量'] = -np.log10(min_qty_df['最小下单量']).round().astype(int)
    default_min_qty = min_qty_df['最小下单量'].max()
    min_qty_df.set_index('合约', inplace=True)
    min_qty_dict = min_qty_df['最小下单量'].to_dict()

    with timer.timer('读取数据'):
        # ['candle_begin_time', 'open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trade_num',
        # 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'avg_price', 'symbol', 'fundingRate',
        # 'funding_rate_raw', '下个周期_avg_price']
        if all_df is None:
            all_df = read_all_df_for_backtest(conf)
        elif isinstance(all_df, str):
            # print('使用swap df')
            all_df: Dict[str, pd.DataFrame] = get_swap_df(all_df)
        else:
            print('跳过读取')

    # 计算选币并输出为目标资金分配
    with timer.timer('calc alloc_ratio'):
        if alloc_ratio is None:
            alloc_ratio = stg.calc_alloc_ratio(all_df, conf.start_date, conf.end_date)
        else:
            print('跳过计算alloc_ratio')
        alloc_ratio_dict: Dict[str, pd.Series] = {
            symbol: group.droplevel('symbol') for symbol, group in alloc_ratio.groupby(level='symbol')
        }

    all_df_info = {}
    for symbol, df in all_df.items():
        if symbol not in alloc_ratio_dict:
            continue
        info_dict = {
            'candle_begin_time': df['candle_begin_time'],
            'symbol': df['symbol'],
            'close': df['close'],
            'next_avg_price': df['avg_price'].shift(-1),
            'next_close': df['close'].shift(-1),
            'next2_funding_rate': df['funding_rate_raw'].shift(-2).fillna(
                0) if conf.enable_funding_rate else np.float64(0),
            'min_qty_prec': pd.Series(data=min_qty_dict.get(symbol, default_min_qty), index=df.index),
            'target_alloc_ratio': pd.Series(
                data=alloc_ratio_dict[symbol].reindex(pd.Index(df['candle_begin_time']), fill_value=0).values,
                index=df.index, copy=False
            )
        }
        df_info_single = pd.DataFrame(data=info_dict, copy=False)

        # 创建筛选条件：当周期或上一个周期的alloc_ratio不为0
        cond_trade = (df_info_single['target_alloc_ratio'] != 0) | (
                df_info_single['target_alloc_ratio'].shift(1, fill_value=0) != 0)
        df_info_single = df_info_single[cond_trade]

        all_df_info[symbol] = df_info_single

    if len(all_df_info) > 0:
        df_info = pd.concat(all_df_info.values(), ignore_index=True, copy=False)
    else:
        info_dict = {
            'candle_begin_time': None, 'symbol': None, 'close': None,
            'next_avg_price': None, 'next_close': None, 'next2_funding_rate': None,
            'min_qty_prec': None, 'target_alloc_ratio': None
        }
        # 创建一个空的DataFrame，其列名为字典的键
        df_info = pd.DataFrame(columns=list(info_dict.keys()))
    # end_date后一天的上周期的alloc_ratio也不为0，应该过滤掉
    df_info = df_info[df_info['candle_begin_time'] < df_info['candle_begin_time'].max()]

    all_symbol_list = sorted(list(set(df_info['symbol'].unique())))
    n_symbol = len(all_symbol_list)
    symbol_to_int = {v: k for k, v in enumerate(all_symbol_list)}
    df_info['symbol_id'] = df_info['symbol'].map(symbol_to_int)

    df_info.sort_values(['candle_begin_time', 'symbol'], inplace=True)

    arr_cbt = df_info['candle_begin_time'].values
    unique_times, group_start = np.unique(arr_cbt, return_index=True)
    group_start = np.append(group_start, arr_cbt.size)

    if len(df_info) == 0:
        account_data = playback_core.AccountData(1)
        account_data.wallet_balance[0] = conf.initial_trade_usdt
        account_data.margin_balance[0] = conf.initial_trade_usdt
        symbol_record = playback_core.SymbolRecord(0)
        return account_data, symbol_record, df_info, unique_times

    try:
        market_data = playback_core.MarketData(
            symbol_id=df_info['symbol_id'].values,
            close=df_info['close'].values,
            next_avg_price=df_info['next_avg_price'].values,
            next_close=df_info['next_close'].values,
            next2_funding_rate=df_info['next2_funding_rate'].values,
            min_qty_prec=df_info['min_qty_prec'].values,
            target_alloc_ratio=df_info['target_alloc_ratio'].values,
            n_symbol=n_symbol,
            n_group=unique_times.size,
            group_start=group_start,
        )
    except Exception as e:
        print('df_info转换market_data失败')
        print(df_info)
        raise e

    with timer.timer('playback_core'):
        account_data, symbol_record = playback_core.playback(
            market_data=market_data,
            init_usdt=np.float64(conf.initial_trade_usdt),
            c_rate=np.float64(conf.c_rate),
            leverage=np.float64(conf.leverage),
            min_margin_ratio=np.float64(conf.min_margin_ratio),
        )
    return account_data, symbol_record, df_info, unique_times


def run_playback(
        stg: base_strategy.BaseStrategy,
        conf: F1PlaybackConfig,
        all_df: Union[Dict[str, pd.DataFrame], str] = None,
        alloc_ratio: pd.Series = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    account_data, symbol_record, df_info, unique_times = run_playback_lite(
        stg=stg, conf=conf, all_df=all_df, alloc_ratio=alloc_ratio
    )
    account_df = pd.DataFrame(index=unique_times, data={
        'totalWalletBalance': account_data.wallet_balance,
        'totalMarginBalance': account_data.margin_balance,
        'totalRealizedProfit': account_data.realized_pnl,
        'totalUnRealizedProfit': account_data.unrealized_pnl,
        'commission': account_data.commission,
        'marginRatio': account_data.margin_ratio,
    }, copy=False)

    if account_df['totalWalletBalance'].min() < 1e-8:
        ind = account_df['totalWalletBalance'].argmin()
        print(f"{account_df.index[ind]}，margin ratio低于{conf.min_margin_ratio}，归零啦！爆仓啦！")

    order_df = pd.DataFrame(data={
        'trade_time': (df_info['candle_begin_time'] + pd.DateOffset(hours=1, minutes=1)).values,
        'symbol': df_info['symbol'].values,
        '当前持仓量': symbol_record.cur_position,
        '实际下单量': symbol_record.order_amount,
        '理想开仓均价': df_info['next_avg_price'].values,
        '交易后持仓量': symbol_record.new_position,
    }, copy=False)
    order_df = order_df[order_df['trade_time'] != order_df['trade_time'].max()]
    order_df = order_df.iloc[:-1]
    order_df = order_df[order_df['实际下单量'] != 0]
    order_df = order_df.set_index(['trade_time', 'symbol'])

    new_pos_value = symbol_record.new_position * df_info['next_close'].values
    long_pos_value = np.where(new_pos_value > 0, new_pos_value, 0)
    short_pos_value = np.where(new_pos_value < 0, -new_pos_value, 0)

    ls_df_raw = pd.DataFrame(data={
        'hold_time': (df_info['candle_begin_time'] + pd.DateOffset(hours=2)).values,
        'long_pos_value': long_pos_value,
        'short_pos_value': short_pos_value,
    }, copy=False)
    ls_df_raw = ls_df_raw.groupby('hold_time', as_index=False).sum()
    ls_df = pd.DataFrame(data={
        'hold_time': ls_df_raw['hold_time'].values[:-1],
        '多头占比': ls_df_raw['long_pos_value'].values[:-1] / account_data.margin_balance[1:],
        '空头占比': ls_df_raw['short_pos_value'].values[:-1] / account_data.margin_balance[1:],
        '资金曲线': account_data.margin_balance[1:] / account_data.margin_balance[0],
    }, copy=False)
    ls_df_head = pd.DataFrame(data={
        'hold_time': [ls_df_raw['hold_time'].values[0] - pd.DateOffset(hours=1)],
        '多头占比': [0.0],
        '空头占比': [0.0],
        '资金曲线': [1.0],
    })
    ls_df = pd.concat([ls_df_head, ls_df], axis=0, ignore_index=True)
    ls_df['多头占比'] = ls_df['多头占比'].round(4)
    ls_df['空头占比'] = ls_df['空头占比'].round(4)
    ls_df.set_index('hold_time', inplace=True)
    ls_df.index.name = None

    long_hold = df_info[long_pos_value > 0].groupby('candle_begin_time')['symbol'].agg(' '.join)
    long_hold = long_hold.reindex(unique_times, fill_value='')
    long_hold = pd.Series(long_hold.values[0:-1])
    short_hold = df_info[short_pos_value > 0].groupby('candle_begin_time')['symbol'].agg(' '.join)
    short_hold = short_hold.reindex(unique_times, fill_value='')
    short_hold = pd.Series(short_hold.values[0:-1])

    turnover_monthly = account_data.turnover_monthly[1:]

    res, curve = function.freestep_evaluate(
        ls_df, long_hold, short_hold,
        month_turnover_rate_list=turnover_monthly, compound_name=conf.compound_name
    )
    commission_loss = (1 - account_df['commission'] / account_df['totalMarginBalance']).cumprod().iloc[-1] - 1

    res['交易费率'] = conf.c_rate * 10000
    res['leverage'] = conf.leverage
    res['手续费磨损净值'] = commission_loss * res['累积净值'].iloc[0]
    final_trade_usdt = round(account_df.iloc[-1]['totalMarginBalance'], 2)
    commission_sum = round(account_df['commission'].sum(), 2)
    d = res.pop('手续费磨损净值')
    res.insert(1, '手续费磨损净值', d)
    logger.info(
        f'初始投入资产: {conf.initial_trade_usdt} U,最终账户资产: {final_trade_usdt} U, 共支付手续费: {-commission_sum} U')
    account_df.index = account_df.index - pd.Timedelta(hours=1)
    curve.index = curve.index - pd.Timedelta(hours=1)
    return res, curve, account_df, order_df
