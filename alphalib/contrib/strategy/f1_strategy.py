import dataclasses
import importlib
from typing import Dict, List, Tuple, Union

import pandas as pd

from alphalib.backtest.playback import timer
from alphalib.common_utils import f1_utils
from alphalib.contrib.strategy import f1_conf_types
from alphalib.contrib.strategy.base_strategy import BaseStrategy
from alphalib.contrib.strategy.f1_conf_types import CalcFactorType, F1FactorConfig, F1FilterParams


@dataclasses.dataclass
class F1StrategyConfig:
    strategy_name: str = 'STRATEGY_NAME'
    hold_period: int = 12
    calc_factor_type: CalcFactorType = 'cross'
    long_factors: List[F1FactorConfig] = dataclasses.field(default_factory=list)
    short_factors: List[F1FactorConfig] = dataclasses.field(default_factory=list)
    filter_before_params: List[F1FilterParams] = dataclasses.field(default_factory=list)
    filter_after_params: List[F1FilterParams] = dataclasses.field(default_factory=list)
    long_weight: float = 1
    short_weight: float = 1
    long_coin_num: Union[int, float] = 1
    short_coin_num: Union[int, float] = 1
    stg_weight: float = 1
    enable_seal999: bool = True

    long_white_list: List = dataclasses.field(default_factory=list)
    short_white_list: List = dataclasses.field(default_factory=list)
    long_black_list: List = dataclasses.field(default_factory=list)
    short_black_list: List = dataclasses.field(default_factory=list)


class F1Strategy(BaseStrategy):
    f1_strategy_config: F1StrategyConfig

    def __init__(self, f1_strategy_config: F1StrategyConfig):
        self.f1_strategy_config = f1_strategy_config
        if abs(self.f1_strategy_config.long_weight + self.f1_strategy_config.short_weight - 2) > 1e-8:
            print(
                '建议设置`long_weight+short_weight==2`。'
                '可通过`F1PlaybackConfig.leverage`控制杠杆，通过`F1StrategyConfig.stg_weight控制组合权重`'
            )
            print(f'当前 long_weight={f1_strategy_config.long_weight} short_weight={f1_strategy_config.short_weight}')

    def calc_alloc_ratio(self, all_df: Dict[str, pd.DataFrame], start_date: str, end_date: str) -> pd.Series:
        # columns = ["symbol", "candle_begin_time", "open", "high", "low", "close", "volume", "quote_volume",
        #            "trade_num", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "fundingRate"]
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        all_df = {symbol: df.copy(deep=False) for symbol, df in all_df.items()}

        factor_conf_list = list(set(self.f1_strategy_config.long_factors + self.f1_strategy_config.short_factors))
        filter_conf_list = list(set(
            x.to_filter_config()
            for x in self.f1_strategy_config.filter_before_params + self.f1_strategy_config.filter_after_params
        ))

        with timer.timer('--calc factors'):
            all_df_factor = {}
            for symbol, df in all_df.items():
                factors_dict = {}
                for factor_conf in factor_conf_list:
                    factor_name = factor_conf.factor_name
                    back_periods = factor_conf.back_periods
                    fdiff_n = factor_conf.fdiff_n
                    factor_col_name = f1_conf_types.factor_to_col_name(factor_conf)
                    factor_mod = importlib.import_module(f'factors.{factor_name}')
                    factors_dict[factor_col_name] = factor_mod.signal(df, back_periods, fdiff_n, factor_col_name)[
                        factor_col_name]

                for filter_factor_conf in filter_conf_list:
                    factor_name = filter_factor_conf.factor_name
                    params = filter_factor_conf.params
                    filter_col_name = filter_factor_conf.to_col_name()
                    filter_mod = importlib.import_module(f'filters.{factor_name}')
                    factors_dict[filter_col_name] = filter_mod.signal(df, params, filter_col_name)[filter_col_name]

                factors_dict = {
                    'candle_begin_time': df['candle_begin_time'],
                    'symbol': df['symbol'],
                    'close': df['close'],
                    **factors_dict
                }
                df_factor = pd.DataFrame(data=factors_dict, copy=False)
                if 1:  # 最好在选币时处理vol过滤
                    mask = df['volume'] > 0
                    if 'avg_price' in df.columns:  # 回测时过滤下周期avg_price为空的数据，实盘时不处理
                        mask_notna = df['avg_price'].shift(-1).notna()
                        mask = mask & mask_notna
                if self.f1_strategy_config.enable_seal999:  # 最好通过前置过滤处理999封印
                    df_factor = df_factor.iloc[999:][mask[999:]]
                else:
                    df_factor = df_factor[mask]

                all_df_factor[symbol] = df_factor

            df_factor_merge = pd.concat(all_df_factor.values(), ignore_index=True, copy=False)

        with timer.timer('--select range and sort'):
            series_cbt = df_factor_merge['candle_begin_time']
            cbts = series_cbt.unique()
            cbts.sort()
            cbts = pd.Series(cbts, copy=False)
            start_idx = cbts.searchsorted(start_date)
            if start_idx >= len(cbts):
                start_cbt = start_date
            else:
                start_idx = max(start_idx - self.f1_strategy_config.hold_period, 0)
                start_cbt = cbts.iloc[start_idx]
            df_factor_merge = df_factor_merge[(series_cbt >= start_cbt) & (series_cbt <= end_date)]
            df_factor_merge = df_factor_merge.sort_values(by=['candle_begin_time', 'symbol'])

        # 提前排除退市币种
        # 首先获取可能退市的币种
        max_time = df_factor_merge['candle_begin_time'].max()
        quit_df = df_factor_merge.groupby('symbol')['candle_begin_time'].max()
        quit_symbols = quit_df[quit_df < max_time].index
        # 退市币种的处理，实盘提前N小时加入黑名单
        quit_time_thresholds = quit_df.loc[quit_symbols] - pd.Timedelta(
            hours=self.f1_strategy_config.hold_period + 1)
        symbol_to_threshold = quit_time_thresholds.to_dict()
        if symbol_to_threshold:
            ths_time = df_factor_merge['symbol'].map(symbol_to_threshold)
            # 使用掩码来筛选数据
            quit_mask = df_factor_merge['symbol'].isin(quit_symbols)
            final_quit_mask = quit_mask & (df_factor_merge['candle_begin_time'] <= ths_time)
            trade_mask = ~quit_mask | final_quit_mask
        else:
            trade_mask = pd.Series(True, index=df_factor_merge.index)

        # 前置过滤
        long_inds, short_inds = self.filter_handle_before(df_factor_merge[trade_mask])

        # 计算因子列
        # 计算多空因子
        if self.f1_strategy_config.long_factors == self.f1_strategy_config.short_factors:
            calc_factor_list = [self.f1_strategy_config.long_factors]
            col_list = ['因子', '因子']
        else:
            calc_factor_list = [self.f1_strategy_config.long_factors,
                                self.f1_strategy_config.short_factors]
            col_list = ['多头因子', '空头因子']
        for factor_conf_list, col in zip(calc_factor_list, col_list):
            factor_list = [f1_conf_types.factor_to_col_name(x) for x in factor_conf_list]
            factor_coef = pd.Series([(-1 if x.descending else 1) * x.weight for x in factor_conf_list],
                                    index=factor_list)
            if self.f1_strategy_config.calc_factor_type == 'cross':
                df_factor_merge[factor_list] = df_factor_merge.groupby('candle_begin_time')[factor_list].rank(
                    ascending=True)
                factor_meds = df_factor_merge.groupby('candle_begin_time')[factor_list].transform('median')
                df_factor_merge[factor_list] = df_factor_merge[factor_list].fillna(factor_meds)
            df_factor_merge[col] = df_factor_merge[factor_list].dot(factor_coef.T)  # use np.dot instead?

        # 计算选币
        cols_for_gen_select = ['candle_begin_time', 'symbol'] + list(set(col_list))
        df_long = df_factor_merge.loc[long_inds, cols_for_gen_select]
        df_short = df_factor_merge.loc[short_inds, cols_for_gen_select]
        rank_long = df_long.groupby('candle_begin_time')[col_list[0]].rank(
            method='first', pct=self.f1_strategy_config.long_coin_num < 1)
        rank_short = df_short.groupby('candle_begin_time')[col_list[1]].rank(
            method='first', pct=self.f1_strategy_config.short_coin_num < 1, ascending=False)
        select_long_mask = (rank_long <= self.f1_strategy_config.long_coin_num)
        select_short_mask = (rank_short <= self.f1_strategy_config.short_coin_num)
        select_long_inds = df_long.index[select_long_mask]
        select_short_inds = df_short.index[select_short_mask]

        # 后置过滤
        select_long_inds, select_short_inds = self.filter_handle_after(
            df_factor_merge.loc[select_long_inds, :],
            df_factor_merge.loc[select_short_inds, :]
        )

        if self.f1_strategy_config.long_coin_num < 1:
            long_coin_num = select_long_mask.groupby(df_long['candle_begin_time']).transform('sum')
            long_coin_num = long_coin_num.loc[select_long_inds]
        else:
            long_coin_num = self.f1_strategy_config.long_coin_num
        if self.f1_strategy_config.short_coin_num < 1:
            short_coin_num = select_short_mask.groupby(df_short['candle_begin_time']).transform('sum')
            short_coin_num = short_coin_num.loc[select_short_inds]
        else:
            short_coin_num = self.f1_strategy_config.short_coin_num

        # 计算单offset资金分配比例
        long_base_coef = 0.5 * self.f1_strategy_config.long_weight / self.f1_strategy_config.hold_period
        short_base_coef = 0.5 * self.f1_strategy_config.short_weight / self.f1_strategy_config.hold_period

        df_factor_merge['alloc_ratio_1offset'] = 0
        df_factor_merge.loc[select_long_inds, 'alloc_ratio_1offset'] += long_base_coef / long_coin_num
        df_factor_merge.loc[select_short_inds, 'alloc_ratio_1offset'] -= short_base_coef / short_coin_num

        # 按照hold_period合并coef
        # 在f1框架的选币逻辑中，给每个offset分配资金后，是按照那个offset当时的close价格计算 target_amount = assign_usdt / close，
        # 相当于乘上调整系数 close_最新 / close_当时
        with timer.timer('--merge alloc_ratio'):
            df_factor_merge['alloc_ratio_1offset'] /= df_factor_merge['close']
            df_factor_merge['alloc_ratio'] = df_factor_merge.groupby('symbol', group_keys=False)[
                'alloc_ratio_1offset'].rolling(
                self.f1_strategy_config.hold_period, min_periods=1).sum().reset_index(level=0, drop=True)
            df_factor_merge['alloc_ratio'] *= df_factor_merge['close']

        res = df_factor_merge[['candle_begin_time', 'symbol', 'alloc_ratio']]
        res = res.set_index(['candle_begin_time', 'symbol'])
        res = res['alloc_ratio']

        cbt_index = res.index.levels[0]
        cbt_index = cbt_index.sort_values()
        start_date_idx = cbt_index.searchsorted(start_date)
        start_date_idx = max(start_date_idx - 1, 0)
        if len(cbt_index) > 0:
            res_start_date = cbt_index[start_date_idx]
        else:
            res_start_date = start_date
        res = res.loc[res.index.get_level_values('candle_begin_time') >= res_start_date]
        return res

    def filter_handle_before(self, df: pd.DataFrame) -> Tuple[pd.Index, pd.Index]:
        df1_mask = ~df['symbol'].isin(self.f1_strategy_config.long_black_list)
        if self.f1_strategy_config.long_white_list:
            df1_mask = df['symbol'].isin(self.f1_strategy_config.long_white_list) & df1_mask

        df2_mask = ~df['symbol'].isin(self.f1_strategy_config.short_black_list)
        if self.f1_strategy_config.short_white_list:
            df2_mask = df['symbol'].isin(self.f1_strategy_config.short_white_list) & df2_mask

        filter_before_params = [
            [x.direction, x.filter_factor, x.filter_type, x.compare_operator, x.filter_value,
             x.rank_ascending, x.filter_after]
            for x in self.f1_strategy_config.filter_before_params
        ]
        filter_before_exec = [f1_utils.filter_generate(param=param) for param in filter_before_params]
        with timer.timer('--do_filter'):
            df1, df2 = f1_utils.do_filter(df[df1_mask], df[df2_mask], filter_before_exec)
        return df1.index, df2.index

    def filter_handle_after(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Tuple[pd.Index, pd.Index]:
        filter_after_params = [
            [x.direction, x.filter_factor, x.filter_type, x.compare_operator, x.filter_value,
             x.rank_ascending, x.filter_after]
            for x in self.f1_strategy_config.filter_after_params
        ]
        filter_after_exec = [f1_utils.filter_generate(param=param) for param in filter_after_params]
        df1 = df1.copy(deep=False)
        df2 = df2.copy(deep=False)
        df1['weight_ratio'] = 1
        df2['weight_ratio'] = 1
        df1, df2 = f1_utils.do_filter(df1, df2, filter_after_exec)
        return df1.index[df1['weight_ratio'] == 1], df2.index[df2['weight_ratio'] == 1]


@dataclasses.dataclass
class F1MultiStrategyConfig:
    multi_strategy_name: str = 'MULTI_STRATEGY_NAME'
    multi_strategy_config: List[F1StrategyConfig] = dataclasses.field(default_factory=list)


class F1MultiStrategy(BaseStrategy):
    stg_list: List[F1Strategy]

    def __init__(self, stg_list: List[F1Strategy]):
        self.stg_list = stg_list

    def calc_alloc_ratio(self, all_df: Dict[str, pd.DataFrame], start_date: str, end_date: str) -> pd.Series:
        res_list = [
            stg.calc_alloc_ratio(all_df=all_df, start_date=start_date, end_date=end_date)
            for stg in self.stg_list
        ]
        weight_list = [stg.f1_strategy_config.stg_weight for stg in self.stg_list]
        res = (pd.concat(res_list, axis=1, copy=False) * weight_list).sum(axis=1) / sum(weight_list)

        return res
