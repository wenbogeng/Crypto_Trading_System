import dataclasses
import importlib
from typing import Dict, List, Tuple, Union

import pandas as pd

from alphalib.backtest.playback import timer
from alphalib.common_utils import f1_utils
from alphalib.contrib.strategy import f1_conf_types
from alphalib.contrib.strategy.base_strategy import BaseStrategy
from alphalib.contrib.strategy.f1_conf_types import CalcFactorType, F1FactorConfig, F1FilterFactorConfig, F1FilterParams


@dataclasses.dataclass
class F1StrategyLegacyConfig:
    strategy_name: str = 'STRATEGY_NAME'
    hold_period: int = 12
    calc_factor_type: CalcFactorType = 'cross'
    factors: List[F1FactorConfig] = dataclasses.field(default_factory=list)
    filter_factors: List[F1FilterFactorConfig] = dataclasses.field(default_factory=list)
    filter_before_params: List[F1FilterParams] = dataclasses.field(default_factory=list)
    filter_after_params: List[F1FilterParams] = dataclasses.field(default_factory=list)
    long_weight: float = 1
    short_weight: float = 1
    select_coin_num: Union[int, float] = 1

    long_white_list: List = dataclasses.field(default_factory=list)
    short_white_list: List = dataclasses.field(default_factory=list)
    long_black_list: List = dataclasses.field(default_factory=list)
    short_black_list: List = dataclasses.field(default_factory=list)


class F1StrategyLegacy(BaseStrategy):
    f1_strategy_config: F1StrategyLegacyConfig

    def __init__(self, f1_strategy_config: F1StrategyLegacyConfig):
        self.f1_strategy_config = f1_strategy_config

    def calc_alloc_ratio(self, all_df: Dict[str, pd.DataFrame], start_date: str, end_date: str) -> pd.Series:
        # columns = ["symbol", "candle_begin_time", "open", "high", "low", "close", "volume", "quote_volume",
        #            "trade_num", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "fundingRate"]
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        all_df = {symbol: df.copy(deep=False) for symbol, df in all_df.items()}
        all_df_factor = {}

        for symbol, df in all_df.items():
            factors_dict = {}
            for factor_conf in self.f1_strategy_config.factors:
                factor_name = factor_conf.factor_name
                back_periods = factor_conf.back_periods
                fdiff_n = factor_conf.fdiff_n
                factor_col_name = f1_conf_types.factor_to_col_name(factor_conf)
                factor_mod = importlib.import_module(f'factors.{factor_name}')
                factors_dict[factor_col_name] = factor_mod.signal(df, back_periods, fdiff_n, factor_col_name)[
                    factor_col_name]
            for filter_factor_conf in self.f1_strategy_config.filter_factors:
                factor_name = filter_factor_conf.factor_name
                params = filter_factor_conf.params
                filter_col_name = filter_factor_conf.to_col_name()
                filter_mod = importlib.import_module(f'filters.{factor_name}')
                df = filter_mod.signal(df, params, filter_col_name)
                factors_dict[filter_col_name] = filter_mod.signal(df, params, filter_col_name)[filter_col_name]

            factors_dict = {
                'candle_begin_time': df['candle_begin_time'],
                'symbol': df['symbol'],
                'close': df['close'],
                **factors_dict
            }
            df_factor = pd.DataFrame(data=factors_dict)
            if 1:  # 最好在选币时处理vol过滤
                mask_vol = df['volume'] > 0
                mask_notna = df['avg_price'].shift(-1).notna()
                mask = mask_vol & mask_notna
            if 1:  # 最好通过前置过滤处理999封印
                df_factor = df_factor.iloc[999:][mask[999:]]

            start_idx = df_factor['candle_begin_time'].searchsorted(pd.to_datetime(start_date))
            start_idx = max(start_idx - self.f1_strategy_config.hold_period, 0)
            end_idx = df_factor['candle_begin_time'].searchsorted(pd.to_datetime(end_date), side='right')
            df_factor = df_factor.iloc[start_idx:end_idx]

            all_df_factor[symbol] = df_factor

        df_factor_merge = pd.concat(all_df_factor.values(), ignore_index=True, copy=False)
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
        ths_time = df_factor_merge['symbol'].map(symbol_to_threshold)
        # 使用掩码来筛选数据
        quit_mask = df_factor_merge['symbol'].isin(quit_symbols)
        final_quit_mask = quit_mask & (df_factor_merge['candle_begin_time'] <= ths_time)
        trade_mask = ~quit_mask | final_quit_mask

        # 前置过滤
        long_inds, short_inds = self.filter_handle_before(df_factor_merge[trade_mask])

        # 计算因子列
        factor_list = [f1_conf_types.factor_to_col_name(x) for x in self.f1_strategy_config.factors]
        factor_coef = pd.Series([(-1 if x.descending else 1) * x.weight for x in self.f1_strategy_config.factors],
                                index=factor_list)
        if self.f1_strategy_config.calc_factor_type == 'cross':
            df_factor_merge[factor_list] = df_factor_merge.groupby('candle_begin_time')[factor_list].rank(
                ascending=True)
            factor_meds = df_factor_merge.groupby('candle_begin_time')[factor_list].transform('median')
            df_factor_merge[factor_list] = df_factor_merge[factor_list].fillna(factor_meds)
        df_factor_merge['因子'] = df_factor_merge[factor_list].dot(factor_coef.T)  # use np.dot instead?

        # 计算选币
        cols_for_gen_select = ['candle_begin_time', 'symbol', '因子']
        df_long = df_factor_merge.loc[long_inds, cols_for_gen_select]
        df_short = df_factor_merge.loc[short_inds, cols_for_gen_select]
        pct = self.f1_strategy_config.select_coin_num < 1
        rank_long = df_long.groupby('candle_begin_time')['因子'].rank(method='first', pct=pct)
        rank_short = df_short.groupby('candle_begin_time')['因子'].rank(method='first', pct=pct, ascending=False)
        select_long_mask = (rank_long <= self.f1_strategy_config.select_coin_num)
        select_short_mask = (rank_short <= self.f1_strategy_config.select_coin_num)
        select_long_inds = df_long.index[select_long_mask]
        select_short_inds = df_short.index[select_short_mask]

        # 后置过滤
        select_long_inds, select_short_inds = self.filter_handle_after(
            df_factor_merge.loc[select_long_inds, :],
            df_factor_merge.loc[select_short_inds, :]
        )

        # 计算单offset资金分配比例
        base_coef = 0.5 / self.f1_strategy_config.hold_period / self.f1_strategy_config.select_coin_num
        df_factor_merge['alloc_ratio_1offset'] = 0
        df_factor_merge.loc[
            select_long_inds, 'alloc_ratio_1offset'] += base_coef * self.f1_strategy_config.long_weight
        df_factor_merge.loc[
            select_short_inds, 'alloc_ratio_1offset'] -= base_coef * self.f1_strategy_config.short_weight

        # 按照hold_period合并coef
        # 在f1框架的选币逻辑中，给每个offset分配资金后，是按照那个offset当时的close价格计算 target_amount = assign_usdt / close，
        # 相当于乘上调整系数 close_最新 / close_当时
        with timer.timer('merge alloc_ratio'):
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
        start_date_idx = cbt_index.searchsorted(pd.to_datetime(start_date))
        start_date_idx = max(start_date_idx - 1, 0)
        res_start_date = cbt_index[start_date_idx]

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
        with timer.timer('do_filter：'):
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
