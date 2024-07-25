from typing import Tuple

import numba as nb
import numba.experimental as nb_exp
import numpy as np
from numpy.typing import NDArray

MARKET_DATA_SPEC = [
    ('symbol_id', nb.int64[:]),
    ('close', nb.float64[:]),
    ('next_avg_price', nb.float64[:]),
    ('next_close', nb.float64[:]),
    ('next2_funding_rate', nb.float64[:]),
    ('min_qty_prec', nb.int64[:]),
    ('target_alloc_ratio', nb.float64[:]),
    ('n_symbol', nb.int64),
    ('n_group', nb.int64),
    ('group_start', nb.int64[:]),
]


@nb_exp.jitclass(MARKET_DATA_SPEC)
class MarketData:
    def __init__(
            self,
            symbol_id: NDArray[np.int64],
            close: NDArray[np.float64],
            next_avg_price: NDArray[np.float64],
            next_close: NDArray[np.float64],
            next2_funding_rate: NDArray[np.float64],
            min_qty_prec: NDArray[np.int64],
            target_alloc_ratio: NDArray[np.float64],
            n_symbol: np.int64, n_group: np.int64, group_start: NDArray[np.int64],
    ):
        self.symbol_id = symbol_id
        self.close = close
        self.next_avg_price = next_avg_price
        self.next_close = next_close
        self.next2_funding_rate = next2_funding_rate
        self.min_qty_prec = min_qty_prec
        self.target_alloc_ratio = target_alloc_ratio
        self.n_symbol = n_symbol
        self.n_group = n_group
        self.group_start = group_start


ACCOUNT_DATA_SPEC = [
    ('n', nb.int64),
    ('wallet_balance', nb.float64[:]),
    ('margin_balance', nb.float64[:]),
    ('realized_pnl', nb.float64[:]),
    ('unrealized_pnl', nb.float64[:]),
    ('commission', nb.float64[:]),
    ('margin_ratio', nb.float64[:]),
    ('turnover_monthly', nb.float64[:]),
]


@nb_exp.jitclass(ACCOUNT_DATA_SPEC)
class AccountData:
    def __init__(self, n: np.int64):
        self.n = n
        self.wallet_balance = np.zeros(n, dtype=np.float64)
        self.margin_balance = np.zeros(n, dtype=np.float64)
        self.realized_pnl = np.zeros(n, dtype=np.float64)
        self.unrealized_pnl = np.zeros(n, dtype=np.float64)
        self.commission = np.zeros(n, dtype=np.float64)
        self.margin_ratio = np.zeros(n, dtype=np.float64)
        self.turnover_monthly = np.zeros(n, dtype=np.float64)


SYMBOL_RECORD_SPEC = [
    ('n', nb.int64),
    ('cur_position', nb.float64[:]),
    ('order_amount', nb.float64[:]),
    ('new_position', nb.float64[:]),
    ('unrealized_pnl', nb.float64[:]),
    ('cur_entry_price', nb.float64[:]),
]


@nb_exp.jitclass(SYMBOL_RECORD_SPEC)
class SymbolRecord:
    def __init__(self, n: np.int64):
        self.n = n
        self.cur_position = np.zeros(n, dtype=np.float64)
        self.order_amount = np.zeros(n, dtype=np.float64)
        self.new_position = np.zeros(n, dtype=np.float64)
        self.unrealized_pnl = np.zeros(n, dtype=np.float64)
        self.cur_entry_price = np.zeros(n, dtype=np.float64)


@nb.njit()
def playback(market_data: MarketData, init_usdt: np.float64, c_rate: np.float64, leverage: np.float64,
             min_margin_ratio: np.float64) -> Tuple[
    AccountData, SymbolRecord
]:
    """
    计算每根K线结束后（收取资金费率后，交易前）账户的状态。
    因此第一根K线对应的为账户初始状态。
    :param market_data: 包含所需的K线相关数据
    :param init_usdt: 初始投入资金
    :param c_rate: 交易费率
    :param leverage: 杠杆
    :return:
    - AccountData: 记录每周期账户的状态
    - SymbolRecord: 记录一些f1回放结果（如下单面板数据）所需的额外信息
    """
    account_data = AccountData(market_data.n_group)
    symbol_record = SymbolRecord(market_data.close.shape[0])
    symbol_pos = np.zeros(market_data.n_symbol, dtype=np.float64)
    symbol_entry_price = np.zeros(market_data.n_symbol, dtype=np.float64)

    next_tot_realized_pnl = 0.0
    next_tot_unrealized_pnl = 0.0
    next_tot_commission = 0.0
    next_tot_hold_value = 0.0
    next_tot_turnover_monthly = 0.0

    for group_idx in range(market_data.n_group):
        # K线闭合，首先更新账户信息
        if group_idx == 0:
            account_data.wallet_balance[0] = init_usdt
            account_data.margin_balance[0] = init_usdt
            account_data.margin_ratio[0] = np.inf
        else:
            account_data.wallet_balance[group_idx] = (
                    account_data.wallet_balance[group_idx - 1] + next_tot_realized_pnl - next_tot_commission
            )
            account_data.margin_balance[group_idx] = account_data.wallet_balance[group_idx] + next_tot_unrealized_pnl
            account_data.realized_pnl[group_idx] = next_tot_realized_pnl
            account_data.unrealized_pnl[group_idx] = next_tot_unrealized_pnl
            account_data.commission[group_idx] = -next_tot_commission
            account_data.margin_ratio[group_idx] = account_data.margin_balance[group_idx] / (
                    np.abs(next_tot_hold_value) + 1e-8)
            account_data.turnover_monthly[group_idx] = next_tot_turnover_monthly
            if account_data.margin_ratio[group_idx] < min_margin_ratio:
                return account_data, symbol_record
        # 开始交易
        all_trade_usdt = account_data.wallet_balance[group_idx] * leverage
        # tot_order_usdt = 0.0
        next_tot_realized_pnl = 0.0
        next_tot_unrealized_pnl = 0.0
        next_tot_commission = 0.0
        next_tot_hold_value = 0.0
        next_tot_turnover_monthly = 0.0
        for idx in range(market_data.group_start[group_idx], market_data.group_start[group_idx + 1]):
            symbol_id = market_data.symbol_id[idx]
            cur_pos = symbol_pos[symbol_id]
            cur_entry_price = symbol_entry_price[symbol_id]

            if market_data.next_close[idx] > 0 and market_data.next_avg_price[idx] > 0 and market_data.close[idx] > 0:
                # 正常计算下单
                target_amount = all_trade_usdt * market_data.target_alloc_ratio[idx] / market_data.close[idx]
                order_amount = target_amount - cur_pos
                min_qty = market_data.min_qty_prec[idx]
                order_amount = np.round(order_amount * (10 ** min_qty)) / (10 ** min_qty)
                order_usdt = np.abs(order_amount) * market_data.close[idx]
                if order_usdt < 5 and target_amount != 0:
                    order_amount = 0.0
                    order_usdt = 0
                trade_price = market_data.next_avg_price[idx]
            else:
                # 如果下根K线异常，说明交易对下架，忽略target_alloc_ratio，直接在当前K线强平（保险起见，也检查了当前K线）
                print(f'触发强平，请检查选币 idx={idx} symbol_id={symbol_id}')
                order_amount = -cur_pos
                if market_data.close[idx] > 0:
                    trade_price = market_data.close[idx]
                else:
                    trade_price = 0
                order_usdt = np.abs(order_amount) * trade_price

            # 统计计划下单资金
            # tot_order_usdt += order_usdt
            # 统计手续费
            next_tot_commission += trade_price * np.abs(order_amount) * c_rate

            # 若反向下单，则会出现已实现仓位
            realized_pos = 0.0
            if np.sign(order_amount) != np.sign(cur_pos):
                realized_pos = min(np.abs(order_amount), np.abs(cur_pos))
            # 已实现盈亏
            next_tot_realized_pnl += realized_pos * (trade_price - cur_entry_price) * np.sign(cur_pos)

            # 交易后新仓位
            new_pos = cur_pos + order_amount
            # 交易后平均新开仓价格
            new_entry_price = 0
            if new_pos != 0:
                if realized_pos == 0:
                    new_entry_price = (cur_pos * cur_entry_price + order_amount * trade_price) / new_pos
                else:
                    if np.abs(order_amount) >= np.abs(cur_pos):
                        new_entry_price = trade_price
                    else:
                        new_entry_price = cur_entry_price

            # TODO: 需要兼容不同period下的月化换手率计算
            # 统计交易的月化换手率
            next_tot_turnover_monthly += order_usdt / all_trade_usdt / 2 * 24 * 30.4

            # 提前计算下根K线结束后的未实现盈亏
            unreal_pnl = 0
            if new_pos != 0:
                unreal_pnl = new_pos * (market_data.next_close[idx] - new_entry_price)
                next_tot_unrealized_pnl += unreal_pnl

            # 这里原版f1回测会对小额持仓清零，并且混用着两套清零逻辑。这里尝试复现f1的结果
            # 第1种清零逻辑：对应原版f1中 symbol_info_[:, 8] 的清零逻辑
            new_pos_ = new_pos
            if np.abs(new_pos_) * market_data.next_close[idx] < 0.001 * all_trade_usdt:
                new_pos_ = 0

            # 提前计算下根K线结束后的持仓价值
            # TODO: 感觉原版f1的计算有点问题，目前写法是abs(sum())，应该是sum(abs())？这里暂时先遵循原版f1的逻辑
            if new_pos_ != 0:
                next_tot_hold_value += new_pos_ * market_data.next_close[idx]

            # 提前计算下根K线结束后的资金费率，并提前计入已实现盈亏。略有点不合理，只是为了和F1(明明资金费率补丁版)一致
            if new_pos != 0:
                next_tot_realized_pnl -= new_pos * market_data.next_close[idx] * market_data.next2_funding_rate[idx]

            # 记录一些f1回放结果（如下单面板等数据.pkl）所需的额外信息
            symbol_record.cur_position[idx] = cur_pos
            symbol_record.order_amount[idx] = order_amount
            symbol_record.new_position[idx] = new_pos_
            symbol_record.unrealized_pnl[idx] = unreal_pnl
            symbol_record.cur_entry_price[idx] = cur_entry_price

            # 第2种清零逻辑：对应原版f1中 symbol_info[:, 8] 的清零逻辑
            if np.abs(new_pos) * market_data.next_close[idx] < 1:
                new_pos = 0
            # 更新symbol账本
            symbol_pos[symbol_id] = new_pos
            symbol_entry_price[symbol_id] = new_entry_price

    return account_data, symbol_record
