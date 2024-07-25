import os
import sys

sys.path.append(os.path.abspath('C:/LoC/量化/alphalib_dev'))

from loguru import logger

from alphalib.backtest.playback import function
from alphalib.contrib.strategy import f1_playback
from strategies import bias as strategy_module

if __name__ == '__main__':
    stg = strategy_module.strategy
    module_name = strategy_module.__name__.split('.')[-1]
    f1_conf = f1_playback.F1PlaybackConfig(
        compound_name=module_name,
        start_date='2022-7-1',
        end_date='2100-1-1',
        c_rate= 0 / 10000,
        leverage=1,
        enable_funding_rate=True
    )

    res, curve, account_df, order_df = f1_playback.run_playback(stg, f1_conf)

    output_path = os.path.join(f1_playback.ROOT_PATH, 'data', 'output')
    save_dir = os.path.join(output_path, module_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    equity_res_path = os.path.join(save_dir, '净值持仓数据.csv')
    res.to_csv(equity_res_path, encoding='gbk')
    curve.to_csv(equity_res_path, encoding='gbk', mode='a')
    account_df.to_csv(os.path.join(save_dir, '虚拟账户数据.csv'), encoding='gbk')
    order_df.to_pickle(os.path.join(save_dir, '下单面板数据.pkl'))

    logger.info(f'\n{res.to_markdown()}')
    function.plot_output(curve, res, save_dir, save_html=True)
    function.plot_log_weekly(curve)
