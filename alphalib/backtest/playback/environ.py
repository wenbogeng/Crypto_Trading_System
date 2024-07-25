import numpy as np

# 打印格式优化标记
tag = 'loc'

# 常规配置
config_type = np.dtype(
    {
        'names': [
            'c_rate',
            'hold_hour_num',
            'long_coin_num',
            'short_coin_num',
            'long_p',
            'short_p',
            'leverage',
            'long_risk_position',
            'initial_trade_usdt',
            'offset_stop_win',
            'offset_stop_loss'],
        'formats': [
            np.float64,
            np.int64,
            np.int64,
            np.int64,
            np.float64,
            np.float64,
            np.float64,
            np.float64,
            np.float64,
            np.float64,
            np.float64,
        ]})
playCfg = np.zeros((1,), dtype=config_type)
