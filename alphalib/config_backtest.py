import os

_ = os.path.abspath(os.path.dirname(__file__))  # 返回当前文件路径
root_path = os.path.abspath(os.path.join(_, '..'))  # 返回根目录文件夹
pickle_path = os.path.join(root_path, 'data', 'pickle_data')
data_path = os.path.join(root_path, 'data', 'factors')
output_path = os.path.join(root_path, 'data', 'output')
min_qty_path = os.path.join(root_path, 'data', 'market', '最小下单量.csv')  # 最小下单量路径
tmp_pkl_path = os.path.join(root_path, 'data', 'tmp')  # 存放参数遍历时产生的临时文件

head_columns = [
    'candle_begin_time',
    'symbol',
    'open',
    'high',
    'low',
    'close',
    'avg_price',
    '下个周期_avg_price',
    'volume',
    'funding_rate_raw',
]

enable_funding_rate = True  # 回测曲线计算中是否启用资金费率
numba_available = True  # 是否开启numba加速
numba_cache = False  # 是否开启numba编译缓存
