import asyncio
import os
import sys
import unittest.mock

sys.path.append(os.path.abspath('C:/LoC/量化/alphalib_dev'))

from alphalib.config_backtest import root_path

# 避免修改原帖中的代码，通过mock的方式屏蔽原帖中的路径创建，转为通过本脚本控制
with unittest.mock.patch('os.makedirs'):
    import alphalib.contrib.data_center as data_center

# data_center.cpu = 4
# CONCURRENCY 的大小影响pkl文件的并发生成，可以根据cpu实际核心数调整
data_center.CONCURRENCY = max(data_center.cpu, 1)
# semaphore 的大小影响下载数据包的并发数，可适当调大
data_center.semaphore = asyncio.Semaphore(value=min(2 * data_center.cpu, 8))
# retry_times 的大小影响下载数据包的重试次数
data_center.retry_times = 10
# api_semaphore 的大小影响请求kline接口和fundingRate接口的并发数，建议小于5
data_center.api_semaphore = asyncio.Semaphore(value=min(2 * data_center.cpu, 3))
data_center.proxy = proxy = 'http://127.0.0.1:23457'
# 如果 use_proxy_download_file为False时下载数据包运行不下去，就尝试打开下面两行注释
# data_center.use_proxy_download_file = True
# data_center.file_proxy = proxy
data_center.root_path = market_path = os.path.join(root_path, 'data/market')
data_center.funding_path = funding_path = os.path.join(market_path, 'funding/')
data_center.openInterestHist_path = openInterestHist_path = os.path.join(market_path, 'openInterestHist/')
data_center.takerlongshortRatio_path = takerlongshortRatio_path = os.path.join(
    market_path, 'takerlongshortRatio/'
)
data_center.pickle_path = pickle_path = os.path.join(root_path, 'data/pickle_data')
trade_type_path = os.path.join(pickle_path, data_center.trade_type)
# 设置整点分钟偏移量，偏移量最小单位为5分钟，可以设置[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]，修改后需要先删除所有pkl文件
data_center.hour_offsets = hour_offsets = []
offset_pkl_paths = [
    os.path.join(f'{pickle_path}_{hour_offset}', data_center.trade_type) for hour_offset in hour_offsets
]
data_center.thunder = False
data_center.force_analyse = True

if __name__ == '__main__':
    for x in [market_path, funding_path, openInterestHist_path, takerlongshortRatio_path, pickle_path,
              trade_type_path] + offset_pkl_paths:
        if not os.path.exists(x):
            os.makedirs(x)

    data_center.run()
