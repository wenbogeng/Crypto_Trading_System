import asyncio
import datetime
import json
import os
import platform
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob
from hashlib import sha256
from itertools import groupby
from operator import itemgetter

import aiofiles
import aiohttp
import matplotlib.pyplot as plt
import pandas as pd
import tqdm
from dateutil.relativedelta import relativedelta
from joblib import Parallel, delayed
from lxml import objectify
from numpy import float64, int64

plt.rcParams["figure.dpi"] = 100
pd_display_rows = 10
pd_display_cols = 100
pd_display_width = 1000
pd.set_option('display.max_rows', pd_display_rows)
pd.set_option('display.min_rows', pd_display_rows)
pd.set_option('display.max_columns', pd_display_cols)
pd.set_option('display.width', pd_display_width)
pd.set_option('display.max_colwidth', pd_display_width)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('expand_frame_repr', False)
os.environ['NUMEXPR_MAX_THREADS'] = "256"

if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# todo 并发数CONCURRENCY，按个人电脑配置修改
cpu = os.cpu_count()
if cpu < 1:
    cpu = 1
CONCURRENCY = max(cpu, 1)
semaphore = asyncio.Semaphore(value=min(2 * cpu, 8))

api_semaphore = asyncio.Semaphore(value=min(2 * cpu, 3))

BASE_URL = 'https://data.binance.vision/'
root_center_url = 'https://s3-ap-northeast-1.amazonaws.com/data.binance.vision'

# 结算期的symbol没有交易数据
# SETTLED_SYMBOLS字典保存了结算的开始时间和结束时间
# 若分析数据完整性有新的symbol报错日内数据不完整，需要更新SETTLED_SYMBOLS
SETTLED_SWAP_SYMBOLS = {
    'ICPUSDT': ['2022-06-10 09:00:00', '2022-09-27 02:30:00'],
    # https://www.binance.com/en/support/announcement/binance-futures-will-launch-usd%E2%93%A2-m-icp-perpetual-contracts-with-up-to-25x-leverage-adabdfbc53344094808a7bea464f101b
    #  'MINAUSDT': ['2023-02-06 03:30:00', '2023-02-07 11:00:00'], # https://www.binance.com/en/support/announcement/binance-futures-to-resume-trading-on-usdt-margined-mina-perpetual-contract-611746e5caf848889b132d9fdde6c47f # noqa: E501
    'BNXUSDT': ['2023-02-11 04:00:00', '2023-02-22 22:45:00'],
    # https://www.binance.com/en/support/announcement/binance-futures-to-relaunch-usd%E2%93%A2-m-bnx-perpetual-contracts-with-up-to-20x-leverage-940d0e48493e4627889c3f46371df70b
    'TLMUSDT': ['2022-06-09 23:59:00', '2023-03-30 12:30:00']
}

SETTLED_SPOT_SYMBOLS = {
}

swap_delist_symbol_set = {'1000BTTCUSDT', 'CVCUSDT', 'DODOUSDT', 'RAYUSDT', 'SCUSDT', 'SRMUSDT', 'LENDUSDT', 'NUUSDT',
                          'LUNAUSDT', 'YFIIUSDT'}

spot_blacklist = []
# todo 设置下载现货（spot）、U本位合约（swap）数据
# trade_type = 'spot'
trade_type = 'swap'

SETTLED_SYMBOLS = None

if trade_type == 'swap':
    # 警告：此处不能改动
    prefix = 'data/futures/um/daily/klines/'
    metrics_prefix = 'data/futures/um/daily/metrics/'
    SETTLED_SYMBOLS = SETTLED_SWAP_SYMBOLS
else:
    # 警告：此处不能改动
    prefix = 'data/spot/daily/klines/'
    SETTLED_SYMBOLS = SETTLED_SPOT_SYMBOLS

# todo 代理设置，无需代理设为None
# proxy = 'http://127.0.0.1:8889'
proxy = 'http://127.0.0.1:23457'
# proxy = None
# todo 配置下载文件是否需要使用代理，1201版本之前默认不使用代理，1201版本默认使用代理，看个人的网络状况配置
# 如果 use_proxy_download_file为False时下载数据包运行不下去，就尝试改为True
use_proxy_download_file = True
file_proxy = proxy if use_proxy_download_file else None

# todo 保存目录
root_path = '/home/moke/market/'
# root_path = 'D:\\market\\'
if not os.path.exists(root_path):
    os.makedirs(root_path)

# todo 配置fundingRate保存目录
funding_path = '/home/moke/market/funding'
# funding_path = 'D:\\market\\funding\\'
if not os.path.exists(funding_path):
    os.makedirs(funding_path)

# todo 配置 合约持仓量 保存目录
openInterestHist_path = '/home/moke/market/openInterestHist'
# openInterestHist_path = 'D:\\market\\openInterestHist\\'
if not os.path.exists(openInterestHist_path):
    os.makedirs(openInterestHist_path)

# todo 配置 合约主动买卖量 保存目录
takerlongshortRatio_path = '/home/moke/market/takerlongshortRatio'
# takerlongshortRatio_path = 'D:\\market\\takerlongshortRatio\\'
if not os.path.exists(takerlongshortRatio_path):
    os.makedirs(takerlongshortRatio_path)

# todo 配置数据中心pkl生成目录
pickle_path = '/home/moke/market/pickle_data_1h'
# pickle_path = 'D:\\market\\pickle_data\\'
# pickle_path = '/media/moke/data/market/binance/pickle_data/'
if not os.path.exists(os.path.join(pickle_path, trade_type)):
    os.makedirs(os.path.join(pickle_path, trade_type))

# todo 设置整点分钟偏移量，偏移量最小单位为5分钟，可以设置[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
hour_offsets = [5, 10]
offset_pkl_paths = [os.path.join(f'{pickle_path}_{hour_offset}', trade_type) for hour_offset in hour_offsets]
# 创建偏移pkl保存目录
[os.makedirs(offset_pkl_path) for offset_pkl_path in offset_pkl_paths if not os.path.exists(offset_pkl_path)]

retry_times = 10
# todo 设置是否快速更新
thunder = True

# todo 设置网络请求失败（超时）时是否打印异常，True: 不打印，False: 打印，建议老用户设为True
blind = False

# todo 设置是否下载 metrics 数据
download_metrics = False
download_metrics = download_metrics and trade_type == 'swap'  # 下载swap合约数据时download_metrics才有效

# todo 设置只运行分析metrics，需要先下载metrics
analyse_metrics = False

# todo 设置是否强制进行完整性分析，False 目录下文件改变才进行，True 强制进行
force_analyse = False

hold_hour = '8h'
rolling_period = 7 * 24

# 全局变量，记录需要完整性分析的目录
need_analyse_set = set()

daily_updated_set = set()

# 是否要更新到最近时间，最近时间为运行脚本的前一个整点，具体以脚本打印日志为准
update_to_now = True

# todo 当上次更新在合并daily数据时异常中断，将daily_err_occur设为True，平时为False
daily_err_occur = False


async def request_session(session, params):
    while True:
        if not thunder:
            await asyncio.sleep(0.2)
        try:
            async with session.get(root_center_url, params=params, proxy=proxy, timeout=20) as response:
                return await response.text()
        except aiohttp.ClientError as ae:
            print('请求失败，继续重试', ae)
        except Exception as e:
            print('请求失败，继续重试', e)


async def request_session_4_list(session, params):
    result = []
    while True:
        if not thunder:
            await asyncio.sleep(0.2)
        try:
            async with session.get(root_center_url, params=params, proxy=proxy, timeout=20) as response:
                data = await response.text()
                root = objectify.fromstring(data.encode('ascii'))
                result.append(root)

                if root.IsTruncated:
                    # 还有下一页
                    params = {
                        'delimiter': '/',
                        'prefix': root.Prefix.text,
                        'marker': root.NextMarker.text
                    }
                    continue  # 继续下一页内容请求
                else:
                    return result
        except aiohttp.ClientError as ae:
            if not blind:
                print('请求失败，继续重试', ae)
        except Exception as e:
            if not blind:
                print('请求失败，继续重试', e)


async def get_symbols(params):
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
        result = await get_symbols_by_session(session, params)
        return result


async def get_symbols_by_session(session, params):
    data = await request_session(session, params)
    root = objectify.fromstring(data.encode('ascii'))
    result = []
    for item in root.CommonPrefixes:
        param = item.Prefix
        s = param.text.split('/')
        result.append(s[len(s) - 2])
    if root.IsTruncated:
        # 下一页的网址
        params['marker'] = root.NextMarker.text
        next = await get_symbols_by_session(session, params)
        result.extend(next)  # 初次循环时，link_lst 包含1000条以上的数据
    return result


def async_get_all_symbols(params):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(get_symbols(params))


def async_get_usdt_symbols(params):
    all_symbols = async_get_all_symbols(params)
    usdt = set()
    [usdt.add(i) for i in all_symbols if i.endswith('USDT')]
    return usdt


def get_download_prefix(trading_type, market_data_type, time_period, symbol, interval):
    trading_type_path = 'data/spot'
    if trading_type == 'swap':
        trading_type_path = 'data/futures/um'
    return f'{trading_type_path}/{time_period}/{market_data_type}/{symbol.upper()}/{interval}/'


def async_get_daily_list(download_folder, symbols, trading_type, data_type, interval):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(download_daily_list(download_folder, symbols, trading_type, data_type, interval))


async def download_daily_list(download_folder, symbols, trading_type, data_type, interval):
    today = datetime.date.today()
    this_month_first_day = datetime.date(today.year, today.month, 1)
    daily_end = this_month_first_day - relativedelta(months=1)

    result = []
    param_list = []
    for symbol in symbols:
        daily_prefix = get_download_prefix(trading_type, data_type, 'daily', symbol, interval)
        checksum_file_name = "{}-{}-{}.zip.CHECKSUM".format(symbol.upper(), interval, daily_end - relativedelta(days=1))
        first_checksum_file_uri = '{}{}'.format(daily_prefix, checksum_file_name)
        param = {
            'delimiter': '/',
            'prefix': daily_prefix,
            'marker': first_checksum_file_uri
        }
        param_list.append(param)
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
        tasks = [asyncio.create_task(request_session(session, p)) for p in param_list]
        await asyncio.wait(tasks)

    for task in tasks:
        data = task.result()

        root = objectify.fromstring(data.encode('ascii'))
        if getattr(root, 'Contents', None) is None:
            continue
        symbol = root.Prefix.text.split('/')[-3]
        local_path = get_local_path(download_folder, trading_type, data_type, 'daily', symbol, interval)
        for item in root.Contents:
            key = item.Key.text
            if key.endswith('CHECKSUM'):
                struct_time = time.strptime(item.LastModified.text, '%Y-%m-%dT%H:%M:%S.%fZ')
                _tmp = {
                    'key': key,
                    'last_modified': time.mktime(struct_time),
                    'local_path': local_path,
                    'interval': interval
                }
                result.append(_tmp)
    return result


def get_local_path(root_path, trading_type, market_data_type, time_period, symbol, interval='5m'):
    trade_type_folder = trading_type + '_' + interval
    path = os.path.join(root_path, trade_type_folder, f'{time_period}_{market_data_type}')

    if symbol:
        path = os.path.join(path, symbol.upper())
    return path


def async_get_monthly_list(download_folder, symbols, trading_type, data_type, interval):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(
        build_download_monthly_list(download_folder, symbols, trading_type, data_type, interval))


async def build_download_monthly_list(download_folder, symbols, trading_type, data_type, interval):
    today = datetime.date.today()
    this_month_first_day = datetime.date(today.year, today.month, 1)
    daily_end = this_month_first_day - relativedelta(months=2)
    end_month = str(daily_end)[0:-3]

    param_list = []
    for symbol in symbols:
        monthly_prefix = get_download_prefix(trading_type, data_type, 'monthly', symbol, interval)
        param = {
            'delimiter': '/',
            'prefix': monthly_prefix
        }
        param_list.append(param)

    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
        tasks = [asyncio.create_task(request_session(session, p)) for p in param_list]
        await asyncio.wait(tasks)

    result = []
    for task in tasks:
        data = task.result()

        root = objectify.fromstring(data.encode('ascii'))
        if getattr(root, 'Contents', None) is None:
            continue
        symbol = root.Prefix.text.split('/')[-3]
        local_path = get_local_path(download_folder, trading_type, data_type, 'monthly', symbol, interval)
        for item in root.Contents:
            key = item.Key.text
            if key.endswith('CHECKSUM') and (key[-20:-13] <= end_month):
                struct_time = time.strptime(item.LastModified.text, '%Y-%m-%dT%H:%M:%S.%fZ')
                _tmp = {
                    'key': key,
                    'last_modified': time.mktime(struct_time),
                    'local_path': local_path
                }
                result.append(_tmp)
    return result


def async_get_metrics_list(download_folder, symbols, trading_type, data_type, interval):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(
        build_download_metrics_list(download_folder, symbols, trading_type, data_type, interval))


async def build_download_metrics_list(download_folder, symbols, trading_type, data_type, interval):
    param_list = []
    for symbol in symbols:
        symbol_prefix = f'{metrics_prefix}{symbol}/'
        param = {
            'delimiter': '/',
            'prefix': symbol_prefix
        }
        param_list.append(param)

    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
        tasks = [asyncio.create_task(request_session_4_list(session, p)) for p in param_list]
        await asyncio.wait(tasks)

    result = []
    for task in tasks:
        datas = task.result()
        for root in datas:
            if getattr(root, 'Contents', None) is None:
                continue
            symbol = root.Prefix.text.split('/')[-2]
            local_path = get_local_path(download_folder, trading_type, data_type, 'daily', symbol, interval)
            for item in root.Contents:
                key = item.Key.text
                if key.endswith('CHECKSUM'):
                    struct_time = time.strptime(item.LastModified.text, '%Y-%m-%dT%H:%M:%S.%fZ')
                    _tmp = {
                        'key': key,
                        'last_modified': time.mktime(struct_time),
                        'local_path': local_path
                    }
                    result.append(_tmp)
    return result


def async_download_file(all_list, error_info_list):
    pbar = tqdm.tqdm(total=len(all_list), ncols=50, mininterval=0.5)
    tasks = download_file(all_list, pbar, error_info_list)
    asyncio.get_event_loop().run_until_complete(asyncio.gather(*tasks))
    pbar.close()


def download_file(params, pbar, error_info_list):
    tasks = []
    for param in params:
        key = param['key']
        download_checksum_url = f'{BASE_URL}{key}'
        sum_file_name = os.path.basename(param['key'])

        if not os.path.exists(param['local_path']):
            os.makedirs(param['local_path'])
        local_path = param['local_path']
        last_modified = param['last_modified']
        local_sum_path = os.path.join(local_path, sum_file_name)
        local_zip_path = os.path.join(local_path, sum_file_name[0:-9])
        if os.path.exists(local_sum_path):
            '''
            这里对checksum文件的更新时间与币安数据中心的更新时间作比较
            '''
            modify_utc_timestamp = datetime.datetime.utcfromtimestamp(os.path.getmtime(local_sum_path)).timestamp()
            if modify_utc_timestamp < last_modified:
                os.remove(local_sum_path)
        if os.path.exists(local_sum_path) and os.path.exists(local_zip_path):
            if not thunder:
                # 本地已有文件，进行校验
                with open(local_sum_path, encoding='utf-8') as in_sum_file:
                    correct_sum = in_sum_file.readline().split(' ')[0]
                sha256Obj = sha256()
                with open(local_zip_path, 'rb') as in_zip_file:
                    sha256Obj.update(in_zip_file.read())
                if correct_sum == sha256Obj.hexdigest().lower():
                    # print(local_zip_path, 'existed and is correct')
                    pbar.update(1)
                    continue  # 继续下一个zip的下载过程
            else:
                # 快速更新模式不校验本地已有文件
                pbar.update(1)
                continue  # 继续下一个zip的下载过程
        if 'monthly' in local_path:
            # 需要数据完整性分析的目录
            need_analyse_set.add(local_path)
        if 'daily_klines' in local_path:
            daily_updated_set.add(local_path)
        tasks.append(
            download(param['local_path'], download_checksum_url, local_sum_path, local_zip_path, pbar, error_info_list))
    return tasks


async def download(local_path, download_checksum_url, local_sum_path, local_zip_path, pbar, error_info_list):
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
        async with semaphore:
            retry = 0
            while True:
                if retry > retry_times and 'daily_klines' in local_path:
                    print('下载daily zip失败次数超过retry_times，当前网络状况不稳定或数据包异常', local_zip_path)
                    break
                try:
                    sum_file = await session.get(download_checksum_url, proxy=file_proxy, timeout=20)
                    sum_file_buffer = await sum_file.read()
                    async with aiofiles.open(local_sum_path, 'wb') as out_sum_file:
                        await out_sum_file.write(sum_file_buffer)

                    zip_file = await session.get(download_checksum_url[0:-9], proxy=file_proxy, timeout=20)
                    zip_file_buffer = await zip_file.read()
                    async with aiofiles.open(local_zip_path, 'wb') as out_zip_file:
                        await out_zip_file.write(zip_file_buffer)

                    async with aiofiles.open(local_sum_path, encoding='utf-8') as in_sum_file:
                        str_sum = await in_sum_file.read()
                        correct_sum = str_sum.split(' ')[0]
                    sha256_obj = sha256()
                    async with aiofiles.open(local_zip_path, 'rb') as in_zip_file:
                        sha256_obj.update(await in_zip_file.read())
                    if correct_sum == sha256_obj.hexdigest().lower():
                        # print(local_zip_path, 'is correct')
                        pbar.update(1)
                        break
                except aiohttp.ClientError as ae:
                    if not blind:
                        print(f'\n\r下载{local_zip_path}失败，继续重试', ae)
                    error_info_list.add(f'下载{local_zip_path}失败，错误原因{ae}，已重试下载，请确认')
                except Exception as e:
                    error_info_list.add(f'下载{local_zip_path}失败，错误类型{type(e)}，已重试下载，请确认')
                retry += 1


def clean_old_daily_zip(local_daily_path, symbols, interval):
    today = datetime.date.today()
    this_month_first_day = datetime.date(today.year, today.month, 1)
    daily_end = this_month_first_day - relativedelta(months=1)

    for symbol in symbols:
        local_daily_symbol_path = os.path.join(local_daily_path, symbol)
        if os.path.exists(local_daily_symbol_path):
            zip_file_path = os.path.join(local_daily_symbol_path,
                                         "{}-{}-{}.zip".format(symbol.upper(), interval, daily_end))
            for item in glob(os.path.join(local_daily_symbol_path, '*')):
                if item < zip_file_path:
                    os.remove(item)
            if not os.listdir(local_daily_symbol_path):
                # 删除空文件夹，即已下架的币种
                os.rmdir(local_daily_symbol_path)


def transfer_daily_to_monthly_and_get_newest(newest_timestamp, daily_list, need_analyse_set):
    sorted_list = sorted(daily_list, key=lambda x: x['local_path'], reverse=False)

    tasks = []

    num = 0
    for local_path, items in groupby(sorted_list, key=itemgetter('local_path')):
        zip_files = []
        day_set = set()
        interval = ''
        symbol = os.path.basename(local_path)
        monthly_path = local_path.replace('daily_', 'monthly_')
        for i in items:
            interval = i['interval']
            zip_files.append(os.path.join(i['local_path'], os.path.basename(i['key'])[0:-9]))
            _day = datetime.datetime.strptime(i['key'][-23:-13], "%Y-%m-%d")
            day_set.add(_day.toordinal())

        days = pd.DataFrame(sorted(list(day_set)))
        if len(days.diff().value_counts()) > 1:
            # daily zip缺失某天或某几天的zip
            need_analyse_set.add(monthly_path)
        df_latest = pd.concat(Parallel(4)(
            delayed(pd.read_csv)(path_, header=None, encoding="utf-8", compression='zip') for path_ in zip_files),
            ignore_index=True)
        df_latest = df_latest[df_latest[0] != 'open_time']
        df_latest = df_latest.astype(dtype={0: int64})
        df_latest.sort_values(by=0)
        latest_monthly_zip = os.path.join(monthly_path, f'{symbol}-{interval}-latest.zip')
        if daily_err_occur or local_path in daily_updated_set or not os.path.exists(latest_monthly_zip) or max((os.path.getmtime(file) for file in zip_files)) > os.path.getmtime(latest_monthly_zip):
            if not os.path.exists(monthly_path):
                os.makedirs(monthly_path)
            compression_options = dict(method='zip', archive_name=f'{symbol}-{interval}-latest.csv')
            df_latest.to_csv(latest_monthly_zip, header=None, index=None, compression=compression_options)

        if update_to_now:
            tasks.append(get_klines_from_aio_api(symbol, interval, int(df_latest.iloc[-1, 0]), newest_timestamp,
                                                 os.path.join(monthly_path, f'{symbol}-{interval}-newest.zip')))
        num += 1
        print(f'\r合并完成数量: {num}', end='')
    print('\n合并结束')
    return tasks


def read_symbol_open_time(symbol, zip_path):
    '''
    只读取open_time 用来进行完整性分析
    '''
    zip_list = glob(os.path.join(zip_path, f'{symbol}*.zip'))
    _df = pd.concat(
        Parallel(CONCURRENCY)(delayed(pd.read_csv)(path_, header=None, encoding="utf-8", compression='zip', usecols=[0],
                                                   names=['open_time'], dtype=str, engine='c'
                                                   ) for path_ in zip_list), ignore_index=True)
    # 过滤表头行
    _df = _df[_df['open_time'] != 'open_time']
    # 规范数据类型，并将时间戳转化为可读时间
    _df = _df.astype(dtype={'open_time': int64})
    _df['candle_begin_time'] = pd.to_datetime(_df['open_time'], unit='ms')
    _df.sort_values(by='open_time', inplace=True)  # 排序
    _df.drop_duplicates(subset=['open_time'], inplace=True, keep='last')  # 去除重复值
    _df.reset_index(drop=True, inplace=True)  # 重置index
    return _df


def analyse_download_data(download_folder, symbols, trading_type, data_type, intervals):
    err_symbols = dict()
    download_err_info = set()
    if len(need_analyse_set) > 0:
        print('需要分析数据完整性的目录：', need_analyse_set)
    print('开始分析数据完整性...')
    pbar = tqdm.tqdm(total=len(symbols) * len(intervals), ncols=50, mininterval=0.5)
    tasks = []
    for symbol in symbols:
        for interval in intervals:
            tasks.extend(
                build_analyse_download_task(download_folder, symbol, trading_type, data_type, interval, err_symbols,
                                            download_err_info))
            pbar.update(1)
    pbar.close()
    asyncio.get_event_loop().run_until_complete(asyncio.gather(*tasks))

    if len(err_symbols) > 0:
        print('数据缺失的交易对数量为', len(err_symbols))
        print(err_symbols)


interval_microsecond = {
    '1m': 60000,
    '5m': 300000
}

interval_param = {
    '1m': relativedelta(minutes=1),
    '5m': relativedelta(minutes=5)
}


async def get_klines_from_aio_api(symbol, interval, start_time, end_time, zip_path, overwrite=False):
    df = pd.DataFrame()
    if not overwrite and os.path.exists(zip_path):
        try:
            df = pd.read_csv(zip_path, header=None, encoding="utf-8", compression='zip')
            df.sort_values(by=0)
            df = df[df.iloc[:, 0] > start_time]
            if df.shape[0] > 0:
                latest_time = int(df.iloc[-1, 0])
                if latest_time + interval_microsecond[interval] == end_time + 1:
                    # 无需更新
                    return
                if latest_time > start_time:
                    start_time = latest_time
        except Exception as e:
            print(zip_path, '读取失败，请根据报错酌情处理，若文件损坏可删除该文件重新运行')
            raise e

    if trade_type == 'spot':
        spot_kline_url = 'https://api.binance.com/api/v3/klines'
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
            async with api_semaphore:
                while start_time < end_time:
                    end = int(start_time) + 999 * interval_microsecond[interval]
                    if end > end_time:
                        end = end_time
                    param = {
                        'symbol': symbol,
                        'interval': interval,
                        'startTime': int(start_time),
                        'endTime': end,
                        'limit': 1000
                    }
                    while True:
                        try:
                            async with session.get(url=spot_kline_url, params=param, proxy=proxy,
                                                   timeout=5) as kline_response:
                                kline = await kline_response.text()
                                _df = pd.DataFrame(json.loads(kline))
                                df = pd.concat([df, _df], ignore_index=True)
                                break
                        except Exception as e:
                            if not blind:
                                print('spot klines请求失败，继续重试', e)
                            continue
                    start_time = end
                if df.shape[0] > 0:
                    compression_options = dict(method='zip', archive_name=f'{symbol}-{interval}-newest.csv')
                    df.to_csv(zip_path, header=None, index=None, compression=compression_options)
    elif trade_type == 'swap':
        swap_kline_url = 'https://fapi.binance.com/fapi/v1/klines'
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
            async with api_semaphore:
                while start_time < end_time:
                    end = int(start_time) + 499 * interval_microsecond[interval]
                    if end > end_time:
                        end = end_time
                    param = {
                        'symbol': symbol,
                        'interval': interval,
                        'startTime': int(start_time),
                        'endTime': end,
                        'limit': 499
                    }
                    while True:
                        response_str = ''
                        try:
                            async with session.get(url=swap_kline_url, params=param, proxy=proxy,
                                                   timeout=5) as kline_response:
                                response_str = kline = await kline_response.text()
                                _df = pd.DataFrame(json.loads(kline))
                                df = pd.concat([df, _df], ignore_index=True)
                                break
                        except Exception as e:
                            if not blind:
                                print(f'swap klines请求失败，返回结果：{response_str}，继续重试', e)
                            continue
                    start_time = end
                if df.shape[0] > 0:
                    compression_options = dict(method='zip', archive_name=f'{symbol}-{interval}-newest.csv')
                    df.to_csv(zip_path, header=None, index=None, compression=compression_options)


def build_analyse_download_task(download_folder, symbol, trading_type, data_type, interval, err_symbols,
                                download_err_info):
    zip_path = get_local_path(download_folder, trading_type, data_type, 'monthly', symbol, interval)
    tasks = []
    if not force_analyse and zip_path not in need_analyse_set:
        # 之前已经分析过数据完整性，本次跳过
        return tasks
    df = read_symbol_open_time(symbol, zip_path)
    df['open_time_diff_1'] = df['open_time'].diff()
    df['open_time_diff_-1'] = df['open_time'].diff(-1)
    df = df[(df['open_time_diff_1'] > interval_microsecond[interval]) | (
        df['open_time_diff_-1'] < -interval_microsecond[interval])]

    if df.size != 0:
        miss_day = []
        msg = []
        df.reset_index(drop=True, inplace=True)
        for row in df.index:
            if row % 2 != 0:
                continue
            start = df.loc[row]['candle_begin_time']
            end = df.loc[row + 1]['candle_begin_time']
            if trading_type == 'swap' and str(start) == '2023-08-16 09:03:00' and str(end) == '2023-08-16 09:06:00':
                # bn合约市场都缺了这几分钟
                continue
            if symbol in SETTLED_SYMBOLS:
                if (str(start + interval_param[interval]) >= SETTLED_SYMBOLS[symbol][0]) and (
                    str(end) <= SETTLED_SYMBOLS[symbol][1]
                ):
                    # 无交易期间的K线不用补全
                    continue
                if SETTLED_SYMBOLS[symbol][0] < str(end) <= SETTLED_SYMBOLS[symbol][1]:
                    end = datetime.datetime.strptime(SETTLED_SYMBOLS[symbol][0][0:10], '%Y-%m-%d') + relativedelta(
                        days=1)
                if SETTLED_SYMBOLS[symbol][0] < str(start) <= SETTLED_SYMBOLS[symbol][1]:
                    start = datetime.datetime.strptime(SETTLED_SYMBOLS[symbol][1][0:10], '%Y-%m-%d') + relativedelta(
                        days=1) - interval_param[interval]
            msg.append(f'\t{start + interval_param[interval]} to {end - interval_param[interval]}')
            if (str(start + interval_param[interval])[-8:] != '00:00:00') or (
                str(end - relativedelta(minutes=1))[-8:] != '23:59:00'
            ):
                print(symbol, f'日内数据不完整，缺失：{start} - {end}')
                if trading_type == 'swap':
                    raise Exception(f'{symbol}日内数据不完整')
                else:
                    # spot 遇到日内缺失的不做处理
                    continue

            while start + interval_param[interval] < end:
                start += relativedelta(days=1)
                miss_day.append(str(start)[0: -9])
        if len(miss_day) != 0:
            print('\n\r', symbol, interval, 'need candles:')
            [print(m) for m in msg]
            print('\tskip days', miss_day)
            err_symbols[symbol] = miss_day
            print('start to download skip zip')

            for day in miss_day:
                local_path = get_local_path(download_folder, trading_type, data_type, 'monthly', symbol, interval)
                sum_name = f'{symbol}-{interval}-{day}.zip.CHECKSUM'
                download_prefix = get_download_prefix(trading_type, data_type, 'daily', symbol, interval)
                tasks.append(download_miss_day_data(symbol, interval, day, local_path, sum_name, download_prefix,
                                                    download_err_info))
    return tasks


async def download_miss_day_data(symbol, interval, day, local_path, sum_name, prefix, download_err_info):
    local_sum_path = os.path.join(local_path, sum_name)
    sum_url = BASE_URL + prefix + sum_name
    local_zip_path = local_sum_path[0:-9]
    err_times = 0
    retry_sum_404 = 0
    retry_zip_404 = 0
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
        async with semaphore:
            while True:
                try:
                    sum_file = await session.get(sum_url, proxy=file_proxy, timeout=20)
                    if sum_file.status == 200:
                        sum_file_buffer = await sum_file.read()
                        async with aiofiles.open(local_sum_path, 'wb') as out_sum_file:
                            await out_sum_file.write(sum_file_buffer)
                    else:
                        retry_sum_404 += 1
                        if retry_sum_404 > 3:
                            # 重试3次确认不存在这个文件
                            break
                        else:
                            continue
                    zip_file = await session.get(sum_url[0:-9], proxy=file_proxy, timeout=20)
                    if zip_file.status == 200:
                        zip_file_buffer = await zip_file.read()
                        async with aiofiles.open(local_zip_path, 'wb') as out_zip_file:
                            await out_zip_file.write(zip_file_buffer)
                    else:
                        retry_zip_404 += 1
                        if retry_zip_404 > 3:
                            # 重试3次确认不存在这个文件
                            break
                        else:
                            continue

                    async with aiofiles.open(local_sum_path, encoding='utf-8') as in_sum_file:
                        str_sum = await in_sum_file.readline()
                        correct_sum = str_sum.split(' ')[0]
                    sha256_obj = sha256()
                    async with aiofiles.open(local_zip_path, 'rb') as in_zip_file:
                        sha256_obj.update(await in_zip_file.read())
                    if correct_sum == sha256_obj.hexdigest().lower():
                        # print(local_zip_path, 'is correct')
                        break
                except aiohttp.ClientError as ae:
                    if not blind:
                        print(f'下载{local_zip_path}失败，继续重试', ae)
                    download_err_info.add(f'下载{local_zip_path}失败，错误原因{ae}，已重试下载，请确认')
                    err_times += 1
                    if err_times > 5:
                        print('补漏下载重试超过5次')
                        raise ae
                except Exception as e:
                    print(f'下载{local_zip_path}失败，错误类型{type(e)}，已重试下载，请确认')
                    download_err_info.add(f'下载{local_zip_path}失败，错误类型{type(e)}，已重试下载，请确认')
                    err_times += 1
                    if err_times > 5:
                        print('补漏下载重试超过5次')
                        raise e
    if retry_sum_404 > 3 or retry_zip_404 > 3:
        print(f'{sum_name} not exist, request from api')
        format_day = datetime.datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc)
        start = int(format_day.timestamp()) * 1000
        end = int((format_day + datetime.timedelta(days=1)).timestamp()) * 1000 - 1
        await get_klines_from_aio_api(symbol, interval, start, end, local_zip_path)


def read_symbol_csv(symbol, zip_path):
    zip_list = glob(os.path.join(zip_path, symbol, f'{symbol}*.zip'))
    # 合并monthly daily 数据
    df = pd.concat(
        Parallel(1)(delayed(pd.read_csv)(path_, header=None, encoding="utf-8", compression='zip',
                                         names=['open_time', 'open', 'high', 'low', 'close', 'volume',
                                                'close_time', 'quote_volume', 'trade_num',
                                                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
                                                'ignore']
                                         ) for path_ in zip_list), ignore_index=True)
    # 过滤表头行
    df = df[df['open_time'] != 'open_time']
    # 规范数据类型，防止计算avg_price报错
    df = df.astype(
        dtype={'open_time': int64, 'open': float, 'high': float, 'low': float, 'close': float, 'volume': float,
               'quote_volume': float,
               'trade_num': int, 'taker_buy_base_asset_volume': float, 'taker_buy_quote_asset_volume': float})
    df['avg_price'] = df['quote_volume'] / df['volume']  # 增加 均价
    # df['candle_begin_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df.drop(columns=['close_time', 'ignore'], inplace=True)
    df.sort_values(by='open_time', inplace=True)  # 排序
    df.drop_duplicates(subset=['open_time'], inplace=True, keep='last')  # 去除重复值
    df.reset_index(drop=True, inplace=True)  # 重置index
    # df.set_index('candle_begin_time', inplace=True)
    return df


def build_hour_offset_pkl(symbol, trading_type, offset, df_big, df_1m):
    '''
    生成分钟偏移的pkl数据，offset=5生成的1h kline数据是从1:05~2:05，但是生成的pkl中的candle_begin_time是1:00，这样回测框架对于偏移是无感知的
    本函数偏移方法是将open_time减去offset分钟，然后生成1h kline数据
    :param symbol: 交易对
    :param trading_type: 交易类型
    :param offset: 小时偏移量
    :param df_big: 5分钟级别的原始数据
    :param df_1m: 1分钟级别的原始数据
    :return:
    '''
    df_big['open_time'] = df_big['open_time'] - offset * 60 * 1000
    df_1m['open_time'] = df_1m['open_time'] - offset * 60 * 1000

    # =将数据转换为1小时周期
    df_big['candle_begin_time'] = pd.to_datetime(df_big['open_time'], unit='ms')
    df_big.set_index('candle_begin_time', inplace=True)
    del df_big['open_time']
    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'quote_volume': 'sum',
        'trade_num': 'sum',
        'taker_buy_base_asset_volume': 'sum',
        'taker_buy_quote_asset_volume': 'sum',
        'avg_price': 'first'
    }
    df = df_big.resample(rule='1H').agg(agg_dict)

    # =针对1小时数据，补全空缺的数据。保证整张表没有空余数据
    # 对开、高、收、低、价格进行补全处理
    df['close'].ffill(inplace=True)
    df['open'].fillna(value=df['close'], inplace=True)
    df['high'].fillna(value=df['close'], inplace=True)
    df['low'].fillna(value=df['close'], inplace=True)
    # 将停盘时间的某些列，数据填补为0
    fill_0_list = ['volume', 'quote_volume', 'trade_num', 'taker_buy_base_asset_volume',
                   'taker_buy_quote_asset_volume']
    df.loc[:, fill_0_list] = df[fill_0_list].fillna(value=0)

    df_1m['candle_begin_time'] = pd.to_datetime(df_1m['open_time'], unit='ms')
    df_1m.set_index('candle_begin_time', inplace=True)
    df['avg_price_1m'] = df_1m['avg_price']
    df['avg_price_5m'] = df['avg_price']

    # =计算最终的均价
    # 默认使用1分钟均价
    df['avg_price'] = df['avg_price_1m']
    # 没有1分钟均价就使用5分钟均价
    df['avg_price'].fillna(value=df['avg_price_5m'], inplace=True)
    # 没有5分钟均价就使用开盘价
    df['avg_price'].fillna(value=df['open'], inplace=True)
    del df['avg_price_5m'], df['avg_price_1m']
    df['symbol'] = symbol.upper()
    symbol = symbol.upper().replace('USDT', '-USDT')

    if trading_type == 'swap':
        # 读取fundingRate
        funding_df = pd.read_feather(os.path.join(funding_path, f'{symbol}.pkl'))
        funding_df = funding_df.astype(dtype={'fundingRate': float})
        funding_df['candle_begin_time'] = pd.to_datetime(funding_df['fundingTime'], unit='ms')
        funding_df.sort_values(by='candle_begin_time', inplace=True)  # 排序
        funding_df.drop_duplicates(subset=['candle_begin_time'], inplace=True, keep='last')  # 去除重复值
        # warning: 由于有了偏移量，这里的fundingRate的时间戳是错位的，需要提前1小时
        funding_df['candle_begin_time'] = pd.to_datetime(funding_df['fundingTime'], unit='ms') - datetime.timedelta(
            hours=1)
        funding_df.reset_index(drop=True, inplace=True)  # 重置index
        funding_df.set_index('candle_begin_time', inplace=True)
        # 合并fundingRate
        df['fundingRate'] = funding_df['fundingRate']
        df['fundingRate'].ffill(inplace=True)
        if False:  # 不再需要对funding_rate_r预处理，保存原始数据即可
            funding_df.reset_index(drop=True, inplace=True)  # 重置index
            funding_df['candle_begin_time'] = pd.to_datetime(funding_df['fundingTime'], unit='ms') - datetime.timedelta(
                hours=1)
            funding_df.set_index('candle_begin_time', inplace=True)
            df['funding_rate_r'] = funding_df['fundingRate']
            df['funding_rate_r'].fillna(value=0, inplace=True)
        df['funding_rate_raw'] = funding_df['fundingRate']

    df.reset_index(inplace=True)
    if not os.path.exists(os.path.join(f'{pickle_path}_{offset}', trade_type)):
        os.makedirs(os.path.join(f'{pickle_path}_{offset}', trade_type))

    original_symbol = symbol.replace('-USDT', 'USDT')
    if trading_type == 'swap' and original_symbol in SETTLED_SWAP_SYMBOLS:
        df_old = df[df['candle_begin_time'] < SETTLED_SWAP_SYMBOLS[original_symbol][0]].copy()
        old_symbol = symbol.replace('-USDT', '1-USDT')
        df_old.loc[:, 'symbol'] = old_symbol.replace('-', '')
        df_old.to_feather(os.path.join(f'{pickle_path}_{offset}', trade_type, f'{old_symbol}.pkl'))
        df_new = df[df['candle_begin_time'] > SETTLED_SWAP_SYMBOLS[original_symbol][1]].copy()
        df_new.reset_index(inplace=True)
        df_new.to_feather(os.path.join(f'{pickle_path}_{offset}', trade_type, f'{symbol}.pkl'))
        return
    df.to_feather(os.path.join(f'{pickle_path}_{offset}', trade_type, f'{symbol}.pkl'))


def data_center_symbol_process(symbol, trading_type, zip_path_1m, zip_path_5m):
    pkl_path = os.path.join(pickle_path, f'{trading_type}', f'{symbol.upper().replace("USDT", "-USDT")}.pkl')
    if not os.path.exists(os.path.join(zip_path_5m, symbol, f'{symbol}-5m-latest.zip')) and os.path.exists(pkl_path):
        # 下架超过2个月的币种，并且之前已经生成过pkl的币种，选择跳过，减少重复工作量
        print('skip pkl', pkl_path)
        df = pd.read_feather(pkl_path, columns=['candle_begin_time', 'symbol', 'avg_price'])
        df['pct_chg'] = df['avg_price'].pct_change(periods=int(hold_hour[:-1]))
        df = df.dropna(subset=['pct_chg'])
        return df[['candle_begin_time', 'symbol', 'pct_chg']]
    df_big = read_symbol_csv(symbol, zip_path_5m)
    # 读取1分钟数据
    df_1m = read_symbol_csv(symbol, zip_path_1m)

    # 生成小时偏移的pkl数据
    [build_hour_offset_pkl(symbol, trading_type, offset, df_big.copy(), df_1m.copy()) for offset in hour_offsets]

    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'quote_volume': 'sum',
        'trade_num': 'sum',
        'taker_buy_base_asset_volume': 'sum',
        'taker_buy_quote_asset_volume': 'sum',
        'avg_price': 'first'
    }
    # =将数据转换为1小时周期
    df_big['candle_begin_time'] = pd.to_datetime(df_big['open_time'], unit='ms')
    df_big.set_index('candle_begin_time', inplace=True)
    del df_big['open_time']
    df = df_big.resample(rule='1H').agg(agg_dict)

    # =针对1小时数据，补全空缺的数据。保证整张表没有空余数据
    # 对开、高、收、低、价格进行补全处理
    df['close'].ffill(inplace=True)
    df['open'].fillna(value=df['close'], inplace=True)
    df['high'].fillna(value=df['close'], inplace=True)
    df['low'].fillna(value=df['close'], inplace=True)
    # 将停盘时间的某些列，数据填补为0
    fill_0_list = ['volume', 'quote_volume', 'trade_num', 'taker_buy_base_asset_volume',
                   'taker_buy_quote_asset_volume']
    df.loc[:, fill_0_list] = df[fill_0_list].fillna(value=0)

    df_1m['candle_begin_time'] = pd.to_datetime(df_1m['open_time'], unit='ms')
    df_1m.set_index('candle_begin_time', inplace=True)
    df['avg_price_1m'] = df_1m['avg_price']
    df['avg_price_5m'] = df['avg_price']

    # =计算最终的均价
    # 默认使用1分钟均价
    df['avg_price'] = df['avg_price_1m']
    # 没有1分钟均价就使用5分钟均价
    df['avg_price'].fillna(value=df['avg_price_5m'], inplace=True)
    # 没有5分钟均价就使用开盘价
    df['avg_price'].fillna(value=df['open'], inplace=True)
    del df['avg_price_5m'], df['avg_price_1m']
    df['symbol'] = symbol.upper()
    symbol = symbol.upper().replace('USDT', '-USDT')

    if trading_type == 'swap':
        # 读取fundingRate
        funding_df = pd.read_feather(os.path.join(funding_path, f'{symbol}.pkl'))
        funding_df = funding_df.astype(dtype={'fundingRate': float})
        funding_df['candle_begin_time'] = pd.to_datetime(funding_df['fundingTime'], unit='ms')
        funding_df.sort_values(by='candle_begin_time', inplace=True)  # 排序
        funding_df.drop_duplicates(subset=['candle_begin_time'], inplace=True, keep='last')  # 去除重复值
        funding_df.reset_index(drop=True, inplace=True)  # 重置index
        funding_df.set_index('candle_begin_time', inplace=True)
        # 合并fundingRate
        df['fundingRate'] = funding_df['fundingRate']
        df['fundingRate'].ffill(inplace=True)
        if False:  # 不再需要对funding_rate_r预处理，保存原始数据即可
            funding_df.reset_index(drop=True, inplace=True)  # 重置index
            funding_df['candle_begin_time'] = pd.to_datetime(funding_df['fundingTime'], unit='ms') - datetime.timedelta(
                hours=1)
            funding_df.set_index('candle_begin_time', inplace=True)
            df['funding_rate_r'] = funding_df['fundingRate']
            df['funding_rate_r'].fillna(value=0, inplace=True)
        df['funding_rate_raw'] = funding_df['fundingRate']

        # 读取 合约持仓量
        ''' # 去掉持仓量数据
        openInterestHist = os.path.join(openInterestHist_path, f'{symbol}.pkl')
        if os.path.exists(openInterestHist):
            openInterestHist_df = pd.read_feather(openInterestHist)
            openInterestHist_df = openInterestHist_df.astype(dtype={'sumOpenInterest': float, 'sumOpenInterestValue': float})
            openInterestHist_df['candle_begin_time'] = pd.to_datetime(openInterestHist_df['timestamp'], unit='ms')
            openInterestHist_df.sort_values(by='candle_begin_time', inplace=True)  # 排序
            openInterestHist_df.drop_duplicates(subset=['candle_begin_time'], inplace=True, keep='last')  # 去除重复值
            openInterestHist_df.reset_index(drop=True, inplace=True)  # 重置index
            openInterestHist_df.set_index('candle_begin_time', inplace=True)
            # 合并 合约持仓量
            df['sumOpenInterest'] = openInterestHist_df['sumOpenInterest'] # 持仓总数量
            df['sumOpenInterestValue'] = openInterestHist_df['sumOpenInterestValue'] # 持仓总价值
        '''

        # 读取 合约主动买卖量
        ''' # 去掉主买主卖数据
        takerlongshortRatio = os.path.join(takerlongshortRatio_path, f'{symbol}.pkl')
        if os.path.exists(takerlongshortRatio):
            takerlongshortRatio_df = pd.read_feather(takerlongshortRatio)
            takerlongshortRatio_df = takerlongshortRatio_df.astype(dtype={'buyVol': float, 'sellVol': float, 'buySellRatio': float})
            takerlongshortRatio_df['candle_begin_time'] = pd.to_datetime(takerlongshortRatio_df['timestamp'], unit='ms')
            takerlongshortRatio_df.sort_values(by='candle_begin_time', inplace=True)  # 排序
            takerlongshortRatio_df.drop_duplicates(subset=['candle_begin_time'], inplace=True, keep='last')  # 去除重复值
            takerlongshortRatio_df.reset_index(drop=True, inplace=True)  # 重置index
            takerlongshortRatio_df.set_index('candle_begin_time', inplace=True)
            # 合并 合约持仓量
            df['buyVol'] = takerlongshortRatio_df['buyVol'] # 主动买入量
            df['sellVol'] = takerlongshortRatio_df['sellVol'] # 主动卖出量
            df['buySellRatio'] = takerlongshortRatio_df['buySellRatio'] # 主买主卖比值
        '''

        # 读取 metrics

    df.reset_index(inplace=True)
    if not os.path.exists(os.path.join(pickle_path, f'{trading_type}')):
        os.makedirs(os.path.join(pickle_path, f'{trading_type}'))

    original_symbol = symbol.replace('-USDT', 'USDT')
    if trading_type == 'swap' and original_symbol in SETTLED_SWAP_SYMBOLS:
        df_old = df[df['candle_begin_time'] < SETTLED_SWAP_SYMBOLS[original_symbol][0]].copy()
        old_symbol = symbol.replace('-USDT', '1-USDT')
        df_old.loc[:, 'symbol'] = old_symbol.replace('-', '')
        df_old.to_feather(os.path.join(pickle_path, f'{trading_type}', f'{old_symbol}.pkl'))
        df_new = df[df['candle_begin_time'] > SETTLED_SWAP_SYMBOLS[original_symbol][1]].copy()
        df_new.reset_index(inplace=True)
        df_new.to_feather(os.path.join(pickle_path, f'{trading_type}', f'{symbol}.pkl'))
        print('pkl process success', symbol)
        df_old['pct_chg'] = df_old['avg_price'].pct_change(periods=int(hold_hour[:-1]))
        df_old = df_old.dropna(subset=['pct_chg'])
        df_new.loc[:, 'pct_chg'] = df_new['avg_price'].pct_change(periods=int(hold_hour[:-1]))
        df_new = df_new.dropna(subset=['pct_chg'])
        return pd.concat(
            [df_old[['candle_begin_time', 'symbol', 'pct_chg']], df_new[['candle_begin_time', 'symbol', 'pct_chg']]],
            ignore_index=True)
    df.to_feather(pkl_path)
    print('pkl process success', symbol)
    df['pct_chg'] = df['avg_price'].pct_change(periods=int(hold_hour[:-1]))
    df = df.dropna(subset=['pct_chg'])
    return df[['candle_begin_time', 'symbol', 'pct_chg']]


def data_center_data_to_pickle_data(trading_type, _njobs, metrics_symbols):
    monthly_zip_path_1m = get_local_path(root_path, trading_type, 'klines', 'monthly', None, '1m')
    daily_zip_path_1m = get_local_path(root_path, trading_type, 'klines', 'daily', None, '1m')
    monthly_zip_path_5m = get_local_path(root_path, trading_type, 'klines', 'monthly', None, '5m')
    monthly_symbols = os.listdir(monthly_zip_path_1m)
    # 剔除monthly_symbols中的.DS_Store
    exclusion = ['.DS_Store']
    monthly_symbols = [symbol for symbol in monthly_symbols if symbol not in exclusion]
    daily_symbol_set = set(os.listdir(daily_zip_path_1m))

    now = datetime.datetime.now()
    start = now - relativedelta(days=25)
    start_timestamp = int(time.mktime(start.timetuple()) * 1000) + 60000
    newest_timestamp = int(time.mktime(now.timetuple()) * 1000) - 1
    if trading_type == 'swap':
        tasks = []
        print('start download other data')
        start_time = datetime.datetime.strptime('2017-09-17 00:00:00', "%Y-%m-%d %H:%M:%S")
        oldest_timestamp = int(time.mktime(start_time.timetuple())) * 1000
        for symbol in monthly_symbols:
            if symbol in daily_symbol_set:
                tasks.append(get_fundingRate_from_aio_api(symbol, oldest_timestamp, newest_timestamp))
                # tasks.append(get_openInterestHist_from_aio_api(symbol, start_timestamp, newest_timestamp))
                # tasks.append(get_takerlongshortRatio_from_aio_api(symbol, start_timestamp, newest_timestamp))
            else:
                tasks.append(get_fundingRate_from_aio_api(symbol, oldest_timestamp, newest_timestamp, True))
                # tasks.append(get_openInterestHist_from_aio_api(symbol, start_timestamp, newest_timestamp, True))
                # tasks.append(get_takerlongshortRatio_from_aio_api(symbol, start_timestamp, newest_timestamp, True))
        if len(tasks) > 0:
            asyncio.get_event_loop().run_until_complete(asyncio.gather(*tasks))

    print('进程池大小', _njobs)

    results = []
    # 创建进程池，最多维护_njobs个线程
    threadpool = ProcessPoolExecutor(_njobs)
    for symbol in monthly_symbols:
        # 串行运行
        # data_center_symbol_process(symbol, trading_type, monthly_zip_path_1m, monthly_zip_path_5m)
        # 并发执行
        future = threadpool.submit(data_center_symbol_process, symbol, trading_type, monthly_zip_path_1m,
                                   monthly_zip_path_5m)
        results.append(future)

    dfa = pd.DataFrame()
    for job in as_completed(results):
        dfa = pd.concat([dfa, job.result()], ignore_index=True)
    threadpool.shutdown(True)

    if trading_type == 'spot':
        # 下载现货数据不展示横截面差异指数
        return
    dfa['pct_rank'] = dfa.groupby('candle_begin_time')['pct_chg'].rank(pct=True, ascending=True)
    dfa = dfa.sort_values('candle_begin_time').reset_index(drop=True)

    df_top_5 = dfa[dfa['pct_rank'] >= 0.97].groupby('candle_begin_time')['pct_chg'].mean()
    df_bot_5 = dfa[dfa['pct_rank'] <= 0.03].groupby('candle_begin_time')['pct_chg'].mean()
    df_top_10 = dfa[dfa['pct_rank'] >= 0.92].groupby('candle_begin_time')['pct_chg'].mean()
    df_bot_10 = dfa[dfa['pct_rank'] <= 0.08].groupby('candle_begin_time')['pct_chg'].mean()

    df_diff: pd.Series = (df_top_5 + df_top_10) / 2 - (df_bot_5 + df_bot_10) / 2
    # df_diff: pd.Series = df_top_5 - df_bot_5
    df = pd.DataFrame()
    df['candle_begin_time'] = df_diff.index
    df['cross_diff'] = df_diff.values
    # 头部会有空值要drop，不要填0
    df = df.dropna(subset=['cross_diff'])
    df['cross_diff'] = df['cross_diff'].ewm(span=rolling_period).mean()
    # 去掉头部几行
    df = df[24:].reset_index(drop=True)
    df.to_feather(f'{trading_type}横截面差异指数.pkl')
    df.set_index('candle_begin_time', inplace=True)
    df.plot(figsize=(16, 9), grid=True)
    plt.show()


async def get_fundingRate_from_aio_api(symbol, start_time, end_time, delist=False):
    df = pd.DataFrame()
    funding_pkl_name = symbol.upper().replace('USDT', '-USDT')
    funding = os.path.join(funding_path, f'{funding_pkl_name}.pkl')
    if os.path.exists(funding):
        if delist:
            # 已下架币种若已存在fundingRate文件，直接返回
            return

        try:
            df = pd.read_feather(funding)
            df.sort_values(by='fundingTime')
            if df.shape[0] > 0:
                latest_time = int(df.iloc[-1, 0]) + 1
                if latest_time > start_time:
                    start_time = latest_time
        except Exception as e:
            print(funding, 'error')
            raise e

    funding_url = 'https://fapi.binance.com/fapi/v1/fundingRate'
    updated = False
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
        while True:
            async with api_semaphore:
                await asyncio.sleep(1)
                param = {
                    'symbol': symbol,
                    'startTime': start_time,
                    'endTime': end_time,
                    'limit': 1000
                }
                try:
                    async with session.get(url=funding_url, params=param, proxy=proxy, timeout=5) as kline_response:
                        f_data = await kline_response.text()
                        if kline_response.status == 200:
                            _df = pd.DataFrame(json.loads(f_data), columns=['fundingTime', 'fundingRate'], index=None)
                            if _df.shape[0] == 0:
                                break
                            _df.sort_values(by='fundingTime', inplace=True)
                            start_time = int(_df.iloc[-1, 0]) + 1
                            _df['fundingTime'] = (_df['fundingTime'] // 1000) * 1000
                            df = pd.concat([df, _df], ignore_index=True)
                            updated = True

                            if _df.shape[0] < 1000:
                                break
                        else:
                            print(f'fundingRate {symbol} error response {f_data}')
                except Exception as e:
                    if not blind:
                        print('swap fundingRate请求失败，小问题不要慌，马上重试', e)
                    continue
        if updated:
            df.to_feather(funding)


async def get_openInterestHist_from_aio_api(symbol, start_time, end_time, delist=False):
    if symbol in swap_delist_symbol_set:
        return
    df = pd.DataFrame()
    openInterestHist_pkl_name = symbol.upper().replace('USDT', '-USDT')
    openInterestHist = os.path.join(openInterestHist_path, f'{openInterestHist_pkl_name}.pkl')
    exists = os.path.exists(openInterestHist)
    if exists:
        if delist:
            # 已下架币种若已存在合约持仓量文件，直接返回
            return

        try:
            df = pd.read_feather(openInterestHist)
            if df.shape[0] > 0:
                df.sort_values(by='timestamp')
                latest_time = int(df.iloc[-1, 0]) + 1
                if latest_time > start_time:
                    start_time = latest_time
        except Exception as e:
            print(openInterestHist, 'error')
            raise e

    openInterestHist_url = 'https://fapi.binance.com/futures/data/openInterestHist'
    updated = False
    limit = 500
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
        while True:
            async with api_semaphore:
                param = {
                    'symbol': symbol,
                    'period': '1h',
                    'startTime': str(start_time),
                    'endTime': str(min(start_time + limit * 60 * 60 * 1000, end_time)),
                    'limit': limit
                }
                try:
                    async with session.get(url=openInterestHist_url, params=param, proxy=proxy,
                                           timeout=5) as kline_response:
                        f_data = await kline_response.text()
                        if kline_response.status == 200:
                            _df = pd.DataFrame(json.loads(f_data),
                                               columns=['timestamp', 'sumOpenInterest', 'sumOpenInterestValue'],
                                               index=None)
                            if _df.shape[0] == 0:
                                if not exists:
                                    param = {
                                        'symbol': symbol,
                                        'period': '1h',
                                        'limit': limit
                                    }
                                    async with session.get(url=openInterestHist_url, params=param, proxy=proxy,
                                                           timeout=5) as new_response:
                                        f_data = await new_response.text()
                                        if new_response.status == 200:
                                            _df = pd.DataFrame(json.loads(f_data),
                                                               columns=['timestamp', 'sumOpenInterest',
                                                                        'sumOpenInterestValue'], index=None)
                                            if _df.shape[0] == 0:
                                                break
                                        else:
                                            print(f'openInterestHist {symbol} error response {f_data}')
                                            break
                                else:
                                    break
                            _df.sort_values(by='timestamp', inplace=True)
                            start_time = int(_df.iloc[-1, 0]) + 1
                            df = pd.concat([df, _df], ignore_index=True)
                            updated = True

                            if _df.shape[0] < 500:
                                break
                        else:
                            print(f'openInterestHist {symbol} error response {f_data}')
                            break
                except Exception as e:
                    if not blind:
                        print('swap openInterestHist 请求失败，小问题不要慌，马上重试', e)
                    continue
        if updated:
            df.to_feather(openInterestHist)


async def get_takerlongshortRatio_from_aio_api(symbol, start_time, end_time, delist=False):
    if symbol in swap_delist_symbol_set:
        return
    df = pd.DataFrame()
    takerlongshortRatio_pkl_name = symbol.upper().replace('USDT', '-USDT')
    takerlongshortRatio = os.path.join(takerlongshortRatio_path, f'{takerlongshortRatio_pkl_name}.pkl')
    exists = os.path.exists(takerlongshortRatio)
    if exists:
        if delist:
            # 已下架币种若已存在合约持仓量文件，直接返回
            return

        try:
            df = pd.read_feather(takerlongshortRatio)
            if df.shape[0] > 0:
                df.sort_values(by='timestamp')
                latest_time = int(df.iloc[-1, 0]) + 1
                if latest_time > start_time:
                    start_time = latest_time
        except Exception as e:
            print(takerlongshortRatio, 'error')
            raise e

    takerlongshortRatio_url = 'https://fapi.binance.com/futures/data/takerlongshortRatio'
    updated = False
    limit = 500
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
        while True:
            async with api_semaphore:
                param = {
                    'symbol': symbol,
                    'period': '1h',
                    'startTime': str(start_time),
                    'endTime': str(min(start_time + limit * 60 * 60 * 1000, end_time)),
                    'limit': limit
                }
                try:
                    async with session.get(url=takerlongshortRatio_url, params=param, proxy=proxy,
                                           timeout=5) as kline_response:
                        f_data = await kline_response.text()
                        if kline_response.status == 200:
                            _df = pd.DataFrame(json.loads(f_data),
                                               columns=['timestamp', 'buyVol', 'sellVol', 'buySellRatio'], index=None)
                            if _df.shape[0] == 0:
                                if not exists:
                                    param = {
                                        'symbol': symbol,
                                        'period': '1h',
                                        'limit': limit
                                    }
                                    async with session.get(url=takerlongshortRatio_url, params=param, proxy=proxy,
                                                           timeout=5) as new_response:
                                        f_data = await new_response.text()
                                        if new_response.status == 200:
                                            _df = pd.DataFrame(json.loads(f_data),
                                                               columns=['timestamp', 'buyVol', 'sellVol',
                                                                        'buySellRatio'], index=None)
                                            if _df.shape[0] == 0:
                                                break
                                        else:
                                            print(f'takerlongshortRatio {symbol} error response {f_data}')
                                            break
                                else:
                                    break
                            _df.sort_values(by='timestamp', inplace=True)
                            start_time = int(_df.iloc[-1, 0]) + 1
                            df = pd.concat([df, _df], ignore_index=True)
                            updated = True

                            if _df.shape[0] < 500:
                                break
                        else:
                            print(f'takerlongshortRatio {symbol} error response {f_data}')
                            break
                except Exception as e:
                    if not blind:
                        print('swap takerlongshortRatio 请求失败，小问题不要慌，马上重试', e)
                    continue
        if updated:
            df.to_feather(takerlongshortRatio)


async def ping():
    """
    对 fapi.binance.com 进行联通测试
    """
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url='https://fapi.binance.com/fapi/v1/ping', proxy=proxy, timeout=5) as response:
                t = await response.text()
                if t == '{}':
                    print('币安接口已连通')
                else:
                    print('币安接口fapi.binance.com连接异常，请检查网络配置后重新运行')
                    exit(0)
        except Exception as e:
            print('币安接口fapi.binance.com无法连接，请检查网络配置后重新运行', e)
            exit(0)


def spot_symbols_filter(symbols):
    others = []
    stable_symbol = ['BKRW', 'USDC', 'USDP', 'TUSD', 'BUSD', 'FDUSD', 'DAI', 'EUR', 'GBP']
    # stable_symbols：稳定币交易对
    stable_symbols = [s + 'USDT' for s in stable_symbol]
    # special_symbols：容易误判的特殊交易对
    special_symbols = ['JUPUSDT']
    pure_spot_symbols = []
    for symbol in symbols:
        if symbol in special_symbols:
            pure_spot_symbols.append(symbol)
            continue
        if symbol.endswith('UPUSDT') or symbol.endswith('DOWNUSDT') or symbol.endswith('BULLUSDT') or symbol.endswith(
            'BEARUSDT'
        ):
            others.append(symbol)
            continue
        if symbol in stable_symbols:
            others.append(symbol)
            continue
        pure_spot_symbols.append(symbol)
    print('过滤掉的现货symbol', others)
    return pure_spot_symbols


def run():
    print(f'CPU核心数: {cpu}')
    start_time = datetime.datetime.now()
    params = {
        'delimiter': '/',
        'prefix': prefix
    }
    symbols = async_get_usdt_symbols(params)
    if trade_type == 'spot':
        symbols = spot_symbols_filter(symbols)
    print('usdt交易对数量', len(symbols))

    metrics_daily_list = []
    metrics_symbols = []
    if trade_type == 'swap':
        # 网络检查
        asyncio.get_event_loop().run_until_complete(ping())
    if download_metrics:
        metrics_param = {
            'delimiter': '/',
            'prefix': metrics_prefix
        }
        metrics_symbols = async_get_usdt_symbols(metrics_param)
        print('metrics 交易对数量', len(metrics_symbols))
        metrics_daily_list = async_get_metrics_list(root_path, metrics_symbols, trade_type, 'metrics', '5m')
        print('metrics 数据包数量', len(metrics_daily_list))

    print('开始获取数据目录')
    download_folder = root_path
    daily_list_1m = async_get_daily_list(download_folder, symbols, trade_type, 'klines', '1m')
    daily_list_5m = async_get_daily_list(download_folder, symbols, trade_type, 'klines', '5m')
    print('daily zip num in latest 2 months =', len(daily_list_1m) + len(daily_list_5m))

    monthly_list_1m = async_get_monthly_list(download_folder, symbols, trade_type, 'klines', '1m')
    monthly_list_5m = async_get_monthly_list(download_folder, symbols, trade_type, 'klines', '5m')
    print('monthly zip num =', len(monthly_list_1m) + len(monthly_list_5m))

    all_list = daily_list_1m + monthly_list_1m + metrics_daily_list + daily_list_5m + monthly_list_5m
    random.shuffle(all_list)  # 打乱monthly和daily的顺序，合理利用网络带宽

    get_time = datetime.datetime.now()
    print('所有数据包个数为', len(all_list), "获取目录耗费 {} s".format((get_time - start_time).seconds))

    print('开始清理daily旧数据...')
    clean_old_daily_zip(get_local_path(download_folder, trade_type, 'klines', 'daily', None, '1m'), symbols, '1m')
    clean_old_daily_zip(get_local_path(download_folder, trade_type, 'klines', 'daily', None, '5m'), symbols, '5m')
    print('清理完成')

    print('start download:')
    error_info_list = set()
    async_download_file(all_list, error_info_list)
    print('need analyse', need_analyse_set)
    if len(error_info_list) > 0:
        print('下载过程发生错误，已完成重试下载，请核实')
        print(error_info_list)
    end_time = datetime.datetime.now()
    print(f'download end cost {(end_time - get_time).seconds} s = {(end_time - get_time).seconds / 60} min')

    print('开始将daily数据合并为monthly数据...')
    # 获取当前时间
    end_time = datetime.datetime.now()
    if end_time.minute < 5:
        # 截止时间，若当前时间离整点没过5分钟则取上个整点，容错处理，防止K线未闭合，15点04分 end_time取14点整
        end_time -= datetime.timedelta(hours=1)
    end_time = end_time.replace(minute=0, second=0, microsecond=0)
    print('最新数据正在更新至', end_time - datetime.timedelta(seconds=1))
    global newest_timestamp
    # 将时间转换为时间戳
    newest_timestamp = int(time.mktime(end_time.timetuple()) * 1000) - 1
    tasks = transfer_daily_to_monthly_and_get_newest(newest_timestamp, daily_list_1m + daily_list_5m, need_analyse_set)
    if len(tasks) > 0:
        asyncio.get_event_loop().run_until_complete(asyncio.gather(*tasks))

    if force_analyse or len(need_analyse_set) > 0:
        analyse_download_data(download_folder, symbols, trade_type, 'klines', ['1m', '5m'])

    print('开始生成PKL数据...')
    data_center_data_to_pickle_data(trade_type, CONCURRENCY, metrics_symbols)
    print(f'结束运行，总耗时{(datetime.datetime.now() - start_time).seconds / 60}min')


def metrics_analyse():
    params = {
        'delimiter': '/',
        'prefix': prefix
    }
    symbols = async_get_usdt_symbols(params)
    print('swap 交易对数量', len(symbols))

    metrics_param = {
        'delimiter': '/',
        'prefix': metrics_prefix
    }
    metrics_symbols = async_get_usdt_symbols(metrics_param)
    print('metrics 交易对数量', len(metrics_symbols))

    print(f'swap 和 metrics 相差 {list(set(symbols) ^ set(metrics_symbols))}')

    analyse_and_resample_metric_data(root_path, metrics_symbols)


def analyse_and_resample_metric_data(download_folder, metrics_symbols):
    """
    读取metrics 5m级别数据，并resample为 1h，并打印缺失情况，最后将数据存储到 metrics_1h目录
    """
    for symbol in sorted(metrics_symbols):
        local_path = get_local_path(download_folder, 'swap', 'metrics', 'daily', symbol, '5m')
        zip_list = glob(os.path.join(local_path, f'{symbol}-metrics*.zip'))
        if not zip_list:
            continue
        m_df = pd.concat(
            Parallel(CONCURRENCY)(
                delayed(pd.read_csv)(
                    path_, header=0, encoding="utf-8", compression='zip',
                    usecols=['create_time', 'sum_open_interest', 'sum_open_interest_value'],
                    dtype={'sum_open_interest': float64, 'sum_open_interest_value': float64},
                    parse_dates=['create_time'], engine='c') for path_ in zip_list), ignore_index=True)
        m_df.sort_values(by='create_time', inplace=True)
        m_df.drop_duplicates(subset=['create_time'], inplace=True, keep='last')  # 去除重复值
        m_df.set_index('create_time', inplace=True)

        monthly_zip_path_5m = get_local_path(root_path, trade_type, 'klines', 'monthly', None, '5m')
        zip_list = glob(os.path.join(monthly_zip_path_5m, symbol, f'{symbol}*.zip'))
        k_df = pd.concat(
            Parallel(CONCURRENCY)(
                delayed(pd.read_csv)(path_, header=None, encoding="utf-8", compression='zip', engine='c',
                                     names=['open_time', 'open', 'high', 'low', 'close', 'volume',
                                            'close_time', 'quote_volume', 'trade_num',
                                            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
                                            'ignore']) for path_ in zip_list), ignore_index=True
        )
        # 过滤表头行
        k_df = k_df[k_df['open_time'] != 'open_time']
        # 规范数据类型，防止计算avg_price报错
        k_df = k_df.astype(
            dtype={'open_time': int64, 'open': float, 'high': float, 'low': float, 'close': float, 'volume': float,
                   'quote_volume': float,
                   'trade_num': int, 'taker_buy_base_asset_volume': float, 'taker_buy_quote_asset_volume': float})
        k_df['candle_begin_time'] = pd.to_datetime(k_df['open_time'], unit='ms')
        del k_df['close_time'], k_df['ignore']
        k_df.sort_values(by='candle_begin_time', inplace=True)  # 排序
        k_df = k_df[
            (k_df['candle_begin_time'] >= '2021-12-01 00:00:00') & (k_df['candle_begin_time'] < '2024-02-11 00:00:00')]
        k_df.drop_duplicates(subset=['candle_begin_time'], inplace=True, keep='last')  # 去除重复值
        k_df.reset_index(drop=True, inplace=True)  # 重置index
        k_df.set_index('candle_begin_time', inplace=True)

        k_df['sum_open_interest_open'] = m_df['sum_open_interest']
        k_df['sum_open_interest_value_open'] = m_df['sum_open_interest_value']
        k_df['sum_open_interest_close'] = m_df['sum_open_interest'].shift(-1)
        k_df['sum_open_interest_value_close'] = m_df['sum_open_interest_value'].shift(-1)
        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'quote_volume': 'sum',
            'trade_num': 'sum',
            'taker_buy_base_asset_volume': 'sum',
            'taker_buy_quote_asset_volume': 'sum',
            'sum_open_interest_open': 'first',
            'sum_open_interest_value_open': 'first',
            'sum_open_interest_close': 'last',
            'sum_open_interest_value_close': 'last',
        }
        df = k_df.resample('1H').agg(agg_dict)
        df = df[df['quote_volume'] > 0]
        empty_data = df[df['sum_open_interest_open'].isnull() | df['sum_open_interest_value_open'].isnull()]
        if not empty_data.empty:
            # print(symbol)
            # print(empty_data)
            a = {str(item).split(' ')[0] for item in empty_data.index}
            print(f'{symbol} miss {sorted(list(a))}')
        # empty_data = k_df[k_df['sum_open_interest_open'].isnull() | k_df['sum_open_interest_value_open'].isnull()]
        # if not empty_data.empty:
        #     print(symbol)
        #     print(empty_data)
        #     exit(0)

        # data = m_df.resample('1H').asfreq()  # 将5min数据重采样为1H周期，缺漏的地方补为NaN
        #
        # empty_data = data[data['sum_open_interest'].isnull() | data['sum_open_interest_value'].isnull()]
        # if not empty_data.empty:
        #     print(empty_data)
        #     a = {str(item).split(' ')[0] for item in empty_data.index}
        #     print(f'{symbol} 缺失持仓量的日期 {sorted(list(a))}')
        # data.reset_index(inplace=True)
        # metrics_folder = os.path.join(download_folder, 'swap_5m', 'metrics_1h')
        # if not os.path.exists(metrics_folder):
        #     os.makedirs(metrics_folder)
        # data.to_feather(os.path.join(metrics_folder, f'{symbol}.pkl'))


if __name__ == "__main__":
    if analyse_metrics:
        metrics_analyse()
        exit(0)
    run()
    df = pd.read_feather(os.path.join(pickle_path, trade_type, 'BTC-USDT.pkl'))
    print(df.head(10))
    print(df.tail(10))
