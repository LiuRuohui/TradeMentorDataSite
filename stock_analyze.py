import io
import os
import re
import sys
import time
import shutil
import logging
import argparse
import traceback
import multiprocessing
import numpy as np
import pandas as pd
import akshare as ak
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl
import warnings

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed  # 添加 ThreadPoolExecutor
from tqdm import tqdm
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from mplfinance.original_flavor import candlestick_ohlc
# 添加 AI 分析相关的导入
from modules.ai_analyzer import get_ai_analysis, save_ai_analysis
from modules.utils import logger
from modules.constants import *

#添加相关可视化ploty的导入
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
pio.templates.default = "simple_white"

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

# 设置中文编码环境
try:
    import locale
    locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')
except locale.Error:
    try:
        locale.setlocale(locale.LC_ALL, 'zh_CN')
    except locale.Error:
        pass

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def set_plot_style() -> None:
    """设置Matplotlib的绘图样式，包括字体、网格、线条宽度等参数"""
    # 使用默认样式
    plt.style.use('default')

    # 设置中文字体
    import platform
    system = platform.system()

    if system == 'Windows':
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    else:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']

    # 用来正常显示负号
    plt.rcParams['axes.unicode_minus'] = False

    # 设置字体大小和样式
    plt.rcParams['font.size'] = FONT_SIZE_BASE
    plt.rcParams['axes.titlesize'] = FONT_SIZE_SUBTITLE
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['figure.titlesize'] = FONT_SIZE_MAIN_TITLE
    plt.rcParams['figure.titleweight'] = 'bold'
    plt.rcParams['legend.fontsize'] = FONT_SIZE_LEGEND
    plt.rcParams['axes.labelsize'] = FONT_SIZE_LABEL

    # 设置坐标轴刻度字体大小
    plt.rcParams['xtick.labelsize'] = FONT_SIZE_TICK
    plt.rcParams['ytick.labelsize'] = FONT_SIZE_TICK

    # 设置网格样式
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.2
    plt.rcParams['grid.linestyle'] = '--'

    # 设置图表布局
    plt.rcParams['figure.constrained_layout.use'] = True

    # 设置线条宽度
    plt.rcParams['lines.linewidth'] = LINE_WIDTH
    plt.rcParams['axes.linewidth'] = LINE_WIDTH

    # 设置坐标轴样式
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.labelweight'] = 'bold'

def get_historical_data(stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取指定股票的历史数据

    Args:
        stock_code: 股票代码
        start_date: 开始日期，格式为'YYYY-MM-DD'
        end_date: 结束日期，格式为'YYYY-MM-DD'

    Returns:
        pd.DataFrame: 包含开盘价、收盘价、最高价、最低价和成交量的历史数据
    """
    try:
        # 从全局变量获取调试模式状态
        use_cache = args.debug

        if use_cache:
            # 检查缓存目录
            cache_dir = os.path.join('data_cache', start_date + '_' + end_date)
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, f'{stock_code}_hist.parquet')

            if os.path.exists(cache_file):
                logger.debug(f"从缓存读取股票 {stock_code} 的历史数据")
                return pd.read_parquet(cache_file)

        # 获取新数据
        hist_data = ak.stock_zh_a_hist(symbol=stock_code,
                                     start_date=start_date,
                                     end_date=end_date)
        if hist_data is None or hist_data.empty:
            return pd.DataFrame()

        # 确保日期列是datetime类型
        hist_data['日期'] = pd.to_datetime(hist_data['日期'])
        hist_data.set_index('日期', inplace=True)

        # 转换列名
        result = hist_data[['开盘', '收盘', '最高', '最低', '成交量']].rename(
            columns={'开盘': 'open', '收盘': 'close', '最高': 'high', 
                    '最低': 'low', '成交量': 'volume'})

        # 如果启用了缓存，保存数据
        if use_cache:
            logger.debug(f"缓存股票 {stock_code} 的历史数据")
            result.to_parquet(cache_file)

        return result

    except Exception as e:
        logger.error(f"获取股票 {stock_code} 历史数据时出错: {str(e)}")
        return pd.DataFrame()

def calculate_technical_indicators(data: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    计算技术指标，包括MA、MACD、KDJ、RSI等

    Args:
        data: 包含历史数据的DataFrame

    Returns:
        Dict[str, pd.Series]: 包含各种技术指标的字典
    """
    try:
        if data.empty:
            return {}

        # 计算移动平均线
        data['ma5'] = data['close'].rolling(window=5).mean()
        data['ma10'] = data['close'].rolling(window=10).mean()
        data['ma20'] = data['close'].rolling(window=20).mean()

        # 计算MACD
        exp12 = data['close'].ewm(span=12, adjust=False).mean()
        exp26 = data['close'].ewm(span=26, adjust=False).mean()
        data['dif'] = exp12 - exp26
        data['dea'] = data['dif'].ewm(span=9, adjust=False).mean()
        data['macd'] = 2 * (data['dif'] - data['dea'])

        # 计算KDJ
        low_9 = data['low'].rolling(window=9).min()
        high_9 = data['high'].rolling(window=9).max()
        rsv = (data['close'] - low_9) / (high_9 - low_9) * 100
        data['k'] = rsv.ewm(span=3, adjust=False).mean()
        data['d'] = data['k'].ewm(span=3, adjust=False).mean()
        data['j'] = 3 * data['k'] - 2 * data['d']

        # 计算RSI
        diff = data['close'].diff()
        gain = (diff.where(diff > 0, 0)).rolling(window=14).mean()
        loss = (-diff.where(diff < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))

        # 计算成交量均线
        data['volume_ma5'] = data['volume'].rolling(window=5).mean()

        return {
            'ma5': data['ma5'],
            'ma10': data['ma10'],
            'ma20': data['ma20'],
            'dif': data['dif'],
            'dea': data['dea'],
            'macd': data['macd'],
            'k': data['k'],
            'd': data['d'],
            'j': data['j'],
            'rsi': data['rsi'],
            'volume_ma5': data['volume_ma5']
        }

    except Exception as e:
        logger.error(f"计算技术指标时出错: {str(e)}")
        return {}

def calculate_stock_score(hist_data: pd.DataFrame, indicators: Dict) -> float:
    """
    根据历史数据和技术指标计算股票评分

    Args:
        hist_data: 历史数据
        indicators: 技术指标

    Returns:
        float: 股票评分，范围0-100
    """
    try:
        # 增加数据长度检查
        min_data_length = 20  # 至少需要20个交易日的数据
        if hist_data.empty or len(hist_data) < min_data_length:
            return 0.0

        # 动态权重计算
        volatility = hist_data['close'].pct_change().std()  # 波动率
        if volatility > 0.05:  # 高波动率时增加技术指标权重
            weights = {'macd': 0.25, 'kdj': 0.25, 'rsi': 0.2, 'ma': 0.2, 'volume': 0.1}
        else:  # 低波动率时增加趋势指标权重
            weights = {'macd': 0.2, 'kdj': 0.2, 'rsi': 0.15, 'ma': 0.3, 'volume': 0.15}

        # 1. MACD指标
        macd_score = 0
        if indicators['macd'][-1] > 0:
            macd_score += 10
        if len(hist_data) > 1 and (indicators['dif'][-1] > indicators['dea'][-1] and 
            indicators['dif'][-2] <= indicators['dea'][-2]):
            macd_score += 10
        macd_score *= weights['macd']

        # 2. KDJ指标
        kdj_score = 0
        if indicators['k'][-1] < 30 and indicators['d'][-1] < 30:
            kdj_score += 10
        if indicators['j'][-1] < 20:
            kdj_score += 5
        if len(hist_data) > 1 and (indicators['k'][-1] > indicators['d'][-1] and 
            indicators['k'][-2] <= indicators['d'][-2]):
            kdj_score += 5
        kdj_score *= weights['kdj']

        # 3. RSI指标
        rsi = indicators['rsi'][-1]
        rsi_score = 0
        if rsi < 30:
            rsi_score = 20
        elif rsi < 40:
            rsi_score = 15
        elif rsi < 50:
            rsi_score = 10
        rsi_score *= weights['rsi']

        # 4. 均线系统
        ma_score = 0
        if (indicators['ma5'][-1] > indicators['ma10'][-1] > 
            indicators['ma20'][-1]):
            ma_score += 10
        latest_price = hist_data['close'][-1]
        if latest_price > indicators['ma5'][-1]:
            ma_score += 4
        if latest_price > indicators['ma10'][-1]:
            ma_score += 3
        if latest_price > indicators['ma20'][-1]:
            ma_score += 3
        ma_score *= weights['ma']

        # 5. 成交量分析
        vol_score = 0
        if hist_data['volume'][-1] > indicators['volume_ma5'][-1]:
            vol_score += 10
        if len(hist_data) >= 3 and (hist_data['volume'][-1] > hist_data['volume'][-2] > 
            hist_data['volume'][-3]):
            vol_score += 10
        vol_score *= weights['volume']

        # 6. 趋势强度指标（新增）
        trend_score = 0
        if len(hist_data) >= 5:
            # 计算短期（5日）和长期（20日）趋势
            short_term = hist_data['close'][-5:].pct_change().mean()
            long_term = hist_data['close'][-20:].pct_change().mean()

            # 趋势强度评分
            if short_term > long_term:  # 短期趋势强于长期趋势
                trend_score += 10
            if hist_data['close'][-1] > hist_data['close'][-5]:  # 近期上涨
                trend_score += 5
            if hist_data['close'][-1] > hist_data['close'][-20]:  # 长期上涨
                trend_score += 5

        # 7. 风险调整（新增）
        risk_adjustment = 1.0
        # 基于波动率调整
        if volatility > 0.1:  # 高波动率
            risk_adjustment *= 0.9
        elif volatility < 0.05:  # 低波动率
            risk_adjustment *= 1.1
        # 基于最大回撤调整
        max_drawdown = (hist_data['close'].cummax() - hist_data['close']).max() / hist_data['close'].cummax().max()
        if max_drawdown > 0.2:  # 最大回撤超过20%
            risk_adjustment *= 0.9

        # 综合评分
        total_score = (macd_score + kdj_score + rsi_score + ma_score + vol_score + trend_score) * risk_adjustment

        # 根据历史表现调整评分
        recent_performance = hist_data['close'][-1] / hist_data['close'][0] - 1
        if recent_performance > 0.1:  # 近期表现良好
            total_score *= 1.1
        elif recent_performance < -0.1:  # 近期表现不佳
            total_score *= 0.9

        return round(min(total_score, 100), 2)  # 确保评分不超过100，并保留2位小数

    except Exception as e:
        logger.error(f"计算股票评分时出错: {str(e)}")
        return 0.0

def analyze_stocks(stock_list: pd.DataFrame, start_date: str, end_date: str, pool_size: int = 8) -> List[Dict]:
    """
    批量分析股票列表

    Args:
        stock_list: 股票列表
        start_date: 开始日期
        end_date: 结束日期
        pool_size: 线程池大小

    Returns:
        List[Dict]: 包含每只股票分析结果的列表
    """
    # 确保stock_list是DataFrame且包含'代码'和'名称'列
    if not isinstance(stock_list, pd.DataFrame) or not {'代码', '名称'}.issubset(stock_list.columns):
        logger.error("股票列表格式不正确")
        return []

    # 过滤掉无效的股票代码
    stock_list = stock_list[stock_list['代码'].notna() & stock_list['代码'].str.match(r'^\d{6}$')]
    total = len(stock_list)

    if total == 0:
        logger.error("没有有效的股票代码")
        return []

    logger.info(f"开始分析 {total} 只股票...")

    logger.info(f"使用 {pool_size} 个线程进行并行处理")

    # 配置文件日志处理器
    log_file = os.path.join(output_dir, 'analysis.log')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')  # 添加 encoding='utf-8'
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # 添加文件处理器
    logger.addHandler(file_handler)

    # 禁用所有控制台输出
    logger.propagate = False

    results = []

    try:
        # 判断数据类型并输出提示
        if is_historical:
            logger.info("使用历史数据进行分析（注意：历史数据不包含市值信息）")
            current_info = None
        else:
            logger.info("使用实时数据进行分析")
            # 预先获取所有股票的实时行情数据
            try:
                current_info = ak.stock_zh_a_spot_em()
                current_info.set_index('代码', inplace=True)
                logger.info("获取实时行情数据成功")
            except Exception as e:
                logger.error(f"获取实时行情数据失败: {str(e)}")
                current_info = None

        # 创建线程池
        with ThreadPoolExecutor(max_workers=pool_size) as executor:
            # 准备任务，将 current_info 和 is_historical 作为参数传递给每个任务
            tasks = [(row['代码'], row['名称'], start_date, end_date, current_info, is_historical)
                    for _, row in stock_list.iterrows()]

            # 提交任务到线程池
            futures = []
            for task in tasks:
                future = executor.submit(analyze_stock_wrapper, task)
                futures.append(future)

            # 使用 tqdm 显示进度
            with tqdm(total=total, desc="分析进度", ncols=100) as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        results.append(result)
                    pbar.update(1)
        sys.stdout.flush()  # 强制刷新输出

    except KeyboardInterrupt:
        logger.info("\n正在终止进程...")
        return results
    finally:
        # 恢复控制台输出
        logger.propagate = True

        # 移除文件处理器
        logger.removeHandler(file_handler)
        file_handler.close()

        # 检查日志文件是否存在且有内容
        if os.path.exists(log_file) and os.path.getsize(log_file) > 0:
            print("\n注意：本次分析可能包含未能成功获取的股票，请查看日志文件：")
            print(f"    {log_file}")
            print("以确认是否有您关注的股票未能成功获取。\n")

    return results

def process_results(results: List[Dict]) -> pd.DataFrame:
    """
    处理分析结果，生成最终的DataFrame

    Args:
        results: 原始分析结果

    Returns:
        pd.DataFrame: 处理后的分析结果
    """
    try:
        # 将结果列表转换为DataFrame
        df = pd.DataFrame(results)

        # 确保所有必需的列都存在
        required_columns = ['代码', '名称', '总市值（亿元）', '起始日价（元）', '截止日价（元）', 
                            '交易所', '得分', '详情']
        if not all(col in df.columns for col in required_columns):
            logger.error("结果数据缺少必需的列")
            return pd.DataFrame()

        # 计算涨幅
        df['涨幅(%)'] = ((df['截止日价（元）'] - df['起始日价（元）']) / df['起始日价（元）'] * 100).round(2)

        # 根据参数选择排序方式
        if args.topgains:
            # 按涨幅降序和代码升序排序
            df = df.sort_values(['涨幅(%)', '代码'], ascending=[False, True])
        else:
            # 按得分降序和代码升序排序
            df = df.sort_values(['得分', '代码'], ascending=[False, True])

        # 重置索引
        df = df.reset_index(drop=True)

        # 格式化数值列
        df['得分'] = df['得分'].round(2)
        try:
            df['总市值（亿元）'] = df['总市值（亿元）'].round(2)
        except:
            # 历史数据没有’总市值（亿元）’列，其值是'N/A'，这里不做处理
            pass
        df['起始日价（元）'] = df['起始日价（元）'].round(2)
        df['截止日价（元）'] = df['截止日价（元）'].round(2)

        # 重新排列列的顺序
        df = df[['代码', '名称', '总市值（亿元）', '起始日价（元）', '截止日价（元）', 
                 '涨幅(%)', '得分', '交易所', '详情']]

        return df

    except Exception as e:
        logger.error(f"处理结果时出错: {str(e)}")
        return pd.DataFrame()

def analyze_stock_wrapper(args: Tuple[str, str, str, str, Optional[pd.DataFrame], bool]) -> Optional[Dict]:
    """
    包装股票分析函数，用于多线程处理

    Args:
        args: 包含股票代码、名称、日期等信息的元组

    Returns:
        Optional[Dict]: 单只股票的分析结果
    """
    try:
        stock_code, stock_name, start_date, end_date, current_info, is_historical = args

        # 获取历史数据
        hist_data = get_historical_data(stock_code, start_date, end_date)
        if hist_data.empty or len(hist_data) < 20:  # 新增数据长度检查
            logger.warning(f"股票 {stock_code} 历史数据不足，跳过分析")
            return None

        # 计算区间涨幅
        # 获取起始日价格和截止日价格
        start_price = hist_data['close'][0]  # 起始价格
        end_price = hist_data['close'][-1]   # 结束价格
        price_change = ((end_price - start_price) / start_price * 100)  # 涨幅百分比

        # 计算技术指标
        indicators = calculate_technical_indicators(hist_data)
        if not indicators:
            return None

        # 获取最新价格和市值数据
        try:
            latest_price = hist_data['close'][-1]  # 总是使用历史数据的最后一个收盘价

            # 根据是否是历史数据来决定市值
            if is_historical:
                market_cap = 'N/A'
            else:
                if current_info is not None and stock_code in current_info.index:
                    market_cap = current_info.loc[stock_code, '总市值'] / 100000000  # 转换为亿元
                else:
                    market_cap = 0.0

        except Exception as e:
            latest_price = hist_data['close'][-1]
            market_cap = 'N/A' if is_historical else 0.0
            logger.warning(f"获取股票 {stock_code} 实时数据失败，使用历史数据最新价格")

        # 计算得分和得分详情
        score = calculate_stock_score(hist_data, indicators)

        # 获取得分详情
        score_details = []
        latest_idx = -1

        # MACD指标得分详情
        macd_score = 0
        if indicators['macd'][latest_idx] > 0:
            macd_score += 10
        if len(hist_data) > 1 and (indicators['dif'][latest_idx] > indicators['dea'][latest_idx] and 
            indicators['dif'][latest_idx-1] <= indicators['dea'][latest_idx-1]):
            macd_score += 10
        score_details.append(f"MACD({macd_score}分)=[柱>0:{10 if indicators['macd'][latest_idx] > 0 else 0}分 + DIF上穿:{10 if macd_score > 10 else 0}分]")

        # KDJ指标得分详情
        kdj_score = 0
        if indicators['k'][latest_idx] < 30 and indicators['d'][latest_idx] < 30:
            kdj_score += 10
        if indicators['j'][latest_idx] < 20:
            kdj_score += 5
        if len(hist_data) > 1 and (indicators['k'][latest_idx] > indicators['d'][latest_idx] and 
            indicators['k'][latest_idx-1] <= indicators['d'][latest_idx-1]):
            kdj_score += 5
        score_details.append(f"KDJ({kdj_score}分)=[KD<30:{10 if kdj_score >= 10 else 0}分 + J<20:{5 if indicators['j'][latest_idx] < 20 else 0}分 + K上穿:{5 if kdj_score-15 == 5 else 0}分]")

        # RSI指标得分详情
        rsi = indicators['rsi'][latest_idx]
        rsi_score = 0
        if rsi < 30:
            rsi_score = 20
        elif rsi < 40:
            rsi_score = 15
        elif rsi < 50:
            rsi_score = 10
        score_details.append(f"RSI({rsi_score}分)=[RSI={rsi:.1f}]")

        # 均线系统得分详情
        ma_score = 0
        if (indicators['ma5'][latest_idx] > indicators['ma10'][latest_idx] > 
            indicators['ma20'][latest_idx]):
            ma_score += 10
        latest_price = hist_data['close'][latest_idx]
        ma_cross_score = 0
        if latest_price > indicators['ma5'][latest_idx]:
            ma_cross_score += 4
        if latest_price > indicators['ma10'][latest_idx]:
            ma_cross_score += 3
        if latest_price > indicators['ma20'][latest_idx]:
            ma_cross_score += 3
        score_details.append(f"均线({ma_score + ma_cross_score}分)=[多头排列:{ma_score}分 + 站上均线:{ma_cross_score}分]")

        # 成交量分析得分详情
        vol_score = 0
        if len(hist_data) >= 1 and hist_data['volume'][latest_idx] > indicators['volume_ma5'][latest_idx]:
            vol_score += 10
        if len(hist_data) >= 3:  # 确保有足够的数据进行比较
            try:
                if (hist_data['volume'][latest_idx] > hist_data['volume'][latest_idx-1] > 
                    hist_data['volume'][latest_idx-2]):
                    vol_score += 10
            except IndexError:
                # 如果出现索引错误，不增加分数
                pass

        score_details.append(f"成交量({vol_score}分)=[量>均量:{10 if vol_score >= 10 else 0}分 + 量增加:{10 if vol_score-10 == 10 else 0}分]")

        # 合并所有得分详情
        score_detail_str = f"总分{round(score, 2)}分 = " + " + ".join(score_details) + f" 涨幅:{round(price_change, 2)}%"

        return {
            '代码': stock_code,
            '名称': stock_name,
            '总市值（亿元）': market_cap,
            '起始日价（元）': start_price,
            '截止日价（元）': latest_price,
            '交易所': 'SH' if stock_code.startswith('6') else 'SZ',
            '得分': round(score, 2),
            '详情': score_detail_str
        }

    except Exception as e:
        logger.error(f"分析股票 {stock_code} 时出错: {str(e)}")
        return None

def generate_stock_charts(stocks: pd.DataFrame,
                         start_date: str,
                         end_date: str,
                         output_dir: str) -> None:
    try:
        stocks = stocks.head(args.k)

        for _, stock in stocks.iterrows():
            code, name = stock['代码'], stock['名称']
            hist = get_historical_data(code, start_date, end_date)
            if hist.empty:
                continue
            ind = calculate_technical_indicators(hist)
            if not ind:
                continue

            # ======================
            # 专业级布局配置
            # ======================
            fig = make_subplots(
                rows=4, cols=2,
                shared_xaxes=True,
                vertical_spacing=0.08,
                horizontal_spacing=0.1,
                specs=[
                    [{"colspan": 2}, None],  # 第1行
                    [{"secondary_y": True}, {"secondary_y": True}],  # 第2行
                    [{"secondary_y": True}, {"secondary_y": True}],  # 第3行
                    [{"secondary_y": True}, {"secondary_y": True}]   # 第4行
                ],
                subplot_titles=(
                    'K线趋势分析',    # 第1行
                    'MACD指标分析',  # 第2行左
                    'KDJ指标分析',   # 第2行右
                    'RSI强弱指标',    # 第3行左
                    '量价关系分析',    # 第3行右
                    '价格趋势线',     # 第4行左
                    '波动率分析'      # 第4行右
                ),
                row_heights=[0.5, 0.2, 0.2, 0.2]
            )

            # ======================
            # 顶部综合信息区
            # ======================
            summary_text = (
                f"◇{code} {name}◇ | "
                f"起始价：{stock['起始日价（元）']:.2f}元 | "
                f"最新价：{stock['截止日价（元）']:.2f}元 | "
                f"区间涨幅：{stock['涨幅(%)']:.2f}% | "
                f"市值规模：{stock['总市值（亿元）']}亿 | "
                f"综合评分：{stock['得分']:.2f}"
            )

            # ======================
            # 核心图表元素
            # ======================
            # ---- 第1行：K线+均线 ----
            # 专业级K线配色
            fig.add_trace(go.Candlestick(
                x=hist.index,
                open=hist['open'],
                high=hist['high'],
                low=hist['low'],
                close=hist['close'],
                increasing_line_color='#E74C3C',  # 专业红
                decreasing_line_color='#2ECC71',  # 专业绿
                name='价格走势'
            ), row=1, col=1)

            # 三重均线系统
            ma_config = [
                ('ma5', '#F39C12', 'MA5'),    # 橙色
                ('ma10', '#3498DB', 'MA10'),  # 蓝色
                ('ma20', '#9B59B6', 'MA20')   # 紫色
            ]
            for ma, color, legend in ma_config:
                fig.add_trace(go.Scatter(
                    x=hist.index,
                    y=ind[ma],
                    line=dict(color=color, width=1.2),
                    name=legend,
                    showlegend=False
                ), row=1, col=1)

            # ---- 第2行左：纯净MACD ----
            fig.add_trace(go.Bar(
                x=hist.index,
                y=ind['macd'],
                marker_color=['#2ecc71' if v <0 else '#e74c3c' for v in ind['macd']],
                name='MACD柱',
                showlegend=False
            ), row=2, col=1)

            fig.add_trace(go.Scatter(
                x=hist.index,
                y=ind['dif'],
                line=dict(color='#3498db', width=1.5),
                name='DIF',
                showlegend=False
            ), row=2, col=1)

            fig.add_trace(go.Scatter(
                x=hist.index,
                y=ind['dea'],
                line=dict(color='#f1c40f', width=1.5),
                name='DEA',
                showlegend=False
            ), row=2, col=1)

            # ---- 第2行右：纯净KDJ ----
            fig.add_trace(go.Scatter(
                x=hist.index,
                y=ind['k'],
                line=dict(color='#e74c3c', width=1.5),
                name='K',
                showlegend=False
            ), row=2, col=2)

            fig.add_trace(go.Scatter(
                x=hist.index,
                y=ind['d'],
                line=dict(color='#2ecc71', width=1.5),
                name='D',
                showlegend=False
            ), row=2, col=2)

            fig.add_trace(go.Scatter(
                x=hist.index,
                y=ind['j'],
                line=dict(color='#3498db', width=1.5),
                name='J',
                showlegend=False
            ), row=2, col=2)

            # ---- 第3行：RSI指标 ----
            fig.add_trace(go.Scatter(
                x=hist.index,
                y=ind['rsi'],
                line=dict(color='#9B59B6', width=1.2),
                name='RSI'
            ), row=3, col=1)
            # 专业参考线
            fig.add_hline(y=30, line=dict(color='#95A5A6', dash='dot'), row=3, col=1)
            fig.add_hline(y=70, line=dict(color='#95A5A6', dash='dot'), row=3, col=1)

            # ---- 第3行：成交量分析 ----
            vol_colors = ['#2ECC71' if hist['close'][i] > hist['open'][i]
                        else '#E74C3C' for i in range(len(hist))]
            fig.add_trace(go.Bar(
                x=hist.index,
                y=hist['volume'],
                marker_color=vol_colors,
                name='成交量'
            ), row=3, col=2)
            fig.add_trace(go.Scatter(
                x=hist.index,
                y=ind['volume_ma5'],
                line=dict(color='#F39C12', width=1.2),
                name='成交量MA5'
            ), row=3, col=2)

            # ---- 第4行：价格趋势 ----
            fig.add_trace(go.Scatter(
                x=hist.index,
                y=hist['close'],
                line=dict(color='#2C3E50', width=1.5),
                name='收盘价'
            ), row=4, col=1)

            # ---- 第4行：波动率分析 ----
            fig.add_trace(go.Scatter(
                x=hist.index,
                y=hist['close'].pct_change().rolling(5).std(),
                line=dict(color='#E67E22', width=1.2),
                name='波动率'
            ), row=4, col=2)

            # ======================
            # 专业样式配置
            # ======================
            fig.update_layout(
                height=1400,
                width=1000,
                title={
                    'text': summary_text,
                    'y':0.97,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': dict(size=14, family='SimHei')
                },
                font=dict(family='SimHei', size=12),
                margin=dict(t=150, b=50, l=50, r=50),  # 增加顶部边距
                plot_bgcolor='rgba(240,240,240,0.1)',
                paper_bgcolor='white',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
            )

            # 增强网格线
            fig.update_xaxes(showgrid=True, gridcolor='rgba(200,200,200,0.2)',rangeslider_visible=False)
            fig.update_yaxes(showgrid=True, gridcolor='rgba(200,200,200,0.2)')

            # 生成标准化文件名
            clean_name = re.sub(r'[\\/*?:"<>|]', "", name)  # 过滤非法字符
            filename = f"{code}_{clean_name}_analysis.html"
            html_path = os.path.join(output_dir, filename)

            # 保存文件
            fig.write_html(
                html_path,
                include_plotlyjs='cdn',
                config={'scrollZoom': True}
            )

    except Exception as e:
        logger.error(f"图表生成失败: {str(e)}")
    # try:
    #     # 只处理前K只股票
    #     stocks = stocks.head(args.k)

    #     for _, stock in stocks.iterrows():
    #         stock_code = stock['代码']
    #         stock_name = stock['名称']

    #         # 获取历史数据
    #         hist_data = get_historical_data(stock_code, start_date, end_date)
    #         if hist_data.empty:
    #             continue

    #         # 计算技术指标
    #         indicators = calculate_technical_indicators(hist_data)
    #         if not indicators:
    #             continue

    #         # 创建图表并设置布局
    #         fig = plt.figure(figsize=(44.70, 35.61))

    #         # 创建网格布局，调整top值给主标题留出足够空间
    #         gs = fig.add_gridspec(8, 2, height_ratios=[0.5, 2, 1, 1, 1, 1, 1, 1], top=0.95)  # 增加一行用于标题

    #         # 设置整张图的主标题（股票信息）
    #         title_ax = fig.add_subplot(gs[0, :])
    #         title_ax.set_axis_off()  # 隐藏坐标轴
    #         title_ax.text(0.5, 0.5, 
    #                      f'{stock_code} {stock_name} - 起始日价: {stock["起始日价（元）"]}元 - 截止日价: {stock["截止日价（元）"]}元 - 涨幅: {stock["涨幅(%)"]}% - 总市值: {stock["总市值（亿元）"]}亿 - 得分: {stock["得分"]}分',
    #                      fontsize=FONT_SIZE_MAIN_TITLE, fontweight='bold', 
    #                      horizontalalignment='center', verticalalignment='center')

    #         # 创建并设置K线子图
    #         ax_k = fig.add_subplot(gs[1, :])
    #         ax_k.set_title('K线图与均线', pad=20, fontsize=FONT_SIZE_SUBTITLE, fontweight='bold')

    #         # 下方6个子图
    #         ax_macd = fig.add_subplot(gs[2:4, 0])  # MACD指标
    #         ax_kdj = fig.add_subplot(gs[2:4, 1])   # KDJ指标
    #         ax_rsi = fig.add_subplot(gs[4:6, 0])   # RSI指标
    #         ax_vol = fig.add_subplot(gs[4:6, 1])   # 成交量分析
    #         ax_trend = fig.add_subplot(gs[6:8, 0])  # 价格趋势
    #         ax_wave = fig.add_subplot(gs[6:8, 1])   # 波动趋势

    #         # 绘制K线图和均线
    #         hist_data['date_num'] = mdates.date2num(hist_data.index.to_pydatetime())
    #         ohlc = hist_data[['date_num', 'open', 'high', 'low', 'close']].values
    #         candlestick_ohlc(ax_k, ohlc, width=0.6, colorup='red', colordown='green')

    #         # 添加均线
    #         ax_k.plot(hist_data.index, indicators['ma5'], label='MA5', linewidth=1)
    #         ax_k.plot(hist_data.index, indicators['ma10'], label='MA10', linewidth=1)
    #         ax_k.plot(hist_data.index, indicators['ma20'], label='MA20', linewidth=1)

    #         # 设置其他子图标题
    #         ax_macd.set_title('MACD指标', fontsize=FONT_SIZE_SUBTITLE, fontweight='bold')
    #         ax_kdj.set_title('KDJ指标', fontsize=FONT_SIZE_SUBTITLE, fontweight='bold')
    #         ax_rsi.set_title('RSI指标', fontsize=FONT_SIZE_SUBTITLE, fontweight='bold')
    #         ax_vol.set_title('成交量分析', fontsize=FONT_SIZE_SUBTITLE, fontweight='bold')
    #         ax_trend.set_title('价格趋势', fontsize=FONT_SIZE_SUBTITLE, fontweight='bold')
    #         ax_wave.set_title('波动趋势', fontsize=FONT_SIZE_SUBTITLE, fontweight='bold')

    #         # 绘制MACD
    #         ax_macd.bar(hist_data.index, indicators['macd'], label='MACD')
    #         ax_macd.plot(hist_data.index, indicators['dif'], label='DIF')
    #         ax_macd.plot(hist_data.index, indicators['dea'], label='DEA')
    #         ax_macd.set_title('MACD指标')
    #         ax_macd.grid(True)
    #         ax_macd.legend()

    #         # 绘制KDJ
    #         ax_kdj.plot(hist_data.index, indicators['k'], label='K')
    #         ax_kdj.plot(hist_data.index, indicators['d'], label='D')
    #         ax_kdj.plot(hist_data.index, indicators['j'], label='J')
    #         ax_kdj.set_title('KDJ指标')
    #         ax_kdj.grid(True)
    #         ax_kdj.legend()

    #         # 绘制RSI
    #         ax_rsi.plot(hist_data.index, indicators['rsi'], label='RSI')
    #         ax_rsi.set_title('RSI指标')
    #         ax_rsi.grid(True)
    #         ax_rsi.legend()

    #         # 绘制成交量
    #         ax_vol.bar(hist_data.index, hist_data['volume'], label='成交量')
    #         ax_vol.plot(hist_data.index, indicators['volume_ma5'], label='MA5', color='red')
    #         ax_vol.set_title('成交量分析')
    #         ax_vol.grid(True)
    #         ax_vol.legend()

    #         # 绘制价格趋势
    #         ax_trend.plot(hist_data.index, hist_data['close'], label='收盘价')
    #         ax_trend.plot(hist_data.index, indicators['ma5'], label='5日均线')
    #         ax_trend.set_title('价格趋势')
    #         ax_trend.grid(True)
    #         ax_trend.legend()

    #         # 绘制波动趋势
    #         daily_return = hist_data['close'].pct_change()
    #         ax_wave.plot(hist_data.index, daily_return.rolling(5).std(), label='5日波动率')
    #         ax_wave.set_title('波动趋势')
    #         ax_wave.grid(True)
    #         ax_wave.legend()

    #         # 调整布局（移除 tight_layout，使用 constrained_layout）
    #         # plt.tight_layout()  # 删除这行

    #         # 保存图表
    #         output_file = os.path.join(output_dir, f'{stock_code}_{stock_name}_analysis.png')
    #         plt.savefig(output_file, dpi=100, bbox_inches='tight')
    #         plt.close()

    #         logger.info(f"已生成 {stock_code} {stock_name} 的技术分析图表")

    # except Exception as e:
    #     logger.error(f"生成技术分析图表时出错: {str(e)}")

def main():
    global args
    global isall
    global is_historical
    global output_dir

    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description=PROGRAM_DESC,
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-k', type=int, default=10, help='显示前K只股票')
    parser.add_argument('-d','--days', type=int, help='从当前时间往回倒退d天开始分析（不能少于30天，默认30天）')
    parser.add_argument('-p', '--parallel', type=int, default=8, help='并行处理的进程数')
    parser.add_argument('--start-date', type=str, help='指定开始日期 (YYYYMMDD)，不能与-d同时使用')
    parser.add_argument('--end-date', type=str, help='指定结束日期 (YYYYMMDD)，不能与-d同时使用')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    parser.add_argument('-g', '--topgains', action='store_true', help='按照区间涨幅排序（仅在分析多只股票时有效）')
    parser.add_argument('-c', '--codes', type=str, help='指定股票代码（用逗号分隔多个代码）')
    parser.add_argument('--ai', action='store_true', help='启用AI分析功能, 需要自己配置config.toml, 请参考:config.example.toml')
    parser.add_argument('--all', action='store_true', help='分析所有股票（默认只分析上证和深证股票）')

    # 解析命令行参数后，设置全局变量
    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)

    # 清除所有现有的处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if args.debug else logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    # 添加处理器
    logger.addHandler(console_handler)

    # 创建文件处理器
    #file_handler = logging.FileHandler('stock_analysis.log', encoding='utf-8')
    #file_handler.setLevel(logging.DEBUG)
    #file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    #file_handler.setFormatter(file_formatter)
    # 添加处理器
    #logger.addHandler(file_handler)

    # 显示免责声明
    logger.info(DISCLAIMER)

    # 参数验证
    if args.k <= 0:
        logger.error("K必须大于0")
        return

    # 检查时间参数冲突
    if args.days and (args.start_date or args.end_date):
        logger.error("不能同时使用-d参数和--start-date/--end-date参数")
        return

    if not args.start_date and args.end_date:
        logger.error("必须提供开始日期: --start-date")
        return

    if args.ai and (args.start_date or args.end_date):
        logger.warning("AI分析仅对实时数据时有意义，对于历史查询结果无效")
        logger.warning("如果希望执行AI分析，请使用'-d'参数")

    # 设置日期范围
    end_date = args.end_date if args.end_date else datetime.now().strftime('%Y%m%d')
    isall=False
    if args.all:
        if args.codes:
            logger.error("不能同时使用--all参数和--codes参数")
            sys.exit(1)
        isall=True

    is_historical = bool(args.start_date or args.end_date)
    if args.start_date:
        start_date = args.start_date
        # 计算时间间隔
        delta = (datetime.strptime(end_date, '%Y%m%d') -
                datetime.strptime(start_date, '%Y%m%d')).days
        if delta < MIN_DAYS:
            logger.error(f"时间间隔不能少于{MIN_DAYS}天")
            return
    else:
        days = MIN_DAYS if args.days is None else args.days
        start_date = (datetime.strptime(end_date, '%Y%m%d') -
                     timedelta(days)).strftime('%Y%m%d')

    logger.info("开始分析股票...")
    logger.info(f"日期范围: {start_date} 至 {end_date}")

    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join('data_archive', timestamp)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"输出目录: {output_dir}")

    # 设置matplotlib样式
    set_plot_style()

    # 获取股票列表
    logger.info("正在获取股票列表...")
    try:
        all_stocks = ak.stock_zh_a_spot_em()
    except Exception as e:
        logger.error(f"获取股票列表失败: {str(e)}")
        sys.exit(1)

    # 处理用户指定的股票代码
    if args.codes:
        # 将输入的股票代码字符串转换为列表
        stock_codes = args.codes.split(',')
        stock_list = all_stocks[all_stocks['代码'].isin(stock_codes)][['代码', '名称']]
        if len(stock_list) < len(stock_codes):
            missing_codes = set(stock_codes) - set(stock_list['代码'])
            logger.warning(f"以下股票代码未找到: {', '.join(missing_codes)}")
    else:
        # 获取所有股票代码
        stock_list = all_stocks[['代码', '名称']]
        logger.info(f"股票总数： {len(stock_list)}")
        st_stocks = stock_list[stock_list['名称'].str.contains('ST|退市')]
        stock_list = stock_list[~stock_list['名称'].str.contains('ST|退市')]
        logger.info(f"排除 'ST以及退市' 股票 {len(st_stocks)} 只, 有效股票 {len(stock_list)} 只")
        # 非'ST|退市'股票
        sh_stocks = stock_list[stock_list['代码'].str.startswith('6') & ~stock_list['代码'].str.startswith('688')]  # 上证主板
        sz_stocks = stock_list[stock_list['代码'].str.startswith('0')]  # 深证主板
        kc_stocks = stock_list[stock_list['代码'].str.startswith('688')]  # 科创板
        cy_stocks = stock_list[stock_list['代码'].str.startswith('3')]  # 创业板
        bj_stocks = stock_list[stock_list['代码'].str.startswith('8') & ~stock_list['代码'].str.startswith('688')]  # 北交所
        b_stocks = stock_list[stock_list['代码'].str.startswith('2') | stock_list['代码'].str.startswith('9')]  # B股
        other_stocks = stock_list[~stock_list['代码'].str.startswith(('0', '2', '3', '6', '8', '9'))]  # 其他股票
        if isall:
            stock_list = stock_list
            logger.info(f"上证主板 {len(sh_stocks)} 只，深证主板 {len(sz_stocks)} 只，"
                        f"科创板 {len(kc_stocks)} 只，创业板 {len(cy_stocks)} 只，"
                        f"北交所 {len(bj_stocks)} 只，B股 {len(b_stocks)} 只，"
                        f"其他股票 {len(other_stocks)} 只")
            logger.info(f"共计 {len(stock_list)} 只")
        else:
            # 使用pd.concat()合并上证和深证股票
            stock_list = pd.concat([sh_stocks, sz_stocks], ignore_index=True)
            logger.info(f"上证主板 {len(sh_stocks)} 只，深证主板 {len(sz_stocks)} 只, 共计 {len(stock_list)} 只")

    # 分析股票
    results = analyze_stocks(stock_list, start_date, end_date, args.parallel)
    # 处理结果
    df = process_results(results)

    # 保存结果
    if not df.empty:
        output_file = os.path.join(output_dir, 'analysis_results.csv')
        df.to_csv(output_file, index=False, encoding='utf_8_sig')
        logger.info(f"分析结果已保存至: {output_file}")

        # 生成图表
        generate_stock_charts(df, start_date, end_date, output_dir)

        # 如果启用了AI分析
        if args.ai:
            if args.start_date or args.end_date:
                logger.warning("AI分析仅对实时数据时有意义，对于历史查询结果无效")
                logger.warning("如果希望执行AI分析，请使用'-d'参数")
                return
            logger.info("正在执行AI分析...")
            try:
                # 只处理生成图表的股票
                chart_stocks = df.head(args.k)
                # 为每个股票创建单独的AI分析文件
                for _, stock in chart_stocks.iterrows():
                    stock_code = stock['代码']
                    stock_name = stock['名称']
                    # 获取图表文件路径
                    chart_file = os.path.join(output_dir, f"{stock_code}_{stock_name}_analysis.png")
                    # 调用AI分析，仅传递必要信息
                    ai_result = get_ai_analysis(chart_file)
                    # 保存为单独的文件
                    ai_output_file = os.path.join(output_dir, f'{stock_code}_{stock_name}_ai_analysis.txt')
                    save_ai_analysis(ai_result, ai_output_file)
                    logger.info(f"已保存 {stock_code} {stock_name} 的 AI 分析结果")
            except Exception as e:
                logger.error(f"AI分析失败: {str(e)}")

    logger.info("分析完成！")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
        logger.debug(traceback.format_exc())
