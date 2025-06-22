import io
import os
import pprint
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
import akshare.utils.func

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
from foreign_stock import serialize_dataframe, deserialize_dataframe

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

def get_historical_data(stock_code: str, start_date: str, end_date: str, debug: bool = False) -> pd.DataFrame:
    """
    获取指定股票的历史数据
    
    Args:
        stock_code: 股票代码
        start_date: 开始日期，格式为'YYYY-MM-DD'
        end_date: 结束日期，格式为'YYYY-MM-DD'
        debug: 是否启用调试模式
    
    Returns:
        pd.DataFrame: 包含开盘价、收盘价、最高价、最低价和成交量的历史数据
    """
    try:
        # 从参数获取调试模式状态
        use_cache = debug
        
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
            hist_data = ak.stock_us_hist(symbol=stock_code,
                                     start_date=start_date,
                                     end_date=end_date)
        if hist_data is None or hist_data.empty:
            return pd.DataFrame()

        logger.debug("结束获取数据")
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
    if data.empty:
        return {}

    # MA
    data['ma5'] = data['close'].rolling(5).mean()
    data['ma10'] = data['close'].rolling(10).mean()
    data['ma20'] = data['close'].rolling(20).mean()

    # MACD
    exp12 = data['close'].ewm(span=12, adjust=False).mean()
    exp26 = data['close'].ewm(span=26, adjust=False).mean()
    data['dif'] = exp12 - exp26
    data['dea'] = data['dif'].ewm(span=9, adjust=False).mean()
    data['macd'] = 2 * (data['dif'] - data['dea'])

    # KDJ
    low_9 = data['low'].rolling(9).min()
    high_9 = data['high'].rolling(9).max()
    rsv = (data['close'] - low_9) / (high_9 - low_9) * 100
    data['k'] = rsv.ewm(span=3, adjust=False).mean()
    data['d'] = data['k'].ewm(span=3, adjust=False).mean()
    data['j'] = 3 * data['k'] - 2 * data['d']

    # RSI
    diff = data['close'].diff()
    gain = diff.clip(lower=0).rolling(14).mean()
    loss = (-diff.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    data['rsi'] = 100 - (100 / (1 + rs))

    # 成交量 MA5
    data['volume_ma5'] = data['volume'].rolling(5).mean()

    return {col: data[col] for col in
            ['ma5', 'ma10', 'ma20', 'dif', 'dea', 'macd',
             'k', 'd', 'j', 'rsi', 'volume_ma5']}

def calculate_stock_score(hist: pd.DataFrame,
                          ind: Dict[str, pd.Series]) -> float:
    try:
        if hist.empty or len(hist) < 15:
            return 0.0

        vol = hist['close'].pct_change().std()
        weights = ({'macd': .25, 'kdj': .25, 'rsi': .20, 'ma': .20, 'volume': .10}
                   if vol > .05 else
                   {'macd': .20, 'kdj': .20, 'rsi': .15, 'ma': .30, 'volume': .15})

        # ========== MACD ==========
        macd_score = 0
        if ind['macd'].iloc[-1] > 0:
            macd_score += 10
        if (ind['dif'].iloc[-1] > ind['dea'].iloc[-1] and
                ind['dif'].iloc[-2] <= ind['dea'].iloc[-2]):
            macd_score += 10
        macd_score *= weights['macd']

        # ========== KDJ ==========
        kdj_score = 0
        if ind['k'].iloc[-1] < 30 and ind['d'].iloc[-1] < 30:
            kdj_score += 10
        if ind['j'].iloc[-1] < 20:
            kdj_score += 5
        if (ind['k'].iloc[-1] > ind['d'].iloc[-1] and
                ind['k'].iloc[-2] <= ind['d'].iloc[-2]):
            kdj_score += 5
        kdj_score *= weights['kdj']

        # ========== RSI ==========
        rsi_val = ind['rsi'].iloc[-1]
        rsi_score = (20 if rsi_val < 30 else
                     15 if rsi_val < 40 else
                     10 if rsi_val < 50 else 0)
        rsi_score *= weights['rsi']

        # ========== 均线 ==========
        ma_score = 0
        if (ind['ma5'].iloc[-1] > ind['ma10'].iloc[-1] >
                ind['ma20'].iloc[-1]):
            ma_score += 10
        last_price = hist['close'].iloc[-1]
        if last_price > ind['ma5'].iloc[-1]:
            ma_score += 4
        if last_price > ind['ma10'].iloc[-1]:
            ma_score += 3
        if last_price > ind['ma20'].iloc[-1]:
            ma_score += 3
        ma_score *= weights['ma']

        # ========== 成交量 ==========
        vol_score = 0
        if hist['volume'].iloc[-1] > ind['volume_ma5'].iloc[-1]:
            vol_score += 10
        if (len(hist) >= 3 and
                hist['volume'].iloc[-1] > hist['volume'].iloc[-2] >
                hist['volume'].iloc[-3]):
            vol_score += 10
        vol_score *= weights['volume']

        # ========== 趋势 ==========
        trend_score = 0
        if len(hist) >= 5:
            short_term = hist['close'].iloc[-5:].pct_change().mean()
            long_term = (hist['close'].iloc[-20:].pct_change().mean()
                         if len(hist) >= 20 else
                         hist['close'].pct_change().mean())
            if short_term > long_term:
                trend_score += 10
            if hist['close'].iloc[-1] > hist['close'].iloc[-5]:
                trend_score += 5
            if len(hist) >= 20 and hist['close'].iloc[-1] > hist['close'].iloc[-20]:
                trend_score += 5

        # ========== 风险调整 ==========
        risk_adj = 1.0
        if vol > .1:
            risk_adj *= .9
        elif vol < .05:
            risk_adj *= 1.1

        max_dd = ((hist['close'].cummax() - hist['close']).max() /
                  hist['close'].cummax().max())
        if max_dd > .2:
            risk_adj *= .9

        total = (macd_score + kdj_score + rsi_score +
                 ma_score + vol_score + trend_score) * risk_adj

        recent = hist['close'].iloc[-1] / hist['close'].iloc[0] - 1
        if recent > .1:
            total *= 1.1
        elif recent < -.1:
            total *= .9

        return round(min(total, 100), 2)

    except Exception as e:
        logger.error(f'计算评分失败: {e}')
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
    tmp = len(stock_list)
    logger.info(f"目前有 {tmp} 只股票...")
    # 过滤掉无效的股票代码

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
                file_path = 'all_stocks.pkl'
                if os.path.exists(file_path):
                    current_info = deserialize_dataframe(file_path)
                else:
                    domestic_info = ak.stock_zh_a_spot_em()
                    foreign_info = ak.stock_us_spot_em()
                    current_info = pd.concat([domestic_info, foreign_info], axis=0, ignore_index=True)
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

        df.info()
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
            # 历史数据没有'总市值（亿元）'列，其值是'N/A'，这里不做处理
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
        hist_data = get_historical_data(stock_code, start_date, end_date, debug=args.debug)
        if hist_data.empty or len(hist_data) < 15:  # 新增数据长度检查
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

def generate_stock_charts(
    stocks: pd.DataFrame,
    start_date: str,
    end_date: str,
    output_dir: str,
    k: int = 10
) -> None:
    try:
        stocks = stocks.head(k)

        for _, stock in stocks.iterrows():
            code, name = stock['代码'], stock['名称']
            hist = get_historical_data(code, start_date, end_date)
            if hist.empty:
                logger.debug(f"股票 {code} {name} 的历史数据为空，跳过分析")
                continue
            ind = calculate_technical_indicators(hist)
            if not ind:
                logger.debug(f"股票 {code} {name} 的技术指标计算失败，跳过分析")
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
                name='价格走势',
                customdata=np.stack((
                    hist.index.strftime('%Y-%m-%d'),
                    hist['open'],
                    hist['high'],
                    hist['low'],
                    hist['close'],
                    hist['volume'],
                    ind['ma5'],
                    ind['ma10'],
                    ind['ma20'],
                    ind['dif'],
                    ind['dea'],
                    ind['macd'],
                    ind['k'],
                    ind['d'],
                    ind['j'],
                    ind['rsi'],
                    ['K线'] * len(hist.index)  # 添加数据类型标识
                ), axis=-1),
                # text=[
                #     f"日期: {date}<br>" +
                #     f"开盘: ¥{open:.2f}<br>" +
                #     f"最高: ¥{high:.2f}<br>" +
                #     f"最低: ¥{low:.2f}<br>" +
                #     f"收盘: ¥{close:.2f}<br>" +
                #     f"成交量: {volume:,.0f}<br>" +
                #     f"MA5: ¥{ma5:.2f}<br>" +
                #     f"MA10: ¥{ma10:.2f}<br>" +
                #     f"MA20: ¥{ma20:.2f}"
                #     for date, open, high, low, close, volume, ma5, ma10, ma20, *_ in zip(
                #         hist.index.strftime('%Y-%m-%d'),
                #         hist['open'],
                #         hist['high'],
                #         hist['low'],
                #         hist['close'],
                #         hist['volume'],
                #         ind['ma5'],
                #         ind['ma10'],
                #         ind['ma20'],
                #         ind['dif'],
                #         ind['dea'],
                #         ind['macd'],
                #         ind['k'],
                #         ind['d'],
                #         ind['j'],
                #         ind['rsi']
                #     )
                # ],
                # hoverinfo='text'
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
                    showlegend=False,
                    customdata=np.stack((
                        hist.index.strftime('%Y-%m-%d'),
                        hist['open'],
                        hist['high'],
                        hist['low'],
                        hist['close'],
                        hist['volume'],
                        ind['ma5'],
                        ind['ma10'],
                        ind['ma20'],
                        ind['dif'],
                        ind['dea'],
                        ind['macd'],
                        ind['k'],
                        ind['d'],
                        ind['j'],
                        ind['rsi'],
                        [legend] * len(hist.index)  # 添加数据类型标识
                    ), axis=-1),
                ), row=1, col=1)

            # 添加点击事件配置
            fig.update_layout(
                clickmode='event+select',
                updatemenus=[{
                    'buttons': [],
                    'showactive': False,
                    'type': 'buttons',
                    'direction': 'right',
                    'visible': False  # 隐藏按钮，只使用点击事件功能
                }]
            )

            # 添加JavaScript回调函数来处理点击事件
            fig.add_annotation(
                text='',
                showarrow=False,
                font=dict(size=12),
                xref='paper',
                yref='paper',
                x=0,
                y=1.1,
                bordercolor='#c7c7c7',
                borderwidth=1,
                borderpad=4,
                bgcolor='#ff7f0e',
                opacity=0.8
            )

            # 添加JavaScript代码来处理点击事件
            fig.update_layout(
                newshape_line_color='#ff0000',
                annotations=[{
                    'text': '',
                    'showarrow': False,
                    'font': {'size': 12},
                    'xref': 'paper',
                    'yref': 'paper',
                    'x': 0,
                    'y': 1.1,
                    'bordercolor': '#c7c7c7',
                    'borderwidth': 1,
                    'borderpad': 4,
                    'bgcolor': '#ff7f0e',
                    'opacity': 0.8
                }]
            )

            # ---- 第2行左：纯净MACD ----
            fig.add_trace(go.Bar(
                x=hist.index,
                y=ind['macd'],
                marker_color=['#2ecc71' if v <0 else '#e74c3c' for v in ind['macd']],
                name='MACD柱',
                showlegend=False,
                customdata=np.stack((
                    hist.index.strftime('%Y-%m-%d'),
                    hist['open'],
                    hist['high'],
                    hist['low'],
                    hist['close'],
                    hist['volume'],
                    ind['ma5'],
                    ind['ma10'],
                    ind['ma20'],
                    ind['dif'],
                    ind['dea'],
                    ind['macd'],
                    ind['k'],
                    ind['d'],
                    ind['j'],
                    ind['rsi'],
                    ['MACD'] * len(hist.index)  # 添加数据类型标识
                ), axis=-1),
            ), row=2, col=1)

            fig.add_trace(go.Scatter(
                x=hist.index,
                y=ind['dif'],
                line=dict(color='#3498db', width=1.5),
                name='DIF',
                showlegend=False,
                customdata=np.stack((
                    hist.index.strftime('%Y-%m-%d'),
                    hist['open'],
                    hist['high'],
                    hist['low'],
                    hist['close'],
                    hist['volume'],
                    ind['ma5'],
                    ind['ma10'],
                    ind['ma20'],
                    ind['dif'],
                    ind['dea'],
                    ind['macd'],
                    ind['k'],
                    ind['d'],
                    ind['j'],
                    ind['rsi'],
                    ['DIF'] * len(hist.index)  # 添加数据类型标识
                ), axis=-1),
            ), row=2, col=1)

            fig.add_trace(go.Scatter(
                x=hist.index,
                y=ind['dea'],
                line=dict(color='#f1c40f', width=1.5),
                name='DEA',
                showlegend=False,
                customdata=np.stack((
                    hist.index.strftime('%Y-%m-%d'),
                    hist['open'],
                    hist['high'],
                    hist['low'],
                    hist['close'],
                    hist['volume'],
                    ind['ma5'],
                    ind['ma10'],
                    ind['ma20'],
                    ind['dif'],
                    ind['dea'],
                    ind['macd'],
                    ind['k'],
                    ind['d'],
                    ind['j'],
                    ind['rsi'],
                    ['DEA'] * len(hist.index)  # 添加数据类型标识
                ), axis=-1),
            ), row=2, col=1)

            # ---- 第2行右：纯净KDJ ----
            fig.add_trace(go.Scatter(
                x=hist.index,
                y=ind['k'],
                line=dict(color='#e74c3c', width=1.5),
                name='K',
                showlegend=False,
                customdata=np.stack((
                    hist.index.strftime('%Y-%m-%d'),
                    hist['open'],
                    hist['high'],
                    hist['low'],
                    hist['close'],
                    hist['volume'],
                    ind['ma5'],
                    ind['ma10'],
                    ind['ma20'],
                    ind['dif'],
                    ind['dea'],
                    ind['macd'],
                    ind['k'],
                    ind['d'],
                    ind['j'],
                    ind['rsi'],
                    ['KDJ-K'] * len(hist.index)  # 添加数据类型标识
                ), axis=-1),
            ), row=2, col=2)

            fig.add_trace(go.Scatter(
                x=hist.index,
                y=ind['d'],
                line=dict(color='#2ecc71', width=1.5),
                name='D',
                showlegend=False,
                customdata=np.stack((
                    hist.index.strftime('%Y-%m-%d'),
                    hist['open'],
                    hist['high'],
                    hist['low'],
                    hist['close'],
                    hist['volume'],
                    ind['ma5'],
                    ind['ma10'],
                    ind['ma20'],
                    ind['dif'],
                    ind['dea'],
                    ind['macd'],
                    ind['k'],
                    ind['d'],
                    ind['j'],
                    ind['rsi'],
                    ['KDJ-D'] * len(hist.index)  # 添加数据类型标识
                ), axis=-1),
            ), row=2, col=2)

            fig.add_trace(go.Scatter(
                x=hist.index,
                y=ind['j'],
                line=dict(color='#3498db', width=1.5),
                name='J',
                showlegend=False,
                customdata=np.stack((
                    hist.index.strftime('%Y-%m-%d'),
                    hist['open'],
                    hist['high'],
                    hist['low'],
                    hist['close'],
                    hist['volume'],
                    ind['ma5'],
                    ind['ma10'],
                    ind['ma20'],
                    ind['dif'],
                    ind['dea'],
                    ind['macd'],
                    ind['k'],
                    ind['d'],
                    ind['j'],
                    ind['rsi'],
                    ['KDJ-J'] * len(hist.index)  # 添加数据类型标识
                ), axis=-1),
            ), row=2, col=2)

            # ---- 第3行：RSI指标 ----
            fig.add_trace(go.Scatter(
                x=hist.index,
                y=ind['rsi'],
                line=dict(color='#9B59B6', width=1.2),
                name='RSI',
                customdata=np.stack((
                    hist.index.strftime('%Y-%m-%d'),
                    hist['open'],
                    hist['high'],
                    hist['low'],
                    hist['close'],
                    hist['volume'],
                    ind['ma5'],
                    ind['ma10'],
                    ind['ma20'],
                    ind['dif'],
                    ind['dea'],
                    ind['macd'],
                    ind['k'],
                    ind['d'],
                    ind['j'],
                    ind['rsi'],
                    ['RSI'] * len(hist.index)  # 添加数据类型标识
                ), axis=-1),
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
                name='成交量',
                customdata=np.stack((
                    hist.index.strftime('%Y-%m-%d'),
                    hist['open'],
                    hist['high'],
                    hist['low'],
                    hist['close'],
                    hist['volume'],
                    ind['ma5'],
                    ind['ma10'],
                    ind['ma20'],
                    ind['dif'],
                    ind['dea'],
                    ind['macd'],
                    ind['k'],
                    ind['d'],
                    ind['j'],
                    ind['rsi'],
                    ['成交量'] * len(hist.index)  # 添加数据类型标识
                ), axis=-1),
            ), row=3, col=2)
            fig.add_trace(go.Scatter(
                x=hist.index,
                y=ind['volume_ma5'],
                line=dict(color='#F39C12', width=1.2),
                name='成交量MA5',
                customdata=np.stack((
                    hist.index.strftime('%Y-%m-%d'),
                    hist['open'],
                    hist['high'],
                    hist['low'],
                    hist['close'],
                    hist['volume'],
                    ind['ma5'],
                    ind['ma10'],
                    ind['ma20'],
                    ind['dif'],
                    ind['dea'],
                    ind['macd'],
                    ind['k'],
                    ind['d'],
                    ind['j'],
                    ind['rsi'],
                    ['成交量MA5'] * len(hist.index)  # 添加数据类型标识
                ), axis=-1),
            ), row=3, col=2)

            # ---- 第4行：价格趋势 ----
            fig.add_trace(go.Scatter(
                x=hist.index,
                y=hist['close'],
                line=dict(color='#2C3E50', width=1.5),
                name='收盘价',
                customdata=np.stack((
                    hist.index.strftime('%Y-%m-%d'),
                    hist['open'],
                    hist['high'],
                    hist['low'],
                    hist['close'],
                    hist['volume'],
                    ind['ma5'],
                    ind['ma10'],
                    ind['ma20'],
                    ind['dif'],
                    ind['dea'],
                    ind['macd'],
                    ind['k'],
                    ind['d'],
                    ind['j'],
                    ind['rsi'],
                    ['收盘价'] * len(hist.index)  # 添加数据类型标识
                ), axis=-1),
            ), row=4, col=1)

            # ---- 第4行：波动率分析 ----
            fig.add_trace(go.Scatter(
                x=hist.index,
                y=hist['close'].pct_change().rolling(5).std(),
                line=dict(color='#E67E22', width=1.2),
                name='波动率',
                customdata=np.stack((
                    hist.index.strftime('%Y-%m-%d'),
                    hist['open'],
                    hist['high'],
                    hist['low'],
                    hist['close'],
                    hist['volume'],
                    ind['ma5'],
                    ind['ma10'],
                    ind['ma20'],
                    ind['dif'],
                    ind['dea'],
                    ind['macd'],
                    ind['k'],
                    ind['d'],
                    ind['j'],
                    ind['rsi'],
                    ['波动率'] * len(hist.index)  # 添加数据类型标识
                ), axis=-1),
            ), row=4, col=2)

            # 显示所有行的x轴（关键修改）
            for row in [2,3,4]:
                fig.update_xaxes(
                    showticklabels=True,  # 显示刻度标签
                    showgrid=True,        # 显示网格线
                    showline=True,        # 显示轴线
                    linecolor='rgba(200,200,200,0.5)',  # 轴线颜色
                    row=row,
                    col=1
                )
                fig.update_xaxes(
                    showticklabels=True,
                    showgrid=True,
                    showline=True,
                    linecolor='rgba(200,200,200,0.5)',
                    row=row,
                    col=2
                )

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
                xaxis=dict(
                    rangeslider=dict(visible=False)  # 确保禁用范围选择器
                )
            )

            # 增强网格线
            fig.update_xaxes(showgrid=True, gridcolor='rgba(200,200,200,0.2)',rangeslider_visible=False)
            fig.update_yaxes(showgrid=True, gridcolor='rgba(200,200,200,0.2)')

            # 生成标准化文件名
            # ----- 生成标准化文件名 -----
            clean_name = re.sub(r'[\\/*?:"<>|]', "", name)

            filename = f"{code}_analysis.html" if clean_name.lower() == str(code).lower() \
                    else f"{code}_{clean_name}_analysis.html"
            html_path = os.path.join(output_dir, filename)

            # 保存文件
            with open(html_path, 'w', encoding='utf-8') as f:
                html_content = fig.to_html(
                    include_plotlyjs='cdn',
                    config={'scrollZoom': True}
                )
                
                # 添加自定义JavaScript代码
                custom_js = """
                <script>
                    document.addEventListener('DOMContentLoaded', function() {
                        // 创建输入控件面板 - 放在右上角
                        var controlPanel = document.createElement('div');
                        controlPanel.id = 'control-panel';
                        controlPanel.style.cssText = 'position:fixed;top:20px;right:20px;background:white;padding:15px;border:1px solid #ccc;border-radius:5px;box-shadow:2px 2px 10px rgba(0,0,0,0.1);z-index:1000;';
                        controlPanel.innerHTML = `
                            <h4 style="margin-top:0;color:#2C3E50;">分析设置</h4>
                            <div style="margin-bottom:10px;">
                                <label for="days-input" style="margin-right:10px;">前后天数:</label>
                                <input type="number" id="days-input" value="3" min="0" max="10" style="width:60px;padding:2px;">
                                <small style="display:block;margin-top:3px;color:#666;">设置点击时显示前后n天的数据</small>
                            </div>
                            <div style="margin-bottom:10px;">
                                <label for="debate-round-input" style="margin-right:10px;">辩论轮数:</label>
                                <input type="number" id="debate-round-input" value="2" min="1" max="5" style="width:60px;padding:2px;">
                                <small style="display:block;margin-top:3px;color:#666;">设置AI智能体辩论的轮数</small>
                            </div>
                        `;
                        document.body.appendChild(controlPanel);
                        
                        // 切换数据详情的可见性
                        function toggleDataDetails(dayId) {
                            var detailsDiv = document.getElementById('details-' + dayId);
                            var toggleBtn = document.getElementById('toggle-' + dayId);
                            if (detailsDiv.style.display === 'none') {
                                detailsDiv.style.display = 'block';
                                toggleBtn.textContent = '▼';
                            } else {
                                detailsDiv.style.display = 'none';
                                toggleBtn.textContent = '▶';
                            }
                        }
                        
                        // 切换分析结果的展开/折叠
                        function toggleAnalysisResult(resultId) {
                            var previewDiv = document.getElementById('preview-' + resultId);
                            var fullDiv = document.getElementById('full-' + resultId);
                            var toggleBtn = document.getElementById('toggle-' + resultId);
                            var streamDiv = document.getElementById('analysis-stream');
                            
                            if (fullDiv.style.display === 'none') {
                                // 展开完整结果
                                previewDiv.style.display = 'none';
                                fullDiv.style.display = 'block';
                                toggleBtn.textContent = '▲';
                            } else {
                                // 折叠显示预览
                                previewDiv.style.display = 'block';
                                fullDiv.style.display = 'none';
                                toggleBtn.textContent = '▼';
                            }
                            
                            // 展开/折叠后自动调整滚动位置
                            setTimeout(function() {
                                if (streamDiv) {
                                    streamDiv.scrollTop = streamDiv.scrollHeight;
                                }
                            }, 100);
                        }
                        
                        // 关闭数据面板并显示控制面板
                        function closeDataPanel() {
                            var dataPanel = document.getElementById('data-panel');
                            var controlPanel = document.getElementById('control-panel');
                            if (dataPanel) {
                                dataPanel.style.display = 'none';
                            }
                            if (controlPanel) {
                                controlPanel.style.display = 'block';
                            }
                        }
                        
                        // 智能分析功能
                        function performSmartAnalysis() {
                            var analysisBtn = document.getElementById('analysis-btn');
                            var analysisResult = document.getElementById('analysis-result');
                            
                            // 显示加载状态
                            analysisBtn.disabled = true;
                            analysisBtn.textContent = '分析中...';
                            analysisBtn.style.background = '#95A5A6';
                            
                            // 获取当前选中数据点的历史数据
                            var currentData = window.currentAnalysisData;
                            if (!currentData) {
                                console.error('没有可用的分析数据');
                                return;
                            }
                            
                            // 获取用户设置的辩论轮数
                            var debateRounds = parseInt(document.getElementById('debate-round-input').value) || 2;
                            
                            // 初始化流式显示界面
                            analysisResult.innerHTML = `
                                <div style="background:#f8f9fa;border-radius:5px;padding:15px;margin-top:10px;">
                                    <h4 style="margin-top:0;color:#2C3E50;border-bottom:1px solid #dee2e6;padding-bottom:8px;">
                                        🤖 AI多智能体分析进行中 (${debateRounds}轮辩论)
                                    </h4>
                                    <div id="analysis-stream" style="background:white;padding:15px;border-radius:3px;border:1px solid #dee2e6;min-height:150px;max-height:400px;overflow-y:auto;">
                                        <div class="status-message" style="color:#666;font-style:italic;">正在连接分析服务...</div>
                                    </div>
                                    <div style="margin-top:10px;font-size:12px;color:#666;">
                                        💡 提示: 多个AI智能体将进行${debateRounds}轮协作分析，请耐心等待完整结果
                                    </div>
                                </div>
                            `;
                            
                            var streamDiv = document.getElementById('analysis-stream');
                            
                            // 按照新的数据格式准备发送的数据
                            var requestData = [];
                            
                            // 添加每一天的数据
                            currentData.priceData.forEach(function(dayPrice, index) {
                                var dayData = {
                                    "type": "price_historical",
                                    "date": dayPrice.date,
                                    "data": {
                                        "symbol": currentData.symbol,
                                        "name": currentData.name,
                                        "open": dayPrice.open,
                                        "high": dayPrice.high,
                                        "low": dayPrice.low,
                                        "close": dayPrice.close,
                                        "volume": dayPrice.volume
                                    }
                                };
                                
                                // 如果是选中的日期，添加技术指标
                                if (dayPrice.date === currentData.selectedDate) {
                                    dayData.data.indicators = currentData.indicators;
                                }
                                
                                requestData.push(dayData);
                            });
                            
                            // 添加配置信息作为最后一个元素
                            requestData.push({
                                "debate_round": debateRounds,
                                "selected_date": currentData.selectedDate,
                                "selected_data": currentData.selectedDataType || "K线"
                            });
                            
                            // 调试：打印发送的数据
                            console.log('发送智能分析数据:', requestData);
                            
                            // 使用fetch发起POST请求启动SSE流
                            fetch('http://localhost:8000/api/v1/debate/stream', {
                                method: 'POST',
                                headers: {
                                    'Content-Type': 'application/json',
                                    'Accept': 'text/event-stream',
                                    'Cache-Control': 'no-cache'
                                },
                                body: JSON.stringify(requestData)
                            })
                            .then(response => {
                                if (!response.ok) {
                                    throw new Error(`HTTP error! status: ${response.status}`);
                                }
                                
                                // 处理SSE流
                                const reader = response.body.getReader();
                                const decoder = new TextDecoder();
                                
                                function processStream() {
                                    return reader.read().then(({ done, value }) => {
                                        if (done) {
                                            // 流结束
                                            analysisBtn.disabled = false;
                                            analysisBtn.textContent = '智能分析';
                                            analysisBtn.style.background = '#3498DB';
                                            
                                            var completeDiv = document.createElement('div');
                                            completeDiv.style.cssText = 'margin:10px 0;padding:10px;background:#f0fff4;border:1px solid #c6f6d5;border-radius:5px;color:#38a169;font-weight:bold;';
                                            completeDiv.innerHTML = '✅ 分析完成';
                                            streamDiv.appendChild(completeDiv);
                                            streamDiv.scrollTop = streamDiv.scrollHeight;
                                            return;
                                        }
                                        
                                        // 处理接收到的数据
                                        const chunk = decoder.decode(value);
                                        const lines = chunk.split('\\n');
                                        
                                        lines.forEach(line => {
                                            if (line.startsWith('data: ')) {
                                                try {
                                                    const jsonData = line.substring(6);
                                                    if (jsonData.trim()) {
                                                        const eventData = JSON.parse(jsonData);
                                                        handleStreamEvent(eventData, streamDiv);
                                                    }
                                                } catch (e) {
                                                    console.error('解析SSE数据失败:', e);
                                                }
                                            }
                                        });
                                        
                                        // 继续读取
                                        return processStream();
                                    });
                                }
                                
                                return processStream();
                            })
                            .catch(error => {
                                console.error('智能分析请求失败:', error);
                                
                                // 恢复按钮状态
                                analysisBtn.disabled = false;
                                analysisBtn.textContent = '智能分析';
                                analysisBtn.style.background = '#3498DB';
                                
                                // 显示错误信息
                                var errorContent = `
                                    <div style="background:#fff5f5;border:1px solid #fed7d7;border-radius:5px;padding:15px;margin-top:10px;">
                                        <h4 style="margin-top:0;color:#E53E3E;">❌ 分析失败</h4>
                                        <p style="margin:5px 0;color:#666;">无法连接到智能分析服务</p>
                                        <p style="margin:5px 0;font-size:12px;color:#999;">错误详情: ${error.message}</p>
                                        <p style="margin:5px 0;font-size:11px;color:#999;">请确保API服务正在运行在 http://localhost:8000</p>
                                    </div>
                                `;
                                analysisResult.innerHTML = errorContent;
                            });
                        }
                        
                        // 轮次管理变量
                        var currentRoundContainer = null;
                        var currentRoundContent = null;
                        
                        // 处理流式事件的函数
                        function handleStreamEvent(eventData, streamDiv) {
                            // 过滤掉heartbeat事件
                            if (eventData.event === 'heartbeat') {
                                return;
                            }
                            
                            // 特殊处理轮次事件
                            if (eventData.event === 'analysis_round_start') {
                                handleRoundStart(eventData, streamDiv);
                                return;
                            } else if (eventData.event === 'analysis_round_complete') {
                                handleRoundComplete(eventData);
                                return;
                            }
                            
                            var messageDiv = document.createElement('div');
                            messageDiv.style.cssText = 'margin:6px 0;padding:8px 12px;border-radius:5px;animation:fadeIn 0.3s ease-in;';
                            
                            // 判断是否需要展示详细内容
                            var needsContent = [
                                'llm_call_complete', 'llm_call_error',
                                'prepare_inputs_start', 'prepare_inputs_complete', 
                                'decision_criteria_check', 'decision_criteria_result',
                                'finalize_decision_complete', 'analysis_round_complete',
                                'agent_task_complete'
                            ].includes(eventData.event);
                            
                            // 根据事件类别设置不同的视觉样式
                            switch(true) {
                                // 工作流级别事件 - 蓝色系
                                case ['workflow_start', 'workflow_complete', 'workflow_error', 'workflow_result'].includes(eventData.event):
                                    handleWorkflowEvent(eventData, messageDiv, needsContent);
                                    break;
                                
                                // 节点级别事件 - 绿色系
                                case eventData.event.includes('prepare_inputs') || 
                                     eventData.event.includes('decision_criteria') || 
                                     eventData.event.includes('finalize_decision'):
                                    handleNodeEvent(eventData, messageDiv, needsContent);
                                    break;
                                
                                // Agent级别事件 - 紫色/橙色系
                                case ['agent_task_start', 'agent_task_complete', 'llm_call_start', 'llm_call_complete', 'llm_call_error'].includes(eventData.event):
                                    handleAgentEvent(eventData, messageDiv, needsContent);
                                    break;
                                
                                // 连接级别事件 - 灰色系
                                case ['connection_established', 'stream_error'].includes(eventData.event):
                                    handleConnectionEvent(eventData, messageDiv, needsContent);
                                    break;
                                
                                // 其他事件
                                default:
                                    handleOtherEvent(eventData, messageDiv, needsContent);
                            }
                            
                            // 判断是否添加到当前轮次容器中
                            var targetContainer = (currentRoundContent && 
                                ['agent_task_start', 'agent_task_complete', 'llm_call_start', 'llm_call_complete', 'llm_call_error'].includes(eventData.event)) ? 
                                currentRoundContent : streamDiv;
                            
                            targetContainer.appendChild(messageDiv);
                            
                            // 平滑滚动到底部，确保始终显示最新输出
                            setTimeout(function() {
                                streamDiv.scrollTop = streamDiv.scrollHeight;
                                streamDiv.scrollTo({
                                    top: streamDiv.scrollHeight,
                                    behavior: 'smooth'
                                });
                            }, 100);
                        }
                        
                        // 工作流级别事件处理
                        function handleWorkflowEvent(eventData, messageDiv, needsContent) {
                            var icons = {
                                'workflow_start': '🚀',
                                'workflow_complete': '✅', 
                                'workflow_error': '❌',
                                'workflow_result': '📋'
                            };
                            
                            messageDiv.style.background = 'linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%)';
                            messageDiv.style.border = '2px solid #2196f3';
                            messageDiv.style.borderLeft = '5px solid #1976d2';
                            
                            var title = getEventTitle(eventData.event, icons[eventData.event] || '🔄');
                            var description = eventData.data.description || eventData.data.message || '';
                            
                            if (needsContent && description) {
                                var contentId = generateContentId();
                                messageDiv.innerHTML = createExpandableContent(contentId, title, description, '#1976d2');
                            } else {
                                messageDiv.innerHTML = `<div style="font-weight:bold;color:#1976d2;font-size:14px;">${title}</div>`;
                            }
                        }
                        
                        // 节点级别事件处理
                        function handleNodeEvent(eventData, messageDiv, needsContent) {
                            var icons = {
                                'prepare_inputs_start': '📥', 'prepare_inputs_complete': '📥',
                                'analysis_round_start': '🔄', 'analysis_round_complete': '🔄',
                                'decision_criteria_check': '⚖️', 'decision_criteria_result': '⚖️',
                                'finalize_decision_start': '🎯', 'finalize_decision_complete': '🎯'
                            };
                            
                            var title = getEventTitle(eventData.event, icons[eventData.event] || '📋');
                            var content = eventData.data.result || eventData.data.summary || eventData.data.description || eventData.data.message || eventData.data.details || '';
                            
                            // 中间过程事件用灰色小字显示
                            var isProcessEvent = ['prepare_inputs_start', 'finalize_decision_start', 'decision_criteria_check'].includes(eventData.event);
                            
                            if (isProcessEvent) {
                                // 灰色小字显示中间过程
                                messageDiv.style.background = 'transparent';
                                messageDiv.style.border = 'none';
                                messageDiv.style.padding = '4px 8px';
                                messageDiv.style.margin = '2px 0';
                                messageDiv.innerHTML = `<div style="font-size:11px;color:#999;font-style:italic;">${title}</div>`;
                            } else {
                                // 重要完成事件正常显示
                                var isComplete = eventData.event.includes('complete') || eventData.event.includes('result');
                                
                                // 特殊处理 finalize_decision_complete
                                if (eventData.event === 'finalize_decision_complete') {
                                    handleFinalDecision(eventData, messageDiv);
                                    return;
                                }
                                
                                messageDiv.style.background = isComplete ? 
                                    'linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%)' : 
                                    'linear-gradient(135deg, #fff3e0 0%, #ffcc02 20%)';
                                messageDiv.style.border = isComplete ? '2px solid #4caf50' : '2px solid #ff9800';
                                messageDiv.style.borderLeft = isComplete ? '5px solid #388e3c' : '5px solid #f57c00';
                                
                                if (needsContent && content) {
                                    var contentId = generateContentId();
                                    var color = isComplete ? '#388e3c' : '#f57c00';
                                    messageDiv.innerHTML = createExpandableContent(contentId, title, content, color);
                                } else {
                                    var color = isComplete ? '#388e3c' : '#f57c00';
                                    messageDiv.innerHTML = `<div style="font-weight:bold;color:${color};font-size:14px;">${title}</div>`;
                                }
                            }
                        }
                        
                        // Agent级别事件处理  
                        function handleAgentEvent(eventData, messageDiv, needsContent) {
                            var icons = {
                                'agent_task_start': '🤖',
                                'agent_task_complete': '✅',
                                'llm_call_start': '🧠',
                                'llm_call_complete': '💭',
                                'llm_call_error': '❌'
                            };
                            
                            var agentRole = eventData.data.agent_role || '';
                            var title = getEventTitle(eventData.event, icons[eventData.event] || '🔧', agentRole);
                            
                            // 获取内容
                            var content = '';
                            if (eventData.event === 'agent_task_complete') {
                                content = eventData.data.result || eventData.data.analysis || eventData.data.response || eventData.data.output || eventData.data.message || eventData.data.content || '';
                            } else {
                                content = eventData.data.result || eventData.data.summary || eventData.data.error || eventData.data.response || '';
                            }
                            
                            // agent_task_complete, llm_call_complete 和 llm_call_error 需要详细展示
                            if (['agent_task_complete', 'llm_call_complete', 'llm_call_error'].includes(eventData.event)) {
                                var isError = eventData.event.includes('error');
                                var isAgentTask = eventData.event === 'agent_task_complete';
                                
                                if (isError) {
                                    messageDiv.style.background = 'linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%)';
                                    messageDiv.style.border = '2px solid #f44336';
                                    messageDiv.style.borderLeft = '5px solid #d32f2f';
                                } else if (isAgentTask) {
                                    messageDiv.style.background = 'linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%)';
                                    messageDiv.style.border = '2px solid #2196f3';
                                    messageDiv.style.borderLeft = '5px solid #1976d2';
                                } else {
                                    messageDiv.style.background = 'linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%)';
                                    messageDiv.style.border = '2px solid #9c27b0';
                                    messageDiv.style.borderLeft = '5px solid #7b1fa2';
                                }
                                
                                if (needsContent && content) {
                                    var contentId = generateContentId();
                                    var color = isError ? '#d32f2f' : (isAgentTask ? '#1976d2' : '#7b1fa2');
                                    
                                    // 为agent_task_complete特殊格式化内容
                                    if (isAgentTask) {
                                        content = formatAgentAnalysis(content);
                                    }
                                    
                                    messageDiv.innerHTML = createExpandableContent(contentId, title, content, color);
                                } else {
                                    var color = isError ? '#d32f2f' : (isAgentTask ? '#1976d2' : '#7b1fa2');
                                    messageDiv.innerHTML = `<div style="font-weight:bold;color:${color};font-size:14px;">${title}</div>`;
                                }
                            } else {
                                // 其他 Agent 事件：灰色小号字体，简洁显示
                                messageDiv.style.background = 'transparent';
                                messageDiv.style.border = 'none';
                                messageDiv.style.padding = '4px 8px';
                                messageDiv.style.margin = '2px 0';
                                messageDiv.innerHTML = `<div style="font-size:11px;color:#999;font-style:italic;">${title}</div>`;
                            }
                        }
                        
                        // 连接级别事件处理
                        function handleConnectionEvent(eventData, messageDiv, needsContent) {
                            var icons = {
                                'connection_established': '🔗',
                                'stream_error': '⚠️'
                            };
                            
                            messageDiv.style.background = 'linear-gradient(135deg, #f5f5f5 0%, #eeeeee 100%)';
                            messageDiv.style.border = '1px solid #bdbdbd';
                            messageDiv.style.borderLeft = '3px solid #757575';
                            messageDiv.style.fontSize = '12px';
                            messageDiv.style.opacity = '0.8';
                            
                            var title = getEventTitle(eventData.event, icons[eventData.event] || '📡');
                            messageDiv.innerHTML = `<div style="font-weight:bold;color:#616161;">${title}</div>`;
                        }
                        
                        // 其他事件处理
                        function handleOtherEvent(eventData, messageDiv, needsContent) {
                            messageDiv.style.background = '#fafafa';
                            messageDiv.style.border = '1px solid #e0e0e0';
                            messageDiv.style.borderLeft = '3px solid #9e9e9e';
                            
                            var title = getEventTitle(eventData.event, '📝');
                            messageDiv.innerHTML = `<div style="font-weight:bold;color:#666;font-size:13px;">${title}</div>`;
                        }
                        
                        // 辅助函数
                        function getEventTitle(eventType, icon, agentRole = '') {
                            var titles = {
                                'workflow_start': '工作流启动',
                                'workflow_complete': '工作流完成', 
                                'workflow_error': '工作流错误',
                                'workflow_result': '最终结果',
                                'prepare_inputs_start': '准备输入数据',
                                'prepare_inputs_complete': '输入数据就绪',
                                'analysis_round_start': '开始分析轮次',
                                'analysis_round_complete': '分析轮次完成',
                                'decision_criteria_check': '检查决策条件',
                                'decision_criteria_result': '决策条件结果',
                                'finalize_decision_start': '开始最终决策',
                                'finalize_decision_complete': '最终决策完成',
                                'agent_task_start': '智能体任务开始',
                                'agent_task_complete': '智能体任务完成',
                                'llm_call_start': 'LLM调用开始',
                                'llm_call_complete': 'LLM分析完成',
                                'llm_call_error': 'LLM调用错误',
                                'connection_established': '连接已建立',
                                'stream_error': '流处理错误'
                            };
                            
                            var baseTitle = titles[eventType] || eventType.replace(/_/g, ' ');
                            return agentRole ? `${icon} ${agentRole} - ${baseTitle}` : `${icon} ${baseTitle}`;
                        }
                        
                        function generateContentId() {
                            return 'content-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
                        }
                        
                        function createExpandableContent(contentId, title, content, color) {
                            var preview = content.length > 120 ? content.substring(0, 120) + '...' : content;
                            var needsToggle = content.length > 120;
                            
                            if (needsToggle) {
                                return `
                                    <div onclick="toggleContent('${contentId}')" style="cursor:pointer;display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                                        <div style="font-weight:bold;color:${color};font-size:14px;">${title}</div>
                                        <span id="toggle-${contentId}" style="font-weight:bold;color:${color};">▼</span>
                                    </div>
                                    <div id="preview-${contentId}" style="font-size:12px;line-height:1.4;color:#666;">
                                        ${preview}
                                    </div>
                                    <div id="full-${contentId}" style="display:none;font-size:12px;line-height:1.4;white-space:pre-wrap;background:rgba(255,255,255,0.8);padding:10px;border-radius:3px;border:1px solid rgba(0,0,0,0.1);max-height:250px;overflow-y:auto;margin-top:8px;">
                                        ${content}
                                    </div>
                                `;
                            } else {
                                return `
                                    <div style="font-weight:bold;color:${color};font-size:14px;margin-bottom:6px;">${title}</div>
                                    <div style="font-size:12px;line-height:1.4;color:#666;">${content}</div>
                                `;
                            }
                        }
                        
                        // 轮次开始处理
                        function handleRoundStart(eventData, streamDiv) {
                            var roundNumber = eventData.data.round || eventData.data.round_number || '?';
                            var roundId = 'round-' + roundNumber + '-' + Date.now();
                            
                            // 创建轮次容器
                            var roundContainer = document.createElement('div');
                            roundContainer.id = roundId;
                            roundContainer.style.cssText = 'margin:10px 0;border:2px solid #ff9800;border-radius:8px;background:linear-gradient(135deg, #fff8e1 0%, #ffecb3 100%);';
                            
                            // 创建轮次标题（可点击）
                            var roundHeader = document.createElement('div');
                            roundHeader.style.cssText = 'padding:12px 15px;cursor:pointer;border-bottom:1px solid #ffcc02;background:rgba(255,152,0,0.1);';
                            roundHeader.onclick = function() { toggleRound(roundId); };
                            roundHeader.innerHTML = `
                                <div style="display:flex;justify-content:space-between;align-items:center;">
                                    <div style="font-weight:bold;color:#ef6c00;font-size:15px;">🔄 第${roundNumber}轮分析 - 进行中</div>
                                    <span id="toggle-${roundId}" style="font-weight:bold;color:#ef6c00;">▼</span>
                                </div>
                            `;
                            
                            // 创建轮次内容容器
                            var roundContent = document.createElement('div');
                            roundContent.id = 'content-' + roundId;
                            roundContent.style.cssText = 'padding:10px 15px;display:block;';
                            
                            roundContainer.appendChild(roundHeader);
                            roundContainer.appendChild(roundContent);
                            streamDiv.appendChild(roundContainer);
                            
                            // 设置当前轮次
                            currentRoundContainer = roundContainer;
                            currentRoundContent = roundContent;
                        }
                        
                        // 轮次完成处理
                        function handleRoundComplete(eventData) {
                            if (currentRoundContainer) {
                                var roundNumber = eventData.data.round || eventData.data.round_number || '?';
                                var summary = eventData.data.summary || eventData.data.result || eventData.data.analysis || '轮次分析完成';
                                
                                // 更新轮次标题
                                var header = currentRoundContainer.querySelector('div');
                                if (header) {
                                    header.innerHTML = `
                                        <div style="display:flex;justify-content:space-between;align-items:center;">
                                            <div style="font-weight:bold;color:#388e3c;font-size:15px;">✅ 第${roundNumber}轮分析 - 已完成</div>
                                            <span id="toggle-${currentRoundContainer.id}" style="font-weight:bold;color:#388e3c;">▲</span>
                                        </div>
                                    `;
                                }
                                
                                // 添加详细的轮次结果
                                if (summary && summary !== '轮次分析完成') {
                                    var contentId = generateContentId();
                                    var summaryDiv = document.createElement('div');
                                    summaryDiv.style.cssText = 'margin-top:10px;padding:12px;background:rgba(76,175,80,0.1);border-radius:6px;border-left:4px solid #4caf50;';
                                    
                                    // 检查是否是详细的分析内容
                                    var isDetailedAnalysis = summary.length > 200 || summary.includes('##') || summary.includes('ASSESSMENT');
                                    
                                    if (isDetailedAnalysis) {
                                        // 详细分析内容，使用可展开格式
                                        var preview = summary.length > 150 ? summary.substring(0, 150) + '...' : summary;
                                        summaryDiv.innerHTML = `
                                            <div onclick="toggleContent('${contentId}')" style="cursor:pointer;display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                                                <div style="font-weight:bold;color:#388e3c;">📋 轮次分析结果</div>
                                                <span id="toggle-${contentId}" style="font-weight:bold;color:#388e3c;">▼</span>
                                            </div>
                                            <div id="preview-${contentId}" style="font-size:12px;line-height:1.4;color:#666;">
                                                ${preview}
                                            </div>
                                            <div id="full-${contentId}" style="display:none;margin-top:10px;padding:15px;background:rgba(255,255,255,0.9);border-radius:6px;border:1px solid #c8e6c9;max-height:400px;overflow-y:auto;">
                                                ${formatAgentAnalysis(summary)}
                                            </div>
                                        `;
                                    } else {
                                        // 简单总结
                                        summaryDiv.innerHTML = `
                                            <div style="font-weight:bold;color:#388e3c;margin-bottom:5px;">📋 轮次总结</div>
                                            <div style="font-size:13px;line-height:1.4;color:#666;">${summary}</div>
                                        `;
                                    }
                                    
                                    currentRoundContent.appendChild(summaryDiv);
                                }
                                
                                // 更新样式为完成状态
                                currentRoundContainer.style.border = '2px solid #4caf50';
                                currentRoundContainer.style.background = 'linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%)';
                                
                                // 清空当前轮次引用
                                currentRoundContainer = null;
                                currentRoundContent = null;
                            }
                        }
                        
                        // 轮次展开/折叠功能
                        function toggleRound(roundId) {
                            var contentDiv = document.getElementById('content-' + roundId);
                            var toggleBtn = document.getElementById('toggle-' + roundId);
                            
                            if (contentDiv.style.display === 'none') {
                                contentDiv.style.display = 'block';
                                toggleBtn.textContent = '▼';
                            } else {
                                contentDiv.style.display = 'none';
                                toggleBtn.textContent = '▶';
                            }
                            
                            // 滚动调整
                            setTimeout(function() {
                                var streamDiv = document.getElementById('analysis-stream');
                                if (streamDiv) {
                                    streamDiv.scrollTop = streamDiv.scrollHeight;
                                }
                            }, 100);
                        }
                        
                        // 处理最终决策显示
                        function handleFinalDecision(eventData, messageDiv) {
                            var data = eventData.data;
                            var contentId = generateContentId();
                            
                            messageDiv.style.background = 'linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%)';
                            messageDiv.style.border = '3px solid #4caf50';
                            messageDiv.style.borderLeft = '6px solid #388e3c';
                            messageDiv.style.boxShadow = '0 4px 12px rgba(76,175,80,0.15)';
                            
                            // 提取关键信息
                            var recommendation = data.recommendation || '无推荐';
                            var confidence = data.final_confidence || 0;
                            var message = data.message || '最终交易决策已确定';
                            var totalRounds = data.total_rounds || 0;
                            var averageScore = data.average_score || 0;
                            var allScores = data.all_scores || [];
                            
                            // 获取操作类型和颜色
                            var actionColor = '#4caf50'; // 默认绿色
                            var actionIcon = '📊';
                            if (recommendation.includes('买入') || recommendation.includes('BUY')) {
                                actionColor = '#f44336'; // 红色
                                actionIcon = '📈';
                            } else if (recommendation.includes('卖出') || recommendation.includes('SELL')) {
                                actionColor = '#2196f3'; // 蓝色
                                actionIcon = '📉';
                            } else if (recommendation.includes('持有') || recommendation.includes('HOLD')) {
                                actionColor = '#ff9800'; // 橙色
                                actionIcon = '📊';
                            }
                            
                            // 信心等级
                            var confidenceLevel = '';
                            var confidenceColor = '';
                            if (confidence >= 8) {
                                confidenceLevel = '高';
                                confidenceColor = '#4caf50';
                            } else if (confidence >= 6) {
                                confidenceLevel = '中';
                                confidenceColor = '#ff9800';
                            } else {
                                confidenceLevel = '低';
                                confidenceColor = '#f44336';
                            }
                            
                            messageDiv.innerHTML = `
                                <div onclick="toggleAnalysisResult('${contentId}')" style="cursor:pointer;display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
                                    <div style="font-weight:bold;color:#388e3c;font-size:16px;">🎯 最终投资决策</div>
                                    <span id="toggle-${contentId}" style="font-weight:bold;color:#388e3c;">▼</span>
                                </div>
                                
                                <!-- 决策概览 -->
                                <div id="preview-${contentId}" style="background:rgba(255,255,255,0.9);padding:15px;border-radius:8px;border:1px solid #c8e6c9;">
                                    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:15px;margin-bottom:15px;">
                                        <div style="text-align:center;padding:10px;background:rgba(76,175,80,0.1);border-radius:6px;">
                                            <div style="font-size:18px;color:${actionColor};font-weight:bold;">${actionIcon}</div>
                                            <div style="font-size:13px;color:#666;margin:3px 0;">操作建议</div>
                                            <div style="font-size:14px;font-weight:bold;color:${actionColor};">${recommendation}</div>
                                        </div>
                                        <div style="text-align:center;padding:10px;background:rgba(76,175,80,0.1);border-radius:6px;">
                                            <div style="font-size:18px;color:${confidenceColor};font-weight:bold;">${confidence}</div>
                                            <div style="font-size:13px;color:#666;margin:3px 0;">信心指数</div>
                                            <div style="font-size:12px;color:${confidenceColor};">${confidenceLevel}信心 (${confidence}/10)</div>
                                        </div>
                                        <div style="text-align:center;padding:10px;background:rgba(76,175,80,0.1);border-radius:6px;">
                                            <div style="font-size:18px;color:#388e3c;font-weight:bold;">${totalRounds}</div>
                                            <div style="font-size:13px;color:#666;margin:3px 0;">分析轮数</div>
                                            <div style="font-size:12px;color:#666;">平均分: ${averageScore}</div>
                                        </div>
                                    </div>
                                    <div style="text-align:center;color:#666;font-size:13px;font-style:italic;">${message}</div>
                                </div>
                                
                                <!-- 详细分析内容 -->
                                <div id="full-${contentId}" style="display:none;margin-top:15px;background:rgba(255,255,255,0.95);padding:20px;border-radius:8px;border:1px solid #c8e6c9;max-height:500px;overflow-y:auto;">
                                    ${formatFinalDecisionContent(data.final_decision || '暂无详细分析')}
                                    
                                    <!-- 技术数据 -->
                                    <div style="margin-top:20px;padding:15px;background:rgba(76,175,80,0.05);border-radius:6px;border-left:4px solid #4caf50;">
                                        <h4 style="margin:0 0 10px 0;color:#388e3c;">📊 分析数据</h4>
                                        <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;font-size:12px;">
                                            <div><strong>总轮数:</strong> ${totalRounds}</div>
                                            <div><strong>平均评分:</strong> ${averageScore}</div>
                                            <div><strong>所有评分:</strong> [${allScores.join(', ')}]</div>
                                            <div><strong>数据来源:</strong> ${data.source || 'unknown'}</div>
                                        </div>
                                        <div style="margin-top:8px;font-size:11px;color:#666;">
                                            <strong>时间戳:</strong> ${data.timestamp ? new Date(data.timestamp * 1000).toLocaleString() : '未知'}
                                        </div>
                                    </div>
                                </div>
                            `;
                        }
                        
                        // 格式化最终决策内容
                        function formatFinalDecisionContent(content) {
                            if (!content) return '<p style="color:#666;font-style:italic;">暂无详细分析内容</p>';
                            
                            // 将Markdown格式转换为HTML
                            var formatted = content
                                .replace(/## ([^\\n]+)/g, '<h3 style="color:#388e3c;margin:20px 0 10px 0;border-bottom:2px solid #c8e6c9;padding-bottom:5px;">$1</h3>')
                                .replace(/### ([^\\n]+)/g, '<h4 style="color:#4caf50;margin:15px 0 8px 0;">$1</h4>')
                                .replace(/\\*\\*([^\\*]+)\\*\\*/g, '<strong style="color:#2c3e50;">$1</strong>')
                                .replace(/\\*([^\\*]+)\\*/g, '<em>$1</em>')
                                .replace(/\\n\\n/g, '</p><p style="margin:10px 0;line-height:1.6;">')
                                .replace(/\\n/g, '<br>')
                                .replace(/^/, '<p style="margin:10px 0;line-height:1.6;">')
                                .replace(/$/, '</p>');
                            
                            // 处理列表
                            formatted = formatted.replace(/(\\d+\\.)\\s/g, '<div style="margin:5px 0 5px 20px;"><span style="color:#4caf50;font-weight:bold;">$1</span> ');
                            formatted = formatted.replace(/([\\r\\n])(\\d+\\.)/g, '$1</div><div style="margin:5px 0 5px 20px;"><span style="color:#4caf50;font-weight:bold;">$2</span> ');
                            
                            return formatted;
                        }
                        
                        // 格式化Agent分析内容
                        function formatAgentAnalysis(content) {
                            if (!content) return '<p style="color:#666;font-style:italic;">暂无分析内容</p>';
                            
                            // 将Markdown格式转换为HTML，针对Agent分析优化
                            var formatted = content
                                // 主要标题
                                .replace(/## ([^\\n]+)/g, '<h3 style="color:#1976d2;margin:18px 0 12px 0;border-bottom:2px solid #bbdefb;padding-bottom:6px;font-size:16px;">📊 $1</h3>')
                                // 次级标题  
                                .replace(/### ([^\\n]+)/g, '<h4 style="color:#1565c0;margin:14px 0 8px 0;font-size:14px;">📋 $1</h4>')
                                // 粗体文本
                                .replace(/\\*\\*([^\\*]+)\\*\\*/g, '<strong style="color:#0d47a1;background:rgba(33,150,243,0.1);padding:1px 3px;border-radius:2px;">$1</strong>')
                                // 斜体文本
                                .replace(/\\*([^\\*]+)\\*/g, '<em style="color:#1976d2;">$1</em>')
                                // 段落处理
                                .replace(/\\n\\n/g, '</p><p style="margin:8px 0;line-height:1.5;color:#333;">')
                                .replace(/\\n/g, '<br>')
                                .replace(/^/, '<p style="margin:8px 0;line-height:1.5;color:#333;">')
                                .replace(/$/, '</p>');
                            
                            // 处理编号列表
                            formatted = formatted.replace(/(\\d+\\.)\\s([^<]+)/g, function(match, number, text) {
                                return `<div style="margin:6px 0 6px 16px;padding:6px 10px;background:rgba(33,150,243,0.08);border-radius:4px;border-left:3px solid #2196f3;">
                                    <span style="color:#1976d2;font-weight:bold;margin-right:8px;">${number}</span>
                                    <span style="color:#333;">${text}</span>
                                </div>`;
                            });
                            
                            // 处理特殊关键词高亮
                            formatted = formatted
                                .replace(/(Score:|Action:|Conviction:|Sizing:|Recommendation:)/g, '<span style="background:#e3f2fd;color:#0d47a1;font-weight:bold;padding:2px 6px;border-radius:3px;margin-right:5px;">$1</span>')
                                .replace(/(BUY|SELL|HOLD)/g, '<span style="background:#1976d2;color:white;font-weight:bold;padding:2px 8px;border-radius:4px;font-size:11px;">$1</span>')
                                .replace(/(HIGH|MEDIUM|LOW)/g, '<span style="background:#bbdefb;color:#0d47a1;font-weight:bold;padding:1px 6px;border-radius:3px;font-size:11px;">$1</span>');
                            
                            // 处理信心等级显示
                            formatted = formatted.replace(/Conviction Level: (\\w+)/g, function(match, level) {
                                var color = level === 'High' ? '#4caf50' : level === 'Medium' ? '#ff9800' : '#f44336';
                                return `<div style="margin:10px 0;padding:8px 12px;background:rgba(33,150,243,0.1);border-radius:6px;border-left:4px solid #2196f3;">
                                    <strong style="color:#1976d2;">信心等级:</strong> 
                                    <span style="color:${color};font-weight:bold;margin-left:8px;">${level}</span>
                                </div>`;
                            });
                            
                            return formatted;
                        }
                        
                        // 将函数添加到全局作用域
                        window.toggleDataDetails = toggleDataDetails;
                        window.toggleAnalysisResult = toggleAnalysisResult;
                        window.toggleRound = toggleRound;
                        window.closeDataPanel = closeDataPanel;
                        window.performSmartAnalysis = performSmartAnalysis;
                        
                        // 获取图表容器
                        var gd = document.querySelector('.plotly-graph-div');
                        
                        // 监听plotly_click事件
                        gd.on('plotly_click', function(data) {
                            if (!data || !data.points || !data.points.length) return;
                            
                            var point = data.points[0];
                            var pointData = point.customdata;
                            var pointIndex = point.pointIndex;
                            
                            // 获取用户设置的天数
                            var nDays = parseInt(document.getElementById('days-input').value) || 3;
                            
                            // 获取所有数据点
                            var allData = point.data.customdata;
                            var totalPoints = allData.length;
                            
                            // 计算前后n天的索引范围
                            var startIndex = Math.max(0, pointIndex - nDays);
                            var endIndex = Math.min(totalPoints - 1, pointIndex + nDays);
                            
                            // 隐藏控制面板
                            var controlPanel = document.getElementById('control-panel');
                            if (controlPanel) {
                                controlPanel.style.display = 'none';
                            }
                            
                            // 创建或更新数据面板 - 增加一倍宽度
                            var dataPanel = document.getElementById('data-panel');
                            if (!dataPanel) {
                                dataPanel = document.createElement('div');
                                dataPanel.id = 'data-panel';
                                dataPanel.style.cssText = 'position:fixed;top:20px;right:20px;background:white;padding:15px;border:1px solid #ccc;border-radius:5px;box-shadow:2px 2px 10px rgba(0,0,0,0.1);z-index:1000;max-width:900px;max-height:80vh;overflow-y:auto;';
                                document.body.appendChild(dataPanel);
                            }
                            
                            // 更新数据面板内容
                            var dataType = pointData[16] || '未知';
                            var content = `
                                <div style="position:sticky;top:0;background:white;border-bottom:1px solid #eee;padding-bottom:8px;margin-bottom:10px;">
                                    <h3 style="margin:0;color:#2C3E50;">数据详情 (前后${nDays}天)</h3>
                                    <p style="margin:5px 0;font-weight:bold;color:#E74C3C;">当前选中: ${dataType} - ${pointData[0]}</p>
                                </div>
                            `;
                            
                            // 准备智能分析需要的数据
                            var analysisData = {
                                selectedDate: pointData[0],
                                selectedDataType: dataType,
                                symbol: '${code}',  // 使用模板变量
                                name: '${name}',    // 使用模板变量
                                priceData: [],
                                indicators: {}
                            };
                            
                            // 收集前后n天的价格数据
                            for (var i = startIndex; i <= endIndex; i++) {
                                var dayData = allData[i];
                                analysisData.priceData.push({
                                    date: dayData[0],
                                    open: parseFloat(dayData[1]),
                                    high: parseFloat(dayData[2]),
                                    low: parseFloat(dayData[3]),
                                    close: parseFloat(dayData[4]),
                                    volume: parseInt(dayData[5])
                                });
                                
                                // 如果是当前选中的日期，记录技术指标
                                if (i === pointIndex) {
                                    analysisData.indicators = {
                                        ma5: parseFloat(dayData[6]),
                                        ma10: parseFloat(dayData[7]),
                                        ma20: parseFloat(dayData[8]),
                                        dif: parseFloat(dayData[9]),
                                        dea: parseFloat(dayData[10]),
                                        macd: parseFloat(dayData[11]),
                                        kdj_k: parseFloat(dayData[12]),
                                        kdj_d: parseFloat(dayData[13]),
                                        kdj_j: parseFloat(dayData[14]),
                                        rsi: parseFloat(dayData[15])
                                    };
                                }
                            }
                            
                            // 存储到全局变量供智能分析使用
                            window.currentAnalysisData = analysisData;
                            
                            // 显示前后n天的数据 - 添加折叠功能
                            for (var i = startIndex; i <= endIndex; i++) {
                                var dayData = allData[i];
                                var isCurrentDay = (i === pointIndex);
                                var dayClass = isCurrentDay ? 'current-day' : 'other-day';
                                var dayStyle = isCurrentDay ? 
                                    'background:#fff3cd;border:2px solid #856404;margin:8px 0;padding:10px;border-radius:5px;' : 
                                    'background:#f8f9fa;border:1px solid #dee2e6;margin:5px 0;padding:8px;border-radius:3px;';
                                
                                var dayId = 'day-' + i;
                                var isExpanded = isCurrentDay; // 默认只展开当前日期
                                
                                content += `
                                    <div class="${dayClass}" style="${dayStyle}">
                                        <div onclick="toggleDataDetails('${dayId}')" style="cursor:pointer;display:flex;justify-content:space-between;align-items:center;">
                                            <h5 style="margin:0;color:#2C3E50;font-weight:bold;">
                                                ${dayData[0]} ${isCurrentDay ? '(选中日期)' : ''}
                                            </h5>
                                            <span id="toggle-${dayId}" style="font-weight:bold;color:#666;">${isExpanded ? '▼' : '▶'}</span>
                                        </div>
                                        <div id="details-${dayId}" style="display:${isExpanded ? 'block' : 'none'};margin-top:8px;">
                                            <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:8px;font-size:12px;margin-bottom:8px;">
                                                <span style="padding:4px;background:rgba(52,152,219,0.1);border-radius:3px;">开盘: ¥${parseFloat(dayData[1]).toFixed(2)}</span>
                                                <span style="padding:4px;background:rgba(231,76,60,0.1);border-radius:3px;">最高: ¥${parseFloat(dayData[2]).toFixed(2)}</span>
                                                <span style="padding:4px;background:rgba(46,204,113,0.1);border-radius:3px;">最低: ¥${parseFloat(dayData[3]).toFixed(2)}</span>
                                                <span style="padding:4px;background:rgba(155,89,182,0.1);border-radius:3px;">收盘: ¥${parseFloat(dayData[4]).toFixed(2)}</span>
                                            </div>
                                            <div style="margin-bottom:8px;font-size:11px;color:#666;padding:4px;background:rgba(149,165,166,0.1);border-radius:3px;">
                                                <span>成交量: ${parseInt(dayData[5]).toLocaleString()}</span>
                                            </div>
                                            
                                            ${isCurrentDay || !isCurrentDay ? `
                                                <div style="border-top:1px solid #ddd;padding-top:8px;">
                                                    <h6 style="margin:0 0 8px 0;color:#2C3E50;">技术指标</h6>
                                                    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;font-size:11px;">
                                                        <div style="padding:4px;background:rgba(243,156,18,0.1);border-radius:3px;">
                                                            <div>MA5: ¥${parseFloat(dayData[6]).toFixed(2)}</div>
                                                            <div>MA10: ¥${parseFloat(dayData[7]).toFixed(2)}</div>
                                                            <div>MA20: ¥${parseFloat(dayData[8]).toFixed(2)}</div>
                                                        </div>
                                                        <div style="padding:4px;background:rgba(52,152,219,0.1);border-radius:3px;">
                                                            <div>MACD: ${parseFloat(dayData[11]).toFixed(2)}</div>
                                                            <div>DIF: ${parseFloat(dayData[9]).toFixed(2)}</div>
                                                            <div>DEA: ${parseFloat(dayData[10]).toFixed(2)}</div>
                                                        </div>
                                                        <div style="padding:4px;background:rgba(231,76,60,0.1);border-radius:3px;">
                                                            <div>KDJ-K: ${parseFloat(dayData[12]).toFixed(2)}</div>
                                                            <div>KDJ-D: ${parseFloat(dayData[13]).toFixed(2)}</div>
                                                            <div>KDJ-J: ${parseFloat(dayData[14]).toFixed(2)}</div>
                                                        </div>
                                                    </div>
                                                    <div style="margin-top:6px;padding:4px;background:rgba(155,89,182,0.1);border-radius:3px;font-size:11px;">
                                                        <span>RSI: ${parseFloat(dayData[15]).toFixed(2)}</span>
                                                    </div>
                                                </div>
                                            ` : ''}
                                        </div>
                                    </div>
                                `;
                            }
                            
                            // 添加智能分析结果显示区域
                            content += `
                                <div id="analysis-result" style="margin-top:15px;">
                                    <!-- 智能分析结果将在这里显示 -->
                                </div>
                            `;
                            
                            // 添加控制按钮
                            content += `
                                <div style="text-align:right;margin-top:15px;position:sticky;bottom:0;background:white;padding-top:8px;border-top:1px solid #eee;">
                                    <button id="analysis-btn" onclick="performSmartAnalysis()" 
                                            style="padding:8px 15px;border:none;background:#3498DB;color:white;border-radius:3px;cursor:pointer;margin-right:10px;">
                                        🤖 智能分析
                                    </button>
                                    <button onclick="closeDataPanel()" 
                                            style="padding:8px 15px;border:none;background:#E74C3C;color:white;border-radius:3px;cursor:pointer;">
                                        关闭
                                    </button>
                                </div>
                            `;
                            
                            dataPanel.innerHTML = content;
                            dataPanel.style.display = 'block';
                        });
                        
                        // 全局函数绑定
                        window.toggleAnalysisResult = function(contentId) {
                            var fullDiv = document.getElementById('full-' + contentId);
                            var toggleBtn = document.getElementById('toggle-' + contentId);
                            
                            if (fullDiv.style.display === 'none') {
                                fullDiv.style.display = 'block';
                                toggleBtn.textContent = '▲';
                            } else {
                                fullDiv.style.display = 'none';
                                toggleBtn.textContent = '▼';
                            }
                            
                            // 滚动调整
                            setTimeout(function() {
                                var streamDiv = document.getElementById('analysis-stream');
                                if (streamDiv) {
                                    streamDiv.scrollTo({
                                        top: streamDiv.scrollHeight,
                                        behavior: 'smooth'
                                    });
                                }
                            }, 100);
                        };
                        
                        // 通用内容展开/折叠函数
                        window.toggleContent = function(contentId) {
                            var fullDiv = document.getElementById('full-' + contentId);
                            var previewDiv = document.getElementById('preview-' + contentId);
                            var toggleBtn = document.getElementById('toggle-' + contentId);
                            
                            if (fullDiv && toggleBtn) {
                                if (fullDiv.style.display === 'none') {
                                    fullDiv.style.display = 'block';
                                    if (previewDiv) previewDiv.style.display = 'none';
                                    toggleBtn.textContent = '▲';
                                } else {
                                    fullDiv.style.display = 'none';
                                    if (previewDiv) previewDiv.style.display = 'block';
                                    toggleBtn.textContent = '▼';
                                }
                                
                                // 滚动调整
                                setTimeout(function() {
                                    var streamDiv = document.getElementById('analysis-stream');
                                    if (streamDiv) {
                                        streamDiv.scrollTo({
                                            top: streamDiv.scrollHeight,
                                            behavior: 'smooth'
                                        });
                                    }
                                }, 100);
                            }
                        };
                        
                    });
                </script>
                """
                
                # 添加CSS动画样式
                custom_css = """
                <style>
                    @keyframes fadeIn {
                        from { 
                            opacity: 0; 
                            transform: translateY(10px); 
                        }
                        to { 
                            opacity: 1; 
                            transform: translateY(0); 
                        }
                    }
                    
                    .message {
                        animation: fadeIn 0.3s ease-in;
                    }
                    
                    .status-message {
                        animation: fadeIn 0.5s ease-in;
                    }
                    
                    /* 滚动条样式 */
                    #analysis-stream::-webkit-scrollbar {
                        width: 8px;
                    }
                    
                    #analysis-stream::-webkit-scrollbar-track {
                        background: #f1f1f1;
                        border-radius: 4px;
                    }
                    
                    #analysis-stream::-webkit-scrollbar-thumb {
                        background: #c1c1c1;
                        border-radius: 4px;
                    }
                    
                    #analysis-stream::-webkit-scrollbar-thumb:hover {
                        background: #a8a8a8;
                    }
                    
                    /* 分析结果展开区域的滚动条 */
                    [id^="full-"]::-webkit-scrollbar {
                        width: 6px;
                    }
                    
                    [id^="full-"]::-webkit-scrollbar-track {
                        background: #f1f1f1;
                        border-radius: 3px;
                    }
                    
                    [id^="full-"]::-webkit-scrollbar-thumb {
                        background: #c1c1c1;
                        border-radius: 3px;
                    }
                    
                    [id^="full-"]::-webkit-scrollbar-thumb:hover {
                        background: #a8a8a8;
                    }
                </style>
                """
                
                # 在HTML内容中插入自定义CSS和JavaScript代码
                html_content = html_content.replace('</head>', f'{custom_css}{custom_js}</head>')
                
                # 写入文件
                f.write(html_content)

    except Exception as e:
        logger.error(f"图表生成失败: {str(e)}")

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
        file_path = 'all_stocks.pkl'
        if os.path.exists(file_path):
            all_stocks = deserialize_dataframe(file_path)
        else:
            domestic_stocks = ak.stock_zh_a_spot_em()
            #获取美股
            foreign_stock = ak.stock_us_spot_em()
            all_stocks = pd.concat([domestic_stocks, foreign_stock], axis=0, ignore_index=True)
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

        try:
            stock_list.to_csv('stock_list.csv', index=False)
            other_stocks.to_csv('other.csv', index=False)
            logger.info("股票列表已成功保存到csv 文件中。")
        except Exception as e:
            logger.error(f"保存股票列表到 CSV 文件时出错: {e}")


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
            
    # 分析股票 这才是重点
    results = analyze_stocks(stock_list, start_date, end_date, args.parallel)
    # 处理结果
    df = process_results(results)
        
    # 保存结果
    if not df.empty:
        output_file = os.path.join(output_dir, 'analysis_results.csv')
        df.to_csv(output_file, index=False, encoding='utf_8_sig')
        logger.info(f"分析结果已保存至: {output_file}")
        
        # 生成图表
        generate_stock_charts(df, start_date, end_date, output_dir, args.k)
        
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
