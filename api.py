from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict
import uvicorn
import os
import shutil
from datetime import datetime, timedelta
import pandas as pd
import json
from typing import Optional, List
from fastapi import Query
from fastapi.encoders import jsonable_encoder   # ❶ 新增
import numpy as np
import akshare as ak
_STOCK_CACHE = "all_stocks.pkl"


# 导入原有功能
from stock_analyze import (
    get_historical_data,
    calculate_technical_indicators,
    calculate_stock_score,
    generate_stock_charts,
    set_plot_style,
    deserialize_dataframe
)

app = FastAPI(
    title="股票分析系统 API",
    description="提供股票数据分析和查询的 RESTful API 服务",
    version="1.0.0"
)

# 创建静态文件目录
STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# 数据模型
class StockAnalysisRequest(BaseModel):
    stock_code: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    days: Optional[int] = 60
    debug: Optional[bool] = False

class BatchAnalysisRequest(BaseModel):
    stock_codes: List[str]
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    days: Optional[int] = 60
    debug: Optional[bool] = False
    topgains: Optional[bool] = False
    k: Optional[int] = 10

class StockListRequest(BaseModel):
    exchange: Optional[str] = None      # SH/SZ/US/HK；None=全部
    refresh:  bool = False              # True=强制刷新 AkShare

# 全局变量
output_dir = None

def _make_output_dir() -> str:
    """按照时间戳创建静态输出目录，返回绝对路径"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(STATIC_DIR, ts)
    os.makedirs(path, exist_ok=True)
    return path

_FIELD_MAP = {
    # A 股、港股接口用中文“代码”“名称”
    "代码": "code",
    "名称": "name",
    # 美股接口返回英文
    "symbol": "code",
    "name": "name",
}

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """把不同接口返回的列重命名为统一的 code / name"""
    rename_dict = {c: _FIELD_MAP[c] for c in df.columns if c in _FIELD_MAP}
    return df.rename(columns=rename_dict)[["code", "name"]]

def _load_stocks(force_refresh: bool) -> pd.DataFrame:
    """加载（必要时抓取）股票列表，保证含 code/name"""
    # ① 若无缓存或要求刷新 → 调 AkShare
    if force_refresh or not os.path.exists(_STOCK_CACHE):
        try:
            df_a  = _standardize_columns(ak.stock_zh_a_spot_em())   # A 股
            df_us = _standardize_columns(ak.stock_us_spot_em())     # 美股
            df_hk = _standardize_columns(ak.stock_hk_spot())        # 港股
        except Exception as e:
            raise HTTPException(500, f"调用 AkShare 获取股票列表失败: {e}")
        df_all = pd.concat([df_a, df_us, df_hk], ignore_index=True)
        df_all.to_pickle(_STOCK_CACHE)
        return df_all

    # ② 读取旧缓存 → 若缺标准列则自动修复
    df_cached = pd.read_pickle(_STOCK_CACHE)
    if "code" not in df_cached.columns or "name" not in df_cached.columns:
        df_cached = _standardize_columns(df_cached)
        df_cached.to_pickle(_STOCK_CACHE)  # 覆盖脏缓存
    return df_cached[["code", "name"]]

@app.get("/")
async def root():
    return {"message": "股票分析系统 API 服务正在运行"}

@app.post("/analyze/single", summary="分析单只股票并生成图表")
async def analyze_single_stock(
    request: StockAnalysisRequest,
    background_tasks: BackgroundTasks
):
    """
    1. 计算区间价格、技术指标、综合得分  
    2. 立即返回 JSON；图表在后台异步生成为 HTML，前端可通过返回的 URL 访问
    """
    try:
        # -------- 计算日期范围 --------
        end_date  = request.end_date or datetime.now().strftime("%Y%m%d")
        start_date = (
            request.start_date or
            (datetime.strptime(end_date, "%Y%m%d") -
             timedelta(days=request.days)).strftime("%Y%m%d")
        )

        # -------- 拉取行情 --------
        hist = get_historical_data(
            request.stock_code, start_date, end_date,
            debug=request.debug
        )
        if hist.empty:
            raise HTTPException(status_code=404, detail="未找到行情数据")

        # -------- 技术指标 + 评分 --------
        indi  = calculate_technical_indicators(hist)
        if not indi:
            raise HTTPException(status_code=500, detail="技术指标计算失败")
        score = calculate_stock_score(hist, indi)

        # -------- 构造 DataFrame（供图表函数使用）--------
        df = pd.DataFrame([{
            "代码": request.stock_code,
            "名称": request.stock_code,          # 若需中文名称，可自行查询
            "总市值（亿元）": "N/A",
            "起始日价（元）": hist['close'].iloc[0],
            "截止日价（元）": hist['close'].iloc[-1],
            "涨幅(%)": (hist['close'].iloc[-1] / hist['close'].iloc[0] - 1) * 100,
            "得分": score
        }])

        # -------- 异步生成图表 --------
        out_dir = _make_output_dir()
        background_tasks.add_task(
            generate_stock_charts,
            df,
            start_date,
            end_date,
            out_dir,
            k=1                              # 只生成这 1 只股票
        )

        # -------- 结果 JSON --------
        return {
            "code": request.stock_code,
            "start_price": round(hist['close'].iloc[0], 2),
            "end_price": round(hist['close'].iloc[-1], 2),
            "change_percent": round((hist['close'].iloc[-1] /
                                      hist['close'].iloc[0] - 1) * 100, 2),
            "score": score,
            "chart_url": f"/static/{os.path.basename(out_dir)}/{request.stock_code}_analysis.html"
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/batch", summary="批量分析多只股票")
async def analyze_batch_stocks(
    request: BatchAnalysisRequest,
    background_tasks: BackgroundTasks
):
    try:
        # -------- 计算日期范围 --------
        end_date  = request.end_date or datetime.now().strftime("%Y%m%d")
        start_date = (
            request.start_date or
            (datetime.strptime(end_date, "%Y%m%d") -
             timedelta(days=request.days)).strftime("%Y%m%d")
        )

        # -------- 循环分析每只股票 --------
        results: list[dict] = []
        for code in set(request.stock_codes):          # 去重
            hist = get_historical_data(code, start_date, end_date,
                                       debug=request.debug)
            if hist.empty or len(hist) < 15:           # 数据太短直接跳过
                continue

            indi  = calculate_technical_indicators(hist)
            if not indi:
                continue

            score = calculate_stock_score(hist, indi)
            change_pct = (hist['close'].iloc[-1] / hist['close'].iloc[0] - 1) * 100

            results.append({
                "代码"       : code,
                "名称"       : code,                   # 如需中文名可自行查表
                "总市值（亿元）" : "N/A",
                "起始日价（元）": float(hist['close'].iloc[0]),
                "截止日价（元）": float(hist['close'].iloc[-1]),
                "涨幅(%)"    : round(change_pct, 2),
                "得分"       : score,
                "交易所"     : "SH" if code.startswith("6") else "SZ",
            })

        if not results:
            raise HTTPException(status_code=404, detail="所有股票均分析失败或无数据")

        df = pd.DataFrame(results)

        # -------- 排序 & 取前 k --------
        if request.topgains:
            df.sort_values(["涨幅(%)", "代码"], ascending=[False, True], inplace=True)
        else:
            df.sort_values(["得分", "代码"], ascending=[False, True], inplace=True)

        df_topk = df.head(request.k)

        # -------- 输出目录 + 异步生成图表 --------
        out_dir = _make_output_dir()
        background_tasks.add_task(
            generate_stock_charts,
            df_topk,
            start_date,
            end_date,
            out_dir,
            k=request.k
        )

        # -------- 保存 CSV --------
        csv_path = os.path.join(out_dir, "analysis_results.csv")
        df.to_csv(csv_path, index=False, encoding="utf_8_sig")

        # -------- 返回 JSON --------
        return {
            "results"   : df_topk.to_dict(orient="records"),
            "csv_url"   : f"/static/{os.path.basename(out_dir)}/analysis_results.csv",
            "charts_dir": f"/static/{os.path.basename(out_dir)}"
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stocks/list", summary="POST 获取股票列表（可选交易所）")
async def get_stock_list(req: StockListRequest):
    """
    **exchange** 留空 = 全部；可选 SH / SZ / US / HK
    **refresh** 设 True 会忽略本地缓存，重新调用 AkShare。
    """
    df = _load_stocks(req.refresh)

    # --- 按交易所过滤 ---
    if req.exchange:
        ex = req.exchange.upper()
        if ex == "SH":
            df = df[df["code"].str.startswith("6")]
        elif ex == "SZ":
            df = df[df["code"].str.startswith(("0", "3"))]
        elif ex == "US":
            # 美股代码示例: 105.GOOG / 106.BABA / 105.AMZN
            df = df[df["code"].str.match(r"^\d{3}\.[A-Za-z]{3,5}$", na=False)]
        elif ex == "HK":
            df = df[df["code"].str.match(r"^\d{5}$", na=False)]
        else:
            raise HTTPException(400, "exchange 仅支持 SH/SZ/US/HK")

    # --- 返回 JSON（已自动将 NaN→None）---
    return {
        "count": len(df),
        "data": jsonable_encoder(df.to_dict("records"))
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)