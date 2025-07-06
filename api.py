from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Header
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
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
from database import db  # 导入数据库模块
from fastapi.templating import Jinja2Templates
from fastapi import Request
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
FORUM_DIR = "forum"
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(FORUM_DIR, exist_ok=True)

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/forum", StaticFiles(directory=FORUM_DIR), name="forum")

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

# 用户认证相关的数据模型
class UserRegisterRequest(BaseModel):
    username: str
    email: Optional[str] = None
    password: str
    user_type: str

class UserLoginRequest(BaseModel):
    username: str
    password: str

class CreatePostRequest(BaseModel):
    title: str
    content: str
    category: str
    tags: List[str]

# 全局变量
output_dir = None

def _make_output_dir() -> str:
    """按照时间戳创建静态输出目录，返回绝对路径"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(STATIC_DIR, ts)
    os.makedirs(path, exist_ok=True)
    return path

_FIELD_MAP = {
    # A 股、港股接口用中文"代码""名称"
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

async def get_current_user(Authorization: Optional[str] = Header(None)) -> Optional[Dict]:
    """获取当前登录用户"""
    if not Authorization:
        return None
    
    try:
        # 从Authorization header中提取token
        if Authorization.startswith("Bearer "):
            token = Authorization[7:]  # 移除"Bearer "前缀
        else:
            token = Authorization
        
        user = db.verify_session(token)
        return user
    except Exception:
        return None

templates = Jinja2Templates(directory="templates")

@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")

@app.get("/login")
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/register")
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.get("/api/status")
async def api_status():
    return {"message": "股票分析系统 API 服务正在运行"}

# 用户认证相关API
@app.post("/api/auth/register", summary="用户注册")
async def register_user(request: UserRegisterRequest):
    """用户注册"""
    try:
        # 验证用户类型
        valid_user_types = db.get_user_types()
        if request.user_type not in valid_user_types:
            raise HTTPException(status_code=400, detail=f"无效的用户类型。可用类型: {', '.join(valid_user_types)}")
        
        result = db.register_user(
            username=request.username,
            email=request.email,
            password=request.password,
            user_type=request.user_type
        )
        
        return {
            "message": "注册成功",
            "user": result["user"]
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"注册失败: {str(e)}")

@app.post("/api/auth/login", summary="用户登录")
async def login_user(request: UserLoginRequest):
    """用户登录"""
    try:
        result = db.login_user(
            username=request.username,
            password=request.password
        )
        
        return {
            "message": "登录成功",
            "user": result["user"],
            "session_token": result["session_token"]
        }
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"登录失败: {str(e)}")

@app.post("/api/auth/logout", summary="用户登出")
async def logout_user(current_user: Optional[Dict] = Depends(get_current_user)):
    """用户登出"""
    if not current_user:
        raise HTTPException(status_code=401, detail="未登录")
    
    try:
        # 这里需要从请求中获取token，暂时返回成功
        return {"message": "登出成功"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"登出失败: {str(e)}")

@app.get("/api/auth/me", summary="获取当前用户信息")
async def get_current_user_info(current_user: Optional[Dict] = Depends(get_current_user)):
    """获取当前登录用户信息"""
    if not current_user:
        raise HTTPException(status_code=401, detail="未登录")
    
    return {"user": current_user}

@app.get("/api/auth/user-types", summary="获取可用用户类型")
async def get_user_types():
    """获取可用的用户类型"""
    return {"user_types": db.get_user_types()}

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
        out_dir = _make_output_dir()
        for code in set(request.stock_codes):          # 去重
            hist = get_historical_data(code, start_date, end_date,
                                       debug=request.debug)
            if hist.empty or len(hist) < 15:           # 数据太短直接跳过
                continue

            indi  = calculate_technical_indicators(hist)
            if not indi:
                continue

            score = calculate_stock_score(hist, indi)

            # -------- 构造 DataFrame（供图表函数使用）--------
            df = pd.DataFrame([{
                "代码": code,
                "名称": code,  # 若需中文名称，可自行查询
                "总市值（亿元）": "N/A",
                "起始日价（元）": hist['close'].iloc[0],
                "截止日价（元）": hist['close'].iloc[-1],
                "涨幅(%)": (hist['close'].iloc[-1] / hist['close'].iloc[0] - 1) * 100,
                "得分": score
            }])

            # -------- 输出目录 + 异步生成图表 --------
            background_tasks.add_task(
                generate_stock_charts,
                df,
                start_date,
                end_date,
                out_dir,
                k=1
            )

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
    **exchange** 留空 = 全部；可选 SH / SZ / US
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
        else:
            raise HTTPException(400, "exchange 仅支持 SH/SZ/US")
    # --- 返回 JSON（已自动将 NaN→None）---
    return {
        "count": len(df),
        "data": jsonable_encoder(df.to_dict("records"))
    }

# 论坛相关的API端点
@app.get("/api/forum/posts")
async def get_forum_posts():
    """获取所有论坛帖子"""
    try:
        posts = db.get_all_posts()
        return {"posts": posts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取帖子失败: {str(e)}")

@app.get("/api/forum/categories")
async def get_forum_categories():
    """获取所有论坛分类"""
    try:
        categories = db.get_categories()
        return {"categories": categories}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取分类失败: {str(e)}")

@app.get("/api/forum/tags")
async def get_forum_tags():
    """获取所有论坛标签"""
    try:
        tags = db.get_tags()
        return {"tags": tags}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取标签失败: {str(e)}")

@app.get("/api/forum/stats")
async def get_forum_stats():
    """获取论坛统计信息"""
    try:
        stats = db.get_forum_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")

@app.post("/api/forum/posts")
async def create_forum_post(
    request: CreatePostRequest,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """创建新帖子（需要登录）"""
    if not current_user:
        raise HTTPException(status_code=401, detail="请先登录后再发布帖子")
    
    try:
        post_id = db.create_post_by_user_id(
            title=request.title,
            content=request.content,
            user_id=current_user['id'],
            category=request.category,
            tags=request.tags
        )
        return {"message": "帖子创建成功", "post_id": post_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建帖子失败: {str(e)}")

@app.get("/api/forum/my-posts")
async def get_my_posts(current_user: Optional[Dict] = Depends(get_current_user)):
    """获取当前用户的帖子（需要登录）"""
    if not current_user:
        raise HTTPException(status_code=401, detail="请先登录")
    
    try:
        posts = db.get_user_posts(current_user['id'])
        return {"posts": posts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取用户帖子失败: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)