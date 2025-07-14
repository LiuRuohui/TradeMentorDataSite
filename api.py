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
    title="Stock Analysis System API",
    description="Provides RESTful API services for stock data analysis and queries",
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
    exchange: Optional[str] = None      # SH/SZ/US/HK; None=all
    refresh:  bool = False              # True=force refresh AkShare

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

class CreateReplyRequest(BaseModel):
    post_id: int
    content: str

# 全局变量
output_dir = None

def _make_output_dir() -> str:
    """Create a static output directory by timestamp and return the absolute path"""
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
    """Rename columns from different sources to unified code/name"""
    rename_dict = {c: _FIELD_MAP[c] for c in df.columns if c in _FIELD_MAP}
    return df.rename(columns=rename_dict)[["code", "name"]]

def _load_stocks(force_refresh: bool) -> pd.DataFrame:
    """Load (or fetch if needed) stock list, ensure code/name columns"""
    # 1. If no cache or force refresh → call AkShare
    if force_refresh or not os.path.exists(_STOCK_CACHE):
        try:
            df_a  = _standardize_columns(ak.stock_zh_a_spot_em())   # A-shares
            df_us = _standardize_columns(ak.stock_us_spot_em())     # US stocks
            df_hk = _standardize_columns(ak.stock_hk_spot())        # HK stocks
        except Exception as e:
            raise HTTPException(500, f"Failed to fetch stock list from AkShare: {e}")
        df_all = pd.concat([df_a, df_us, df_hk], ignore_index=True)
        df_all.to_pickle(_STOCK_CACHE)
        return df_all

    # 2. Read old cache → auto-fix if missing standard columns
    df_cached = pd.read_pickle(_STOCK_CACHE)
    if "code" not in df_cached.columns or "name" not in df_cached.columns:
        df_cached = _standardize_columns(df_cached)
        df_cached.to_pickle(_STOCK_CACHE)  # Overwrite dirty cache
    return df_cached[["code", "name"]]

async def get_current_user(Authorization: Optional[str] = Header(None)) -> Optional[Dict]:
    """Get current logged-in user"""
    if not Authorization:
        return None
    
    try:
        # Extract token from Authorization header
        if Authorization.startswith("Bearer "):
            token = Authorization[7:]
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
    return {"message": "Stock Analysis System API service is running"}

# 用户认证相关API
@app.post("/api/auth/register", summary="User Registration")
async def register_user(request: UserRegisterRequest):
    """User registration"""
    try:
        # Validate user type
        valid_user_types = db.get_user_types()
        if request.user_type not in valid_user_types:
            raise HTTPException(status_code=400, detail=f"Invalid user type. Available types: {', '.join(valid_user_types)}")
        
        result = db.register_user(
            username=request.username,
            email=request.email,
            password=request.password,
            user_type=request.user_type
        )
        
        return {
            "message": "Registration successful",
            "user": result["user"]
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@app.post("/api/auth/login", summary="User Login")
async def login_user(request: UserLoginRequest):
    """User login"""
    try:
        result = db.login_user(
            username=request.username,
            password=request.password
        )
        
        return {
            "message": "Login successful",
            "user": result["user"],
            "session_token": result["session_token"]
        }
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")

@app.post("/api/auth/logout", summary="User Logout")
async def logout_user(current_user: Optional[Dict] = Depends(get_current_user)):
    """User logout"""
    if not current_user:
        raise HTTPException(status_code=401, detail="Not logged in")
    
    try:
        # Here you need to get the token from the request, temporarily return success
        return {"message": "Logout successful"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Logout failed: {str(e)}")

@app.get("/api/auth/me", summary="Get Current User Info")
async def get_current_user_info(current_user: Optional[Dict] = Depends(get_current_user)):
    """Get current logged-in user info"""
    if not current_user:
        raise HTTPException(status_code=401, detail="Not logged in")
    
    return {"user": current_user}

@app.get("/api/auth/user-types", summary="Get Available User Types")
async def get_user_types():
    """Get available user types"""
    return {"user_types": db.get_user_types()}

@app.post("/analyze/single", summary="Analyze Single Stock and Generate Chart")
async def analyze_single_stock(
    request: StockAnalysisRequest,
    background_tasks: BackgroundTasks
):
    """
    1. Calculate price range, technical indicators, and comprehensive score
    2. Return JSON immediately; chart is generated asynchronously as HTML, frontend can access via returned URL
    """
    try:
        # Calculate date range
        end_date  = request.end_date or datetime.now().strftime("%Y%m%d")
        start_date = (
            request.start_date or
            (datetime.strptime(end_date, "%Y%m%d") -
             timedelta(days=request.days)).strftime("%Y%m%d")
        )

        # Fetch historical data
        hist = get_historical_data(
            request.stock_code, start_date, end_date,
            debug=request.debug
        )
        if hist.empty:
            raise HTTPException(status_code=404, detail="No historical data found")

        # Technical indicators + score
        indi  = calculate_technical_indicators(hist)
        if not indi:
            raise HTTPException(status_code=500, detail="Technical indicator calculation failed")
        score = calculate_stock_score(hist, indi)

        # Build DataFrame (for chart function)
        df = pd.DataFrame([{
            "代码": request.stock_code,
            "名称": request.stock_code,          # 若需中文名称，可自行查询
            "总市值（亿元）": "N/A",
            "起始日价（元）": hist['close'].iloc[0],
            "截止日价（元）": hist['close'].iloc[-1],
            "涨幅(%)": (hist['close'].iloc[-1] / hist['close'].iloc[0] - 1) * 100,
            "得分": score
        }])

        # Async generate chart
        out_dir = _make_output_dir()
        background_tasks.add_task(
            generate_stock_charts,
            df,
            start_date,
            end_date,
            out_dir,
            k=1                              # Only generate for this stock
        )

        # Result JSON
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

@app.post("/analyze/batch", summary="Batch Analyze Multiple Stocks")
async def analyze_batch_stocks(
    request: BatchAnalysisRequest,
    background_tasks: BackgroundTasks
):
    try:
        # Calculate date range
        end_date  = request.end_date or datetime.now().strftime("%Y%m%d")
        start_date = (
            request.start_date or
            (datetime.strptime(end_date, "%Y%m%d") -
             timedelta(days=request.days)).strftime("%Y%m%d")
        )

        # Analyze each stock
        results: list[dict] = []
        out_dir = _make_output_dir()
        for code in set(request.stock_codes):          # Remove duplicates
            hist = get_historical_data(code, start_date, end_date,
                                       debug=request.debug)
            if hist.empty or len(hist) < 15:           # Skip if data too short
                continue

            indi  = calculate_technical_indicators(hist)
            if not indi:
                continue

            score = calculate_stock_score(hist, indi)

            # Build DataFrame (for chart function)
            df = pd.DataFrame([{
            "代码": code,
            "名称": code,          # 若需中文名称，可自行查询
            "总市值（亿元）": "N/A",
            "起始日价（元）": hist['close'].iloc[0],
            "截止日价（元）": hist['close'].iloc[-1],
            "涨幅(%)": (hist['close'].iloc[-1] / hist['close'].iloc[0] - 1) * 100,
            "得分": score
        }])

            # Output dir + async generate chart
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
            raise HTTPException(status_code=404, detail="All stocks failed to analyze or no data")

        df = pd.DataFrame(results)

        # Sort & take top k
        if request.topgains:
            df.sort_values(["涨幅(%)", "代码"], ascending=[False, True], inplace=True)
        else:
            df.sort_values(["得分", "代码"], ascending=[False, True], inplace=True)

        df_topk = df.head(request.k)

        # Save CSV
        csv_path = os.path.join(out_dir, "analysis_results.csv")
        df.to_csv(csv_path, index=False, encoding="utf_8_sig")

        # Return JSON
        return {
            "results"   : df_topk.to_dict(orient="records"),
            "csv_url"   : f"/static/{os.path.basename(out_dir)}/analysis_results.csv",
            "charts_dir": f"/static/{os.path.basename(out_dir)}"
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stocks/list", summary="POST Get Stock List (Optional Exchange)")
async def get_stock_list(req: StockListRequest):
    """
    **exchange** leave blank = all; optional SH / SZ / US
    **refresh** set True will ignore local cache and call AkShare again.
    """
    df = _load_stocks(req.refresh)

    # Filter by exchange
    if req.exchange:
        ex = req.exchange.upper()
        if ex == "SH":
            df = df[df["code"].str.startswith("6")]
        elif ex == "SZ":
            df = df[df["code"].str.startswith(("0", "3"))]
        elif ex == "US":
            # US stock code example: 105.GOOG / 106.BABA / 105.AMZN
            df = df[df["code"].str.match(r"^\d{3}\.[A-Za-z]{3,5}$", na=False)]
        else:
            raise HTTPException(400, "exchange only supports SH/SZ/US")
    # Return JSON (NaN→None automatically)
    return {
        "count": len(df),
        "data": jsonable_encoder(df.to_dict("records"))
    }

# 论坛相关的API端点
@app.get("/api/forum/posts")
async def get_forum_posts():
    """Get all forum posts"""
    try:
        posts = db.get_all_posts()
        return {"posts": posts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get posts: {str(e)}")

@app.get("/api/forum/categories")
async def get_forum_categories():
    """Get all forum categories"""
    try:
        categories = db.get_categories()
        return {"categories": categories}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get categories: {str(e)}")

@app.get("/api/forum/tags")
async def get_forum_tags():
    """Get all forum tags"""
    try:
        tags = db.get_tags()
        return {"tags": tags}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get tags: {str(e)}")

@app.get("/api/forum/stats")
async def get_forum_stats():
    """Get forum statistics"""
    try:
        stats = db.get_forum_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")

@app.post("/api/forum/posts")
async def create_forum_post(
    request: CreatePostRequest,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """Create a new post (login required)"""
    if not current_user:
        raise HTTPException(status_code=401, detail="Please login before posting")
    
    try:
        post_id = db.create_post_by_user_id(
            title=request.title,
            content=request.content,
            user_id=current_user['id'],
            category=request.category,
            tags=request.tags
        )
        return {"message": "Post created successfully", "post_id": post_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create post: {str(e)}")

@app.get("/api/forum/my-posts")
async def get_my_posts(current_user: Optional[Dict] = Depends(get_current_user)):
    """Get current user's posts (login required)"""
    if not current_user:
        raise HTTPException(status_code=401, detail="Please login first")
    
    try:
        posts = db.get_user_posts(current_user['id'])
        return {"posts": posts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get user posts: {str(e)}")

@app.get("/api/forum/posts/{post_id}")
async def get_forum_post_detail(post_id: int):
    """Get forum post detail by ID"""
    try:
        post = db.get_post_by_id(post_id)
        if not post:
            raise HTTPException(status_code=404, detail="Post not found")
        return {"post": post}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get post detail: {str(e)}")

@app.get("/api/forum/posts/{post_id}/replies")
async def get_post_replies(post_id: int):
    """获取某个帖子的所有评论"""
    try:
        replies = db.get_replies_by_post_id(post_id)
        return {"replies": replies}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get replies: {str(e)}")

@app.post("/api/forum/posts/{post_id}/replies")
async def add_post_reply(post_id: int, request: CreateReplyRequest, current_user: Optional[Dict] = Depends(get_current_user)):
    """添加评论（需登录）"""
    if not current_user:
        raise HTTPException(status_code=401, detail="Please login before commenting")
    try:
        reply_id = db.add_reply(post_id, current_user['id'], request.content)
        return {"message": "Reply added", "reply_id": reply_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add reply: {str(e)}")

@app.post("/api/forum/posts/{post_id}/like")
async def like_post(post_id: int):
    """给帖子点赞，返回最新点赞数"""
    try:
        likes = db.like_post(post_id)
        return {"likes": likes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to like post: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)