from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Optional, List
from pydantic import BaseModel
import spacy
import os
from dotenv import load_dotenv
from database import db, init_db
from bson import ObjectId

# Load environment variables
load_dotenv()

# 加载spaCy模型
nlp = spacy.load("zh_core_web_sm")

# 配置
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# 创建FastAPI应用
app = FastAPI()

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 密码加密
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# 模型定义
class User(BaseModel):
    username: str
    disabled: Optional[bool] = None
    created_at: Optional[datetime] = None

class UserInDB(User):
    hashed_password: str

class UserCreate(BaseModel):
    username: str
    password: str

class UserUpdate(BaseModel):
    new_password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class TextAnalysisRequest(BaseModel):
    text: str

class AnalysisResult(BaseModel):
    id: str
    text: str
    result: dict
    timestamp: datetime
    username: str

class UserStats(BaseModel):
    today: int
    week: int
    total: int

# 验证用户
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

async def get_user(username: str):
    user_dict = await db.users.find_one({"username": username})
    if user_dict:
        return UserInDB(**user_dict)
    return None

async def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    user = await get_user(username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user

# 创建访问令牌
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# 获取当前用户
async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="无效的认证凭据",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = await get_user(username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

# 注册路由
@app.post("/api/auth/register", response_model=User)
async def register_user(user: UserCreate):
    if await get_user(user.username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="用户名已存在"
        )
    
    hashed_password = pwd_context.hash(user.password)
    user_dict = {
        "username": user.username,
        "hashed_password": hashed_password,
        "disabled": False,
        "created_at": datetime.utcnow()
    }
    
    await db.users.insert_one(user_dict)
    return User(**user_dict)

# 登录路由
@app.post("/api/auth/login", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# 修改密码路由
@app.post("/api/auth/change-password")
async def change_password(
    user_update: UserUpdate,
    current_user: User = Depends(get_current_user)
):
    user = await get_user(current_user.username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="用户不存在"
        )
    
    hashed_password = pwd_context.hash(user_update.new_password)
    await db.users.update_one(
        {"username": current_user.username},
        {"$set": {"hashed_password": hashed_password}}
    )
    return {"message": "密码修改成功"}

# 获取用户信息路由
@app.get("/api/users/profile", response_model=User)
async def get_user_profile(current_user: User = Depends(get_current_user)):
    user = await get_user(current_user.username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="用户不存在"
        )
    return User(**user.dict())

# 文本分析路由
@app.post("/api/nlp/analyze", response_model=AnalysisResult)
async def analyze_text(
    request: TextAnalysisRequest,
    current_user: User = Depends(get_current_user)
):
    doc = nlp(request.text)
    
    # 进行基本的NLP分析
    analysis = {
        "entities": [(ent.text, ent.label_) for ent in doc.ents],
        "tokens": [token.text for token in doc],
        "pos_tags": [(token.text, token.pos_) for token in doc],
    }
    
    # 保存分析结果
    result_dict = {
        "text": request.text,
        "result": analysis,
        "timestamp": datetime.utcnow(),
        "username": current_user.username
    }
    
    result = await db.analysis_results.insert_one(result_dict)
    result_dict["id"] = str(result.inserted_id)
    
    return AnalysisResult(**result_dict)

# 获取分析历史路由
@app.get("/api/nlp/history", response_model=List[AnalysisResult])
async def get_analysis_history(
    current_user: User = Depends(get_current_user),
    limit: int = 10
):
    results = await db.analysis_results.find(
        {"username": current_user.username}
    ).sort("timestamp", -1).limit(limit).to_list(length=limit)
    
    return [AnalysisResult(**{**result, "id": str(result["_id"])}) for result in results]

# 获取用户统计信息路由
@app.get("/api/users/stats", response_model=UserStats)
async def get_user_stats(current_user: User = Depends(get_current_user)):
    now = datetime.utcnow()
    today_start = datetime(now.year, now.month, now.day)
    week_start = today_start - timedelta(days=now.weekday())
    
    total = await db.analysis_results.count_documents({"username": current_user.username})
    today = await db.analysis_results.count_documents({
        "username": current_user.username,
        "timestamp": {"$gte": today_start}
    })
    week = await db.analysis_results.count_documents({
        "username": current_user.username,
        "timestamp": {"$gte": week_start}
    })
    
    return UserStats(today=today, week=week, total=total)

@app.on_event("startup")
async def startup_db_client():
    init_db()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 