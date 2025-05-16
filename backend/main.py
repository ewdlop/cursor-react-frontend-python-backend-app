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
from collections import Counter
from textblob import TextBlob
import jieba.analyse
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
from nltk.util import ngrams
from nltk.translate.bleu_score import sentence_bleu
from nltk.metrics.distance import edit_distance
from langdetect import detect, LangDetectException
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

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
    features: Optional[List[str]] = ["basic"]  # 可选的分析特征列表
    compare_text: Optional[str] = None  # 用于文本比较的第二个文本

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
    analysis = {}
    
    # 基本分析
    if "basic" in request.features:
        analysis["entities"] = [(ent.text, ent.label_) for ent in doc.ents]
        analysis["tokens"] = [token.text for token in doc]
        analysis["pos_tags"] = [(token.text, token.pos_) for token in doc]
    
    # 情感分析
    if "sentiment" in request.features:
        blob = TextBlob(request.text)
        analysis["sentiment"] = {
            "polarity": blob.sentiment.polarity,
            "subjectivity": blob.sentiment.subjectivity
        }
    
    # 关键词提取
    if "keywords" in request.features:
        keywords = jieba.analyse.extract_tags(request.text, topK=5, withWeight=True)
        analysis["keywords"] = [(word, weight) for word, weight in keywords]
    
    # 词频统计
    if "word_freq" in request.features:
        words = [token.text for token in doc if not token.is_stop and not token.is_punct]
        word_freq = Counter(words)
        analysis["word_frequency"] = dict(word_freq.most_common(10))
    
    # 依存句法分析
    if "dependencies" in request.features:
        analysis["dependencies"] = [(token.text, token.dep_, token.head.text) for token in doc]
    
    # NLTK 新增功能
    if "nltk" in request.features:
        # 句子分割
        sentences = sent_tokenize(request.text)
        analysis["sentences"] = sentences
        
        # 词形还原
        lemmatizer = WordNetLemmatizer()
        tokens = word_tokenize(request.text)
        lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
        analysis["lemmatized"] = lemmatized
        
        # 词性标注
        pos_tags = nltk.pos_tag(tokens)
        analysis["nltk_pos_tags"] = pos_tags
        
        # 停用词过滤
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in tokens if word.lower() not in stop_words]
        analysis["filtered_words"] = filtered_words
        
        # N-gram 分析
        bigrams = list(ngrams(tokens, 2))
        trigrams = list(ngrams(tokens, 3))
        analysis["bigrams"] = bigrams[:10]  # 只返回前10个
        analysis["trigrams"] = trigrams[:10]
        
        # 词频分布
        fdist = FreqDist(tokens)
        analysis["word_frequency_nltk"] = dict(fdist.most_common(10))
        
        # 文本统计
        analysis["text_stats"] = {
            "sentence_count": len(sentences),
            "word_count": len(tokens),
            "unique_words": len(set(tokens)),
            "avg_sentence_length": len(tokens) / len(sentences) if sentences else 0
        }
    
    # 新增功能：文本摘要
    if "summarization" in request.features:
        sentences = sent_tokenize(request.text)
        # 使用TF-IDF进行简单的文本摘要
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sentences)
        sentence_scores = tfidf_matrix.sum(axis=1).A1
        top_sentences = np.argsort(sentence_scores)[-3:]  # 选择得分最高的3个句子
        summary = [sentences[i] for i in sorted(top_sentences)]
        analysis["summary"] = summary
    
    # 新增功能：语言检测
    if "language" in request.features:
        try:
            lang = detect(request.text)
            analysis["language"] = lang
        except LangDetectException:
            analysis["language"] = "unknown"
    
    # 新增功能：文本相似度分析
    if "similarity" in request.features and request.compare_text:
        # 使用TF-IDF和余弦相似度
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([request.text, request.compare_text])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        analysis["similarity"] = {
            "cosine_similarity": float(similarity),
            "edit_distance": edit_distance(request.text, request.compare_text)
        }
    
    # 新增功能：文本可读性分析
    if "readability" in request.features:
        sentences = sent_tokenize(request.text)
        words = word_tokenize(request.text)
        avg_sentence_length = len(words) / len(sentences)
        unique_words = len(set(words))
        analysis["readability"] = {
            "avg_sentence_length": avg_sentence_length,
            "unique_word_ratio": unique_words / len(words),
            "sentence_count": len(sentences),
            "word_count": len(words)
        }
    
    # 新增功能：命名实体关系分析
    if "entity_relations" in request.features:
        entity_relations = []
        for ent1 in doc.ents:
            for ent2 in doc.ents:
                if ent1 != ent2:
                    # 检查实体间是否存在依存关系
                    for token in doc:
                        if token.text == ent1.text:
                            for child in token.children:
                                if child.text == ent2.text:
                                    entity_relations.append({
                                        "entity1": ent1.text,
                                        "entity2": ent2.text,
                                        "relation": child.dep_
                                    })
        analysis["entity_relations"] = entity_relations
    
    # 新增功能：文本分类（简单实现）
    if "text_classification" in request.features:
        # 使用TextBlob进行简单的情感分类
        blob = TextBlob(request.text)
        sentiment = blob.sentiment.polarity
        if sentiment > 0.3:
            category = "positive"
        elif sentiment < -0.3:
            category = "negative"
        else:
            category = "neutral"
        analysis["text_classification"] = {
            "category": category,
            "confidence": abs(sentiment)
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