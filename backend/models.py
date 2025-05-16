from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Dict, Any

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

class TextGenerationRequest(BaseModel):
    prompt: str
    max_length: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    num_return_sequences: int = 1

class TextGenerationResult(BaseModel):
    id: str
    prompt: str
    generated_text: str
    timestamp: str
    username: str
    settings: Dict[str, Any] 