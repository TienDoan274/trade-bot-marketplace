# file: security.py

import os
import sys
from datetime import datetime, timedelta
from typing import Optional, Annotated

from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import crud, models, schemas
from core.database import get_db

# Lấy các biến từ .env hoặc dùng giá trị mặc định
SECRET_KEY = os.getenv("SECRET_KEY", "a_super_secret_key_for_dev")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 # 24 giờ

# Cấu hình mã hóa mật khẩu
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Cấu hình OAuth2
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Kiểm tra mật khẩu thô với mật khẩu đã mã hóa."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Mã hóa mật khẩu."""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Tạo một JWT access token mới."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# --- Dependencies để bảo vệ Endpoints ---

async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)], db: Session = Depends(get_db)) -> models.User:
    """
    Giải mã token, lấy user_id và truy xuất thông tin user từ DB.
    Đây là dependency cơ bản nhất để xác thực.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = crud.get_user(db, user_id=int(user_id))
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: Annotated[models.User, Depends(get_current_user)]) -> models.User:
    """Kiểm tra xem user có bị vô hiệu hóa hay không (nếu có trường is_active)."""
    # if current_user.disabled:
    #     raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

async def get_current_active_developer(current_user: Annotated[models.User, Depends(get_current_active_user)]) -> models.User:
    """Yêu cầu người dùng phải có vai trò là DEVELOPER hoặc ADMIN."""
    # In development mode, bypass role check
    if os.getenv("DEVELOPMENT_MODE", "false").lower() == "true":
        return current_user
    
    if current_user.role.value not in ["DEVELOPER", "ADMIN"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Operation not permitted for your role"
        )
    return current_user

async def get_current_active_admin(current_user: Annotated[models.User, Depends(get_current_active_user)]) -> models.User:
    """Yêu cầu người dùng phải có vai trò là ADMIN."""
    if current_user.role.value != "ADMIN":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You must be an admin to perform this action"
        )
    return current_user