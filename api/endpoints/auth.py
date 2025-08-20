import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# file: api/endpoints/auth.py

from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from typing import Annotated, List, Optional

from core import crud, schemas, security
from core.database import get_db

router = APIRouter()

@router.post("/register", response_model=schemas.UserInDB)
def register_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    """Register a new user"""
    db_user = crud.get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    return crud.create_user(db=db, user=user)

@router.post("/token")
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: Session = Depends(get_db)
):
    """Login and get access token"""
    user = crud.get_user_by_email(db, email=form_data.username)
    if not user or not security.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = security.create_access_token(
        data={"sub": str(user.id), "role": user.role.value}
    )
    return {
        "access_token": access_token, 
        "token_type": "bearer",
        "user_id": user.id,
        "email": user.email,
        "role": user.role.value
    }

@router.get("/me", response_model=schemas.UserProfile)
def get_current_user_profile(
    current_user: schemas.UserInDB = Depends(security.get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get current user profile with statistics"""
    return crud.get_user_profile(db, user_id=current_user.id)

@router.put("/me", response_model=schemas.UserInDB)
def update_user_profile(
    user_update: schemas.UserUpdate,
    current_user: schemas.UserInDB = Depends(security.get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update current user profile"""
    return crud.update_user(db, user_id=current_user.id, user_update=user_update)

@router.get("/users", response_model=List[schemas.UserInDB])
def list_users(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: schemas.UserInDB = Depends(security.get_current_active_admin)
):
    """List all users (admin only)"""
    return crud.get_users(db, skip=skip, limit=limit)

@router.get("/users/{user_id}", response_model=schemas.UserProfile)
def get_user_profile(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: schemas.UserInDB = Depends(security.get_current_active_user)
):
    """Get user profile by ID"""
    if current_user.id != user_id and current_user.role != schemas.UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    
    user_profile = crud.get_user_profile(db, user_id=user_id)
    if not user_profile:
        raise HTTPException(status_code=404, detail="User not found")
    return user_profile

# --- Exchange Credentials Endpoints ---
@router.get("/me/credentials", response_model=List[schemas.ExchangeCredentialsPublic])
def get_my_exchange_credentials(
    exchange: Optional[str] = Query(None, description="Filter by exchange"),
    is_testnet: Optional[bool] = Query(None, description="Filter by testnet/mainnet"),
    current_user: schemas.UserInDB = Depends(security.get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get current user's exchange credentials"""
    credentials = crud.get_user_exchange_credentials(
        db, user_id=current_user.id, exchange=exchange, is_testnet=is_testnet
    )
    
    # Convert to public view (hide sensitive data)
    result = []
    for cred in credentials:
        result.append(schemas.ExchangeCredentialsPublic(
            id=cred.id,
            exchange=cred.exchange,
            is_testnet=cred.is_testnet,
            is_active=cred.is_active,
            validation_status=cred.validation_status,
            last_validated=cred.last_validated,
            api_key_preview=cred.api_key[:8] + "..." if len(cred.api_key) > 8 else cred.api_key
        ))
    
    return result

@router.post("/me/credentials", response_model=schemas.ExchangeCredentialsPublic, status_code=status.HTTP_201_CREATED)
def create_exchange_credentials(
    credentials: schemas.ExchangeCredentialsCreate,
    current_user: schemas.UserInDB = Depends(security.get_current_active_user),
    db: Session = Depends(get_db)
):
    """Add new exchange credentials"""
    try:
        new_credentials = crud.create_exchange_credentials(
            db, credentials=credentials, user_id=current_user.id
        )
        
        # Return public view
        return schemas.ExchangeCredentialsPublic(
            id=new_credentials.id,
            exchange=new_credentials.exchange,
            is_testnet=new_credentials.is_testnet,
            is_active=new_credentials.is_active,
            validation_status=new_credentials.validation_status,
            last_validated=new_credentials.last_validated,
            api_key_preview=new_credentials.api_key[:8] + "..." if len(new_credentials.api_key) > 8 else new_credentials.api_key
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create credentials: {str(e)}")

@router.put("/me/credentials/{credentials_id}", response_model=schemas.ExchangeCredentialsPublic)
def update_exchange_credentials(
    credentials_id: int,
    credentials_update: schemas.ExchangeCredentialsUpdate,
    current_user: schemas.UserInDB = Depends(security.get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update exchange credentials"""
    updated_credentials = crud.update_exchange_credentials(
        db, credentials_id=credentials_id, user_id=current_user.id, credentials_update=credentials_update
    )
    
    if not updated_credentials:
        raise HTTPException(status_code=404, detail="Credentials not found")
    
    # Return public view
    return schemas.ExchangeCredentialsPublic(
        id=updated_credentials.id,
        exchange=updated_credentials.exchange,
        is_testnet=updated_credentials.is_testnet,
        is_active=updated_credentials.is_active,
        validation_status=updated_credentials.validation_status,
        last_validated=updated_credentials.last_validated,
        api_key_preview=updated_credentials.api_key[:8] + "..." if len(updated_credentials.api_key) > 8 else updated_credentials.api_key
    )

@router.delete("/me/credentials/{credentials_id}")
def delete_exchange_credentials(
    credentials_id: int,
    current_user: schemas.UserInDB = Depends(security.get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete exchange credentials"""
    success = crud.delete_exchange_credentials(
        db, credentials_id=credentials_id, user_id=current_user.id
    )
    
    if not success:
        raise HTTPException(status_code=404, detail="Credentials not found")
    
    return {"message": "Credentials deleted successfully"}

@router.post("/me/credentials/{credentials_id}/validate", response_model=schemas.ExchangeCredentialsValidation)
def validate_exchange_credentials(
    credentials_id: int,
    current_user: schemas.UserInDB = Depends(security.get_current_active_user),
    db: Session = Depends(get_db)
):
    """Validate specific exchange credentials"""
    # Get credentials
    credentials = crud.get_exchange_credentials_by_id(
        db, credentials_id=credentials_id, user_id=current_user.id
    )
    
    if not credentials:
        raise HTTPException(status_code=404, detail="Credentials not found")
    
    # Validate using exchange factory
    from services.exchange_factory import ExchangeFactory
    
    is_valid, message = ExchangeFactory.validate_exchange_credentials(
        exchange_name=credentials.exchange.value,
        api_key=credentials.api_key,
        api_secret=credentials.api_secret,
        testnet=credentials.is_testnet
    )
    
    # Update validation status in database
    crud.update_credentials_validation(db, credentials_id, is_valid, message)
    
    return schemas.ExchangeCredentialsValidation(
        valid=is_valid,
        message=message,
        exchange=credentials.exchange,
        is_testnet=credentials.is_testnet
    )