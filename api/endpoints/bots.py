import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# file: api/endpoints/bots.py

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
from decimal import Decimal
import json

from core import crud, models, schemas, security
from core.database import get_db
from core.bot_manager import BotManager
from services.s3_manager import S3Manager

# Initialize managers
bot_manager = BotManager()
s3_manager = S3Manager()

router = APIRouter()

# --- Public endpoints ---
@router.get("/", response_model=schemas.BotListResponse)
def get_public_bots(
    skip: int = 0,
    limit: int = 50,
    category_id: Optional[int] = None,
    search: Optional[str] = None,
    sort_by: str = "created_at",
    order: str = "desc",
    db: Session = Depends(get_db)
):
    """Get public approved bots with pagination and filtering"""
    bots, total = crud.get_public_bots(
        db, skip=skip, limit=limit, category_id=category_id, 
        search=search, sort_by=sort_by, order=order
    )
    return {
        "bots": bots,
        "total": total,
        "page": skip // limit + 1,
        "per_page": limit
    }

@router.get("/{bot_id}", response_model=schemas.BotWithDeveloper)
def get_bot_details(bot_id: int, db: Session = Depends(get_db)):
    """Get detailed information about a specific bot"""
    bot = crud.get_bot_with_developer(db, bot_id=bot_id)
    if not bot:
        raise HTTPException(status_code=404, detail="Bot not found")
    return bot

@router.get("/{bot_id}/reviews", response_model=List[schemas.BotReviewWithUser])
def get_bot_reviews(
    bot_id: int,
    skip: int = 0,
    limit: int = 20,
    db: Session = Depends(get_db)
):
    """Get reviews for a specific bot"""
    return crud.get_bot_reviews(db, bot_id=bot_id, skip=skip, limit=limit)

@router.get("/{bot_id}/performance", response_model=schemas.PerformanceResponse)
def get_bot_performance(
    bot_id: int,
    days: int = 30,
    db: Session = Depends(get_db)
):
    """Get bot performance metrics"""
    return crud.get_bot_performance_metrics(db, bot_id=bot_id, days=days)

# --- Bot categories ---
@router.get("/categories/", response_model=List[schemas.BotCategoryInDB])
def get_bot_categories(db: Session = Depends(get_db)):
    """Get all bot categories"""
    return crud.get_bot_categories(db)

@router.post("/categories/", response_model=schemas.BotCategoryInDB)
def create_bot_category(
    category: schemas.BotCategoryCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(security.get_current_active_admin)
):
    """Create a new bot category (admin only)"""
    return crud.create_bot_category(db, category=category)

# --- Developer endpoints ---
@router.post("/", response_model=schemas.BotInDB, status_code=status.HTTP_201_CREATED)
def submit_new_bot(
    bot_in: schemas.BotCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(security.get_current_active_developer)
):
    """Developer submit a new bot, bot will be in PENDING status"""
    return crud.create_bot(db=db, bot=bot_in, developer_id=current_user.id)

@router.post("/with-code", response_model=schemas.BotInDB, status_code=status.HTTP_201_CREATED)
async def submit_new_bot_with_code(
    name: str = Form(...),
    description: str = Form(...),
    category_id: int = Form(...),
    price_per_month: float = Form(0.0),
    is_free: bool = Form(True),
    bot_type: str = Form("TECHNICAL"),
    version: str = Form("1.0.0"),
    config_schema: str = Form("{}"),
    default_config: str = Form("{}"),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(security.get_current_active_developer)
):
    """Developer submit a new bot with code file"""
    try:
        # Validate file type
        if not file.filename or not file.filename.endswith('.py'):
            raise HTTPException(status_code=400, detail="Only Python files are allowed")
        
        # Read file content
        content = await file.read()
        code_content = content.decode('utf-8')
        
        # Validate bot code
        validation_result = bot_manager.validate_bot_code(code_content)
        if not validation_result["valid"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Bot code validation failed: {validation_result['error']}"
            )
        
        # Parse bot type
        try:
            bot_type_enum = schemas.BotType(bot_type)
        except ValueError:
            bot_type_enum = schemas.BotType.TECHNICAL
        
        # Create bot data
        bot_data = schemas.BotCreate(
            name=name,
            description=description,
            category_id=category_id,
            version=version,
            price_per_month=Decimal(str(price_per_month)),
            is_free=is_free,
            bot_type=bot_type_enum,
            config_schema=json.loads(config_schema),
            default_config=json.loads(default_config)
        )
        
        # Create bot with S3 upload
        bot_record = crud.save_bot_with_s3(
            db=db,
            bot_data=bot_data,
            developer_id=current_user.id,
            file_content=code_content,
            file_name=file.filename
        )
        
        return bot_record
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in config fields")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create bot: {str(e)}")

@router.get("/me/bots", response_model=List[schemas.BotInDB])
def get_my_bots(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(security.get_current_active_developer)
):
    """Developer get list of their own bots"""
    return crud.get_bots_by_developer(db=db, developer_id=current_user.id)

@router.put("/{bot_id}", response_model=schemas.BotInDB)
def update_my_bot(
    bot_id: int,
    bot_in: schemas.BotUpdate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(security.get_current_active_developer)
):
    """Developer update their own bot"""
    db_bot = crud.get_bot_by_id(db, bot_id)
    if not db_bot:
        raise HTTPException(status_code=404, detail="Bot not found")
    if db_bot.developer_id != current_user.id and current_user.role != schemas.UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    return crud.update_bot(db=db, bot_id=bot_id, bot_update=bot_in)

@router.delete("/{bot_id}", response_model=schemas.BotInDB)
def archive_my_bot(
    bot_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(security.get_current_active_developer)
):
    """Developer archive their own bot"""
    db_bot = crud.get_bot_by_id(db, bot_id)
    if not db_bot:
        raise HTTPException(status_code=404, detail="Bot not found")
    if db_bot.developer_id != current_user.id and current_user.role != schemas.UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    return crud.update_bot_status(db=db, bot_id=bot_id, status=schemas.BotStatus.ARCHIVED)

@router.post("/{bot_id}/upload", response_model=schemas.BotInDB)
async def upload_bot_code(
    bot_id: int,
    file: UploadFile = File(...),
    version: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(security.get_current_active_developer)
):
    """Upload bot code file to S3"""
    try:
        db_bot = crud.get_bot_by_id(db, bot_id)
        if not db_bot:
            raise HTTPException(status_code=404, detail="Bot not found")
        if db_bot.developer_id != current_user.id and current_user.role != schemas.UserRole.ADMIN:
            raise HTTPException(status_code=403, detail="Not enough permissions")
        
        # Validate file type
        if not file.filename or not file.filename.endswith('.py'):
            raise HTTPException(status_code=400, detail="Only Python files are allowed")
        
        # Read file content
        content = await file.read()
        code_content = content.decode('utf-8')
        
        # Validate bot code
        validation_result = bot_manager.validate_bot_code(code_content)
        if not validation_result["valid"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Bot code validation failed: {validation_result['error']}"
            )
        
        # Upload to S3
        upload_result = crud.save_bot_file_to_s3(
            db=db,
            bot_id=bot_id,
            file_content=content,
            file_name=file.filename,
            version=version or None
        )
        
        # Update bot status back to PENDING for review
        crud.update_bot_status(db=db, bot_id=bot_id, status=schemas.BotStatus.PENDING)
        
        # Get updated bot
        updated_bot = crud.get_bot_by_id(db, bot_id)
        
        return updated_bot
        
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be valid UTF-8 text")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload bot code: {str(e)}")

@router.get("/{bot_id}/versions", response_model=List[str])
def get_bot_versions(
    bot_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(security.get_current_active_developer)
):
    """Get all versions of a bot from S3"""
    try:
        db_bot = crud.get_bot_by_id(db, bot_id)
        if not db_bot:
            raise HTTPException(status_code=404, detail="Bot not found")
        if db_bot.developer_id != current_user.id and current_user.role != schemas.UserRole.ADMIN:
            raise HTTPException(status_code=403, detail="Not enough permissions")
        
        # Return current version as fallback
        return [db_bot.version] if db_bot.version else ["1.0.0"]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get bot versions: {str(e)}")

@router.get("/{bot_id}/download/{version}")
def download_bot_code(
    bot_id: int,
    version: str,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(security.get_current_active_developer)
):
    """Download bot code from S3"""
    try:
        db_bot = crud.get_bot_by_id(db, bot_id)
        if not db_bot:
            raise HTTPException(status_code=404, detail="Bot not found")
        if db_bot.developer_id != current_user.id and current_user.role != schemas.UserRole.ADMIN:
            raise HTTPException(status_code=403, detail="Not enough permissions")
        
        code_content = s3_manager.download_bot_code(bot_id, version)
        
        return {
            "bot_id": bot_id,
            "version": version,
            "code_content": code_content,
            "download_time": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download bot code: {str(e)}")

@router.post("/{bot_id}/test")
def test_bot_code(
    bot_id: int,
    test_data: dict = {"symbol": "BTCUSDT", "amount": 100},
    version: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(security.get_current_active_developer)
):
    """Test bot code from S3"""
    try:
        db_bot = crud.get_bot_by_id(db, bot_id)
        if not db_bot:
            raise HTTPException(status_code=404, detail="Bot not found")
        if db_bot.developer_id != current_user.id and current_user.role != schemas.UserRole.ADMIN:
            raise HTTPException(status_code=403, detail="Not enough permissions")
        
        # Load bot from S3
        bot_instance = crud.load_bot_from_s3(
            bot_id=bot_id,
            version=version,
            user_config=test_data,
            user_api_keys={
                'key': current_user.api_key,
                'secret': current_user.api_secret
            }
        )
        
        if not bot_instance:
            raise HTTPException(status_code=400, detail="Failed to load bot from S3")
        
        # Test bot
        test_result = bot_manager.test_bot(bot_id, test_data)
        
        return {
            "bot_id": bot_id,
            "version": version or "latest",
            "test_result": test_result,
            "test_time": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to test bot: {str(e)}")

# --- User endpoints ---
@router.post("/{bot_id}/reviews", response_model=schemas.BotReviewInDB)
def create_bot_review(
    bot_id: int,
    review: schemas.BotReviewCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(security.get_current_active_user)
):
    """Create a review for a bot"""
    # Check if bot exists and is approved
    bot = crud.get_bot_by_id(db, bot_id)
    if not bot or bot.status != schemas.BotStatus.APPROVED:
        raise HTTPException(status_code=404, detail="Bot not found or not approved")
    
    # Check if user has already reviewed this bot
    existing_review = crud.get_user_bot_review(db, user_id=current_user.id, bot_id=bot_id)
    if existing_review:
        raise HTTPException(status_code=400, detail="You have already reviewed this bot")
    
    # Check if user has subscribed to this bot
    has_subscription = crud.user_has_bot_subscription(db, user_id=current_user.id, bot_id=bot_id)
    if not has_subscription:
        raise HTTPException(status_code=400, detail="You must subscribe to this bot before reviewing")
    
    return crud.create_bot_review(db, review=review, user_id=current_user.id, bot_id=bot_id)

@router.put("/{bot_id}/reviews/{review_id}", response_model=schemas.BotReviewInDB)
def update_bot_review(
    bot_id: int,
    review_id: int,
    review_update: schemas.BotReviewBase,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(security.get_current_active_user)
):
    """Update a bot review"""
    review = crud.get_bot_review(db, review_id=review_id)
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    if review.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    
    return crud.update_bot_review(db, review_id=review_id, review_update=review_update)

@router.delete("/{bot_id}/reviews/{review_id}")
def delete_bot_review(
    bot_id: int,
    review_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(security.get_current_active_user)
):
    """Delete a bot review"""
    review = crud.get_bot_review(db, review_id=review_id)
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    if review.user_id != current_user.id and current_user.role != schemas.UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    
    crud.delete_bot_review(db, review_id=review_id)
    return {"message": "Review deleted successfully"}