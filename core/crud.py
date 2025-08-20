# file: crud.py

import sys
import os
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import func, desc, asc, and_, or_
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime, timedelta
from decimal import Decimal
import hashlib
import json
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import models, schemas, security

# Lazy initialization - only create when needed
_s3_manager = None
_bot_manager = None

def get_s3_manager():
    """Get S3Manager instance (lazy initialization)"""
    global _s3_manager
    if _s3_manager is None:
        try:
            from services.s3_manager import S3Manager
            _s3_manager = S3Manager()
        except Exception as e:
            print(f"Warning: Could not initialize S3Manager: {e}")
            _s3_manager = None
    return _s3_manager

def get_bot_manager():
    """Get BotManager instance (lazy initialization)"""
    global _bot_manager
    if _bot_manager is None:
        try:
            from core.bot_manager import BotManager
            _bot_manager = BotManager()
        except Exception as e:
            print(f"Warning: Could not initialize BotManager: {e}")
            _bot_manager = None
    return _bot_manager

# --- User CRUD ---
def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()

def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()

def get_users(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.User).offset(skip).limit(limit).all()

def create_user(db: Session, user: schemas.UserCreate):
    hashed_password = security.get_password_hash(user.password)
    db_user = models.User(
        email=user.email,
        hashed_password=hashed_password,
        role=user.role,
        developer_name=user.developer_name,
        developer_bio=user.developer_bio,
        developer_website=user.developer_website
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def update_user(db: Session, user_id: int, user_update: schemas.UserUpdate):
    db_user = get_user(db, user_id)
    if db_user:
        update_data = user_update.dict(exclude_unset=True)
        for key, value in update_data.items():
            setattr(db_user, key, value)
        db.commit()
        db.refresh(db_user)
    return db_user

def get_user_profile(db: Session, user_id: int):
    user = get_user(db, user_id)
    if not user:
        return None
    
    # Get subscription counts
    total_subscriptions = db.query(models.Subscription).filter(
        models.Subscription.user_id == user_id
    ).count()
    
    active_subscriptions = db.query(models.Subscription).filter(
        models.Subscription.user_id == user_id,
        models.Subscription.status == models.SubscriptionStatus.ACTIVE
    ).count()
    
    # Get bot development stats
    total_developed_bots = db.query(models.Bot).filter(
        models.Bot.developer_id == user_id
    ).count()
    
    approved_bots = db.query(models.Bot).filter(
        models.Bot.developer_id == user_id,
        models.Bot.status == models.BotStatus.APPROVED
    ).count()
    
    # Create profile response
    profile = schemas.UserProfile(
        id=user.id,
        email=user.email,
        role=user.role,
        is_active=user.is_active,
        developer_name=user.developer_name,
        developer_bio=user.developer_bio,
        developer_website=user.developer_website,
        created_at=user.created_at,
        updated_at=user.updated_at,
        total_subscriptions=total_subscriptions,
        active_subscriptions=active_subscriptions,
        total_developed_bots=total_developed_bots,
        approved_bots=approved_bots
    )
    
    return profile

def update_user_status(db: Session, user_id: int, user_update: schemas.AdminUserUpdate):
    db_user = get_user(db, user_id)
    if db_user:
        db_user.is_active = user_update.is_active
        if user_update.role:
            db_user.role = user_update.role
        db.commit()
        db.refresh(db_user)
    return db_user

def get_all_users(db: Session, skip: int = 0, limit: int = 100, role_filter: Optional[schemas.UserRole] = None, active_only: bool = False):
    query = db.query(models.User)
    
    if role_filter:
        query = query.filter(models.User.role == role_filter)
    
    if active_only:
        query = query.filter(models.User.is_active == True)
    
    return query.offset(skip).limit(limit).all()

def update_user_by_admin(db: Session, user_id: int, user_update: schemas.AdminUserUpdate):
    return update_user_status(db, user_id, user_update)

def delete_user(db: Session, user_id: int):
    user = get_user(db, user_id)
    if user:
        db.delete(user)
        db.commit()
    return True

# --- Bot Category CRUD ---
def get_bot_categories(db: Session):
    return db.query(models.BotCategory).all()

def create_bot_category(db: Session, category: schemas.BotCategoryCreate):
    db_category = models.BotCategory(**category.dict())
    db.add(db_category)
    db.commit()
    db.refresh(db_category)
    return db_category

# --- Bot CRUD ---
def create_bot(db: Session, bot: schemas.BotCreate, developer_id: int, status: schemas.BotStatus = schemas.BotStatus.PENDING, approved_by: Optional[int] = None):
    db_bot = models.Bot(
        **bot.dict(),
        developer_id=developer_id,
        status=status,
        approved_by=approved_by,
        approved_at=datetime.utcnow() if status == schemas.BotStatus.APPROVED else None
    )
    db.add(db_bot)
    db.commit()
    db.refresh(db_bot)
    return db_bot

def get_bot_by_id(db: Session, bot_id: int):
    return db.query(models.Bot).filter(models.Bot.id == bot_id).first()

def get_bot_with_developer(db: Session, bot_id: int):
    return db.query(models.Bot).join(models.User, models.Bot.developer_id == models.User.id).filter(models.Bot.id == bot_id).first()

def get_approved_bots(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Bot).filter(
        models.Bot.status == models.BotStatus.APPROVED
    ).offset(skip).limit(limit).all()

def get_public_bots(db: Session, skip: int = 0, limit: int = 50, category_id: Optional[int] = None, search: Optional[str] = None, sort_by: str = "created_at", order: str = "desc"):
    query = db.query(models.Bot).filter(models.Bot.status == schemas.BotStatus.APPROVED)
    
    if category_id:
        query = query.filter(models.Bot.category_id == category_id)
    
    if search:
        search_term = f"%{search}%"
        query = query.filter(
            or_(
                models.Bot.name.ilike(search_term),
                models.Bot.description.ilike(search_term)
            )
        )
    
    # Apply sorting
    if sort_by == "created_at":
        query = query.order_by(desc(models.Bot.created_at) if order == "desc" else asc(models.Bot.created_at))
    elif sort_by == "rating":
        query = query.order_by(desc(models.Bot.average_rating) if order == "desc" else asc(models.Bot.average_rating))
    elif sort_by == "price":
        query = query.order_by(desc(models.Bot.price_per_month) if order == "desc" else asc(models.Bot.price_per_month))
    
    total = query.count()
    bots = query.offset(skip).limit(limit).all()
    
    return bots, total

def get_bots_by_developer(db: Session, developer_id: int):
    return db.query(models.Bot).filter(models.Bot.developer_id == developer_id).all()

def get_all_bots(db: Session, skip: int = 0, limit: int = 100, status_filter: Optional[schemas.BotStatus] = None):
    query = db.query(models.Bot)
    
    if status_filter:
        query = query.filter(models.Bot.status == status_filter)
    
    return query.offset(skip).limit(limit).all()

def get_bots_by_status(db: Session, status: schemas.BotStatus):
    return db.query(models.Bot).filter(models.Bot.status == status).all()

def update_bot(db: Session, bot_id: int, bot_update: schemas.BotUpdate):
    db_bot = get_bot_by_id(db, bot_id)
    if db_bot:
        update_data = bot_update.dict(exclude_unset=True)
        for key, value in update_data.items():
            setattr(db_bot, key, value)
        db.commit()
        db.refresh(db_bot)
    return db_bot

def update_bot_status(db: Session, bot_id: int, status: schemas.BotStatus):
    db_bot = get_bot_by_id(db, bot_id)
    if db_bot:
        db_bot.status = status
        db.commit()
        db.refresh(db_bot)
    return db_bot

def update_bot_approval(db: Session, bot_id: int, status: schemas.BotStatus, approved_by: int, approval_notes: Optional[str] = None):
    db_bot = get_bot_by_id(db, bot_id)
    if db_bot:
        db_bot.status = status
        db_bot.approved_by = approved_by
        db_bot.approved_at = datetime.utcnow()
        if approval_notes:
            db_bot.approval_notes = approval_notes
        db.commit()
        db.refresh(db_bot)
    return db_bot

def save_bot_file_to_s3(db: Session, bot_id: int, file_content: bytes, file_name: str, version: Optional[str] = None) -> Dict[str, Any]:
    """Save bot file to S3 and return upload information"""
    try:
        # Upload to S3
        code_content = file_content.decode('utf-8')
        upload_result = get_s3_manager().upload_bot_code(
            bot_id=bot_id,
            code_content=code_content,
            filename=file_name,
            version=version
        )
        
        # Update bot record with S3 information
        db_bot = get_bot_by_id(db, bot_id)
        if db_bot:
            db_bot.code_path = upload_result['s3_key']
            db_bot.version = upload_result['version']
            db.commit()
            db.refresh(db_bot)
        
        return upload_result
        
    except Exception as e:
        raise Exception(f"Failed to save bot file to S3: {str(e)}")

def save_bot_file(db: Session, bot_id: int, file_content: bytes, file_name: str):
    """Legacy function - now uses S3"""
    upload_result = save_bot_file_to_s3(db, bot_id, file_content, file_name)
    return upload_result['s3_key']

def update_bot_code_path(db: Session, bot_id: int, code_path: str):
    """Update bot code path (S3 key)"""
    db_bot = get_bot_by_id(db, bot_id)
    if db_bot:
        db_bot.code_path = code_path
        db.commit()
        db.refresh(db_bot)
    return db_bot

def save_bot_with_s3(db: Session, bot_data: schemas.BotCreate, developer_id: int, 
                     file_content: Optional[str] = None, file_name: str = "bot.py", 
                     version: Optional[str] = None) -> models.Bot:
    """Create bot and upload code to S3 in one operation"""
    try:
        # Create bot record first
        db_bot = create_bot(db, bot_data, developer_id)
        
        # Upload code to S3 if provided
        if file_content:
            upload_result = get_s3_manager().upload_bot_code(
                bot_id=db_bot.id,
                code_content=file_content,
                filename=file_name,
                version=version
            )
            
            # Update bot with S3 information
            db_bot.code_path = upload_result['s3_key']
            db_bot.version = upload_result['version']
            db.commit()
            db.refresh(db_bot)
        
        return db_bot
        
    except Exception as e:
        # Rollback if something fails
        db.rollback()
        raise Exception(f"Failed to create bot with S3: {str(e)}")

def load_bot_from_s3(bot_id: int, version: Optional[str] = None, user_config: Optional[Dict[str, Any]] = None, 
                     user_api_keys: Optional[Dict[str, str]] = None):
    """Load bot from S3 using BotManager"""
    try:
        return get_bot_manager().load_bot_from_s3(bot_id, version, user_config, user_api_keys)
    except Exception as e:
        raise Exception(f"Failed to load bot from S3: {str(e)}")

def get_bot_s3_info(db: Session, bot_id: int) -> Optional[Dict[str, Any]]:
    """Get bot S3 information"""
    db_bot = get_bot_by_id(db, bot_id)
    if not db_bot:
        return None
    
    return {
        "bot_id": bot_id,
        "s3_key": db_bot.code_path,
        "version": db_bot.version,
        "updated_at": db_bot.updated_at
    }

def get_active_subscriptions_for_bot(db: Session, bot_id: int):
    return db.query(models.Subscription).filter(
        models.Subscription.bot_id == bot_id,
        models.Subscription.status == models.SubscriptionStatus.ACTIVE
    ).all()

def delete_bot(db: Session, bot_id: int):
    """Delete bot and its S3 files"""
    try:
        bot = get_bot_by_id(db, bot_id)
        if bot:
            # Delete from S3 first
            if bot.code_path:
                try:
                    get_s3_manager().delete_bot_files(bot_id)
                except Exception as e:
                    # Log but don't fail if S3 deletion fails
                    print(f"Warning: Failed to delete S3 files for bot {bot_id}: {e}")
            
            # Delete from database
            db.delete(bot)
            db.commit()
            
            # Clear from bot manager cache
            get_bot_manager().unload_bot(bot_id)
            
        return True
    except Exception as e:
        db.rollback()
        raise Exception(f"Failed to delete bot: {str(e)}")

# --- Bot Review CRUD ---
def create_bot_review(db: Session, review: schemas.BotReviewCreate, user_id: int, bot_id: int):
    db_review = models.BotReview(
        user_id=user_id,
        bot_id=bot_id,
        rating=review.rating,
        review_text=review.review_text
    )
    db.add(db_review)
    db.commit()
    db.refresh(db_review)
    
    # Update bot's average rating
    update_bot_rating(db, bot_id)
    
    return db_review

def get_bot_reviews(db: Session, bot_id: int, skip: int = 0, limit: int = 20):
    return db.query(models.BotReview).options(
        joinedload(models.BotReview.user)
    ).filter(models.BotReview.bot_id == bot_id).offset(skip).limit(limit).all()

def get_user_bot_review(db: Session, user_id: int, bot_id: int):
    return db.query(models.BotReview).filter(
        models.BotReview.user_id == user_id,
        models.BotReview.bot_id == bot_id
    ).first()

def get_bot_review(db: Session, review_id: int):
    return db.query(models.BotReview).filter(models.BotReview.id == review_id).first()

def update_bot_review(db: Session, review_id: int, review_update: schemas.BotReviewBase):
    db_review = get_bot_review(db, review_id)
    if db_review:
        update_data = review_update.dict(exclude_unset=True)
        for key, value in update_data.items():
            setattr(db_review, key, value)
        db.commit()
        db.refresh(db_review)
        
        # Update bot's average rating
        update_bot_rating(db, db_review.bot_id)
    
    return db_review

def delete_bot_review(db: Session, review_id: int):
    review = get_bot_review(db, review_id)
    if review:
        bot_id = review.bot_id
        db.delete(review)
        db.commit()
        
        # Update bot's average rating
        update_bot_rating(db, bot_id)
    
    return True

def update_bot_rating(db: Session, bot_id: int):
    """Update bot's average rating and total reviews"""
    avg_rating = db.query(func.avg(models.BotReview.rating)).filter(
        models.BotReview.bot_id == bot_id
    ).scalar() or 0.0
    
    total_reviews = db.query(models.BotReview).filter(
        models.BotReview.bot_id == bot_id
    ).count()
    
    bot = get_bot_by_id(db, bot_id)
    if bot:
        bot.average_rating = float(avg_rating)
        bot.total_reviews = total_reviews
        db.commit()

def get_all_reviews(db: Session, skip: int = 0, limit: int = 100, bot_id: Optional[int] = None):
    query = db.query(models.BotReview).options(
        joinedload(models.BotReview.user)
    )
    
    if bot_id:
        query = query.filter(models.BotReview.bot_id == bot_id)
    
    return query.offset(skip).limit(limit).all()

def user_has_bot_subscription(db: Session, user_id: int, bot_id: int):
    return db.query(models.Subscription).filter(
        models.Subscription.user_id == user_id,
        models.Subscription.bot_id == bot_id
    ).first() is not None

# --- Subscription CRUD ---
def create_subscription(db: Session, sub: schemas.SubscriptionCreate, user_id: int):
    db_sub = models.Subscription(
        user_id=user_id,
        bot_id=sub.bot_id,
        instance_name=sub.instance_name,
        trading_pair=sub.trading_pair,
        timeframe=sub.timeframe,
        strategy_config=sub.strategy_config,
        execution_config=sub.execution_config.dict(),
        risk_config=sub.risk_config.dict(),
        is_testnet=sub.is_testnet,
        is_trial=sub.is_trial,
        status=models.SubscriptionStatus.ACTIVE
    )
    db.add(db_sub)
    db.commit()
    db.refresh(db_sub)
    
    # Update bot subscriber count
    update_bot_subscriber_count(db, sub.bot_id)
    
    return db_sub

def create_trial_subscription(db: Session, trial: schemas.SubscriptionTrialCreate, user_id: int):
    """Create a trial subscription with default safe settings for testnet"""
    from datetime import datetime, timedelta
    
    # Check if user already has an ACTIVE trial for this bot
    existing_active_trial = db.query(models.Subscription).filter(
        models.Subscription.user_id == user_id,
        models.Subscription.bot_id == trial.bot_id,
        models.Subscription.is_trial == True,
        models.Subscription.status == models.SubscriptionStatus.ACTIVE
    ).first()
    
    if existing_active_trial:
        raise ValueError("User already has an active trial subscription for this bot. Please cancel the existing trial first.")
    
    # Create trial subscription with safe defaults
    trial_expires = datetime.utcnow() + timedelta(hours=trial.trial_duration_hours)
    
    # Default safe execution config for trials
    default_execution_config = {
        "initial_balance": 1000.0,  # $1000 testnet balance
        "max_position_size": 0.1,   # 10% max position
        "order_type": "market",
        "slippage_tolerance": 0.001
    }
    
    # Default safe risk config for trials
    default_risk_config = {
        "max_loss_per_trade": 0.02,  # 2% max loss per trade
        "max_daily_loss": 0.05,      # 5% max daily loss
        "stop_loss_percentage": 0.02,
        "take_profit_percentage": 0.04,
        "trailing_stop": False
    }
    
    db_trial = models.Subscription(
        user_id=user_id,
        bot_id=trial.bot_id,
        instance_name=trial.instance_name,
        exchange_type=trial.exchange_type.value,
        trading_pair=trial.trading_pair,
        timeframe=trial.timeframe,
        strategy_config={},  # Use bot's default config
        execution_config=default_execution_config,
        risk_config=default_risk_config,
        is_testnet=True,     # Always testnet for trials
        is_trial=True,       # Mark as trial
        trial_expires_at=trial_expires,
        expires_at=trial_expires,  # Same as trial expiry
        status=models.SubscriptionStatus.ACTIVE
    )
    
    db.add(db_trial)
    db.commit()
    db.refresh(db_trial)
    
    # Update bot subscriber count
    update_bot_subscriber_count(db, trial.bot_id)
    
    return db_trial

def get_subscription_by_id(db: Session, sub_id: int):
    return db.query(models.Subscription).filter(models.Subscription.id == sub_id).first()

# --- Exchange Credentials CRUD ---
def create_exchange_credentials(db: Session, credentials: schemas.ExchangeCredentialsCreate, user_id: int):
    """Create new exchange credentials for a user"""
    # Check if credentials already exist for this exchange + testnet combination
    existing = db.query(models.ExchangeCredentials).filter(
        models.ExchangeCredentials.user_id == user_id,
        models.ExchangeCredentials.exchange == credentials.exchange,
        models.ExchangeCredentials.is_testnet == credentials.is_testnet
    ).first()
    
    if existing:
        raise ValueError(f"Credentials for {credentials.exchange.value} ({'testnet' if credentials.is_testnet else 'mainnet'}) already exist")
    
    db_credentials = models.ExchangeCredentials(
        user_id=user_id,
        exchange=credentials.exchange,
        api_key=credentials.api_key,
        api_secret=credentials.api_secret,
        api_passphrase=credentials.api_passphrase,
        is_testnet=credentials.is_testnet,
        validation_status="pending"
    )
    db.add(db_credentials)
    db.commit()
    db.refresh(db_credentials)
    return db_credentials

def get_user_exchange_credentials(db: Session, user_id: int, exchange: str = None, is_testnet: bool = None):
    """Get user's exchange credentials with optional filtering"""
    query = db.query(models.ExchangeCredentials).filter(
        models.ExchangeCredentials.user_id == user_id,
        models.ExchangeCredentials.is_active == True
    )
    
    if exchange:
        query = query.filter(models.ExchangeCredentials.exchange == exchange)
    if is_testnet is not None:
        query = query.filter(models.ExchangeCredentials.is_testnet == is_testnet)
    
    return query.all()

def get_exchange_credentials_by_id(db: Session, credentials_id: int, user_id: int):
    """Get specific exchange credentials by ID (user must own them)"""
    return db.query(models.ExchangeCredentials).filter(
        models.ExchangeCredentials.id == credentials_id,
        models.ExchangeCredentials.user_id == user_id
    ).first()

def update_exchange_credentials(db: Session, credentials_id: int, user_id: int, credentials_update: schemas.ExchangeCredentialsUpdate):
    """Update exchange credentials"""
    db_credentials = get_exchange_credentials_by_id(db, credentials_id, user_id)
    if not db_credentials:
        return None
    
    update_data = credentials_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_credentials, field, value)
    
    # Reset validation status if API keys changed
    if 'api_key' in update_data or 'api_secret' in update_data:
        db_credentials.validation_status = "pending"
        db_credentials.last_validated = None
    
    db.commit()
    db.refresh(db_credentials)
    return db_credentials

def delete_exchange_credentials(db: Session, credentials_id: int, user_id: int):
    """Delete exchange credentials"""
    db_credentials = get_exchange_credentials_by_id(db, credentials_id, user_id)
    if not db_credentials:
        return False
    
    db.delete(db_credentials)
    db.commit()
    return True

def update_credentials_validation(db: Session, credentials_id: int, is_valid: bool, message: str):
    """Update validation status of credentials"""
    from datetime import datetime
    
    db_credentials = db.query(models.ExchangeCredentials).filter(
        models.ExchangeCredentials.id == credentials_id
    ).first()
    
    if db_credentials:
        db_credentials.validation_status = "valid" if is_valid else "invalid"
        db_credentials.last_validated = datetime.utcnow()
        db.commit()
        db.refresh(db_credentials)
    
    return db_credentials

def get_subscription_with_bot(db: Session, subscription_id: int):
    return db.query(models.Subscription).options(
        joinedload(models.Subscription.bot)
    ).filter(models.Subscription.id == subscription_id).first()

def get_user_subscriptions(db: Session, user_id: int):
    return db.query(models.Subscription).filter(models.Subscription.user_id == user_id).all()

def get_user_subscriptions_paginated(db: Session, user_id: int, skip: int = 0, limit: int = 50, status_filter: Optional[schemas.SubscriptionStatus] = None):
    query = db.query(models.Subscription).options(
        joinedload(models.Subscription.bot)
    ).filter(models.Subscription.user_id == user_id)
    
    if status_filter:
        query = query.filter(models.Subscription.status == status_filter)
    
    total = query.count()
    subscriptions = query.offset(skip).limit(limit).all()
    
    return subscriptions, total

def get_user_subscription_by_name(db: Session, user_id: int, instance_name: str):
    return db.query(models.Subscription).filter(
        models.Subscription.user_id == user_id,
        models.Subscription.instance_name == instance_name,
        models.Subscription.status == models.SubscriptionStatus.ACTIVE
    ).first()

def update_subscription(db: Session, subscription_id: int, sub_update: schemas.SubscriptionUpdate):
    db_sub = get_subscription_by_id(db, subscription_id)
    if db_sub:
        update_data = sub_update.dict(exclude_unset=True)
        for key, value in update_data.items():
            if key in ['execution_config', 'risk_config'] and value:
                setattr(db_sub, key, value.dict() if hasattr(value, 'dict') else value)
            else:
                setattr(db_sub, key, value)
        db.commit()
        db.refresh(db_sub)
    return db_sub

def update_subscription_status(db: Session, sub_id: int, status: schemas.SubscriptionStatus):
    db_sub = get_subscription_by_id(db, sub_id)
    if db_sub:
        old_status = db_sub.status
        db_sub.status = status
        db.commit()
        db.refresh(db_sub)
        
        # Update bot subscriber count if subscription was cancelled
        if old_status == models.SubscriptionStatus.ACTIVE and status == models.SubscriptionStatus.CANCELLED:
            update_bot_subscriber_count(db, db_sub.bot_id)
    
    return db_sub

def get_all_subscriptions(db: Session, skip: int = 0, limit: int = 100, status_filter: Optional[schemas.SubscriptionStatus] = None):
    query = db.query(models.Subscription).options(
        joinedload(models.Subscription.bot)
    )
    
    if status_filter:
        query = query.filter(models.Subscription.status == status_filter)
    
    return query.offset(skip).limit(limit).all()

def get_active_subscriptions_for_user(db: Session, user_id: int):
    return db.query(models.Subscription).filter(
        models.Subscription.user_id == user_id,
        models.Subscription.status == models.SubscriptionStatus.ACTIVE
    ).all()

def get_active_subscriptions(db: Session):
    """Get all active subscriptions for scheduling"""
    return db.query(models.Subscription).options(
        joinedload(models.Subscription.bot),
        joinedload(models.Subscription.user)
    ).filter(
        models.Subscription.status == models.SubscriptionStatus.ACTIVE
    ).all()

def update_subscription_next_run(db: Session, subscription_id: int, next_run: datetime):
    """Update subscription's next run time"""
    subscription = get_subscription_by_id(db, subscription_id)
    if subscription:
        subscription.next_run_at = next_run
        db.commit()
        return subscription
    return None

def update_bot_subscriber_count(db: Session, bot_id: int):
    """Update bot's total subscriber count"""
    total_subscribers = db.query(models.Subscription).filter(
        models.Subscription.bot_id == bot_id,
        models.Subscription.status == models.SubscriptionStatus.ACTIVE
    ).count()
    
    bot = get_bot_by_id(db, bot_id)
    if bot:
        bot.total_subscribers = total_subscribers
        db.commit()

def close_open_trades_for_subscription(db: Session, subscription_id: int):
    """Close all open trades for a subscription"""
    open_trades = db.query(models.Trade).filter(
        models.Trade.subscription_id == subscription_id,
        models.Trade.status == models.TradeStatus.OPEN
    ).all()
    
    for trade in open_trades:
        trade.status = models.TradeStatus.CLOSED
        trade.exit_time = datetime.utcnow()
        # You would calculate exit_price and pnl here based on current market price
    
    db.commit()

# --- Trade CRUD ---
def create_trade(db: Session, trade: schemas.TradeCreate):
    db_trade = models.Trade(**trade.dict())
    db.add(db_trade)
    db.commit()
    db.refresh(db_trade)
    
    # Update subscription trade count
    update_subscription_trade_stats(db, trade.subscription_id)
    
    return db_trade

def get_subscription_trades_paginated(db: Session, subscription_id: int, skip: int = 0, limit: int = 50, status_filter: Optional[schemas.TradeStatus] = None):
    query = db.query(models.Trade).filter(models.Trade.subscription_id == subscription_id)
    
    if status_filter:
        query = query.filter(models.Trade.status == status_filter)
    
    total = query.count()
    trades = query.order_by(desc(models.Trade.entry_time)).offset(skip).limit(limit).all()
    
    return trades, total

def update_subscription_trade_stats(db: Session, subscription_id: int):
    """Update subscription trade statistics"""
    total_trades = db.query(models.Trade).filter(
        models.Trade.subscription_id == subscription_id
    ).count()
    
    winning_trades = db.query(models.Trade).filter(
        models.Trade.subscription_id == subscription_id,
        models.Trade.pnl > 0
    ).count()
    
    total_pnl = db.query(func.sum(models.Trade.pnl)).filter(
        models.Trade.subscription_id == subscription_id
    ).scalar() or Decimal('0.0')
    
    subscription = get_subscription_by_id(db, subscription_id)
    if subscription:
        subscription.total_trades = total_trades
        subscription.winning_trades = winning_trades
        subscription.total_pnl = total_pnl
        db.commit()

# --- Performance CRUD ---
def get_bot_performance_metrics(db: Session, bot_id: int, days: int = 30):
    # This would calculate performance metrics for a specific bot
    # Implementation depends on your specific requirements
    pass

def get_subscription_performance_metrics(db: Session, subscription_id: int, days: int = 30):
    # This would calculate performance metrics for a specific subscription
    # Implementation depends on your specific requirements
    pass

def log_bot_action(db: Session, subscription_id: int, action: str, details: str = None):
    """Log bot action to performance log table"""
    try:
        log_entry = models.PerformanceLog(
            subscription_id=subscription_id,
            action=action,
            price=0.0,  # Default price for non-trade actions
            quantity=0.0,  # Default quantity for non-trade actions  
            balance=0.0,  # Default balance for non-trade actions
            signal_data={"details": details} if details else {}
        )
        db.add(log_entry)
        db.commit()
        db.refresh(log_entry)
        return log_entry
    except Exception as e:
        print(f"Failed to log bot action: {e}")
        db.rollback()
        return None

def get_subscription_logs(db: Session, subscription_id: int, skip: int = 0, limit: int = 100, action_filter: Optional[str] = None):
    query = db.query(models.PerformanceLog).filter(
        models.PerformanceLog.subscription_id == subscription_id
    )
    
    if action_filter:
        query = query.filter(models.PerformanceLog.action == action_filter)
    
    return query.order_by(desc(models.PerformanceLog.timestamp)).offset(skip).limit(limit).all()

# --- Admin CRUD ---
def get_admin_stats(db: Session):
    total_users = db.query(models.User).count()
    total_developers = db.query(models.User).filter(models.User.role == models.UserRole.DEVELOPER).count()
    total_bots = db.query(models.Bot).count()
    approved_bots = db.query(models.Bot).filter(models.Bot.status == models.BotStatus.APPROVED).count()
    pending_bots = db.query(models.Bot).filter(models.Bot.status == models.BotStatus.PENDING).count()
    total_subscriptions = db.query(models.Subscription).count()
    active_subscriptions = db.query(models.Subscription).filter(
        models.Subscription.status == models.SubscriptionStatus.ACTIVE
    ).count()
    total_trades = db.query(models.Trade).count()
    
    # Calculate total revenue (you would need to implement payment tracking)
    total_revenue = Decimal('0.0')  # Placeholder
    
    return schemas.AdminStats(
        total_users=total_users,
        total_developers=total_developers,
        total_bots=total_bots,
        approved_bots=approved_bots,
        pending_bots=pending_bots,
        total_subscriptions=total_subscriptions,
        active_subscriptions=active_subscriptions,
        total_trades=total_trades,
        total_revenue=total_revenue
    )

def get_bot_performance_overview(db: Session, days: int = 30, limit: int = 50):
    # Implementation for bot performance overview
    pass

def get_user_performance_overview(db: Session, days: int = 30, limit: int = 50):
    # Implementation for user performance overview
    pass

def get_system_health(db: Session):
    # Implementation for system health check
    return {"status": "healthy", "database": "connected"}

def get_system_performance_logs(db: Session, skip: int = 0, limit: int = 100):
    # Implementation for system performance logs
    return []

# --- Pricing Plan CRUD ---
def create_pricing_plan(db: Session, plan_data: schemas.PricingPlanCreate, bot_id: int):
    """Create a new pricing plan for a bot"""
    db_plan = models.BotPricingPlan(
        bot_id=bot_id,
        **plan_data.dict()
    )
    db.add(db_plan)
    db.commit()
    db.refresh(db_plan)
    return db_plan

def get_bot_pricing_plans(db: Session, bot_id: int, active_only: bool = True):
    """Get all pricing plans for a bot"""
    query = db.query(models.BotPricingPlan).filter(models.BotPricingPlan.bot_id == bot_id)
    if active_only:
        query = query.filter(models.BotPricingPlan.is_active == True)
    return query.order_by(models.BotPricingPlan.price_per_month).all()

def get_pricing_plan_by_id(db: Session, plan_id: int):
    """Get a specific pricing plan"""
    return db.query(models.BotPricingPlan).filter(models.BotPricingPlan.id == plan_id).first()

def update_pricing_plan(db: Session, plan_id: int, plan_data: schemas.PricingPlanUpdate):
    """Update a pricing plan"""
    db_plan = get_pricing_plan_by_id(db, plan_id)
    if not db_plan:
        return None
    
    update_data = plan_data.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_plan, field, value)
    
    db.commit()
    db.refresh(db_plan)
    return db_plan

def delete_pricing_plan(db: Session, plan_id: int):
    """Delete a pricing plan (soft delete by setting is_active=False)"""
    db_plan = get_pricing_plan_by_id(db, plan_id)
    if not db_plan:
        return False
    
    db_plan.is_active = False
    db.commit()
    return True

# --- Promotion CRUD ---
def create_promotion(db: Session, promotion_data: schemas.PromotionCreate, bot_id: int, created_by: int):
    """Create a new promotion for a bot"""
    db_promotion = models.BotPromotion(
        bot_id=bot_id,
        created_by=created_by,
        **promotion_data.dict()
    )
    db.add(db_promotion)
    db.commit()
    db.refresh(db_promotion)
    return db_promotion

def get_bot_promotions(db: Session, bot_id: int, active_only: bool = True):
    """Get all promotions for a bot"""
    query = db.query(models.BotPromotion).filter(models.BotPromotion.bot_id == bot_id)
    if active_only:
        query = query.filter(models.BotPromotion.is_active == True)
    return query.all()

def get_promotion_by_code(db: Session, promotion_code: str):
    """Get a promotion by its code"""
    return db.query(models.BotPromotion).filter(
        models.BotPromotion.promotion_code == promotion_code,
        models.BotPromotion.is_active == True
    ).first()

def validate_promotion(db: Session, promotion_code: str, bot_id: int, user_id: int):
    """Validate if a promotion code can be used"""
    promotion = get_promotion_by_code(db, promotion_code)
    if not promotion:
        return None, "Invalid promotion code"
    
    if promotion.bot_id != bot_id:
        return None, "Promotion code not valid for this bot"
    
    if promotion.used_count >= promotion.max_uses:
        return None, "Promotion code usage limit exceeded"
    
    now = datetime.utcnow()
    if now < promotion.valid_from or now > promotion.valid_until:
        return None, "Promotion code expired or not yet valid"
    
    return promotion, None

def use_promotion(db: Session, promotion_id: int):
    """Mark a promotion as used"""
    promotion = db.query(models.BotPromotion).filter(models.BotPromotion.id == promotion_id).first()
    if promotion:
        promotion.used_count += 1
        db.commit()
        return True
    return False

# --- Invoice CRUD ---
def create_invoice(db: Session, invoice_data: schemas.InvoiceCreate):
    """Create a new invoice"""
    # Generate invoice number
    invoice_number = f"INV-{datetime.utcnow().strftime('%Y%m%d')}-{invoice_data.subscription_id:06d}"
    
    db_invoice = models.SubscriptionInvoice(
        invoice_number=invoice_number,
        **invoice_data.dict()
    )
    db.add(db_invoice)
    db.commit()
    db.refresh(db_invoice)
    return db_invoice

def get_user_invoices(db: Session, user_id: int, skip: int = 0, limit: int = 50):
    """Get invoices for a user"""
    return db.query(models.SubscriptionInvoice).filter(
        models.SubscriptionInvoice.user_id == user_id
    ).offset(skip).limit(limit).all()

def get_subscription_invoices(db: Session, subscription_id: int):
    """Get all invoices for a subscription"""
    return db.query(models.SubscriptionInvoice).filter(
        models.SubscriptionInvoice.subscription_id == subscription_id
    ).order_by(models.SubscriptionInvoice.created_at.desc()).all()

def update_invoice_status(db: Session, invoice_id: int, status: str, payment_date: datetime = None):
    """Update invoice status"""
    invoice = db.query(models.SubscriptionInvoice).filter(models.SubscriptionInvoice.id == invoice_id).first()
    if invoice:
        invoice.status = status
        if payment_date:
            invoice.payment_date = payment_date
        db.commit()
        return invoice
    return None

# --- Enhanced Subscription CRUD ---
def create_subscription_with_plan(
    db: Session, 
    sub: schemas.SubscriptionCreate, 
    user_id: int, 
    pricing_plan_id: int,
    promotion_code: str = None
):
    """Create subscription with pricing plan and promotion"""
    now = datetime.utcnow()
    
    # Get pricing plan
    pricing_plan = get_pricing_plan_by_id(db, pricing_plan_id)
    if not pricing_plan:
        raise ValueError(f"Pricing plan {pricing_plan_id} not found")
    
    # Validate promotion if provided
    promotion = None
    if promotion_code:
        promotion = validate_promotion(db, promotion_code, sub.bot_id, user_id)
        if not promotion:
            raise ValueError(f"Invalid promotion code: {promotion_code}")
    
    # Calculate trial period
    trial_expires = None
    is_trial = False
    if pricing_plan.trial_days > 0:
        trial_expires = now + timedelta(days=pricing_plan.trial_days)
        expires_at = trial_expires
        is_trial = True
    else:
        expires_at = now + timedelta(days=30)  # Default 30 days
        is_trial = False
    
    # Create subscription
    db_sub = models.Subscription(
        user_id=user_id,
        bot_id=sub.bot_id,
        # pricing_plan_id=pricing_plan_id,  # Tạm thời comment out
        instance_name=sub.instance_name,
        trading_pair=sub.trading_pair,
        timeframe=sub.timeframe,
        strategy_config=sub.strategy_config,
        execution_config=sub.execution_config.dict(),
        risk_config=sub.risk_config.dict(),
        is_testnet=sub.is_testnet,
        is_trial=is_trial,
        trial_expires_at=trial_expires if is_trial else None,
        expires_at=expires_at,
        status=models.SubscriptionStatus.ACTIVE,
        billing_cycle="MONTHLY",
        next_billing_date=expires_at
    )
    db.add(db_sub)
    db.commit()
    db.refresh(db_sub)
    
    # Create invoice
    base_price = pricing_plan.price_per_month
    discount_amount = Decimal('0.00')
    
    if promotion:
        if promotion.discount_type == "PERCENTAGE":
            discount_amount = base_price * (promotion.discount_value / 100)
        elif promotion.discount_type == "FIXED_AMOUNT":
            discount_amount = promotion.discount_value
        elif promotion.discount_type == "FREE_TRIAL":
            discount_amount = base_price
        
        # Mark promotion as used
        use_promotion(db, promotion.id)
    
    final_amount = base_price - discount_amount
    
    invoice_data = schemas.InvoiceCreate(
        subscription_id=db_sub.id,
        user_id=user_id,
        amount=base_price,
        base_price=base_price,
        discount_amount=discount_amount,
        final_amount=final_amount,
        billing_period_start=now,
        billing_period_end=expires_at,
        promotion_code=promotion_code if promotion else None,
        promotion_discount=discount_amount
    )
    
    create_invoice(db, invoice_data)
    
    # Update bot subscriber count
    update_bot_subscriber_count(db, sub.bot_id)
    
    return db_sub

# --- Pricing Calculation Functions ---
def calculate_subscription_price(
    pricing_plan: models.BotPricingPlan,
    billing_cycle: str = "MONTHLY",
    promotion: models.BotPromotion = None
) -> Dict[str, Decimal]:
    """Calculate subscription price with discounts"""
    if billing_cycle == "YEARLY" and pricing_plan.price_per_year:
        base_price = pricing_plan.price_per_year
    elif billing_cycle == "QUARTERLY" and pricing_plan.price_per_quarter:
        base_price = pricing_plan.price_per_quarter
    else:
        base_price = pricing_plan.price_per_month
    
    discount_amount = Decimal('0.00')
    
    if promotion:
        if promotion.discount_type == "PERCENTAGE":
            discount_amount = base_price * (promotion.discount_value / 100)
        elif promotion.discount_type == "FIXED_AMOUNT":
            discount_amount = promotion.discount_value
        elif promotion.discount_type == "FREE_TRIAL":
            discount_amount = base_price
    
    final_amount = base_price - discount_amount
    
    return {
        "base_price": base_price,
        "discount_amount": discount_amount,
        "final_amount": final_amount,
        "billing_cycle": billing_cycle
    }

def cleanup_old_bot_actions(cutoff_date: datetime) -> int:
    """Clean up old performance logs"""
    try:
        from core.database import SessionLocal
        db = SessionLocal()
        
        # Delete performance logs older than cutoff date
        deleted_count = db.query(models.PerformanceLog).filter(
            models.PerformanceLog.timestamp < cutoff_date
        ).delete()
        
        db.commit()
        db.close()
        
        return deleted_count
        
    except Exception as e:
        return 0

# Bot Draft CRUD operations
def create_bot_draft(db: Session, user_id: int, name: str, description: str, bot_code: str, category_id: Optional[int] = None) -> models.BotDraft:
    """Create a new bot draft"""
    db_draft = models.BotDraft(
        user_id=user_id,
        name=name,
        description=description,
        bot_code=bot_code,
        category_id=category_id
    )
    db.add(db_draft)
    db.commit()
    db.refresh(db_draft)
    return db_draft

def get_bot_draft(db: Session, draft_id: int, user_id: int) -> Optional[models.BotDraft]:
    """Get a specific bot draft by ID and user ID"""
    return db.query(models.BotDraft).filter(
        models.BotDraft.id == draft_id,
        models.BotDraft.user_id == user_id
    ).first()

def get_user_bot_drafts(db: Session, user_id: int, skip: int = 0, limit: int = 100) -> List[models.BotDraft]:
    """Get all bot drafts for a user"""
    return db.query(models.BotDraft).filter(
        models.BotDraft.user_id == user_id
    ).order_by(models.BotDraft.updated_at.desc()).offset(skip).limit(limit).all()

def update_bot_draft(db: Session, draft_id: int, user_id: int, name: str = None, description: str = None, bot_code: str = None, category_id: int = None) -> Optional[models.BotDraft]:
    """Update a bot draft"""
    db_draft = get_bot_draft(db, draft_id, user_id)
    if not db_draft:
        return None
    
    if name is not None:
        db_draft.name = name
    if description is not None:
        db_draft.description = description
    if bot_code is not None:
        db_draft.bot_code = bot_code
    if category_id is not None:
        db_draft.category_id = category_id
    
    db_draft.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(db_draft)
    return db_draft

def delete_bot_draft(db: Session, draft_id: int, user_id: int) -> bool:
    """Delete a bot draft"""
    db_draft = get_bot_draft(db, draft_id, user_id)
    if not db_draft:
        return False
    
    db.delete(db_draft)
    db.commit()
    return True