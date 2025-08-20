import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# file: api/endpoints/subscriptions.py

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional
import logging

from core import crud, models, schemas, security
from core.database import get_db
from core.tasks import run_bot_logic
from services.s3_manager import S3Manager
from core.bot_manager import BotManager
from services.exchange_factory import ExchangeFactory

# Initialize logger
logger = logging.getLogger(__name__)

# Initialize managers
s3_manager = S3Manager()
bot_manager = BotManager()

router = APIRouter()

@router.get("/", response_model=schemas.SubscriptionListResponse)
def get_user_subscriptions(
    skip: int = 0,
    limit: int = 50,
    status_filter: Optional[schemas.SubscriptionStatus] = None,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(security.get_current_active_user)
):
    """Get user's subscriptions with pagination and filtering"""
    subscriptions, total = crud.get_user_subscriptions_paginated(
        db, user_id=current_user.id, skip=skip, limit=limit, status_filter=status_filter
    )
    return {
        "subscriptions": subscriptions,
        "total": total,
        "page": skip // limit + 1,
        "per_page": limit
    }

@router.get("/{subscription_id}", response_model=schemas.SubscriptionWithBot)
def get_subscription_details(
    subscription_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(security.get_current_active_user)
):
    """Get detailed information about a specific subscription"""
    subscription = crud.get_subscription_with_bot(db, subscription_id=subscription_id)
    if not subscription:
        raise HTTPException(status_code=404, detail="Subscription not found")
    if subscription.user_id != current_user.id and current_user.role != schemas.UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    return subscription

@router.post("/", response_model=schemas.SubscriptionResponse, status_code=status.HTTP_201_CREATED)
def create_subscription(
    sub_in: schemas.SubscriptionCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(security.get_current_active_user)
):
    """User subscribes to a bot"""
    try:
        # Check if bot exists and is approved
        bot = crud.get_bot_by_id(db, sub_in.bot_id)
        if not bot or bot.status != schemas.BotStatus.APPROVED:
            raise HTTPException(status_code=404, detail="Bot not found or not approved for use")

        # Check if user has API credentials
        if not current_user.api_key or not current_user.api_secret:
            raise HTTPException(
                status_code=400, 
                detail="Please configure your API credentials before subscribing to a bot"
            )
        
        # Check if user already has a subscription with the same instance name
        existing_sub = crud.get_user_subscription_by_name(
            db, user_id=current_user.id, instance_name=sub_in.instance_name
        )
        if existing_sub:
            raise HTTPException(
                status_code=400,
                detail="You already have a subscription with this instance name"
            )
        
        # Verify bot can be loaded from S3
        bot_s3_info = crud.get_bot_s3_info(db, sub_in.bot_id)
        if not bot_s3_info or not bot_s3_info.get('s3_key'):
            raise HTTPException(
                status_code=400,
                detail="Bot code not found in storage. Please contact the developer."
            )
        
        # Test bot loading from S3
        try:
            test_bot = crud.load_bot_from_s3(
                bot_id=sub_in.bot_id,
                version=bot.version,
                user_config=sub_in.strategy_config,
                user_api_keys={
                    'key': current_user.api_key,
                    'secret': current_user.api_secret
                }
            )
            
            if not test_bot:
                raise HTTPException(
                    status_code=400,
                    detail="Bot failed to load from storage. Please contact the developer."
                )
                
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Bot validation failed: {str(e)}"
            )
        
        # Create subscription
        subscription = crud.create_subscription(db, sub=sub_in, user_id=current_user.id)
        
        # Start the bot execution cycle
        run_bot_logic.apply_async(args=[subscription.id], countdown=10)
        
        return schemas.SubscriptionResponse(
            subscription_id=subscription.id,
            status=subscription.status.value,
            message="Bot subscription created successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create subscription: {str(e)}"
        )

@router.post("/trial", response_model=schemas.SubscriptionResponse, status_code=status.HTTP_201_CREATED)
def create_trial_subscription(
    trial_in: schemas.SubscriptionTrialCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(security.get_current_active_user)
):
    """Create a trial subscription for testing a bot with testnet"""
    try:
        # Check if bot exists and is approved
        bot = crud.get_bot_by_id(db, trial_in.bot_id)
        
        # Debug logging
        logger.info(f"Trial subscription request for bot_id={trial_in.bot_id}")
        logger.info(f"Bot found: {bot is not None}")
        if bot:
            logger.info(f"Bot status: {bot.status}")
            logger.info(f"Bot status type: {type(bot.status)}")
            logger.info(f"Expected status: {models.BotStatus.APPROVED}")
            logger.info(f"Expected status type: {type(models.BotStatus.APPROVED)}")
            logger.info(f"Status match: {bot.status == models.BotStatus.APPROVED}")
        
        if not bot or bot.status != models.BotStatus.APPROVED:
            raise HTTPException(status_code=404, detail="Bot not found or not approved for trial")

        # Check if user has exchange credentials for the requested exchange and testnet
        user_credentials = crud.get_user_exchange_credentials(
            db, 
            user_id=current_user.id, 
            exchange=trial_in.exchange_type.value,
            is_testnet=True  # Always testnet for trials
        )
        
        if not user_credentials:
            raise HTTPException(
                status_code=400,
                detail=f"No testnet API credentials found for {trial_in.exchange_type.value}. Please add your exchange credentials first."
            )
        
        # Use the first valid credentials
        credentials = user_credentials[0]
        
        # Validate API credentials work with testnet
        is_valid, message = ExchangeFactory.validate_exchange_credentials(
            exchange_name=credentials.exchange.value,
            api_key=credentials.api_key,
            api_secret=credentials.api_secret,
            testnet=True  # Always testnet for trials
        )
        
        if not is_valid:
            # Update validation status
            crud.update_credentials_validation(db, credentials.id, False, message)
            raise HTTPException(
                status_code=400,
                detail=f"Testnet API validation failed: {message}. Please check your credentials."
            )
        
        # Update validation status as valid
        crud.update_credentials_validation(db, credentials.id, True, "Validation successful")
        
        # Check if user already has an ACTIVE trial for this bot
        existing_active_trial = db.query(models.Subscription).filter(
            models.Subscription.user_id == current_user.id,
            models.Subscription.bot_id == trial_in.bot_id,
            models.Subscription.is_trial == True,
            models.Subscription.status == models.SubscriptionStatus.ACTIVE
        ).first()
        
        if existing_active_trial:
            raise HTTPException(
                status_code=400,
                detail="You already have an active trial for this bot. Please cancel the existing trial first."
            )
        
        # Check if user already has an ACTIVE subscription with the same instance name
        existing_sub = db.query(models.Subscription).filter(
            models.Subscription.user_id == current_user.id,
            models.Subscription.instance_name == trial_in.instance_name,
            models.Subscription.status == models.SubscriptionStatus.ACTIVE
        ).first()
        
        if existing_sub:
            raise HTTPException(
                status_code=400,
                detail="You already have an active subscription with this instance name. Please cancel it first or use a different name."
            )
        
        # Verify bot can be loaded from S3
        bot_s3_info = crud.get_bot_s3_info(db, trial_in.bot_id)
        if not bot_s3_info or not bot_s3_info.get('s3_key'):
            raise HTTPException(
                status_code=400,
                detail="Bot code not found in storage. Please contact the developer."
            )
        
        # Create trial subscription with safe defaults
        trial_subscription = crud.create_trial_subscription(db, trial=trial_in, user_id=current_user.id)
        
        # Start the bot execution cycle for trial (with extra safety checks)
        run_bot_logic.apply_async(args=[trial_subscription.id], countdown=30)  # Longer delay for trials
        
        return schemas.SubscriptionResponse(
            subscription_id=trial_subscription.id,
            status=trial_subscription.status.value,
            message=f"Trial subscription created! Bot will run on testnet for {trial_in.trial_duration_hours} hours with safe defaults."
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create trial subscription: {str(e)}"
        )

@router.post("/with-bot-version", response_model=schemas.SubscriptionResponse, status_code=status.HTTP_201_CREATED)
def create_subscription_with_version(
    sub_in: schemas.SubscriptionCreate,
    bot_version: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(security.get_current_active_user)
):
    """User subscribes to a specific version of a bot"""
    try:
        # Check if bot exists and is approved
        bot = crud.get_bot_by_id(db, sub_in.bot_id)
        if not bot or bot.status != schemas.BotStatus.APPROVED:
            raise HTTPException(status_code=404, detail="Bot not found or not approved for use")

        # Check if user has API credentials
        if not current_user.api_key or not current_user.api_secret:
            raise HTTPException(
                status_code=400, 
                detail="Please configure your API credentials before subscribing to a bot"
            )
        
        # Check if user already has a subscription with the same instance name
        existing_sub = crud.get_user_subscription_by_name(
            db, user_id=current_user.id, instance_name=sub_in.instance_name
        )
        if existing_sub:
            raise HTTPException(
                status_code=400,
                detail="You already have a subscription with this instance name"
            )
        
        # Use specified version or default to bot's current version
        version_to_use = bot_version or bot.version
        
        # Test bot loading from S3 with specific version
        try:
            test_bot = crud.load_bot_from_s3(
                bot_id=sub_in.bot_id,
                version=version_to_use,
                user_config=sub_in.strategy_config,
                user_api_keys={
                    'key': current_user.api_key,
                    'secret': current_user.api_secret
                }
            )
            
            if not test_bot:
                raise HTTPException(
                    status_code=400,
                    detail=f"Bot version {version_to_use} failed to load from storage."
                )
                
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Bot version {version_to_use} validation failed: {str(e)}"
            )
        
        # Create subscription with specific version
        subscription = crud.create_subscription(db, sub=sub_in, user_id=current_user.id)
        
        # Store the specific version in subscription metadata if needed
        subscription.metadata = subscription.metadata or {}
        subscription.metadata['bot_version'] = version_to_use
        db.commit()
        
        # Start the bot execution cycle
        run_bot_logic.apply_async(args=[subscription.id], countdown=10)
    
        return schemas.SubscriptionResponse(
            subscription_id=subscription.id,
            status=subscription.status.value,
            message=f"Bot subscription created successfully with version {version_to_use}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create subscription with version: {str(e)}"
        )

@router.put("/{subscription_id}", response_model=schemas.SubscriptionInDB)
def update_subscription(
    subscription_id: int,
    sub_update: schemas.SubscriptionUpdate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(security.get_current_active_user)
):
    """Update subscription settings"""
    subscription = crud.get_subscription_by_id(db, subscription_id)
    if not subscription:
        raise HTTPException(status_code=404, detail="Subscription not found")
    if subscription.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    
    # Can only update if subscription is paused or active
    if subscription.status not in [models.SubscriptionStatus.ACTIVE, models.SubscriptionStatus.PAUSED]:
        raise HTTPException(
            status_code=400,
            detail="Can only update active or paused subscriptions"
        )
    
    return crud.update_subscription(db, subscription_id=subscription_id, sub_update=sub_update)

@router.post("/{subscription_id}/pause", response_model=schemas.SubscriptionResponse)
def pause_subscription(
    subscription_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(security.get_current_active_user)
):
    """Pause a subscription"""
    subscription = crud.get_subscription_by_id(db, subscription_id)
    if not subscription:
        raise HTTPException(status_code=404, detail="Subscription not found")
    if subscription.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    
    # Fix: Use models.SubscriptionStatus instead of schemas.SubscriptionStatus
    if subscription.status != models.SubscriptionStatus.ACTIVE:
        raise HTTPException(status_code=400, detail="Can only pause active subscriptions")
    
    updated_sub = crud.update_subscription_status(db, subscription_id, schemas.SubscriptionStatus.PAUSED)
    return schemas.SubscriptionResponse(
        subscription_id=updated_sub.id,
        status=updated_sub.status.value,
        message="Subscription paused successfully"
    )

@router.post("/{subscription_id}/resume", response_model=schemas.SubscriptionResponse)
def resume_subscription(
    subscription_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(security.get_current_active_user)
):
    """Resume a paused subscription"""
    try:
        subscription = crud.get_subscription_by_id(db, subscription_id)
        if not subscription:
            raise HTTPException(status_code=404, detail="Subscription not found")
        if subscription.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not enough permissions")
        
        if subscription.status != models.SubscriptionStatus.PAUSED:
            raise HTTPException(status_code=400, detail="Can only resume paused subscriptions")
        
        # Verify bot can still be loaded from S3
        bot_version = subscription.metadata.get('bot_version') if subscription.metadata else None
        try:
            test_bot = crud.load_bot_from_s3(
                bot_id=subscription.bot_id,
                version=bot_version,
                user_config=subscription.strategy_config,
                user_api_keys={
                    'key': current_user.api_key,
                    'secret': current_user.api_secret
                }
            )
            
            if not test_bot:
                raise HTTPException(
                    status_code=400,
                    detail="Bot cannot be loaded from storage. Please contact support."
                )
                
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Bot validation failed: {str(e)}"
            )
        
        updated_sub = crud.update_subscription_status(db, subscription_id, schemas.SubscriptionStatus.ACTIVE)
    
        # Restart the bot execution cycle
        run_bot_logic.apply_async(args=[subscription_id], countdown=5)
        
        return schemas.SubscriptionResponse(
            subscription_id=updated_sub.id,
            status=updated_sub.status.value,
            message="Subscription resumed successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to resume subscription: {str(e)}"
        )

@router.post("/{subscription_id}/cancel", response_model=schemas.SubscriptionResponse)
def cancel_subscription(
    subscription_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(security.get_current_active_user)
):
    """Cancel a subscription"""
    subscription = crud.get_subscription_by_id(db, subscription_id)
    if not subscription:
        raise HTTPException(status_code=404, detail="Subscription not found")
    if subscription.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    
    if subscription.status == schemas.SubscriptionStatus.CANCELLED:
        raise HTTPException(status_code=400, detail="Subscription is already cancelled")
    
    # Close any open trades
    crud.close_open_trades_for_subscription(db, subscription_id)
    
    updated_sub = crud.update_subscription_status(db, subscription_id, schemas.SubscriptionStatus.CANCELLED)
    return schemas.SubscriptionResponse(
        subscription_id=updated_sub.id,
        status=updated_sub.status.value,
        message="Subscription cancelled successfully"
    )

@router.get("/{subscription_id}/trades", response_model=schemas.TradeListResponse)
def get_subscription_trades(
    subscription_id: int,
    skip: int = 0,
    limit: int = 50,
    status_filter: Optional[schemas.TradeStatus] = None,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(security.get_current_active_user)
):
    """Get trades for a specific subscription"""
    subscription = crud.get_subscription_by_id(db, subscription_id)
    if not subscription:
        raise HTTPException(status_code=404, detail="Subscription not found")
    if subscription.user_id != current_user.id and current_user.role != schemas.UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    
    trades, total = crud.get_subscription_trades_paginated(
        db, subscription_id=subscription_id, skip=skip, limit=limit, status_filter=status_filter
    )
    return {
        "trades": trades,
        "total": total,
        "page": skip // limit + 1,
        "per_page": limit
    }

@router.get("/{subscription_id}/performance", response_model=schemas.PerformanceResponse)
def get_subscription_performance(
    subscription_id: int,
    days: int = 30,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(security.get_current_active_user)
):
    """Get performance metrics for a subscription"""
    subscription = crud.get_subscription_by_id(db, subscription_id)
    if not subscription:
        raise HTTPException(status_code=404, detail="Subscription not found")
    if subscription.user_id != current_user.id and current_user.role != schemas.UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    
    return crud.get_subscription_performance_metrics(db, subscription_id=subscription_id, days=days)

@router.get("/{subscription_id}/logs", response_model=List[schemas.PerformanceLogInDB])
def get_subscription_logs(
    subscription_id: int,
    skip: int = 0,
    limit: int = 100,
    action_filter: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(security.get_current_active_user)
):
    """Get performance logs for a subscription"""
    subscription = crud.get_subscription_by_id(db, subscription_id)
    if not subscription:
        raise HTTPException(status_code=404, detail="Subscription not found")
    if subscription.user_id != current_user.id and current_user.role != schemas.UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    
    return crud.get_subscription_logs(
        db, subscription_id=subscription_id, skip=skip, limit=limit, action_filter=action_filter
    )

@router.post("/{subscription_id}/force-run")
def force_run_subscription(
    subscription_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(security.get_current_active_user)
):
    """Force run a subscription immediately (for testing)"""
    subscription = crud.get_subscription_by_id(db, subscription_id)
    if not subscription:
        raise HTTPException(status_code=404, detail="Subscription not found")
    if subscription.user_id != current_user.id and current_user.role != schemas.UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    
    if subscription.status != models.SubscriptionStatus.ACTIVE:
        raise HTTPException(status_code=400, detail="Can only force run active subscriptions")
    
    # Trigger immediate execution
    task = run_bot_logic.apply_async(args=[subscription_id])
    
    return {"message": "Bot execution triggered", "task_id": task.id}