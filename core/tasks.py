import logging
import traceback
import time
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from decimal import Decimal
import importlib.util
import inspect
import os
import tempfile
import sys
import types

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.celery_app import app
from sqlalchemy.orm import Session

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_bot(subscription):
    """Initialize bot from subscription - Load from S3"""
    try:
        # Import here to avoid circular imports
        from core import models
        from core import schemas
        from core.database import SessionLocal
        from services.s3_manager import S3Manager
        from core.bot_base_classes import get_base_classes
        
        # Initialize S3 manager
        s3_manager = S3Manager()
        
        # Get bot information
        bot_id = subscription.bot.id
        logger.info(f"Initializing bot {bot_id} from S3...")
        
        # Get latest version from S3
        try:
            latest_version = s3_manager.get_latest_version(bot_id, "code")
            logger.info(f"Using latest version: {latest_version}")
        except Exception as e:
            logger.error(f"Could not get latest version for bot {bot_id}: {e}")
            return None
        
        # Download bot code from S3
        try:
            code_content = s3_manager.download_bot_code(bot_id, latest_version)
            logger.info(f"Downloaded bot code from S3: {len(code_content)} characters")
        except Exception as e:
            logger.error(f"Failed to download bot code from S3: {e}")
            return None
        
        # Create temporary file to execute the code
        import tempfile
        import os
        
        # Load base classes from bot_sdk folder
        base_classes = get_base_classes()
        
        # Combine base classes with downloaded bot code
        full_code = base_classes + "\n" + code_content
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(full_code)
            temp_file_path = f.name
        
        try:
            # Load bot module from temporary file
            spec = importlib.util.spec_from_file_location("bot_module", temp_file_path)
            if not spec or not spec.loader:
                logger.error("Could not create module spec")
                return None
            
            bot_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(bot_module)
            
            # Find bot class in the module
            bot_class = None
            for attr_name in dir(bot_module):
                attr = getattr(bot_module, attr_name)
                if (inspect.isclass(attr) and 
                    hasattr(attr, 'execute_algorithm') and 
                    attr_name != 'CustomBot'):
                    bot_class = attr
                    break
            
            if not bot_class:
                logger.error("No valid bot class found in module")
                return None
            
            # Prepare bot configuration
            bot_config = {
                'short_window': 50,
                'long_window': 200,
                'position_size': 0.3,
                'min_volume_threshold': 1000000,
                'volatility_threshold': 0.05
            }
            
            # Override with subscription config if available
            if subscription.strategy_config:
                bot_config.update(subscription.strategy_config)
            
            # Prepare API keys (mock for now, real implementation would get from user)
            api_keys = {
                'exchange': subscription.exchange_type.value if subscription.exchange_type else 'binance',
                'key': 'test_key',  # Would be real API key in production
                'secret': 'test_secret',  # Would be real API secret in production  
                'testnet': subscription.is_testnet if subscription.is_testnet else True
            }
            
            # Try to initialize bot with new constructor format first
            try:
                bot_instance = bot_class(bot_config, api_keys)
                logger.info(f"Successfully initialized bot with new constructor: {bot_class.__name__} v{latest_version}")
            except TypeError as e:
                # Fallback to old constructor format (no parameters)
                if "missing" in str(e) and "required positional arguments" in str(e):
                    try:
                        bot_instance = bot_class()
                        logger.info(f"Successfully initialized bot with old constructor: {bot_class.__name__} v{latest_version}")
                        # Set config manually if the bot has attributes for it
                        if hasattr(bot_instance, 'short_window'):
                            bot_instance.short_window = bot_config.get('short_window', 50)
                        if hasattr(bot_instance, 'long_window'):
                            bot_instance.long_window = bot_config.get('long_window', 200)
                        if hasattr(bot_instance, 'position_size'):
                            bot_instance.position_size = bot_config.get('position_size', 0.3)
                    except Exception as fallback_error:
                        logger.error(f"Both new and old constructor failed: {fallback_error}")
                        return None
                else:
                    logger.error(f"Bot initialization failed: {e}")
                    return None
            
            return bot_instance
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except:
                pass
        
    except Exception as e:
        logger.error(f"Error initializing bot: {e}")
        logger.error(traceback.format_exc())
        return None

def execute_trade_action(exchange_client, action, subscription, market_context):
    """
    Execute trade action using the new prediction cycle logic.
    
    Args:
        exchange_client: Exchange client instance
        action: Action dictionary from bot
        subscription: Subscription object
        market_context: Market context information
        
    Returns:
        Trade details dictionary
    """
    try:
        # Get trading pair
        trading_pair = subscription.trading_pair or 'BTC/USDT'
        exchange_symbol = trading_pair.replace('/', '')
        
        # Get current price
        current_price = market_context.get('current_price', 0)
        if not current_price:
            current_price = exchange_client.get_current_price(trading_pair)
        
        # Get balance info
        try:
            balance = market_context.get('balance', {})
            base_asset = trading_pair.split('/')[0]  # BTC from BTC/USDT
            quote_asset = trading_pair.split('/')[1]  # USDT from BTC/USDT
            
            base_balance = balance.get(base_asset, 0)
            quote_balance = balance.get(quote_asset, 0)
            
            logger.info(f"Balance - {base_asset}: {base_balance}, {quote_asset}: {quote_balance}")
            
        except Exception as e:
            logger.warning(f"Could not get balance info: {e}")
            base_balance = 0
            quote_balance = 0
        
        # Execute trade based on action
        if action['action'] == "BUY":
            # Calculate buy amount
            trade_amount = action['amount']
            
            # Check if we have enough quote currency
            if trade_amount > quote_balance:
                logger.warning(f"Insufficient {quote_asset} balance for buy order")
                return {
                    'success': False,
                    'error': f"Insufficient {quote_asset} balance"
                }
            
            # Place buy order
            try:
                # Calculate quantity based on trade amount
                quantity = trade_amount / current_price
                
                # Use exchange's calculate_quantity method for proper precision
                quantity_str, quantity_info = exchange_client.calculate_quantity(
                    symbol=exchange_symbol,
                    side="BUY",
                    amount=trade_amount,
                    price=current_price
                )
                
                order = exchange_client.create_market_order(
                    symbol=exchange_symbol,
                    side="BUY",
                    quantity=quantity_str
                )
                
                logger.info(f"Buy order executed: {order}")
                
                return {
                    'success': True,
                    'order_id': getattr(order, 'order_id', 'N/A'),
                    'quantity': quantity_str,
                    'current_price': current_price,
                    'usdt_value': trade_amount,
                    'base_asset': base_asset
                }
                
            except Exception as e:
                logger.error(f"Buy order failed: {e}")
                return {
                    'success': False,
                    'error': str(e)
                }
                
        elif action['action'] == "SELL":
            # Calculate sell amount
            trade_amount = action['amount']
            
            # Check if we have enough base currency
            if trade_amount > base_balance:
                logger.warning(f"Insufficient {base_asset} balance for sell order")
                return {
                    'success': False,
                    'error': f"Insufficient {base_asset} balance"
                }
            
            # Place sell order
            try:
                # Use exchange's calculate_quantity method for proper precision
                quantity_str, quantity_info = exchange_client.calculate_quantity(
                    symbol=exchange_symbol,
                    side="SELL",
                    amount=trade_amount,
                    price=current_price
                )
                
                order = exchange_client.create_market_order(
                    symbol=exchange_symbol,
                    side="SELL",
                    quantity=quantity_str
                )
                
                logger.info(f"Sell order executed: {order}")
                
                return {
                    'success': True,
                    'order_id': getattr(order, 'order_id', 'N/A'),
                    'quantity': quantity_str,
                    'current_price': current_price,
                    'usdt_value': trade_amount,
                    'base_asset': base_asset
                }
                
            except Exception as e:
                logger.error(f"Sell order failed: {e}")
                return {
                    'success': False,
                    'error': str(e)
                }
        
        elif action['action'] == "HOLD":
            return {
                'success': True,
                'message': 'No trade executed (HOLD signal)',
                'action': 'HOLD'
            }
        
        else:
            return {
                'success': False,
                'error': f"Unknown action: {action['action']}"
            }
            
    except Exception as e:
        logger.error(f"Error executing trade action: {e}")
        return {
            'success': False,
            'error': str(e)
        }

@app.task
def send_trade_notification(user_email: str, bot_name: str, action: dict, trade_details: dict, subscription):
    """
    Send trade notification email to user.
    
    Args:
        user_email: User's email address
        bot_name: Name of the bot
        action: Action dictionary from bot
        trade_details: Trade execution details
        subscription: Subscription object
    """
    try:
        from services.sendgrid_email_service import SendGridEmailService
        from services.gmail_smtp_service import GmailSMTPService
        
        # Prepare email content
        subject = f"Bot {bot_name} - {action.get('action', 'ACTION')} Signal"
        
        # Create email body
        body = f"""
        Bot: {bot_name}
        Action: {action.get('action', 'N/A')}
        Reason: {action.get('reason', 'N/A')}
        Amount: {action.get('amount', 'N/A')}
        Confidence: {action.get('confidence', 'N/A')}
        
        Trade Details:
        Success: {trade_details.get('success', 'N/A')}
        """
        
        if not trade_details.get('success'):
            body += f"Error: {trade_details.get('error', 'N/A')}"
        else:
            body += f"Order ID: {trade_details.get('order_id', 'N/A')}"
            body += f"Quantity: {trade_details.get('quantity', 'N/A')}"
            body += f"Price: ${trade_details.get('current_price', 'N/A')}"
        
        # Try SendGrid first
        try:
            sendgrid_service = SendGridEmailService()
            success = sendgrid_service.send_email(user_email, subject, body)
            if success:
                logger.info(f"Trade notification sent via SendGrid to {user_email}")
                return
        except Exception as e:
            logger.warning(f"SendGrid failed: {e}")
        
        # Fallback to Gmail SMTP
        try:
            gmail_service = GmailSMTPService()
            success = gmail_service.send_email(user_email, subject, body)
            if success:
                logger.info(f"Trade notification sent via Gmail SMTP to {user_email}")
                return
        except Exception as e:
            logger.warning(f"Gmail SMTP failed: {e}")
        
        logger.error(f"All email services failed for trade notification to {user_email}")
        
    except Exception as e:
        logger.error(f"Error sending trade notification: {e}")

@app.task
def send_sendgrid_notification(email: str, bot_name: str, action: str, details: dict):
    """Send SendGrid notification"""
    try:
        from services.sendgrid_email_service import SendGridEmailService
        
        sendgrid_service = SendGridEmailService()
        success = sendgrid_service.send_trade_notification(email, bot_name, action, details)
        
        if success:
            logger.info(f"SendGrid notification sent to {email}")
        else:
            logger.error(f"Failed to send SendGrid notification to {email}")
            
    except Exception as e:
        logger.error(f"Error sending SendGrid notification: {e}")

@app.task
def test_task():
    """Test task for debugging"""
    logger.info("Test task executed successfully")
    return "Test task completed"

@app.task
def run_bot_logic(subscription_id: int):
    """
    Execute bot logic for a specific subscription.
    Now uses prediction_cycle instead of timeframe for execution timing.
    """
    try:
        db = next(get_db())
        
        # Get subscription details
        subscription = crud.get_subscription(db, subscription_id)
        if not subscription:
            logger.error(f"Subscription {subscription_id} not found")
            return
        
        if subscription.status != "ACTIVE":
            logger.info(f"Subscription {subscription_id} is not active (status: {subscription.status})")
            return
        
        # Check if it's time to run based on prediction_cycle
        current_time = datetime.utcnow()
        
        # Get bot details
        bot = crud.get_bot(db, subscription.bot_id)
        if not bot:
            logger.error(f"Bot {subscription.bot_id} not found")
            return
        
        # Download bot code from S3
        try:
            bot_code = s3_manager.download_bot_code(bot.id, bot.version)
            logger.info(f"Downloaded bot code: {bot.s3_key} ({len(bot_code)} characters)")
        except Exception as e:
            logger.error(f"Failed to download bot code: {e}")
            return
        
        # Load bot classes
        try:
            # Load Action class from bot_sdk
            action_module = importlib.import_module('bots.bot_sdk.Action')
            Action = getattr(action_module, 'Action')
            logger.info("Loaded Action class from bot_sdk")
            
            # Load CustomBot base class from bot_sdk
            custom_bot_module = importlib.import_module('bots.bot_sdk.CustomBot')
            CustomBot = getattr(custom_bot_module, 'CustomBot')
            logger.info("Loaded CustomBot base class from bot_sdk")
            
            logger.info("Successfully loaded base classes from bot_sdk")
        except Exception as e:
            logger.error(f"Failed to load bot SDK classes: {e}")
            return
        
        # Create bot instance
        try:
            # Create a temporary module for the bot
            bot_module = types.ModuleType(f"bot_{bot.id}")
            exec(bot_code, bot_module.__dict__)
            
            # Find the bot class (should inherit from CustomBot)
            bot_class = None
            for attr_name in dir(bot_module):
                attr = getattr(bot_module, attr_name)
                if (inspect.isclass(attr) and 
                    issubclass(attr, CustomBot) and 
                    attr != CustomBot):
                    bot_class = attr
                    break
            
            if not bot_class:
                logger.error("No valid bot class found in code")
                return
            
            bot_config = {
                'prediction_cycle': subscription.timeframe,
                'prediction_cycle_configurable': True,
                'max_data_points': 1000,
                'required_warmup_periods': 50,
                **(subscription.strategy_config or {}),
                **(subscription.execution_config or {})
            }
            
            bot_instance = bot_class(bot_config)
            logger.info(f"Successfully initialized bot with new constructor: {bot_class.__name__} v{bot.version}")
            
            # Validate prediction cycle if configurable
            if bot_instance.prediction_cycle_configurable:
                if hasattr(bot_instance, 'validate_prediction_cycle'):
                    if not bot_instance.validate_prediction_cycle(bot_instance.prediction_cycle):
                        logger.error(f"Invalid prediction cycle: {bot_instance.prediction_cycle}")
                        if hasattr(bot_instance, 'get_supported_prediction_cycles'):
                            logger.info(f"Supported cycles: {bot_instance.get_supported_prediction_cycles()}")
                        if hasattr(bot_instance, 'get_recommended_prediction_cycle'):
                            logger.info(f"Recommended cycle: {bot_instance.get_recommended_prediction_cycle()}")
                        return
            else:
                logger.info(f"Bot uses fixed prediction cycle: {bot_instance.prediction_cycle}")
            
        except Exception as e:
            logger.error(f"Failed to create bot instance: {e}")
            return
        
        # Check if it's time to execute prediction
        if not bot_instance.should_execute_prediction(current_time):
            logger.info(f"Bot {bot.id} prediction cycle not ready yet. Next execution in {bot_instance.prediction_cycle}")
            return
        
        # Get exchange credentials
        user = crud.get_user(db, subscription.user_id)
        if not user:
            logger.error(f"User {subscription.user_id} not found")
            return
        
        exchange_creds = crud.get_exchange_credentials(db, user.id, subscription.exchange_type)
        if not exchange_creds:
            logger.error(f"No exchange credentials found for user {user.email} on {subscription.exchange_type}")
            return
        
        logger.info(f"Found exchange credentials for {subscription.exchange_type} (testnet={subscription.is_testnet})")
        
        # Create exchange client
        try:
            exchange_factory = ExchangeFactory()
            exchange_client = exchange_factory.create_exchange(
                exchange_type=subscription.exchange_type,
                api_key=exchange_creds.api_key,
                api_secret=exchange_creds.api_secret,
                testnet=subscription.is_testnet
            )
            logger.info(f"Creating exchange client for subscription {subscription_id} (testnet={subscription.is_testnet})")
        except Exception as e:
            logger.error(f"Failed to create exchange client: {e}")
            return
        
        # Set exchange client in bot instance
        bot_instance.exchange_client = exchange_client
        
        # Execute bot algorithm
        try:
            # Create market context
            market_context = {
                'symbol': subscription.trading_pair,
                'timeframe': subscription.timeframe,
                'current_price': exchange_client.get_current_price(subscription.trading_pair),
                'balance': exchange_client.get_balance(),
                'subscription_id': subscription_id
            }
            
            # Execute bot algorithm
            action_result = bot_instance.execute_algorithm(current_time, market_context)
            
            if action_result and bot_instance.validate_signal(action_result):
                logger.info(f"Bot {bot.name} executed with action: {action_result['action']}, value: {action_result['amount']}, reason: {action_result['reason']}")
                
                # Execute trade action
                trade_details = execute_trade_action(
                    exchange_client=exchange_client,
                    action=action_result,
                    subscription=subscription,
                    market_context=market_context
                )
                
                # Update bot performance
                bot_instance.update_performance(trade_details)
                bot_instance.last_action = action_result
                bot_instance.last_action_time = current_time
                
                # Send email notification
                send_trade_notification(
                    user_email=user.email,
                    bot_name=bot.name,
                    action=action_result,
                    trade_details=trade_details,
                    subscription=subscription
                )
                
            else:
                logger.info(f"Bot {bot.name} executed but no valid action generated")
                
        except Exception as e:
            logger.error(f"Error executing bot algorithm: {e}")
            # Send error notification
            send_trade_notification(
                user_email=user.email,
                bot_name=bot.name,
                action={'action': 'ERROR', 'reason': str(e)},
                trade_details={'success': False, 'error': str(e)},
                subscription=subscription
            )
        
        # Update subscription last_run_at
        crud.update_subscription_last_run(db, subscription_id, current_time)
        
        logger.info(f"Task run_bot_logic[{subscription_id}] completed successfully")
        
    except Exception as e:
        logger.error(f"Error in run_bot_logic: {e}")
        logger.error(traceback.format_exc())

@app.task
def schedule_active_bots():
    """Schedule active bots for execution"""
    try:
        from core.database import SessionLocal
        from core import crud
        from core import models
        
        db = SessionLocal()
        
        try:
            # Get all active subscriptions
            active_subscriptions = crud.get_active_subscriptions(db)
            
            for subscription in active_subscriptions:
                # Check if it's time to run this bot
                should_run = False
                
                if subscription.next_run_at:
                    # If next_run_at is set, check if it's time to run
                    should_run = subscription.next_run_at <= datetime.utcnow()
                else:
                    # If next_run_at is NULL, run immediately
                    should_run = True
                    logger.info(f"Subscription {subscription.id} has no next_run_at, scheduling immediately")
                
                if should_run:
                    logger.info(f"Scheduling bot execution for subscription {subscription.id}")
                    
                    # Queue the bot execution task
                    run_bot_logic.delay(subscription.id)
                    
                    # Update next run time based on timeframe
                    if subscription.timeframe == "1m":
                        next_run = datetime.utcnow() + timedelta(minutes=1)
                    elif subscription.timeframe == "5m":
                        next_run = datetime.utcnow() + timedelta(minutes=5)
                    elif subscription.timeframe == "15m":
                        next_run = datetime.utcnow() + timedelta(minutes=15)
                    elif subscription.timeframe == "1h":
                        next_run = datetime.utcnow() + timedelta(hours=1)
                    elif subscription.timeframe == "4h":
                        next_run = datetime.utcnow() + timedelta(hours=4)
                    elif subscription.timeframe == "1d":
                        next_run = datetime.utcnow() + timedelta(days=1)
                    else:
                        next_run = datetime.utcnow() + timedelta(hours=1)  # Default to 1 hour
                    
                    crud.update_subscription_next_run(db, subscription.id, next_run)
                    logger.info(f"Updated next_run_at for subscription {subscription.id} to {next_run}")
                else:
                    logger.debug(f"Subscription {subscription.id} not ready to run yet. Next run: {subscription.next_run_at}")
                    
        finally:
            db.close()

    except Exception as e:
        logger.error(f"Error in schedule_active_bots: {e}")
        logger.error(traceback.format_exc())

@app.task
def cleanup_old_logs():
    """Clean up old bot action logs"""
    try:
        from core.database import SessionLocal
        from core import crud
        
        # Clean up logs older than 30 days
        cutoff_date = datetime.utcnow() - timedelta(days=30)
        deleted_count = crud.cleanup_old_bot_actions(cutoff_date)
        
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old bot action logs")
            
    except Exception as e:
        logger.error(f"Error in cleanup_old_logs: {e}")
        logger.error(traceback.format_exc())

@app.task
def send_email_notification(email: str, subject: str, body: str):
    """Send email notification"""
    try:
        from services.sendgrid_email_service import SendGridEmailService
        from services.gmail_smtp_service import GmailSMTPService
        
        # Try SendGrid first
        try:
            sendgrid_service = SendGridEmailService()
            success = sendgrid_service.send_email(email, subject, body)
            if success:
                logger.info(f"Email sent via SendGrid to {email}")
                return
        except Exception as e:
            logger.warning(f"SendGrid failed: {e}")
        
        # Fallback to Gmail SMTP
        try:
            gmail_service = GmailSMTPService()
            success = gmail_service.send_email(email, subject, body)
            if success:
                logger.info(f"Email sent via Gmail SMTP to {email}")
                return
        except Exception as e:
            logger.warning(f"Gmail SMTP failed: {e}")
        
        logger.error(f"All email services failed for {email}")
        
    except Exception as e:
        logger.error(f"Error sending email notification: {e}")