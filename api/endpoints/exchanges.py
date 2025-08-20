import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


"""
Exchange API endpoints
Handles exchange integration, credentials validation, and supported exchange listing
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session

from core import models, schemas
from core.database import get_db
from core.security import get_current_user
from services.exchange_factory import ExchangeFactory
import logging

logger = logging.getLogger(__name__)
security = HTTPBearer()

router = APIRouter()

@router.get("/supported", response_model=List[str])
async def get_supported_exchanges():
    """Get list of supported exchanges"""
    try:
        exchanges = ExchangeFactory.get_supported_exchanges()
        return exchanges
    except Exception as e:
        logger.error(f"Error getting supported exchanges: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get supported exchanges"
        )

@router.get("/capabilities/{exchange_name}")
async def get_exchange_capabilities(exchange_name: str):
    """Get capabilities for specific exchange"""
    try:
        capabilities = ExchangeFactory.get_exchange_capabilities(exchange_name)
        return {
            "exchange": exchange_name.upper(),
            "spot_trading": capabilities.spot_trading,
            "futures_trading": capabilities.futures_trading,
            "margin_trading": capabilities.margin_trading,
            "stop_loss_orders": capabilities.stop_loss_orders,
            "take_profit_orders": capabilities.take_profit_orders,
            "advanced_orders": capabilities.advanced_orders,
            "api_key_permissions": capabilities.api_key_permissions or []
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error getting exchange capabilities: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get exchange capabilities"
        )

@router.get("/validate-credentials", response_model=schemas.ExchangeValidationResponse)
async def validate_credentials_get(
    exchange_name: str,
    api_key: str,
    api_secret: str,
    testnet: bool = True,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Validate exchange API credentials via GET with query parameters"""
    try:
        logger.info(f"Validating {exchange_name} credentials for user {current_user.email} (testnet={testnet})")
        
        is_valid, message = ExchangeFactory.validate_exchange_credentials(
            exchange_name=exchange_name,
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet
        )
        
        # Get additional account info if validation successful
        account_info = None
        if is_valid and exchange_name.upper() == "BINANCE":
            try:
                from services.binance_integration import BinanceIntegration
                client = BinanceIntegration(api_key, api_secret, testnet)
                
                # Get basic account info
                account_data = client.get_account_info()
                account_info = {
                    "accountType": account_data.get("accountType", "UNKNOWN"),
                    "canTrade": account_data.get("canTrade", False),
                    "canWithdraw": account_data.get("canWithdraw", False),
                    "canDeposit": account_data.get("canDeposit", False),
                    "permissions": account_data.get("permissions", []),
                    "balanceCount": len(account_data.get("balances", []))
                }
            except Exception as e:
                logger.warning(f"Could not get account info: {e}")
        
        return {
            "valid": is_valid,
            "message": message,
            "exchange": exchange_name.upper(),
            "testnet": testnet,
            "account_info": account_info
        }
        
    except Exception as e:
        logger.error(f"Error validating credentials: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to validate credentials: {str(e)}"
        )

@router.post("/validate-credentials-json", response_model=schemas.ExchangeValidationResponse)
async def validate_credentials_post(
    credentials: schemas.ExchangeCredentials,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Validate exchange API credentials"""
    try:
        logger.info(f"Validating {credentials.exchange_name} credentials for user {current_user.email}")
        
        is_valid, message = ExchangeFactory.validate_exchange_credentials(
            exchange_name=credentials.exchange_name,
            api_key=credentials.api_key,
            api_secret=credentials.api_secret,
            testnet=credentials.testnet
        )
        
        # Get additional account info if validation successful
        account_info = None
        if is_valid and credentials.exchange_name.upper() == "BINANCE":
            try:
                from services.binance_integration import BinanceIntegration
                client = BinanceIntegration(credentials.api_key, credentials.api_secret, credentials.testnet)
                
                # Get basic account info
                account_data = client.get_account_info()
                account_info = {
                    "accountType": account_data.get("accountType", "UNKNOWN"),
                    "canTrade": account_data.get("canTrade", False),
                    "canWithdraw": account_data.get("canWithdraw", False),
                    "canDeposit": account_data.get("canDeposit", False),
                    "permissions": account_data.get("permissions", []),
                    "balanceCount": len(account_data.get("balances", []))
                }
            except Exception as e:
                logger.warning(f"Could not get account info: {e}")
        
        return {
            "valid": is_valid,
            "message": message,
            "exchange": credentials.exchange_name.upper(),
            "testnet": credentials.testnet,
            "account_info": account_info
        }
        
    except Exception as e:
        logger.error(f"Error validating credentials: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to validate credentials: {str(e)}"
        )

@router.post("/validate-credentials-debug")
async def validate_credentials_debug(
    credentials: schemas.ExchangeCredentials,
    current_user: models.User = Depends(get_current_user)
):
    """Debug version of validate credentials with detailed error information"""
    try:
        if credentials.exchange_name.upper() != "BINANCE":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Debug validation only supported for Binance"
            )
        
        from services.binance_integration import BinanceIntegration
        import logging
        
        # Enable debug logging
        logging.getLogger("bot_marketplace.binance_integration").setLevel(logging.DEBUG)
        
        client = BinanceIntegration(credentials.api_key, credentials.api_secret, credentials.testnet)
        
        debug_info = {
            "api_key_length": len(credentials.api_key),
            "api_secret_length": len(credentials.api_secret),
            "testnet": credentials.testnet,
            "base_url": client.base_url,
            "steps": []
        }
        
        # Step 1: Test connectivity
        try:
            debug_info["steps"].append("Testing connectivity...")
            connectivity = client.test_connectivity()
            debug_info["connectivity"] = connectivity
            debug_info["steps"].append(f"Connectivity: {'✓ SUCCESS' if connectivity else '✗ FAILED'}")
        except Exception as e:
            debug_info["connectivity"] = False
            debug_info["connectivity_error"] = str(e)
            debug_info["steps"].append(f"Connectivity: ✗ FAILED - {str(e)}")
        
        # Step 2: Get server time
        try:
            debug_info["steps"].append("Getting server time...")
            server_time = client._get_server_time()
            debug_info["server_time"] = server_time
            debug_info["steps"].append(f"Server time: ✓ {server_time}")
        except Exception as e:
            debug_info["server_time_error"] = str(e)
            debug_info["steps"].append(f"Server time: ✗ FAILED - {str(e)}")
        
        # Step 3: Test account access
        try:
            debug_info["steps"].append("Testing account access...")
            account_info = client.get_account_info()
            debug_info["account_access"] = True
            debug_info["account_info"] = {
                "accountType": account_info.get("accountType"),
                "canTrade": account_info.get("canTrade"),
                "permissions": account_info.get("permissions", []),
                "balances_count": len(account_info.get("balances", []))
            }
            debug_info["steps"].append("Account access: ✓ SUCCESS")
        except Exception as e:
            debug_info["account_access"] = False
            debug_info["account_error"] = str(e)
            debug_info["steps"].append(f"Account access: ✗ FAILED - {str(e)}")
        
        # Overall result
        debug_info["overall_success"] = debug_info.get("connectivity", False) and debug_info.get("account_access", False)
        
        return debug_info
        
    except Exception as e:
        logger.error(f"Debug validation error: {e}")
        return {
            "error": str(e),
            "success": False
        }

@router.post("/test-connection")
async def test_exchange_connection(
    exchange_name: str,
    api_key: str,
    api_secret: str,
    testnet: bool = True,
    current_user: models.User = Depends(get_current_user)
):
    """Test connection to exchange and get basic account info"""
    try:
        # First validate credentials
        is_valid, message = ExchangeFactory.validate_exchange_credentials(
            exchange_name=exchange_name,
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet
        )
        
        if not is_valid:
            return {
                "connected": False,
                "message": message
            }
        
        # Create exchange client and get account info
        exchange = ExchangeFactory.create_exchange(
            exchange_name=exchange_name,
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet
        )
        
        # Get basic account information
        account_info = exchange.get_account_info()
        
        # Get USDT balance as example
        try:
            balance = exchange.get_balance("USDT")
            balance_info = {
                "asset": balance.asset,
                "free": balance.free,
                "locked": balance.locked
            }
        except Exception:
            balance_info = None
        
        return {
            "connected": True,
            "message": "Connection successful",
            "exchange": exchange_name.upper(),
            "testnet": testnet,
            "account_type": account_info.get("accountType", "UNKNOWN"),
            "can_trade": account_info.get("canTrade", False),
            "can_withdraw": account_info.get("canWithdraw", False),
            "can_deposit": account_info.get("canDeposit", False),
            "sample_balance": balance_info,
            "capabilities": ExchangeFactory.get_exchange_capabilities(exchange_name).__dict__
        }
        
    except Exception as e:
        logger.error(f"Error testing exchange connection: {e}")
        return {
            "connected": False,
            "message": f"Connection failed: {str(e)}",
            "exchange": exchange_name.upper(),
            "testnet": testnet
        }

@router.get("/trading-pairs/{exchange_name}")
async def get_trading_pairs(
    exchange_name: str,
    current_user: models.User = Depends(get_current_user)
):
    """Get available trading pairs for exchange (requires user's API credentials)"""
    try:
        if not current_user.api_key or not current_user.api_secret:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User API credentials not configured"
            )
        
        exchange = ExchangeFactory.create_exchange(
            exchange_name=exchange_name,
            api_key=current_user.api_key,
            api_secret=current_user.api_secret,
            testnet=True  # Use testnet for safety
        )
        
        # For now, return common trading pairs
        # In a real implementation, you'd fetch from exchange API
        common_pairs = [
            "BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "XRP/USDT",
            "SOL/USDT", "DOT/USDT", "DOGE/USDT", "AVAX/USDT", "MATIC/USDT",
            "LINK/USDT", "UNI/USDT", "LTC/USDT", "BCH/USDT", "FIL/USDT"
        ]
        
        return {
            "exchange": exchange_name.upper(),
            "trading_pairs": common_pairs,
            "count": len(common_pairs)
        }
        
    except Exception as e:
        logger.error(f"Error getting trading pairs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get trading pairs: {str(e)}"
        )

@router.get("/balance/{exchange_name}")
async def get_user_balance(
    exchange_name: str,
    asset: str = "USDT",
    current_user: models.User = Depends(get_current_user)
):
    """Get user balance for specific asset on exchange"""
    try:
        if not current_user.api_key or not current_user.api_secret:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User API credentials not configured"
            )
        
        exchange = ExchangeFactory.create_exchange(
            exchange_name=exchange_name,
            api_key=current_user.api_key,
            api_secret=current_user.api_secret,
            testnet=True
        )
        
        balance = exchange.get_balance(asset)
        
        return {
            "exchange": exchange_name.upper(),
            "asset": balance.asset,
            "free": balance.free,
            "locked": balance.locked,
            "total": str(float(balance.free) + float(balance.locked))
        }
        
    except Exception as e:
        logger.error(f"Error getting user balance: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get balance: {str(e)}"
        )

@router.get("/account-info/{exchange_name}")
async def get_account_info(
    exchange_name: str,
    current_user: models.User = Depends(get_current_user)
):
    """Get detailed account information"""
    try:
        if not current_user.api_key or not current_user.api_secret:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User API credentials not configured"
            )
        
        exchange = ExchangeFactory.create_exchange(
            exchange_name=exchange_name,
            api_key=current_user.api_key,
            api_secret=current_user.api_secret,
            testnet=True
        )
        
        account_info = exchange.get_account_info()
        
        # Get top balances (non-zero)
        balances = []
        if 'balances' in account_info:
            for balance in account_info['balances']:
                free_balance = float(balance.get('free', 0))
                locked_balance = float(balance.get('locked', 0))
                if free_balance > 0 or locked_balance > 0:
                    balances.append({
                        "asset": balance['asset'],
                        "free": balance['free'],
                        "locked": balance['locked'],
                        "total": str(free_balance + locked_balance)
                    })
        
        return {
            "exchange": exchange_name.upper(),
            "account_type": account_info.get("accountType", "UNKNOWN"),
            "can_trade": account_info.get("canTrade", False),
            "can_withdraw": account_info.get("canWithdraw", False),
            "can_deposit": account_info.get("canDeposit", False),
            "permissions": account_info.get("permissions", []),
            "balances": balances[:10],  # Limit to top 10
            "update_time": account_info.get("updateTime")
        }
        
    except Exception as e:
        logger.error(f"Error getting account info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get account info: {str(e)}"
        ) 