from fastapi import APIRouter, Depends, HTTPException, Form, WebSocket, Query, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import Any, Optional, List, Dict
from pydantic import BaseModel
import os
import json
import traceback
import inspect
from datetime import datetime, timedelta
import asyncio
import random
import pandas as pd
import numpy as np

# Core imports
from core.database import get_db
from core import models, schemas
from core.security import get_current_active_developer

# Optional imports for development
try:
    from services.exchange_factory import ExchangeFactory
    EXCHANGE_FACTORY_AVAILABLE = True
except ImportError:
    EXCHANGE_FACTORY_AVAILABLE = False
    print("Warning: ExchangeFactory not available")

try:
    from services.s3_manager import S3Manager
    S3_MANAGER_AVAILABLE = True
except ImportError:
    S3_MANAGER_AVAILABLE = False
    print("Warning: S3Manager not available")

try:
    from core.bot_manager import BotManager
    BOT_MANAGER_AVAILABLE = True
except ImportError:
    BOT_MANAGER_AVAILABLE = False
    print("Warning: BotManager not available")

try:
    from core.bot_sdk import CustomBot
    BOT_SDK_AVAILABLE = True
except ImportError:
    BOT_SDK_AVAILABLE = False
    print("Warning: BotSDK not available")

# Optional data science imports
try:
    import pandas as pd
    import numpy as np
    DATA_SCIENCE_AVAILABLE = True
except ImportError:
    DATA_SCIENCE_AVAILABLE = False
    print("Warning: pandas/numpy not available")

# Optional crud import
try:
    from core import crud
    CRUD_AVAILABLE = True
except ImportError:
    CRUD_AVAILABLE = False
    print("Warning: crud not available")

router = APIRouter(tags=["Developer Sandbox"])

# Pydantic models for request validation
class TestBotCodeRequest(BaseModel):
    bot_code: str
    exchange_type: str = "BINANCE"
    symbol: str = "BTC/USDT"
    timeframe: str = "5m"
    config: Dict[str, Any] = {}

# Enforce developer auth for all endpoints in this router
def get_dev_user():
    """Simple dependency for development mode"""
    if os.getenv("DEVELOPMENT_MODE", "false").lower() == "true":
        # In development mode, return a dummy user
        class DummyUser:
            def __init__(self):
                self.id = 1
                self.email = "developer@test.com"
                self.role = type('Role', (), {'value': 'DEVELOPER'})()
        return DummyUser()
    else:
        # In production mode, use real authentication
        return get_current_active_developer

def get_current_user_dep(
    current_user: Any = Depends(get_current_active_developer)
) -> Any:
    """Dependency to get current user, handling development mode"""
    if os.getenv("DEVELOPMENT_MODE", "false").lower() == "true":
        # In development mode, we need to decode the token manually
        from fastapi import Request
        from core.security import decode_access_token
        
        # Get the request to access headers
        request = Request
        auth_header = request.headers.get("Authorization")
        
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            try:
                # Decode the token to get user_id
                payload = decode_access_token(token)
                user_id = int(payload.get("sub"))
                
                # Get the actual user from database
                from core.database import SessionLocal
                db = SessionLocal()
                try:
                    actual_user = db.query(models.User).filter(models.User.id == user_id).first()
                    if actual_user:
                        print(f"DEBUG: Found logged-in user: {actual_user.id} - {actual_user.email}")
                        return actual_user
                finally:
                    db.close()
            except Exception as e:
                print(f"DEBUG: Error decoding token: {e}")
        
        # Fallback: get first developer user (for backward compatibility)
        from core.database import SessionLocal
        db = SessionLocal()
        try:
            actual_user = db.query(models.User).filter(models.User.role == models.UserRole.DEVELOPER).first()
            if actual_user:
                print(f"DEBUG: Fallback to first developer: {actual_user.id} - {actual_user.email}")
                return actual_user
        finally:
            db.close()
    else:
        return current_user

def get_current_user_from_token(
    request: Request,
    db: Session = Depends(get_db)
) -> Any:
    """Get current user by decoding JWT token from request headers"""
    if os.getenv("DEVELOPMENT_MODE", "false").lower() == "true":
        auth_header = request.headers.get("Authorization")
        
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            try:
                # Decode the token to get user_id using the same method as security.py
                import jwt
                from core.security import SECRET_KEY, ALGORITHM
                
                payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
                user_id = int(payload.get("sub"))
                
                # Get the actual user from database
                actual_user = db.query(models.User).filter(models.User.id == user_id).first()
                if actual_user:
                    print(f"DEBUG: Found logged-in user: {actual_user.id} - {actual_user.email}")
                    return actual_user
            except Exception as e:
                print(f"DEBUG: Error decoding token: {e}")
        
        # Fallback: get first developer user
        actual_user = db.query(models.User).filter(models.User.role == models.UserRole.DEVELOPER).first()
        if actual_user:
            print(f"DEBUG: Fallback to first developer: {actual_user.id} - {actual_user.email}")
            return actual_user
    
    # For non-development mode, use the regular authentication
    return get_current_active_developer(request, db)

# In-memory storage for bot code (in production, use Redis or database)
bot_code_storage = {}

class BotSandbox:
    """Sandbox environment for bot development and testing"""
    
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.temp_dir = None
        self.test_file = None
        self.bot_instance = None
        self.test_results = []
        self.backtest_data = []
    
    def setup_temp_environment(self, bot_code: str) -> bool:
        """Setup a temporary environment for testing the bot"""
        try:
            import tempfile
            import os
            import sys
            
            self.temp_dir = tempfile.mkdtemp(prefix=f"bot_sandbox_{self.user_id}_")
            test_file = os.path.join(self.temp_dir, 'test_bot.py')
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(bot_code)
            self.test_file = test_file
            
            # Add temp directory to Python path at highest priority
            if self.temp_dir in sys.path:
                sys.path.remove(self.temp_dir)
            sys.path.insert(0, self.temp_dir)
            
            # Clear cached module to force re-import and syntax check
            if 'test_bot' in sys.modules:
                try:
                    del sys.modules['test_bot']
                except Exception:
                    pass
            
            return True
        except Exception as e:
            self.test_results.append(f"Setup failed: {str(e)}")
            return False
    
    def _find_bot_class(self, mod) -> Any:
        """Find a bot class in module: prefer classes with execute_algorithm"""
        bot_cls = None
        for name, obj in inspect.getmembers(mod, inspect.isclass):
            if obj.__module__ == mod.__name__ and hasattr(obj, 'execute_algorithm'):
                bot_cls = obj
                break
        return bot_cls
    
    def test_bot_import(self) -> bool:
        """Test if bot can be imported successfully"""
        try:
            import importlib
            mod = importlib.import_module('test_bot')
            # Find suitable bot class
            bot_cls = self._find_bot_class(mod)
            if bot_cls is None:
                self.test_results.append("❌ No suitable bot class found (requires execute_algorithm)")
                return False
            
            # Try instantiate with (config, api_keys) then fallback to (config) then ()
            config = {}
            api_keys = {"api_key": "x", "api_secret": "y"}
            inst = None
            try:
                inst = bot_cls(config, api_keys)
            except Exception:
                try:
                    inst = bot_cls(config)
                except Exception:
                    inst = bot_cls()
            self.bot_instance = inst
            
            ok_methods = []
            if hasattr(self.bot_instance, 'execute_algorithm'):
                ok_methods.append('execute_algorithm')
            if hasattr(self.bot_instance, 'get_configuration_schema'):
                ok_methods.append('get_configuration_schema')
            self.test_results.append(f"✅ Bot import successful ({', '.join(ok_methods)})")
            return True
            
        except Exception as e:
            self.test_results.append(f"❌ Bot import failed: {str(e)}")
            return False
    
    def test_bot_initialization(self, config: Dict[str, Any]) -> bool:
        """Test bot initialization with config"""
        try:
            if not self.bot_instance:
                raise Exception("Bot not loaded")
            
            # If initialize exists, call it; otherwise rely on constructor
            if hasattr(self.bot_instance, 'initialize'):
                self.bot_instance.initialize(config)
            self.test_results.append("✅ Bot initialization successful")
            return True
            
        except Exception as e:
            self.test_results.append(f"❌ Bot initialization failed: {str(e)}")
            return False
    
    def test_signal_generation(self, market_data: Dict[str, Any]) -> bool:
        """Test signal/action generation"""
        try:
            if not self.bot_instance:
                raise Exception("Bot not loaded")
            
            action_obj = None
            if hasattr(self.bot_instance, 'execute_algorithm'):
                # Build a minimal DataFrame from candles if possible
                import pandas as pd
                closes = []
                for c in market_data.get('candles', []):
                    try:
                        closes.append(float(c[4]))
                    except Exception:
                        continue
                if not closes:
                    closes = [float(market_data.get('current_price', 0))]
                df = pd.DataFrame({'close': closes})
                action_obj = self.bot_instance.execute_algorithm(df, market_data.get('timeframe', '1m'), {})
            elif hasattr(self.bot_instance, 'generate_signal'):
                action_obj = self.bot_instance.generate_signal(market_data)
            else:
                raise Exception("Bot has neither execute_algorithm nor generate_signal")
            
            # Validate result
            if isinstance(action_obj, dict):
                self.test_results.append("✅ Action generated (dict)")
                return True
            if hasattr(action_obj, 'to_dict'):
                _ = action_obj.to_dict()
                self.test_results.append("✅ Action generated (object with to_dict)")
                return True
            # Fallback: check common attrs
            if all(hasattr(action_obj, k) for k in ('action', 'reason')):
                self.test_results.append("✅ Action generated (object)")
                return True
            raise Exception("Action format not recognized")
            
        except Exception as e:
            self.test_results.append(f"❌ Action test failed: {str(e)}")
            return False
    
    def run_backtest(self, exchange_type: str, symbol: str, timeframe: str, 
                    start_date: str, end_date: str, initial_balance: float = 10000,
                    fee_rate: float = 0.001,
                    trade_amount_type: str = "FULL_BALANCE",  # FULL_BALANCE | PERCENT_BALANCE | FIXED_QUOTE | FIXED_BASE
                    trade_amount_value: float = 100.0,
                    prediction_cycle: str = "5m",
                    data_fetch_timeframe: str = "5m",
                    data_fetch_limit: int = 100) -> Dict[str, Any]:
        """Run backtest with historical data"""
        try:
            print(f"Starting backtest for {symbol} on {timeframe} from {start_date} to {end_date}")
            print(f"Bot controls: prediction_cycle={prediction_cycle}, data_fetch={data_fetch_timeframe} ({data_fetch_limit} candles)")
            
            if not self.bot_instance:
                raise Exception("Bot not loaded")
            if not EXCHANGE_FACTORY_AVAILABLE:
                raise Exception("ExchangeFactory not available")
            
            # Set bot control settings
            if hasattr(self.bot_instance, 'prediction_cycle'):
                self.bot_instance.prediction_cycle = prediction_cycle
            if hasattr(self.bot_instance, 'data_fetch_timeframe'):
                self.bot_instance.data_fetch_timeframe = data_fetch_timeframe
            if hasattr(self.bot_instance, 'data_fetch_limit'):
                self.bot_instance.data_fetch_limit = data_fetch_limit
            
            print("Creating exchange client...")
            exchange_factory = ExchangeFactory()
            exchange = exchange_factory.create_exchange(exchange_type, use_testnet=True)
            
            print("Fetching historical data...")
            start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            all_candles = []
            current_dt = start_dt
            max_candles = 3000
            while current_dt < end_dt and len(all_candles) < max_candles:
                try:
                    candles = exchange.fetch_ohlcv(symbol, data_fetch_timeframe, limit=1000)
                except TypeError:
                    candles = exchange.fetch_ohlcv(symbol, data_fetch_timeframe, since=int(current_dt.timestamp() * 1000), limit=1000)
                if not candles:
                    break
                all_candles.extend(candles)
                if len(all_candles) >= max_candles:
                    all_candles = all_candles[:max_candles]
                    break
                current_dt = datetime.fromtimestamp(candles[-1][0] / 1000, tz=timezone.utc) + timedelta(minutes=1)
            
            print(f"Fetched {len(all_candles)} candles using {data_fetch_timeframe} timeframe")
            
            if len(all_candles) < 50:
                raise Exception("Insufficient historical data")

            balance = float(initial_balance)
            position = 0.0
            trades = []
            signals = []

            def compute_trade_quote(current_balance: float, price: float) -> (float, float):
                # returns (trade_quote, trade_base) before fees
                nonlocal trade_amount_type, trade_amount_value
                if trade_amount_type == "PERCENT_BALANCE":
                    q = max(0.0, min(current_balance, current_balance * (float(trade_amount_value) / 100.0)))
                    return (q, q / price)
                elif trade_amount_type == "FIXED_QUOTE":
                    q = max(0.0, min(current_balance, float(trade_amount_value)))
                    return (q, q / price)
                elif trade_amount_type == "FIXED_BASE":
                    b = max(0.0, float(trade_amount_value))
                    return (b * price, b)
                # FULL_BALANCE default
                return (current_balance, current_balance / price)

            print(f"Running backtest simulation with prediction cycle: {prediction_cycle}")
            
            # Calculate prediction cycle interval in minutes
            cycle_minutes = {
                "1s": 1/60, "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, 
                "1h": 60, "2h": 120, "4h": 240, "6h": 360, "8h": 480, "12h": 720,
                "1d": 1440, "3d": 4320, "1w": 10080, "1M": 43200
            }
            prediction_interval = cycle_minutes.get(prediction_cycle, 5)
            
            # Run simulation based on prediction cycle
            last_prediction_time = None
            for i in range(50, len(all_candles)):
                candle = all_candles[i]
                timestamp = datetime.fromtimestamp(candle[0] / 1000, tz=timezone.utc)
                current_price = float(candle[4])

                # Check if it's time for a prediction based on prediction cycle
                if last_prediction_time is None or (timestamp - last_prediction_time).total_seconds() >= prediction_interval * 60:
                    last_prediction_time = timestamp
                    
                    # Build DataFrame window for execute_algorithm
                    signal: Dict[str, Any] = {}
                    if hasattr(self.bot_instance, 'execute_algorithm'):
                        import pandas as pd
                        # Use data_fetch_limit for window size
                        window = all_candles[max(0, i-data_fetch_limit):i+1]
                        closes = [float(c[4]) for c in window if len(c) > 4]
                        df = pd.DataFrame({'close': closes})
                        res = self.bot_instance.execute_algorithm(df, data_fetch_timeframe, {})
                        if isinstance(res, dict):
                            signal = res
                        elif hasattr(res, 'to_dict'):
                            signal = res.to_dict()
                        else:
                            signal = {'action': getattr(res, 'action', None), 'reason': getattr(res, 'reason', '')}
                    elif hasattr(self.bot_instance, 'generate_signal'):
                        market_data = {'symbol': symbol, 'timeframe': data_fetch_timeframe, 'candles': all_candles[:i+1], 'current_price': current_price, 'timestamp': timestamp.isoformat()}
                        signal = self.bot_instance.generate_signal(market_data)
                    else:
                        raise Exception("Bot has neither execute_algorithm nor generate_signal")

                    sig_action = (signal or {}).get('action')
                    balance_before = balance
                    position_before = position
                    fee_quote = 0.0
                    base_delta = 0.0
                    quote_delta = 0.0

                    if sig_action == 'BUY' and position == 0.0:
                        trade_quote, _ = compute_trade_quote(balance, current_price)
                        fee_quote = trade_quote * float(fee_rate)
                        net_quote = max(0.0, trade_quote - fee_quote)
                        base_bought = net_quote / current_price if current_price > 0 else 0.0
                        position += base_bought
                        balance -= trade_quote
                        base_delta = base_bought
                        quote_delta = -trade_quote
                        trades.append({
                            'timestamp': timestamp.isoformat(),
                            'action': 'BUY',
                            'price': current_price,
                            'quote_amount': trade_quote,
                            'base_amount': base_bought,
                            'fee_quote': fee_quote,
                            'fee_rate': float(fee_rate),
                            'position_size_pct': (trade_quote / balance_before * 100.0) if balance_before > 0 else 0.0,
                            'balance_before': balance_before,
                            'balance_after': balance,
                        })
                        print(f"BUY: ${current_price:.2f}, Amount: ${trade_quote:.2f}, Fee: ${fee_quote:.4f}")
                        
                    elif sig_action == 'SELL' and position > 0.0:
                        proceeds_quote = position * current_price
                        fee_quote = proceeds_quote * float(fee_rate)
                        net_quote = proceeds_quote - fee_quote
                        balance += net_quote
                        base_sold = position
                        base_delta = -base_sold
                        quote_delta = net_quote
                        position = 0.0
                        pnl = balance - balance_before
                        trades.append({
                            'timestamp': timestamp.isoformat(),
                            'action': 'SELL',
                            'price': current_price,
                            'quote_amount': net_quote,
                            'base_amount': base_sold,
                            'fee_quote': fee_quote,
                            'fee_rate': float(fee_rate),
                            'position_size_pct': 0.0,
                            'balance_before': balance_before,
                            'balance_after': balance,
                            'pnl_quote': pnl
                        })
                        print(f"SELL: ${current_price:.2f}, Amount: ${net_quote:.2f}, PnL: ${pnl:.2f}")

                    signals.append({'timestamp': timestamp.isoformat(), 'price': current_price, 'signal': signal})

            final_balance = balance + (position * float(all_candles[-1][4]))
            total_return = ((final_balance - initial_balance) / initial_balance) * 100
            
            print(f"Backtest completed: Initial: ${initial_balance:.2f}, Final: ${final_balance:.2f}, Return: {total_return:.2f}%, Trades: {len(trades)}")
            
            return {
                'initial_balance': initial_balance,
                'final_balance': final_balance,
                'total_return': total_return,
                'trades': trades,
                'signals': signals,
                'total_trades': len(trades),
                'fee_rate': float(fee_rate),
                'sizing': {'type': trade_amount_type, 'value': trade_amount_value},
                'bot_controls': {
                    'prediction_cycle': prediction_cycle,
                    'data_fetch_timeframe': data_fetch_timeframe,
                    'data_fetch_limit': data_fetch_limit
                }
            }
        except Exception as e:
            print(f"Backtest failed: {e}")
            raise Exception(f"Backtest failed: {str(e)}")

    def run_live_test(self, exchange_type: str, symbol: str, timeframe: str, duration_minutes: int = 60,
                      initial_balance: float = 10000,
                      fee_rate: float = 0.001,
                      trade_amount_type: str = "PERCENT_BALANCE",
                      trade_amount_value: float = 10.0,
                      start_immediately: bool = False) -> Dict[str, Any]:
        """Run live test paced by timeframe. Waits until each candle closes before making a decision.
        This sandbox implementation still runs synchronously with sleeps to simulate realtime pacing.
        """
        try:
            if not self.bot_instance:
                raise Exception("Bot not loaded")
            exchange_factory = ExchangeFactory()
            exchange = exchange_factory.create_exchange(exchange_type, use_testnet=True)

            # Fetch initial candles for context
            try:
                candles = exchange.fetch_ohlcv(symbol, timeframe, limit=200)
            except TypeError:
                candles = exchange.fetch_ohlcv(symbol, timeframe)
            if not candles or len(candles) < 50:
                raise Exception("Insufficient market data")

            balance = float(initial_balance)
            position = 0.0
            signals: List[Dict[str, Any]] = []
            trades: List[Dict[str, Any]] = []

            # Determine timeframe seconds
            tf_map = {"1m":60, "3m":180, "5m":300, "15m":900, "30m":1800, "1h":3600, "4h":14400, "1d":86400}
            tf_sec = tf_map.get(timeframe, 60)

            # Compute next candle close time from last candle timestamp
            last_ts_ms = int(candles[-1][0])  # ms
            last_close = datetime.fromtimestamp(last_ts_ms/1000, tz=timezone.utc)
            # Snap to next boundary
            next_close = last_close + timedelta(seconds=tf_sec)

            # If not starting immediately, wait to the next boundary
            if not start_immediately:
                now = datetime.now(timezone.utc)
                if next_close > now:
                    import time as _t
                    _t.sleep(min((next_close - now).total_seconds(), 5))  # cap sleep to 5s for sandbox responsiveness

            # How many iterations to run paced by timeframe
            max_iters = max(1, int(duration_minutes * 60 / tf_sec))

            def compute_trade_quote(current_balance: float, price: float) -> (float, float):
                nonlocal trade_amount_type, trade_amount_value
                if trade_amount_type == "PERCENT_BALANCE":
                    q = max(0.0, min(current_balance, current_balance * (float(trade_amount_value) / 100.0)))
                    return (q, q / price)
                elif trade_amount_type == "FIXED_QUOTE":
                    q = max(0.0, min(current_balance, float(trade_amount_value)))
                    return (q, q / price)
                elif trade_amount_type == "FIXED_BASE":
                    b = max(0.0, float(trade_amount_value))
                    return (b * price, b)
                return (current_balance, current_balance / price)

            import time as _t
            for i in range(max_iters):
                # Wait until candle close boundary
                now = datetime.now(timezone.utc)
                if now < next_close:
                    _t.sleep(min((next_close - now).total_seconds(), 5))
                # After waiting, fetch latest candles and compute signal based on the just-closed candle
                try:
                    current_candles = exchange.fetch_ohlcv(symbol, timeframe, limit=200)
                except TypeError:
                    current_candles = exchange.fetch_ohlcv(symbol, timeframe)
                if not current_candles:
                    break
                current_price = float(current_candles[-1][4])

                signal: Dict[str, Any] = {}
                if hasattr(self.bot_instance, 'execute_algorithm'):
                    import pandas as pd
                    closes = [float(c[4]) for c in current_candles if len(c) > 4]
                    df = pd.DataFrame({'close': closes})
                    res = self.bot_instance.execute_algorithm(df, timeframe, {})
                    if isinstance(res, dict):
                        signal = res
                    elif hasattr(res, 'to_dict'):
                        signal = res.to_dict()
                    else:
                        signal = {'action': getattr(res, 'action', None), 'reason': getattr(res, 'reason', '')}
                elif hasattr(self.bot_instance, 'generate_signal'):
                    market_data = {'symbol': symbol, 'timeframe': timeframe, 'candles': current_candles, 'current_price': current_price, 'timestamp': datetime.utcnow().isoformat()}
                    signal = self.bot_instance.generate_signal(market_data)
                else:
                    raise Exception("Bot has neither execute_algorithm nor generate_signal")

                sig_action = (signal or {}).get('action')
                balance_before = balance
                fee_quote = 0.0
                if sig_action == 'BUY' and position == 0.0:
                    trade_quote, _ = compute_trade_quote(balance, current_price)
                    fee_quote = trade_quote * float(fee_rate)
                    net_quote = max(0.0, trade_quote - fee_quote)
                    base_bought = net_quote / current_price if current_price > 0 else 0.0
                    position += base_bought
                    balance -= trade_quote
                    trades.append({
                        'timestamp': datetime.utcnow().isoformat(),
                        'action': 'BUY',
                        'price': current_price,
                        'quote_amount': trade_quote,
                        'base_amount': base_bought,
                        'fee_quote': fee_quote,
                        'fee_rate': float(fee_rate),
                        'position_size_pct': (trade_quote / balance_before * 100.0) if balance_before > 0 else 0.0,
                        'balance_before': balance_before,
                        'balance_after': balance,
                    })
                elif sig_action == 'SELL' and position > 0.0:
                    proceeds_quote = position * current_price
                    fee_quote = proceeds_quote * float(fee_rate)
                    net_quote = proceeds_quote - fee_quote
                    balance += net_quote
                    trades.append({
                        'timestamp': datetime.utcnow().isoformat(),
                        'action': 'SELL',
                        'price': current_price,
                        'quote_amount': net_quote,
                        'base_amount': position,
                        'fee_quote': fee_quote,
                        'fee_rate': float(fee_rate),
                        'position_size_pct': 0.0,
                        'balance_before': balance_before,
                        'balance_after': balance,
                    })
                    position = 0.0

                signals.append({'timestamp': datetime.utcnow().isoformat(), 'price': current_price, 'signal': signal})

                # Advance next_close to upcoming timeframe boundary
                next_close = next_close + timedelta(seconds=tf_sec)

            return {
                'duration_minutes': duration_minutes,
                'signals': signals,
                'trades': trades,
                'total_signals': len(signals),
                'fee_rate': float(fee_rate),
                'sizing': {'type': trade_amount_type, 'value': trade_amount_value},
                'final_balance': balance + position * current_price
            }
        except Exception as e:
            raise Exception(f"Live test failed: {str(e)}")
    
    def cleanup(self):
        """Clean up temporary environment"""
        try:
            if self.temp_dir and os.path.exists(self.temp_dir):
                import shutil
                shutil.rmtree(self.temp_dir)
            
            # Remove from Python path
            if self.temp_dir in sys.path:
                sys.path.remove(self.temp_dir)
                
        except Exception as e:
            print(f"Cleanup error: {e}")

    def test_bot_code(self, bot_code: str, exchange_type: str, symbol: str, timeframe: str, config: str) -> dict:
        """Test bot code for syntax and basic functionality"""
        try:
            # Parse config if provided
            try:
                config_dict = json.loads(config) if config else {}
            except json.JSONDecodeError as e:
                return {
                    "success": False,
                    "error": "Invalid configuration JSON",
                    "details": {
                        "error_type": "JSONDecodeError",
                        "error_message": str(e),
                        "line_info": f"JSON parsing error at position {e.pos}: {e.msg}",
                        "context": config[:100] + "..." if len(config) > 100 else config
                    },
                    "message": f"❌ Configuration error: Invalid JSON format"
                }
            
            # Test syntax first
            try:
                compile(bot_code, '<string>', 'exec')
            except SyntaxError as e:
                # Enhanced syntax error reporting with better line detection
                lines = bot_code.split('\n')
                
                # Get the problematic line
                error_line = ""
                if e.lineno <= len(lines):
                    error_line = lines[e.lineno - 1]
                else:
                    error_line = "Line number out of range"
                
                # Create visual indicator for the error position
                indicator = ""
                if e.offset and e.text:
                    # Calculate the actual offset in the line
                    actual_offset = min(e.offset - 1, len(error_line))
                    indicator = ' ' * actual_offset + '^'
                
                # Get context lines (3 lines before and 2 lines after for better context)
                context_start = max(0, e.lineno - 4)
                context_end = min(len(lines), e.lineno + 2)
                context_lines = lines[context_start:context_end]
                
                context_display = []
                for i, line in enumerate(context_lines):
                    line_num = context_start + i + 1
                    marker = ">>> " if line_num == e.lineno else "    "
                    # Highlight the error line
                    if line_num == e.lineno:
                        context_display.append(f"{marker}{line_num:3d}: {line}")
                        if indicator:
                            context_display.append(f"     {indicator}")
                    else:
                        context_display.append(f"{marker}{line_num:3d}: {line}")
                
                # Provide more specific error information
                error_details = {
                    "error_type": "SyntaxError",
                    "error_message": e.msg,
                    "line_number": e.lineno,
                    "column": e.offset,
                    "error_line": error_line,
                    "indicator": indicator,
                    "context": "\n".join(context_display),
                    "full_error": f"SyntaxError: {e.msg} at line {e.lineno}, column {e.offset}",
                    "suggestion": self._get_syntax_error_suggestion(e.msg, e.lineno, error_line)
                }
                
                return {
                    "success": False,
                    "error": "Syntax error detected",
                    "details": error_details,
                    "message": f"❌ Syntax Error at line {e.lineno}: {e.msg}"
                }
            
            # Test import and class structure
            try:
                # Create a temporary module
                temp_module = types.ModuleType("test_bot")
                
                # Execute the code in the temporary module
                exec(bot_code, temp_module.__dict__)
                
                # Find bot classes
                bot_classes = []
                for attr_name in dir(temp_module):
                    attr = getattr(temp_module, attr_name)
                    if (inspect.isclass(attr) and 
                        hasattr(attr, '__bases__') and 
                        any('CustomBot' in str(base) for base in attr.__bases__)):
                        bot_classes.append(attr)
                
                if not bot_classes:
                    return {
                        "success": False,
                        "error": "No valid bot class found",
                        "details": {
                            "error_type": "ClassNotFoundError",
                            "error_message": "No class inheriting from CustomBot found in the code",
                            "available_classes": [name for name in dir(temp_module) if inspect.isclass(getattr(temp_module, name))],
                            "suggestion": "Make sure your bot class inherits from CustomBot"
                        },
                        "message": "❌ No valid bot class found - must inherit from CustomBot"
                    }
                
                # Test bot initialization
                bot_class = bot_classes[0]  # Use the first found bot class
                
                # Test configuration
                test_config = {
                    'prediction_cycle': '15m',
                    'prediction_cycle_configurable': True,
                    'max_data_points': 1000,
                    'required_warmup_periods': 50
                }
                
                try:
                    bot_instance = bot_class(test_config)
                except Exception as e:
                    return {
                        "success": False,
                        "error": "Bot initialization failed",
                        "details": {
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                            "bot_class": bot_class.__name__,
                            "test_config": test_config,
                            "suggestion": "Check your bot's __init__ method and ensure it handles the required parameters"
                        },
                        "message": f"❌ Bot initialization failed: {type(e).__name__} - {str(e)}"
                    }
                
                # Test required methods
                required_methods = ['execute_algorithm', 'get_configuration_schema']
                missing_methods = []
                
                for method in required_methods:
                    if not hasattr(bot_instance, method):
                        missing_methods.append(method)
                
                if missing_methods:
                    return {
                        "success": False,
                        "error": "Missing required methods",
                        "details": {
                            "error_type": "MissingMethodError",
                            "error_message": f"Bot class is missing required methods: {', '.join(missing_methods)}",
                            "bot_class": bot_class.__name__,
                            "missing_methods": missing_methods,
                            "suggestion": f"Implement the following methods: {', '.join(missing_methods)}"
                        },
                        "message": f"❌ Missing required methods: {', '.join(missing_methods)}"
                    }
                
                # Get configuration schema
                try:
                    config_schema = bot_instance.get_configuration_schema()
                except Exception as e:
                    config_schema = {}
                    print(f"Could not get configuration schema: {e}")
                
                # Test execute_algorithm with minimal data
                try:
                    # Create a minimal exchange client for testing
                    test_exchange = self._create_test_exchange(exchange_type)
                    bot_instance.exchange_client = test_exchange
                    
                    # Test with minimal data and check signature
                    from datetime import datetime
                    
                    # Get method signature
                    sig = inspect.signature(bot_instance.execute_algorithm)
                    param_count = len(sig.parameters)
                    
                    # Test with different parameter counts
                    if param_count == 2:  # self, market_data
                        test_result = bot_instance.execute_algorithm(datetime.utcnow())
                    elif param_count == 3:  # self, market_data, timeframe
                        test_result = bot_instance.execute_algorithm(datetime.utcnow(), "5m")
                    elif param_count == 4:  # self, market_data, timeframe, config
                        test_result = bot_instance.execute_algorithm(datetime.utcnow(), "5m", {})
                    else:
                        test_result = f"Unexpected signature: {param_count} parameters, expected 2-4"
                    
                    if test_result is None:
                        test_result = "No action returned (None)"
                    elif isinstance(test_result, dict) and not test_result:
                        test_result = "Empty action returned ({})"
                    
                except TypeError as e:
                    # Handle signature mismatch
                    test_result = f"Signature error: {str(e)}"
                    print(f"Bot signature test failed: {e}")
                except Exception as e:
                    test_result = f"Error during execution: {type(e).__name__} - {str(e)}"
                    print(f"Bot execution test failed: {e}")
                
                # Get prediction cycle info
                prediction_cycle_info = {
                    "current_cycle": getattr(bot_instance, 'prediction_cycle', 'Unknown'),
                    "configurable": getattr(bot_instance, 'prediction_cycle_configurable', False),
                    "supported_cycles": bot_instance.get_supported_prediction_cycles() if hasattr(bot_instance, 'get_supported_prediction_cycles') else [],
                    "recommended_cycle": bot_instance.get_recommended_prediction_cycle() if hasattr(bot_instance, 'get_recommended_prediction_cycle') else getattr(bot_instance, 'prediction_cycle', 'Unknown')
                }
                
                return {
                    "success": True,
                    "message": "✅ Bot code is valid!",
                    "bot_class_name": bot_class.__name__,
                    "config_schema": config_schema,
                    "test_result": test_result,
                    "prediction_cycle_info": prediction_cycle_info,
                    "details": {
                        "methods_found": [method for method in dir(bot_instance) if not method.startswith('_')],
                        "inheritance": [base.__name__ for base in bot_class.__bases__],
                        "module_info": f"Successfully imported from {temp_module.__name__}"
                    }
                }
                
            except Exception as e:
                # Handle import/execution errors with enhanced debugging
                import traceback
                import sys
                
                # Get detailed error information
                exc_type, exc_value, exc_traceback = sys.exc_info()
                tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
                tb_text = ''.join(tb_lines)
                
                # Analyze the error to provide better suggestions
                error_analysis = self._analyze_runtime_error(e, tb_text)
                
                return {
                    "success": False,
                    "error": "Bot code execution failed",
                    "details": {
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "full_traceback": tb_text,
                        "error_analysis": error_analysis,
                        "suggestion": error_analysis.get('suggestion', "Check your code for runtime errors, missing imports, or undefined variables")
                    },
                    "message": f"❌ Execution failed: {type(e).__name__} - {str(e)}"
                }
                
        except Exception as e:
            # Catch any other unexpected errors
            import traceback
            tb_lines = traceback.format_exception(type(e), e, e.__traceback__)
            tb_text = ''.join(tb_lines)
            
            return {
                "success": False,
                "error": "Unexpected error during testing",
                "details": {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "full_traceback": tb_text
                },
                "message": f"❌ Unexpected error: {type(e).__name__} - {str(e)}"
            } 

    def _get_syntax_error_suggestion(self, error_msg: str, line_num: int, error_line: str) -> str:
        """Provide helpful suggestions for common syntax errors"""
        error_msg_lower = error_msg.lower()
        error_line_lower = error_line.lower()
        
        suggestions = []
        
        # Common syntax error patterns
        if "unexpected indent" in error_msg_lower:
            suggestions.append("Check indentation - Python is very strict about spaces vs tabs")
            suggestions.append("Make sure all lines in the same block have the same indentation level")
        elif "invalid syntax" in error_msg_lower:
            if ":" in error_line and not any(keyword in error_line_lower for keyword in ["if", "else", "elif", "for", "while", "def", "class", "try", "except", "finally", "with"]):
                suggestions.append("Missing colon (:) after control flow statements like if, for, def, class")
            elif "=" in error_line and "==" not in error_line and "!=" not in error_line and ">=" not in error_line and "<=" not in error_line:
                suggestions.append("Check if you meant to use == (comparison) instead of = (assignment)")
        elif "eol while scanning string literal" in error_msg_lower:
            suggestions.append("Unclosed string - check for missing quotes at the end of the line")
            suggestions.append("Make sure all quotes are properly paired")
        elif "unexpected eof while parsing" in error_msg_lower:
            suggestions.append("Missing closing parenthesis, bracket, or brace")
            suggestions.append("Check if all opening symbols have matching closing symbols")
        elif "invalid character in identifier" in error_msg_lower:
            suggestions.append("Check for special characters or invisible characters in your code")
            suggestions.append("Make sure you're using standard ASCII characters")
        
        # Add general suggestions
        if not suggestions:
            suggestions.append("Check the line above and below for missing punctuation")
            suggestions.append("Verify that all parentheses, brackets, and braces are properly closed")
            suggestions.append("Ensure proper indentation throughout your code")
        
        # Add line-specific suggestions
        suggestions.append(f"Focus on line {line_num} and the surrounding context")
        
        return " • ".join(suggestions) 

    def _analyze_runtime_error(self, error: Exception, traceback_text: str) -> dict:
        """Analyze runtime errors to provide better suggestions"""
        error_type = type(error).__name__
        error_msg = str(error).lower()
        tb_text = traceback_text.lower()
        
        analysis = {
            "suggestion": "Check your code for runtime errors, missing imports, or undefined variables",
            "common_causes": [],
            "debugging_tips": []
        }
        
        # Analyze common error patterns
        if "nameerror" in error_type.lower():
            if "is not defined" in error_msg:
                analysis["suggestion"] = "Variable or function is not defined - check spelling and scope"
                analysis["common_causes"].append("Misspelled variable or function name")
                analysis["common_causes"].append("Variable used before definition")
                analysis["common_causes"].append("Missing import statement")
            elif "name" in error_msg:
                analysis["suggestion"] = "Check variable names and imports"
        
        elif "attributeerror" in error_type.lower():
            if "object has no attribute" in error_msg:
                analysis["suggestion"] = "Object doesn't have the method or attribute you're trying to use"
                analysis["common_causes"].append("Wrong object type")
                analysis["common_causes"].append("Method name misspelled")
                analysis["common_causes"].append("Object not properly initialized")
        
        elif "typeerror" in error_type.lower():
            if "unsupported operand type" in error_msg:
                analysis["suggestion"] = "Cannot perform operation between incompatible types"
                analysis["common_causes"].append("Mixing different data types")
                analysis["common_causes"].append("Using string where number expected")
            elif "argument" in error_msg:
                analysis["suggestion"] = "Wrong number or type of arguments passed to function"
                analysis["common_causes"].append("Missing required arguments")
                analysis["common_causes"].append("Wrong argument types")
        
        elif "importerror" in error_type.lower() or "modulenotfounderror" in error_type.lower():
            analysis["suggestion"] = "Module or package not found - check imports and dependencies"
            analysis["common_causes"].append("Package not installed")
            analysis["common_causes"].append("Wrong import path")
            analysis["common_causes"].append("Virtual environment not activated")
        
        elif "indentationerror" in error_type.lower():
            analysis["suggestion"] = "Python indentation error - check spaces vs tabs"
            analysis["common_causes"].append("Mixed tabs and spaces")
            analysis["common_causes"].append("Inconsistent indentation levels")
            analysis["common_causes"].append("Missing indentation after colon")
        
        # Add debugging tips based on error type
        analysis["debugging_tips"].append("Check the line mentioned in the error message")
        analysis["debugging_tips"].append("Verify all variables are defined before use")
        analysis["debugging_tips"].append("Ensure proper indentation throughout your code")
        analysis["debugging_tips"].append("Check that all parentheses, brackets, and braces are properly closed")
        
        # Add specific tips for common patterns
        if "exchange_client" in tb_text:
            analysis["debugging_tips"].append("Make sure exchange_client is properly initialized")
        if "execute_algorithm" in tb_text:
            analysis["debugging_tips"].append("Check that execute_algorithm method returns valid data")
        if "get_configuration_schema" in tb_text:
            analysis["debugging_tips"].append("Ensure get_configuration_schema returns a valid dictionary")
        
        return analysis 

    def _create_test_exchange(self, exchange_type: str):
        """Create a test exchange client for testing purposes"""
        try:
            if not EXCHANGE_FACTORY_AVAILABLE:
                # Return a mock exchange if ExchangeFactory is not available
                return MockExchange(exchange_type)
            
            exchange_factory = ExchangeFactory()
            return exchange_factory.create_exchange(exchange_type, use_testnet=True)
        except Exception as e:
            print(f"Error creating test exchange: {e}")
            # Return a mock exchange as fallback
            return MockExchange(exchange_type)

    def load_bot_code(self, bot_code: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Load bot code and create instance"""
        try:
            # Setup temporary environment
            if not self.setup_temp_environment(bot_code):
                return {
                    "success": False,
                    "error": "Failed to setup environment",
                    "message": "❌ Failed to setup testing environment"
                }
            
            # Test bot import
            if not self.test_bot_import():
                return {
                    "success": False,
                    "error": "Bot import failed",
                    "message": "❌ Bot import failed"
                }
            
            # Create bot instance
            try:
                import importlib
                temp_module = importlib.import_module('test_bot')
                bot_class = self._find_bot_class(temp_module)
                
                if bot_class is None:
                    return {
                        "success": False,
                        "error": "No suitable bot class found",
                        "message": "❌ No suitable bot class found (requires execute_algorithm)"
                    }
                
                # Initialize bot with config
                config = config or {}
                try:
                    self.bot_instance = bot_class(config)
                except Exception as e:
                    return {
                        "success": False,
                        "error": "Bot initialization failed",
                        "message": f"❌ Bot initialization failed: {type(e).__name__} - {str(e)}"
                    }
                
                return {
                    "success": True,
                    "message": "✅ Bot loaded successfully",
                    "bot_class_name": bot_class.__name__
                }
                
            except Exception as e:
                return {
                    "success": False,
                    "error": "Bot loading failed",
                    "message": f"❌ Bot loading failed: {type(e).__name__} - {str(e)}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": "Bot loading failed",
                "message": f"❌ Bot loading failed: {type(e).__name__} - {str(e)}"
            }


class MockExchange:
    """Mock exchange for testing when real exchange is not available"""
    
    def __init__(self, exchange_type: str):
        self.exchange_type = exchange_type
        self.name = f"Mock{exchange_type}"
    
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100):
        """Return mock OHLCV data"""
        import time
        current_time = int(time.time() * 1000)
        
        # Generate mock data
        mock_data = []
        base_price = 50000  # Mock BTC price
        
        for i in range(limit):
            timestamp = current_time - (limit - i) * 60000  # 1 minute intervals
            price = base_price + (i * 10)  # Slight price increase
            mock_data.append([
                timestamp,  # timestamp
                price * 0.999,  # open
                price * 1.002,  # high
                price * 0.998,  # low
                price,  # close
                1000.0  # volume
            ])
        
        return mock_data
    
    def get_klines(self, symbol: str, timeframe: str, limit: int = 100):
        """Alias for fetch_ohlcv to match expected interface"""
        return self.fetch_ohlcv(symbol, timeframe, limit)
    
    def fetch_ticker(self, symbol: str):
        """Return mock ticker data"""
        return {
            'symbol': symbol,
            'last': 50000.0,
            'bid': 49990.0,
            'ask': 50010.0,
            'volume': 1000.0
        }

@router.get("/template-bot/{filename}")
async def get_template_bot(filename: str):
    """Get a template bot file content"""
    try:
        # Resolve templates directory
        templates_dir = _resolve_templates_dir()
        template_path = os.path.join(templates_dir, filename)
        
        # Security check: ensure the file is within templates directory
        if not os.path.abspath(template_path).startswith(os.path.abspath(templates_dir)):
            raise HTTPException(status_code=400, detail="Invalid file path")
        
        # Check if file exists
        if not os.path.exists(template_path):
            raise HTTPException(status_code=404, detail=f"Template file {filename} not found")
        
        # Read and return file content
        with open(template_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return {
            "success": True,
            "filename": filename,
            "content": content,
            "size": len(content)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        # Use print instead of logger since logger is not defined
        print(f"Error serving template bot {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading template file: {str(e)}")

@router.post("/test-bot-code")
async def test_bot_code(
    bot_code: str = Form(...),
    config: str = Form("{}"),
    exchange_type: str = Form("BINANCE"),
    symbol: str = Form("BTC/USDT"),
    timeframe: str = Form("5m"),
    test_real_data: bool = Form(False),
    current_user: Any = Depends(get_current_user_dep),
    db: Session = Depends(get_db)
):
    """Test bot code for syntax and basic functionality"""
    try:
        # Create a temporary environment for testing
        sandbox = BotSandbox(current_user.id if hasattr(current_user, 'id') else 1)
        
        # Parse config
        try:
            config_dict = json.loads(config) if config else {}
        except json.JSONDecodeError:
            return {
                "success": False,
                "error": "Invalid configuration JSON",
                "message": "❌ Configuration error: Invalid JSON format"
            }
        
        # Test bot code compilation and basic structure
        result = sandbox.test_bot_code(
            bot_code,
            exchange_type,
            symbol,
            timeframe,
            config
        )
        
        return result
        
    except Exception as e:
        # Enhanced error logging with detailed information
        import traceback
        import sys
        
        # Get detailed error information
        exc_type, exc_value, exc_traceback = sys.exc_info()
        
        # Format the traceback
        tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        tb_text = ''.join(tb_lines)
        
        # Try to extract line number from syntax errors
        line_info = ""
        if isinstance(e, SyntaxError):
            line_info = f"Syntax Error at line {e.lineno}, column {e.offset}: {e.text}"
            if e.text:
                line_info += f"\nContext: {e.text.strip()}"
                if e.offset and e.text:
                    line_info += f"\n{' ' * (e.offset - 1)}^"
        
        # Create detailed error response
        error_details = {
            "error_type": exc_type.__name__ if exc_type else "Unknown",
            "error_message": str(e),
            "line_info": line_info,
            "full_traceback": tb_text,
            "error_location": f"{exc_type.__name__}: {str(e)}" if exc_type else str(e)
        }
        
        # Use print instead of logger since logger is not defined
        print(f"Bot code test failed: {error_details}")
        
        return {
            "success": False,
            "error": "Bot code test failed",
            "details": error_details,
            "message": f"❌ Test failed: {error_details['error_type']} - {error_details['error_message']}"
        }


@router.post("/backtest")
async def run_backtest(
    bot_code: str = Form(...),
    config: str = Form("{}"),
    exchange_type: str = Form("BINANCE"),
    symbol: str = Form("BTC/USDT"),
    timeframe: str = Form("5m"),
    start_date: str = Form(...),
    end_date: str = Form(...),
    initial_balance: float = Form(10000.0),
    prediction_cycle: str = Form("5m"),
    data_fetch_timeframe: str = Form("5m"),
    data_fetch_limit: int = Form(100),
    current_user: Any = Depends(get_current_user_dep)
):
    """Run backtest with historical data"""
    try:
        print(f"Backtest request received: {symbol} {timeframe} {start_date} to {end_date}")
        print(f"Bot controls: prediction_cycle={prediction_cycle}, data_fetch={data_fetch_timeframe} ({data_fetch_limit} candles)")
        
        # Create a temporary environment for testing
        sandbox = BotSandbox(current_user.id if hasattr(current_user, 'id') else 1)
        
        # Parse config
        try:
            config_dict = json.loads(config) if config else {}
        except json.JSONDecodeError:
            return {
                "success": False,
                "error": "Invalid configuration JSON",
                "message": "❌ Configuration error: Invalid JSON format"
            }
        
        # Add bot control settings to config
        config_dict.update({
            'prediction_cycle': prediction_cycle,
            'data_fetch_timeframe': data_fetch_timeframe,
            'data_fetch_limit': data_fetch_limit,
            'prediction_cycle_configurable': True # Assuming configurable for backtest
        })
        
        # Load bot code first
        load_result = sandbox.load_bot_code(bot_code, config_dict)
        if not load_result.get("success", False):
            return load_result
        
        # Run backtest with bot control settings
        result = sandbox.run_backtest(
            exchange_type,
            symbol,
            timeframe,
            start_date,
            end_date,
            initial_balance,
            prediction_cycle=prediction_cycle,
            data_fetch_timeframe=data_fetch_timeframe,
            data_fetch_limit=data_fetch_limit
        )
        
        # Add success flag to result
        result['success'] = True
        result['message'] = f"✅ Backtest completed: {result.get('total_trades', 0)} trades, {result.get('total_return', 0):.2f}% return"
        result['bot_controls'] = {
            'prediction_cycle': prediction_cycle,
            'data_fetch_timeframe': data_fetch_timeframe,
            'data_fetch_limit': data_fetch_limit,
            'prediction_cycle_configurable': True
        }
        
        print(f"Backtest completed successfully: {result.get('total_trades', 0)} trades")
        return result
        
    except Exception as e:
        print(f"Backtest failed: {e}")
        return {
            "success": False,
            "error": "Backtest failed",
            "message": f"❌ Backtest failed: {str(e)}"
        }


@router.post("/live-test")
async def run_live_test(
    bot_code: str = Form(...),
    config: str = Form("{}"),
    exchange_type: str = Form("BINANCE"),
    symbol: str = Form("BTC/USDT"),
    timeframe: str = Form("5m"),
    duration_minutes: int = Form(10),
    current_user: Any = Depends(get_current_user_dep)
):
    """Run live test with real-time data"""
    try:
        # Create a temporary environment for testing
        sandbox = BotSandbox(current_user.id if hasattr(current_user, 'id') else 1)
        
        # Parse config
        try:
            config_dict = json.loads(config) if config else {}
        except json.JSONDecodeError:
            return {
                "success": False,
                "error": "Invalid configuration JSON",
                "message": "❌ Configuration error: Invalid JSON format"
            }
        
        # Load bot code first
        load_result = sandbox.load_bot_code(bot_code, config_dict)
        if not load_result.get("success", False):
            return load_result
        
        # Run live test
        result = sandbox.run_live_test(
            exchange_type,
            symbol,
            timeframe,
            duration_minutes
        )
        
        return result
        
    except Exception as e:
        print(f"Live test failed: {e}")
        return {
            "success": False,
            "error": "Live test failed",
            "message": f"❌ Live test failed: {str(e)}"
        }


@router.websocket("/live/ws")
async def live_test_ws(websocket: WebSocket):
    """WebSocket endpoint for live test with authentication"""
    await websocket.accept()
    
    try:
        # Get query parameters
        query_params = websocket.query_params
        exchange_type = query_params.get("exchange_type", "BINANCE")
        symbol = query_params.get("symbol", "BTC/USDT")
        timeframe = query_params.get("timeframe", "5m")
        initial_balance = float(query_params.get("initial_balance", "10000"))
        fee_rate = float(query_params.get("fee_rate", "0.001"))
        trade_amount_type = query_params.get("trade_amount_type", "PERCENT_BALANCE")
        trade_amount_value = float(query_params.get("trade_amount_value", "10"))
        
        # Bot control settings
        prediction_cycle = query_params.get("prediction_cycle", "5m")
        data_fetch_timeframe = query_params.get("data_fetch_timeframe", "5m")
        data_fetch_limit = int(query_params.get("data_fetch_limit", "100"))
        prediction_cycle_configurable = query_params.get("prediction_cycle_configurable", "true").lower() == "true"
        
        # Send initial connection message
        await websocket.send_text(json.dumps({
            "type": "connection",
            "message": f"Connected to live test: {symbol} on {timeframe}",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "exchange_type": exchange_type,
                "symbol": symbol,
                "timeframe": timeframe,
                "initial_balance": initial_balance,
                "fee_rate": fee_rate,
                "trade_amount_type": trade_amount_type,
                "trade_amount_value": trade_amount_value,
                "prediction_cycle": prediction_cycle,
                "data_fetch_timeframe": data_fetch_timeframe,
                "data_fetch_limit": data_fetch_limit,
                "prediction_cycle_configurable": prediction_cycle_configurable
            }
        }))
        
        # Calculate prediction cycle interval
        cycle_minutes = {
            "1s": 1/60, "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, 
            "1h": 60, "2h": 120, "4h": 240, "6h": 360, "8h": 480, "12h": 720,
            "1d": 1440, "3d": 4320, "1w": 10080, "1M": 43200
        }
        prediction_interval = cycle_minutes.get(prediction_cycle, 5)
        
        # Simulate live trading updates based on prediction cycle
        import time
        import random
        
        balance = initial_balance
        position = 0.0
        trade_count = 0
        last_prediction_time = None
        
        for i in range(20):  # Simulate 20 updates
            try:
                current_time = datetime.now()
                
                # Check if it's time for a prediction based on prediction cycle
                if last_prediction_time is None or (current_time - last_prediction_time).total_seconds() >= prediction_interval * 60:
                    last_prediction_time = current_time
                    
                    # Simulate market data
                    base_price = 50000 + (random.random() - 0.5) * 1000
                    current_price = base_price + (random.random() - 0.5) * 100
                    
                    # Simulate trading signals based on prediction cycle
                    signal_type = random.choice(['BUY', 'SELL', 'HOLD'])
                    
                    if signal_type == 'BUY' and position == 0:
                        trade_amount = balance * (trade_amount_value / 100) if trade_amount_type == "PERCENT_BALANCE" else trade_amount_value
                        fee = trade_amount * fee_rate
                        position = (trade_amount - fee) / current_price
                        balance -= trade_amount
                        trade_count += 1
                        
                        await websocket.send_text(json.dumps({
                            "type": "trade",
                            "action": "BUY",
                            "price": current_price,
                            "amount": trade_amount,
                            "fee": fee,
                            "balance": balance,
                            "position": position,
                            "prediction_cycle": prediction_cycle,
                            "timestamp": current_time.isoformat()
                        }))
                        
                    elif signal_type == 'SELL' and position > 0:
                        sell_amount = position * current_price
                        fee = sell_amount * fee_rate
                        balance += sell_amount - fee
                        pnl = sell_amount - fee - (position * base_price)
                        position = 0
                        trade_count += 1
                        
                        await websocket.send_text(json.dumps({
                            "type": "trade",
                            "action": "SELL",
                            "price": current_price,
                            "amount": sell_amount,
                            "fee": fee,
                            "pnl": pnl,
                            "balance": balance,
                            "position": position,
                            "prediction_cycle": prediction_cycle,
                            "timestamp": current_time.isoformat()
                        }))
                    
                    else:
                        # Send market update
                        await websocket.send_text(json.dumps({
                            "type": "market_update",
                            "price": current_price,
                            "signal": signal_type,
                            "balance": balance,
                            "position": position,
                            "prediction_cycle": prediction_cycle,
                            "timestamp": current_time.isoformat()
                        }))
                
                # Wait based on prediction cycle
                await asyncio.sleep(min(prediction_interval * 60, 5))  # Cap at 5 seconds for responsiveness
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"Error in live test: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                }))
        
        # Send final summary
        final_balance = balance + (position * current_price)
        total_return = ((final_balance - initial_balance) / initial_balance) * 100
        
        await websocket.send_text(json.dumps({
            "type": "summary",
            "initial_balance": initial_balance,
            "final_balance": final_balance,
            "total_return": total_return,
            "total_trades": trade_count,
            "prediction_cycle": prediction_cycle,
            "data_fetch_timeframe": data_fetch_timeframe,
            "timestamp": datetime.now().isoformat()
        }))
                
    except WebSocketDisconnect:
        print("Live test WebSocket client disconnected")
    except Exception as e:
        print(f"Live test WebSocket error: {e}")
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": f"Live test error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }))
        except:
            pass


@router.post("/save-bot-code")
async def save_bot_code(
    name: str = Form(...),
    description: str = Form(""),
    bot_code: str = Form(...),
    category_id: Optional[int] = Form(None),
    db: Session = Depends(get_db),
    current_user: Any = Depends(get_current_user_from_token)
):
    """Save bot code as a draft in database"""
    try:
        # Use the actual user ID from current_user
        user_id = current_user.id
        print(f"Saving draft for user_id: {user_id}")
        
        # Create bot draft in database
        draft = crud.create_bot_draft(
            db=db,
            user_id=user_id,
            name=name,
            description=description,
            bot_code=bot_code,
            category_id=category_id
        )
        
        return {
            "success": True,
            "message": "✅ Bot code saved successfully",
            "draft": {
                "id": draft.id,
                "name": draft.name,
                "description": draft.description,
                "created_at": draft.created_at.isoformat(),
                "updated_at": draft.updated_at.isoformat()
            }
        }
        
    except Exception as e:
        print(f"Error saving bot code: {e}")
        return {
            "success": False,
            "error": "Failed to save bot code",
            "message": f"❌ Save failed: {str(e)}"
        }


@router.get("/list-saved-bots")
async def list_saved_bots(
    db: Session = Depends(get_db),
    current_user: Any = Depends(get_current_user_from_token)
):
    """List all saved bot drafts for the user from database"""
    try:
        # Use the actual user ID from current_user
        user_id = current_user.id
        print(f"Listing drafts for user_id: {user_id}")
        
        drafts = crud.get_user_bot_drafts(db, user_id)
        print(f"DEBUG: Found {len(drafts)} drafts from database")
        
        draft_list = []
        for draft in drafts:
            draft_list.append({
                "id": draft.id,
                "name": draft.name,
                "description": draft.description,
                "created_at": draft.created_at.isoformat(),
                "updated_at": draft.updated_at.isoformat(),
                "category_id": draft.category_id
            })
        
        print(f"DEBUG: Returning {len(draft_list)} drafts")
        
        return {
            "success": True,
            "drafts": draft_list,
            "total": len(draft_list)
        }
        
    except Exception as e:
        print(f"Error listing saved bots: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": "Failed to list saved bots",
            "message": f"❌ List failed: {str(e)}"
        }


@router.get("/load-bot-code/{draft_id}")
async def load_bot_code(
    draft_id: int,
    db: Session = Depends(get_db),
    current_user: Any = Depends(get_current_user_dep)
):
    """Load a specific bot draft from database"""
    try:
        user_id = current_user.id if hasattr(current_user, 'id') else 1
        
        draft = crud.get_bot_draft(db, draft_id, user_id)
        
        if not draft:
            return {
                "success": False,
                "error": "Draft not found",
                "message": "❌ Draft not found or access denied"
            }
        
        return {
            "success": True,
            "draft": {
                "id": draft.id,
                "name": draft.name,
                "description": draft.description,
                "bot_code": draft.bot_code,
                "category_id": draft.category_id,
                "created_at": draft.created_at.isoformat(),
                "updated_at": draft.updated_at.isoformat()
            }
        }
        
    except Exception as e:
        print(f"Error loading bot code: {e}")
        return {
            "success": False,
            "error": "Failed to load bot code",
            "message": f"❌ Load failed: {str(e)}"
        }


@router.delete("/delete-bot-code/{draft_id}")
async def delete_bot_code(
    draft_id: int,
    db: Session = Depends(get_db),
    current_user: Any = Depends(get_current_user_dep)
):
    """Delete a specific bot draft from database"""
    try:
        user_id = current_user.id if hasattr(current_user, 'id') else 1
        
        success = crud.delete_bot_draft(db, draft_id, user_id)
        
        if not success:
            return {
                "success": False,
                "error": "Draft not found",
                "message": "❌ Draft not found or access denied"
            }
        
        return {
            "success": True,
            "message": "✅ Draft deleted successfully"
        }
        
    except Exception as e:
        print(f"Error deleting bot code: {e}")
        return {
            "success": False,
            "error": "Failed to delete bot code",
            "message": f"❌ Delete failed: {str(e)}"
        }


def _resolve_templates_dir() -> str:
    """Resolve the path to the templates directory."""
    # Try multiple possible paths
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "..", "..", "templates", "bots"),
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "templates", "bots"),
        os.path.join(os.getcwd(), "templates", "bots"),
        os.path.join(os.getcwd(), "bot_marketplace", "templates", "bots"),
        os.environ.get("TEMPLATES_DIR", ""),  # Allow override via environment variable
    ]
    
    for path in possible_paths:
        if path and os.path.exists(path):
            return path
    
    # Fallback to current directory
    return os.path.join(os.getcwd(), "templates", "bots") 

@router.get("/_health")
async def health_check():
    """Health check endpoint - no auth required"""
    return {"included": True, "status": "healthy"}

@router.get("/test-auth")
async def test_auth(
    current_user: Any = Depends(get_current_user_dep)
):
    """Test endpoint to check authentication"""
    return {
        "success": True,
        "user_info": {
            "id": getattr(current_user, 'id', 'No ID'),
            "email": getattr(current_user, 'email', 'No email'),
            "role": getattr(current_user, 'role', 'No role'),
            "type": str(type(current_user))
        },
        "development_mode": os.getenv("DEVELOPMENT_MODE", "false")
    } 

@router.get("/decode-token")
async def decode_token_endpoint(
    request: Request
):
    """Decode JWT token from Authorization header and return user information"""
    try:
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return {
                "success": False,
                "error": "No valid Authorization header"
            }
        
        token = auth_header.split(" ")[1]
        import jwt
        from core.security import SECRET_KEY, ALGORITHM
        
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        return {
            "success": True,
            "user_id": int(payload.get("sub")),
            "role": payload.get("role"),
            "payload": payload
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        } 

@router.get("/debug-user")
async def debug_user(
    current_user: Any = Depends(get_current_user_from_token)
):
    """Debug endpoint to show current user information"""
    return {
        "success": True,
        "user_info": {
            "id": getattr(current_user, 'id', 'No ID'),
            "email": getattr(current_user, 'email', 'No email'),
            "role": getattr(current_user, 'role', 'No role'),
            "type": str(type(current_user))
        },
        "development_mode": os.getenv("DEVELOPMENT_MODE", "false")
    } 

@router.get("/test-token")
async def test_token(
    request: Request
):
    """Test endpoint to decode current token"""
    try:
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return {
                "success": False,
                "error": "No Authorization header found"
            }
        
        token = auth_header.split(" ")[1]
        print(f"DEBUG: Token received: {token[:50]}...")
        
        # Decode token manually using the same method as security.py
        import jwt
        from core.security import SECRET_KEY, ALGORITHM
        
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        print(f"DEBUG: Token payload: {payload}")
        
        return {
            "success": True,
            "token_preview": token[:50] + "...",
            "payload": payload,
            "user_id": payload.get("sub"),
            "role": payload.get("role")
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        } 