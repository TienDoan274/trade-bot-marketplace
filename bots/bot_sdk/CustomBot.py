"""
Enhanced CustomBot SDK
Provides complete trading bot framework with exchange integration and data processing
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List
from datetime import datetime, timedelta
import logging

class CustomBot(ABC):
    """
    Abstract base class for custom trading bots.
    
    Key Changes:
    - prediction_cycle: How often the bot should predict actions (independent of data fetching)
    - Developers can fetch data whenever they want using exchange client
    - No automatic data injection - developers control their own data pipeline
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the bot with configuration.
        
        Args:
            config: Bot configuration dictionary containing:
                - prediction_cycle: Time interval for action prediction (e.g., "5m", "1h", "4h")
                - prediction_cycle_configurable: Whether users can change prediction_cycle (default: True)
                - max_data_points: Maximum number of candles to fetch (optional, default 1000)
                - required_warmup_periods: Minimum candles needed before trading (optional, default 50)
                - Other custom parameters...
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Prediction cycle configuration
        self.prediction_cycle = config.get('prediction_cycle', '1h')
        self.prediction_cycle_configurable = config.get('prediction_cycle_configurable', True)
        self.prediction_cycle_seconds = self._parse_timeframe(self.prediction_cycle)
        
        # Data fetching configuration (optional - developers can override)
        self.max_data_points = config.get('max_data_points', 1000)
        self.required_warmup_periods = config.get('required_warmup_periods', 50)
        
        # Trading state
        self.last_action = None
        self.last_action_time = None
        self.is_initialized = False
        
        # Exchange client (will be set by the system)
        self.exchange_client = None
        
        # Validate prediction cycle if configurable
        if self.prediction_cycle_configurable:
            if hasattr(self, 'validate_prediction_cycle'):
                if not self.validate_prediction_cycle(self.prediction_cycle):
                    supported = self.get_supported_prediction_cycles() if hasattr(self, 'get_supported_prediction_cycles') else []
                    recommended = self.get_recommended_prediction_cycle() if hasattr(self, 'get_recommended_prediction_cycle') else self.prediction_cycle
                    raise ValueError(f"Invalid prediction cycle: {self.prediction_cycle}. Supported: {supported}. Recommended: {recommended}")
        
        # Initialize the bot
        self.initialize()
        self.is_initialized = True
        
        self.logger.info(f"{self.__class__.__name__} initialized with prediction cycle: {self.prediction_cycle}")
        if self.prediction_cycle_configurable:
            self.logger.info(f"Prediction cycle is configurable by users")
        else:
            self.logger.info(f"Prediction cycle is fixed and cannot be changed by users")
    
    def _parse_timeframe(self, timeframe: str) -> int:
        """Parse timeframe string to seconds."""
        tf_map = {
            "1m": 60, "3m": 180, "5m": 300, "15m": 900,
            "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400
        }
        return tf_map.get(timeframe, 3600)  # Default to 1h if invalid
    
    @abstractmethod
    def initialize(self):
        """
        Initialize bot-specific parameters and state.
        Called once during bot initialization.
        """
        pass
    
    @abstractmethod
    def execute_algorithm(self, current_time: datetime, market_context: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """
        Main algorithm method that predicts actions.
        
        Args:
            current_time: Current timestamp when this method is called
            market_context: Optional context information (current price, volume, etc.)
            
        Returns:
            Action dictionary or None if no action
            Example: {
                "action": "BUY",  # or "SELL", "HOLD"
                "amount": 100,     # Amount to trade
                "reason": "RSI oversold condition",
                "confidence": 0.85
            }
        """
        pass
    
    def should_execute_prediction(self, current_time: datetime) -> bool:
        """
        Check if it's time to execute prediction based on prediction cycle.
        
        Args:
            current_time: Current timestamp
            
        Returns:
            True if prediction should be executed, False otherwise
        """
        if not self.last_action_time:
            return True  # First execution
        
        time_since_last = (current_time - self.last_action_time).total_seconds()
        return time_since_last >= self.prediction_cycle_seconds
    
    def get_market_data(self, symbol: str, timeframe: str, limit: int = None) -> pd.DataFrame:
        """
        Fetch market data from exchange.
        Developers can use this method to get data whenever they need it.
        
        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            timeframe: Data timeframe (e.g., "1m", "5m", "1h")
            limit: Number of candles to fetch (defaults to self.max_data_points)
            
        Returns:
            DataFrame with OHLCV data
        """
        if not self.exchange_client:
            raise RuntimeError("Exchange client not initialized")
        
        limit = limit or self.max_data_points
        
        try:
            # Fetch klines data
            klines = self.exchange_client.get_klines(
                symbol=symbol,
                interval=timeframe,
                limit=limit
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert types
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Sort by timestamp (oldest first)
            df.sort_index(inplace=True)
            
            self.logger.info(f"Fetched {len(df)} candles for {symbol} on {timeframe}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching market data: {e}")
            return pd.DataFrame()
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for a symbol.
        
        Args:
            symbol: Trading pair
            
        Returns:
            Current price or None if error
        """
        if not self.exchange_client:
            return None
        
        try:
            ticker = self.exchange_client.get_ticker(symbol=symbol)
            return float(ticker['last'])
        except Exception as e:
            self.logger.error(f"Error getting current price: {e}")
            return None
    
    def get_balance(self, asset: str = None) -> Dict[str, float]:
        """
        Get account balance.
        
        Args:
            asset: Specific asset to get balance for (e.g., "BTC", "USDT")
            
        Returns:
            Balance dictionary
        """
        if not self.exchange_client:
            return {}
        
        try:
            balance = self.exchange_client.get_balance()
            if asset:
                return {asset: balance.get(asset, 0.0)}
            return balance
        except Exception as e:
            self.logger.error(f"Error getting balance: {e}")
            return {}
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess market data. Developers can override this method.
        
        Args:
            data: Raw market data DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        if data.empty:
            return data
        
        # Basic preprocessing
        processed_data = data.copy()
        
        # Handle missing values
        processed_data = processed_data.ffill().bfill()
        
        # Add basic technical indicators
        processed_data = self._add_basic_indicators(processed_data)
        
        return processed_data
    
    def _add_basic_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add basic technical indicators."""
        if data.empty:
            return data
        
        # Simple Moving Averages
        data['sma_20'] = data['close'].rolling(window=20).mean()
        data['sma_50'] = data['close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        data['ema_12'] = data['close'].ewm(span=12).mean()
        data['ema_26'] = data['close'].ewm(span=26).mean()
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        data['macd'] = data['ema_12'] - data['ema_26']
        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        data['macd_histogram'] = data['macd'] - data['macd_signal']
        
        # Bollinger Bands
        data['bb_middle'] = data['close'].rolling(window=20).mean()
        bb_std = data['close'].rolling(window=20).std()
        data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
        data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
        
        # Volume indicators (only if volume column exists)
        if 'volume' in data.columns:
            data['volume_sma'] = data['volume'].rolling(window=20).mean()
            data['volume_ratio'] = data['volume'] / data['volume_sma']
        else:
            # Create dummy volume data if not available
            data['volume'] = 1000.0  # Default volume
            data['volume_sma'] = 1000.0
            data['volume_ratio'] = 1.0
        
        return data
    
    def validate_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Validate trading signal. Developers can override this method.
        
        Args:
            signal: Signal dictionary from execute_algorithm
            
        Returns:
            True if signal is valid, False otherwise
        """
        if not signal:
            return False
        
        required_fields = ['action', 'amount', 'reason']
        if not all(field in signal for field in required_fields):
            return False
        
        if signal['action'] not in ['BUY', 'SELL', 'HOLD']:
            return False
        
        if signal['amount'] <= 0:
            return False
        
        return True
    
    def update_performance(self, action_result: Dict[str, Any]):
        """
        Update bot performance metrics. Developers can override this method.
        
        Args:
            action_result: Result of executed action
        """
        if action_result and 'success' in action_result:
            if action_result['success']:
                self.logger.info(f"Action executed successfully: {action_result}")
            else:
                self.logger.warning(f"Action failed: {action_result}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics. Developers can override this method.
        
        Returns:
            Dictionary of performance metrics
        """
        return {
            'total_actions': 0,
            'successful_actions': 0,
            'failed_actions': 0,
            'last_action_time': self.last_action_time,
            'prediction_cycle': self.prediction_cycle,
            'prediction_cycle_configurable': self.prediction_cycle_configurable
        }
    
    def get_supported_prediction_cycles(self) -> List[str]:
        """
        Get list of supported prediction cycles. Developers can override this method.
        
        Returns:
            List of supported prediction cycle strings
        """
        return ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
    
    def get_recommended_prediction_cycle(self) -> str:
        """
        Get recommended prediction cycle. Developers can override this method.
        
        Returns:
            Recommended prediction cycle string
        """
        return "15m"
    
    def validate_prediction_cycle(self, cycle: str) -> bool:
        """
        Validate prediction cycle. Developers can override this method.
        
        Args:
            cycle: Prediction cycle to validate
            
        Returns:
            True if valid, False otherwise
        """
        supported = self.get_supported_prediction_cycles()
        return cycle in supported
    
    def get_adaptive_parameters(self) -> Dict[str, Any]:
        """
        Get adaptive parameters based on current prediction cycle. 
        Developers can override this method to provide cycle-specific parameters.
        
        Returns:
            Dictionary of adaptive parameters
        """
        # Default adaptive parameters
        cycle_params = {
            "1m": {
                "data_timeframe": "1m",
                "data_limit": 100,
                "description": "Scalping - Very frequent signals"
            },
            "5m": {
                "data_timeframe": "1m",
                "data_limit": 300,
                "description": "Day Trading - Frequent signals"
            },
            "15m": {
                "data_timeframe": "5m",
                "data_limit": 200,
                "description": "Swing Trading - Standard signals"
            },
            "30m": {
                "data_timeframe": "5m",
                "data_limit": 300,
                "description": "Swing Trading - Moderate signals"
            },
            "1h": {
                "data_timeframe": "15m",
                "data_limit": 100,
                "description": "Position Trading - Conservative signals"
            },
            "4h": {
                "data_timeframe": "1h",
                "data_limit": 50,
                "description": "Long-term - Very conservative signals"
            },
            "1d": {
                "data_timeframe": "4h",
                "data_limit": 30,
                "description": "Long-term - Extremely conservative signals"
            }
        }
        
        return cycle_params.get(self.prediction_cycle, cycle_params["15m"])
    
    def cleanup(self):
        """
        Cleanup resources. Developers can override this method.
        """
        self.logger.info(f"{self.__class__.__name__} cleanup completed")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup() 