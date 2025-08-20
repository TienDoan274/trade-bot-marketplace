import pandas as pd
import numpy as np
from typing import Dict, Any, List
from bots.bot_sdk.CustomBot import CustomBot
from bots.bot_sdk.Action import SimpleAction, AmountType
import logging

logger = logging.getLogger(__name__)

class BollingerBandsBot(CustomBot):
    """
    Bollinger Bands Trading Bot
    
    Buy near lower band, sell near upper band, with optional RSI confirmation.
    """
    
    def __init__(self, config: Dict[str, Any], api_keys: Dict[str, str]):
        super().__init__(config, api_keys)
        self.name = "Bollinger Bands Bot"
        self.version = "1.0.0"
        self.description = "Bollinger Bands strategy with configurable parameters"
        self.author = "Bot Marketplace"
        
        # Apply configuration
        self.bb_period = config.get("bb_period", 20)
        self.bb_std_dev = config.get("bb_std_dev", 2.0)
        self.use_rsi_confirmation = config.get("use_rsi_confirmation", True)
        self.rsi_period = config.get("rsi_period", 14)
        self.rsi_overbought = config.get("rsi_overbought", 70.0)
        self.rsi_oversold = config.get("rsi_oversold", 30.0)
        self.allocation_percentage = config.get("allocation_percentage", 10.0)

    def initialize(self):
        """Initialize bot-specific parameters."""
        # Bot configuration
        self.symbol = "BTC/USDT"
        
        # Get adaptive parameters based on prediction cycle
        self.adaptive_params = self.get_adaptive_parameters()
        self.data_fetch_timeframe = self.adaptive_params.get("data_timeframe", "5m")
        self.data_fetch_limit = self.adaptive_params.get("data_limit", 100)
        
        # Bollinger Bands parameters
        self.bb_period = 20
        self.bb_std_dev = 2
        self.trade_amount = 100
        
        # Strategy state
        self.last_position = "NONE"
        
        self.logger.info(f"BollingerBandsBot initialized with prediction_cycle: {self.prediction_cycle}")
        self.logger.info(f"Data fetch: {self.data_fetch_limit} candles on {self.data_fetch_timeframe}")
        self.logger.info(f"Strategy: {self.adaptive_params.get('description', 'Bollinger Bands')}")
    
    def get_supported_prediction_cycles(self) -> List[str]:
        """Danh sách chu kỳ được hỗ trợ."""
        return ["5m", "15m", "1h", "4h"]
    
    def get_recommended_prediction_cycle(self) -> str:
        """Chu kỳ được khuyến nghị."""
        return "15m"
    
    def validate_prediction_cycle(self, cycle: str) -> bool:
        """Validation cho chu kỳ mới."""
        supported = self.get_supported_prediction_cycles()
        if cycle not in supported:
            self.logger.warning(f"Chu kỳ {cycle} không được hỗ trợ. Hỗ trợ: {supported}")
            return False
        return True

    def get_configuration_schema(self) -> Dict[str, Any]:
        """Flat config schema used by UI to render inputs"""
        return {
            "bb_period": {"type": "int", "default": 20, "min": 10, "max": 50, "description": "Period for Bollinger Bands"},
            "bb_std_dev": {"type": "float", "default": 2.0, "min": 1.0, "max": 3.0, "description": "Standard deviation multiplier"},
            "use_rsi_confirmation": {"type": "bool", "default": True, "description": "Use RSI for confirmation"},
            "rsi_period": {"type": "int", "default": 14, "min": 5, "max": 30, "description": "RSI period"},
            "rsi_overbought": {"type": "float", "default": 70.0, "min": 50.0, "max": 90.0, "description": "RSI overbought threshold"},
            "rsi_oversold": {"type": "float", "default": 30.0, "min": 10.0, "max": 50.0, "description": "RSI oversold threshold"},
            "allocation_percentage": {"type": "float", "default": 10.0, "min": 1.0, "max": 50.0, "description": "Percentage of balance per trade"},
        }

    def calculate_bollinger_bands(self, prices: List[float], period: int, std_dev: float) -> Dict[str, float]:
        if len(prices) < period:
            return None
        recent = prices[-period:]
        sma = float(np.mean(recent))
        std = float(np.std(recent))
        return {"upper": sma + (std_dev * std), "middle": sma, "lower": sma - (std_dev * std)}

    def calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        if len(prices) < period + 1:
            return None
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100 - (100 / (1 + rs)))

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess market data and add Bollinger Bands indicators."""
        try:
            # Use parent class preprocessing
            processed_data = super().preprocess_data(data)
            
            # Add Bollinger Bands if not already present
            if 'bb_upper' not in processed_data.columns:
                bb_middle = processed_data['close'].rolling(window=20).mean()
                bb_std = processed_data['close'].rolling(window=20).std()
                processed_data['bb_upper'] = bb_middle + (bb_std * 2)
                processed_data['bb_lower'] = bb_middle - (bb_std * 2)
                processed_data['bb_middle'] = bb_middle
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error preprocessing data: {e}")
            return data

    def execute_algorithm(self, market_data, timeframe=None, config=None):
        """
        Main algorithm method called every prediction_cycle.
        
        Args:
            market_data: DataFrame with OHLCV data or current timestamp
            timeframe: Trading timeframe (optional)
            config: Additional configuration (optional)
            
        Returns:
            Action dictionary or None
        """
        try:
            # Handle different input types
            if isinstance(market_data, pd.DataFrame):
                # Backend is passing DataFrame directly
                processed_data = self.preprocess_data(market_data)
            else:
                # Backend is passing timestamp, we need to fetch data
                current_time = market_data
                market_data = self.get_market_data(
                    symbol=self.symbol,
                    timeframe=self.data_fetch_timeframe,
                    limit=self.data_fetch_limit
                )
                
                if market_data.empty:
                    self.logger.warning("No market data available")
                    return None
                
                processed_data = self.preprocess_data(market_data)
            
            # Get current price and Bollinger Bands
            current_price = processed_data['close'].iloc[-1]
            bb_upper = processed_data['bb_upper'].iloc[-1]
            bb_lower = processed_data['bb_lower'].iloc[-1]
            bb_middle = processed_data['bb_middle'].iloc[-1]
            
            self.logger.info(f"Current Price: ${current_price:.2f}, BB Upper: ${bb_upper:.2f}, BB Lower: ${bb_lower:.2f}")
            
            # Trading logic based on Bollinger Bands
            action = None
            
            if current_price <= bb_lower and self.position != "LONG":
                # Price at or below lower band - potential buy signal
                action = {
                    "action": "BUY",
                    "amount": self.trade_amount,
                    "reason": f"Price at lower Bollinger Band (${current_price:.2f} <= ${bb_lower:.2f})",
                    "confidence": 0.8,
                    "price": current_price
                }
                self.position = "LONG"
                self.logger.info(f"BUY signal: Price at lower Bollinger Band")
                
            elif current_price >= bb_upper and self.position != "SHORT":
                # Price at or above upper band - potential sell signal
                action = {
                    "action": "SELL",
                    "amount": self.trade_amount,
                    "reason": f"Price at upper Bollinger Band (${current_price:.2f} >= ${bb_upper:.2f})",
                    "confidence": 0.8,
                    "price": current_price
                }
                self.position = "SHORT"
                self.logger.info(f"SELL signal: Price at upper Bollinger Band")
            
            else:
                # No action - HOLD
                action = {
                    "action": "HOLD",
                    "amount": 0,
                    "reason": f"Price within Bollinger Bands (${bb_lower:.2f} < ${current_price:.2f} < ${bb_upper:.2f})",
                    "confidence": 0.5,
                    "price": current_price
                }
                self.logger.info(f"HOLD: Price within Bollinger Bands")
            
            # Store last action for next iteration
            self.last_action = action
            
            return action
            
        except Exception as e:
            self.logger.error(f"Error in execute_algorithm: {e}")
            return None 