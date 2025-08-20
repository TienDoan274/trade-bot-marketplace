"""
RSI Trading Bot using new prediction_cycle logic
This bot demonstrates RSI-based trading with flexible data fetching
"""

from bots.bot_sdk.CustomBot import CustomBot
import pandas as pd
import numpy as np
from typing import List

class RSIBot(CustomBot):
    """
    RSI-based trading bot using new prediction_cycle logic.
    
    Key Features:
    - prediction_cycle: Configurable (how often to predict actions)
    - data_fetch_timeframe: Independent of prediction cycle
    - RSI oversold/overbought strategy
    - Configurable parameters
    """
    
    def initialize(self):
        """Initialize bot-specific parameters."""
        # Bot configuration
        self.symbol = "BTC/USDT"
        
        # Get adaptive parameters based on prediction cycle
        self.adaptive_params = self.get_adaptive_parameters()
        self.data_fetch_timeframe = self.adaptive_params.get("data_timeframe", "5m")
        self.data_fetch_limit = self.adaptive_params.get("data_limit", 100)
        
        # RSI parameters
        self.rsi_period = 14
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.trade_amount = 100               # USDT amount to trade
        
        # Strategy state
        self.last_rsi = None
        self.position = "NONE"  # NONE, LONG, SHORT
        
        self.logger.info(f"RSIBot initialized with prediction_cycle: {self.prediction_cycle}")
        self.logger.info(f"Data fetch: {self.data_fetch_limit} candles on {self.data_fetch_timeframe}")
        self.logger.info(f"Strategy: {self.adaptive_params.get('description', 'Standard RSI')}")
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess market data and add RSI indicator."""
        try:
            # Use parent class preprocessing
            processed_data = super().preprocess_data(data)
            
            # Add RSI if not already present
            if 'rsi' not in processed_data.columns:
                delta = processed_data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
                rs = gain / loss
                processed_data['rsi'] = 100 - (100 / (1 + rs))
            
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
            
            # Get current RSI
            current_rsi = processed_data['rsi'].iloc[-1]
            current_price = processed_data['close'].iloc[-1]
            
            self.logger.info(f"Current RSI: {current_rsi:.2f}, Price: ${current_price:.2f}")
            
            # Trading logic based on RSI
            action = None
            
            if current_rsi < self.rsi_oversold and self.position != "LONG":
                # RSI oversold - potential buy signal
                action = {
                    "action": "BUY",
                    "amount": self.trade_amount,
                    "reason": f"RSI oversold ({current_rsi:.2f} < {self.rsi_oversold})",
                    "confidence": 0.8,
                    "price": current_price
                }
                self.position = "LONG"
                self.logger.info(f"BUY signal: RSI oversold at {current_rsi:.2f}")
                
            elif current_rsi > self.rsi_overbought and self.position != "SHORT":
                # RSI overbought - potential sell signal
                action = {
                    "action": "SELL",
                    "amount": self.trade_amount,
                    "reason": f"RSI overbought ({current_rsi:.2f} > {self.rsi_overbought})",
                    "confidence": 0.8,
                    "price": current_price
                }
                self.position = "SHORT"
                self.logger.info(f"SELL signal: RSI overbought at {current_rsi:.2f}")
            
            else:
                # No action - HOLD
                action = {
                    "action": "HOLD",
                    "amount": 0,
                    "reason": f"RSI neutral ({current_rsi:.2f})",
                    "confidence": 0.5,
                    "price": current_price
                }
                self.logger.info(f"HOLD: RSI neutral at {current_rsi:.2f}")
            
            # Store last action for next iteration
            self.last_action = action
            
            return action
            
        except Exception as e:
            self.logger.error(f"Error in execute_algorithm: {e}")
            return None
    
    def get_configuration_schema(self):
        """
        Define configuration parameters that users can customize.
        """
        return {
            "prediction_cycle": {
                "type": "select",
                "label": "Prediction Cycle",
                "options": [
                    {"value": "1m", "label": "1 minute"},
                    {"value": "5m", "label": "5 minutes"},
                    {"value": "15m", "label": "15 minutes"},
                    {"value": "1h", "label": "1 hour"},
                    {"value": "4h", "label": "4 hours"},
                    {"value": "1d", "label": "1 day"}
                ],
                "default": "5m",
                "description": "How often the bot should predict actions"
            },
            "data_fetch_timeframe": {
                "type": "select",
                "label": "Data Fetch Timeframe",
                "options": [
                    {"value": "1m", "label": "1 minute"},
                    {"value": "5m", "label": "5 minutes"},
                    {"value": "15m", "label": "15 minutes"},
                    {"value": "1h", "label": "1 hour"}
                ],
                "default": "5m",
                "description": "Timeframe for fetching market data (independent of prediction cycle)"
            },
            "data_fetch_limit": {
                "type": "number",
                "label": "Data Fetch Limit",
                "min": 50,
                "max": 1000,
                "default": 100,
                "description": "Number of candles to fetch for analysis"
            },
            "rsi_period": {
                "type": "number",
                "label": "RSI Period",
                "min": 10,
                "max": 30,
                "default": 14,
                "description": "Period for RSI calculation"
            },
            "rsi_oversold": {
                "type": "number",
                "label": "RSI Oversold Threshold",
                "min": 20,
                "max": 40,
                "default": 30,
                "description": "RSI level considered oversold"
            },
            "rsi_overbought": {
                "type": "number",
                "label": "RSI Overbought Threshold",
                "min": 60,
                "max": 80,
                "default": 70,
                "description": "RSI level considered overbought"
            },
            "trade_amount": {
                "type": "number",
                "label": "Trade Amount (USDT)",
                "min": 10,
                "max": 10000,
                "default": 100,
                "description": "Amount to trade in USDT"
            }
        }
    
    def get_action_sample(self):
        """Return sample action for documentation."""
        return {
            "action": "BUY",
            "amount": 100,
            "reason": "RSI oversold (25.5 < 30)",
            "confidence": 0.8,
            "price": 45000.0
        } 

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