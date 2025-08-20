"""
Template Bot using new prediction_cycle logic
This bot demonstrates how to:
1. Set a prediction cycle (how often to predict actions)
2. Fetch data independently using exchange client
3. Control data fetching frequency and timeframe
4. Execute algorithm only when prediction cycle is ready
"""

from bots.bot_sdk.CustomBot import CustomBot
from bots.bot_sdk.Action import SimpleAction
import pandas as pd
import numpy as np

class PredictionCycleBot(CustomBot):
    """
    Example bot that demonstrates the new prediction_cycle logic.
    
    Key Features:
    - prediction_cycle: "5m" (predicts actions every 5 minutes)
    - data_fetch_timeframe: "1m" (fetches 1-minute data for analysis)
    - data_fetch_limit: 200 (fetches 200 candles for analysis)
    - Developers control when and how much data to fetch
    """
    
    def initialize(self):
        """Initialize bot-specific parameters."""
        # Bot configuration
        self.symbol = "BTC/USDT"
        self.data_fetch_timeframe = "1m"      # Fetch 1-minute data
        self.data_fetch_limit = 200           # Fetch 200 candles
        self.rsi_period = 14
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.trade_amount = 100               # USDT amount to trade
        
        # Strategy state
        self.last_rsi = None
        self.position = "NONE"  # NONE, LONG, SHORT
        
        self.logger.info(f"PredictionCycleBot initialized with prediction_cycle: {self.prediction_cycle}")
        self.logger.info(f"Data fetch: {self.data_fetch_limit} candles on {self.data_fetch_timeframe}")
    
    def execute_algorithm(self, current_time, market_context=None):
        """
        Main algorithm method called every prediction_cycle.
        
        Args:
            current_time: Current timestamp
            market_context: Market context (symbol, current_price, balance, etc.)
            
        Returns:
            Action dictionary or None
        """
        try:
            # Get current market data using exchange client
            # This is independent of prediction_cycle - we fetch when we need it
            market_data = self.get_market_data(
                symbol=self.symbol,
                timeframe=self.data_fetch_timeframe,
                limit=self.data_fetch_limit
            )
            
            if market_data.empty:
                self.logger.warning("No market data available")
                return None
            
            # Preprocess data
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
            
            # Store last RSI for next iteration
            self.last_rsi = current_rsi
            
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
                "default": "1m",
                "description": "Timeframe for fetching market data (independent of prediction cycle)"
            },
            "data_fetch_limit": {
                "type": "number",
                "label": "Data Fetch Limit",
                "min": 50,
                "max": 1000,
                "default": 200,
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