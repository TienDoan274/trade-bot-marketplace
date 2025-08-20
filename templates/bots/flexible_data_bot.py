"""
Flexible Data Fetching Bot
This bot demonstrates advanced data fetching strategies:
1. Multiple timeframes for different analysis purposes
2. Dynamic data amount based on market conditions
3. Custom data preprocessing
4. Adaptive prediction cycles
"""

from bots.bot_sdk.CustomBot import CustomBot
from bots.bot_sdk.Action import SimpleAction
import pandas as pd
import numpy as np

class FlexibleDataBot(CustomBot):
    """
    Advanced bot that demonstrates flexible data fetching strategies.
    
    Key Features:
    - prediction_cycle: "15m" (predicts actions every 15 minutes)
    - Multiple data timeframes: 1m for short-term, 1h for trend analysis
    - Dynamic data amounts based on volatility
    - Custom technical indicators
    """
    
    def initialize(self):
        """Initialize bot-specific parameters."""
        # Prediction cycle (how often to execute algorithm)
        self.prediction_cycle = "15m"
        
        # Data fetching strategies
        self.short_term_timeframe = "1m"      # For immediate price action
        self.trend_timeframe = "1h"           # For trend analysis
        self.short_term_limit = 100           # 100 candles for short-term
        self.trend_limit = 50                 # 50 candles for trend
        
        # Trading parameters
        self.symbol = "BTC/USDT"
        self.trade_amount = 200               # USDT amount
        self.volatility_threshold = 0.02     # 2% volatility threshold
        
        # Strategy state
        self.last_action = None
        self.trend_direction = "NEUTRAL"      # UP, DOWN, NEUTRAL
        
        self.logger.info(f"FlexibleDataBot initialized with prediction_cycle: {self.prediction_cycle}")
        self.logger.info(f"Short-term data: {self.short_term_limit} candles on {self.short_term_timeframe}")
        self.logger.info(f"Trend data: {self.trend_limit} candles on {self.trend_timeframe}")
    
    def execute_algorithm(self, current_time, market_context=None):
        """
        Main algorithm method called every prediction_cycle.
        Fetches data using multiple strategies and timeframes.
        """
        try:
            # Strategy 1: Get short-term data for immediate analysis
            short_term_data = self.get_market_data(
                symbol=self.symbol,
                timeframe=self.short_term_timeframe,
                limit=self.short_term_limit
            )
            
            # Strategy 2: Get trend data for longer-term analysis
            trend_data = self.get_market_data(
                symbol=self.symbol,
                timeframe=self.trend_timeframe,
                limit=self.trend_limit
            )
            
            if short_term_data.empty or trend_data.empty:
                self.logger.warning("Insufficient market data available")
                return None
            
            # Process both datasets
            short_processed = self.preprocess_data(short_term_data)
            trend_processed = self.preprocess_data(trend_data)
            
            # Calculate indicators
            short_indicators = self.calculate_short_term_indicators(short_processed)
            trend_indicators = self.calculate_trend_indicators(trend_processed)
            
            # Get current market conditions
            current_price = short_processed['close'].iloc[-1]
            volatility = self.calculate_volatility(short_processed)
            
            self.logger.info(f"Current price: ${current_price:.2f}, Volatility: {volatility:.2%}")
            self.logger.info(f"Trend direction: {trend_indicators['trend']}")
            
            # Generate trading signal based on multiple timeframes
            action = self.generate_trading_signal(
                short_indicators, 
                trend_indicators, 
                current_price, 
                volatility
            )
            
            return action
            
        except Exception as e:
            self.logger.error(f"Error in execute_algorithm: {e}")
            return None
    
    def calculate_short_term_indicators(self, data):
        """Calculate indicators for short-term analysis."""
        indicators = {}
        
        # RSI
        indicators['rsi'] = data['rsi'].iloc[-1]
        
        # MACD
        indicators['macd'] = data['macd'].iloc[-1]
        indicators['macd_signal'] = data['macd_signal'].iloc[-1]
        indicators['macd_histogram'] = data['macd_histogram'].iloc[-1]
        
        # Bollinger Bands position
        current_price = data['close'].iloc[-1]
        bb_upper = data['bb_upper'].iloc[-1]
        bb_lower = data['bb_lower'].iloc[-1]
        bb_middle = data['bb_middle'].iloc[-1]
        
        if current_price > bb_upper:
            indicators['bb_position'] = "ABOVE_UPPER"
        elif current_price < bb_lower:
            indicators['bb_position'] = "BELOW_LOWER"
        else:
            indicators['bb_position'] = "INSIDE"
        
        # Volume analysis
        indicators['volume_ratio'] = data['volume_ratio'].iloc[-1]
        
        return indicators
    
    def calculate_trend_indicators(self, data):
        """Calculate indicators for trend analysis."""
        indicators = {}
        
        # Moving averages
        sma_20 = data['sma_20'].iloc[-1]
        sma_50 = data['sma_50'].iloc[-1]
        ema_12 = data['ema_12'].iloc[-1]
        ema_26 = data['ema_26'].iloc[-1]
        
        # Trend direction
        if sma_20 > sma_50 and ema_12 > ema_26:
            indicators['trend'] = "UP"
        elif sma_20 < sma_50 and ema_12 < ema_26:
            indicators['trend'] = "DOWN"
        else:
            indicators['trend'] = "NEUTRAL"
        
        # Trend strength
        price_change = (data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0]
        indicators['trend_strength'] = abs(price_change)
        
        # Support and resistance levels
        indicators['support'] = data['low'].rolling(window=20).min().iloc[-1]
        indicators['resistance'] = data['high'].rolling(window=20).max().iloc[-1]
        
        return indicators
    
    def calculate_volatility(self, data, window=20):
        """Calculate price volatility."""
        returns = data['close'].pct_change().dropna()
        return returns.rolling(window=window).std().iloc[-1]
    
    def generate_trading_signal(self, short_indicators, trend_indicators, current_price, volatility):
        """Generate trading signal based on multiple indicators."""
        
        # High volatility strategy
        if volatility > self.volatility_threshold:
            return self.generate_volatility_strategy(short_indicators, current_price)
        
        # Trend following strategy
        else:
            return self.generate_trend_strategy(short_indicators, trend_indicators, current_price)
    
    def generate_volatility_strategy(self, short_indicators, current_price):
        """Strategy for high volatility periods."""
        
        # RSI extremes
        if short_indicators['rsi'] < 25:
            return {
                "action": "BUY",
                "amount": self.trade_amount,
                "reason": f"High volatility + RSI oversold ({short_indicators['rsi']:.1f})",
                "confidence": 0.85,
                "price": current_price
            }
        
        elif short_indicators['rsi'] > 75:
            return {
                "action": "SELL",
                "amount": self.trade_amount,
                "reason": f"High volatility + RSI overbought ({short_indicators['rsi']:.1f})",
                "confidence": 0.85,
                "price": current_price
            }
        
        # Bollinger Bands extremes
        elif short_indicators['bb_position'] == "BELOW_LOWER":
            return {
                "action": "BUY",
                "amount": self.trade_amount,
                "reason": "High volatility + Price below BB lower band",
                "confidence": 0.80,
                "price": current_price
            }
        
        elif short_indicators['bb_position'] == "ABOVE_UPPER":
            return {
                "action": "SELL",
                "amount": self.trade_amount,
                "reason": "High volatility + Price above BB upper band",
                "confidence": 0.80,
                "price": current_price
            }
        
        return {
            "action": "HOLD",
            "amount": 0,
            "reason": "High volatility but no clear signal",
            "confidence": 0.60,
            "price": current_price
        }
    
    def generate_trend_strategy(self, short_indicators, trend_indicators, current_price):
        """Strategy for low volatility, trending periods."""
        
        # Trend following with confirmation
        if trend_indicators['trend'] == "UP":
            # Look for pullback to buy
            if (short_indicators['rsi'] < 40 and 
                short_indicators['bb_position'] == "INSIDE" and
                short_indicators['macd_histogram'] > 0):
                
                return {
                    "action": "BUY",
                    "amount": self.trade_amount,
                    "reason": f"Uptrend pullback: RSI({short_indicators['rsi']:.1f}) + MACD positive",
                    "confidence": 0.75,
                    "price": current_price
                }
        
        elif trend_indicators['trend'] == "DOWN":
            # Look for bounce to sell
            if (short_indicators['rsi'] > 60 and 
                short_indicators['bb_position'] == "INSIDE" and
                short_indicators['macd_histogram'] < 0):
                
                return {
                    "action": "SELL",
                    "amount": self.trade_amount,
                    "reason": f"Downtrend bounce: RSI({short_indicators['rsi']:.1f}) + MACD negative",
                    "confidence": 0.75,
                    "price": current_price
                }
        
        return {
            "action": "HOLD",
            "amount": 0,
            "reason": f"Trend: {trend_indicators['trend']}, waiting for confirmation",
            "confidence": 0.65,
            "price": current_price
        }
    
    def get_configuration_schema(self):
        """Define configuration parameters."""
        return {
            "prediction_cycle": {
                "type": "select",
                "label": "Prediction Cycle",
                "options": [
                    {"value": "5m", "label": "5 minutes"},
                    {"value": "15m", "label": "15 minutes"},
                    {"value": "30m", "label": "30 minutes"},
                    {"value": "1h", "label": "1 hour"}
                ],
                "default": "15m",
                "description": "How often the bot should predict actions"
            },
            "short_term_timeframe": {
                "type": "select",
                "label": "Short-term Timeframe",
                "options": [
                    {"value": "1m", "label": "1 minute"},
                    {"value": "5m", "label": "5 minutes"}
                ],
                "default": "1m",
                "description": "Timeframe for immediate price action analysis"
            },
            "trend_timeframe": {
                "type": "select",
                "label": "Trend Timeframe",
                "options": [
                    {"value": "1h", "label": "1 hour"},
                    {"value": "4h", "label": "4 hours"},
                    {"value": "1d", "label": "1 day"}
                ],
                "default": "1h",
                "description": "Timeframe for trend analysis"
            },
            "short_term_limit": {
                "type": "number",
                "label": "Short-term Data Limit",
                "min": 50,
                "max": 500,
                "default": 100,
                "description": "Number of short-term candles to fetch"
            },
            "trend_limit": {
                "type": "number",
                "label": "Trend Data Limit",
                "min": 20,
                "max": 200,
                "default": 50,
                "description": "Number of trend candles to fetch"
            },
            "volatility_threshold": {
                "type": "number",
                "label": "Volatility Threshold",
                "min": 0.01,
                "max": 0.10,
                "step": 0.01,
                "default": 0.02,
                "description": "Threshold for high volatility strategy (as decimal)"
            },
            "trade_amount": {
                "type": "number",
                "label": "Trade Amount (USDT)",
                "min": 50,
                "max": 10000,
                "default": 200,
                "description": "Amount to trade in USDT"
            }
        }
    
    def get_action_sample(self):
        """Return sample action for documentation."""
        return {
            "action": "BUY",
            "amount": 200,
            "reason": "Uptrend pullback: RSI(35.2) + MACD positive",
            "confidence": 0.75,
            "price": 45000.0
        } 