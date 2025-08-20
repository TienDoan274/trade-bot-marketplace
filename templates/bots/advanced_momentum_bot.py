"""
Advanced Momentum Bot - Version 1.0.0
Advanced strategy using multiple indicators with various order types.
Demonstrates AdvancedAction usage with limit orders, stop losses, and take profits.
"""

from bots.bot_sdk.CustomBot import CustomBot
from bots.bot_sdk.Action import AdvancedAction, AmountType, OrderType, ActionType
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

class AdvancedMomentumBot(CustomBot):
    """
    Advanced Momentum Strategy Bot
    - Uses RSI, MACD, and Bollinger Bands
    - Implements limit orders, stop losses, and take profits
    - Advanced position sizing and risk management
    """
    
    def __init__(self, config: Dict[str, Any], api_keys: Dict[str, str]):
        super().__init__(config, api_keys)
        
        # Bot metadata
        self.bot_name = "Advanced Momentum Bot"
        self.description = "Advanced momentum strategy with multiple indicators and order types"
        self.version = "1.0.0"
        self.bot_type = "ADVANCED"
        
        # Strategy parameters
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_oversold = config.get('rsi_oversold', 30)
        self.rsi_overbought = config.get('rsi_overbought', 70)
        self.macd_fast = config.get('macd_fast', 12)
        self.macd_slow = config.get('macd_slow', 26)
        self.macd_signal = config.get('macd_signal', 9)
        self.bb_period = config.get('bb_period', 20)
        self.bb_std = config.get('bb_std', 2)
        self.allocation_percentage = config.get('allocation_percentage', 15.0)
        self.use_limit_orders = config.get('use_limit_orders', True)
        self.limit_order_offset = config.get('limit_order_offset', 0.5)  # 0.5% offset
        self.enable_stop_loss = config.get('enable_stop_loss', True)
        self.stop_loss_percentage = config.get('stop_loss_percentage', 2.0)
        self.enable_take_profit = config.get('enable_take_profit', True)
        self.take_profit_percentage = config.get('take_profit_percentage', 5.0)
        
        logger.info(f"AdvancedMomentumBot v{self.version} initialized")
    
    def initialize(self):
        """Initialize bot-specific parameters."""
        # Bot configuration
        self.symbol = "BTC/USDT"
        
        # Get adaptive parameters based on prediction cycle
        self.adaptive_params = self.get_adaptive_parameters()
        self.data_fetch_timeframe = self.adaptive_params.get("data_timeframe", "5m")
        self.data_fetch_limit = self.adaptive_params.get("data_limit", 100)
        
        # Strategy parameters
        self.rsi_period = 14
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.trade_amount = 100
        
        # Strategy state
        self.last_signal = None
        
        self.logger.info(f"AdvancedMomentumBot initialized with prediction_cycle: {self.prediction_cycle}")
        self.logger.info(f"Data fetch: {self.data_fetch_limit} candles on {self.data_fetch_timeframe}")
        self.logger.info(f"Strategy: {self.adaptive_params.get('description', 'Advanced Momentum')}")
    
    def get_supported_prediction_cycles(self) -> List[str]:
        """Danh sách chu kỳ được hỗ trợ."""
        return ["15m", "1h", "4h"]
    
    def get_recommended_prediction_cycle(self) -> str:
        """Chu kỳ được khuyến nghị."""
        return "1h"
    
    def validate_prediction_cycle(self, cycle: str) -> bool:
        """Validation cho chu kỳ mới."""
        supported = self.get_supported_prediction_cycles()
        if cycle not in supported:
            self.logger.warning(f"Chu kỳ {cycle} không được hỗ trợ. Hỗ trợ: {supported}")
            return False
        return True
    
    def calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """Calculate MACD indicator"""
        ema_fast = data['close'].ewm(span=fast).mean()
        ema_slow = data['close'].ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def calculate_bollinger_bands(self, data: pd.DataFrame, period: int = 20, std: int = 2) -> tuple:
        """Calculate Bollinger Bands"""
        sma = data['close'].rolling(window=period).mean()
        std_dev = data['close'].rolling(window=period).std()
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        return upper_band, sma, lower_band
    
    def analyze_market_conditions(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market conditions using multiple indicators"""
        # Calculate indicators
        rsi = self.calculate_rsi(data, self.rsi_period)
        macd_line, signal_line, histogram = self.calculate_macd(data, self.macd_fast, self.macd_slow, self.macd_signal)
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(data, self.bb_period, self.bb_std)
        
        current_price = data['close'].iloc[-1]
        current_rsi = rsi.iloc[-1]
        current_macd = macd_line.iloc[-1]
        current_signal = signal_line.iloc[-1]
        current_bb_upper = bb_upper.iloc[-1]
        current_bb_lower = bb_lower.iloc[-1]
        
        # Analyze conditions
        rsi_oversold = current_rsi < self.rsi_oversold
        rsi_overbought = current_rsi > self.rsi_overbought
        macd_bullish = current_macd > current_signal and histogram.iloc[-1] > 0
        macd_bearish = current_macd < current_signal and histogram.iloc[-1] < 0
        price_near_bb_lower = current_price <= current_bb_lower * 1.01
        price_near_bb_upper = current_price >= current_bb_upper * 0.99
        
        return {
            'current_price': current_price,
            'rsi': current_rsi,
            'macd': current_macd,
            'signal': current_signal,
            'bb_upper': current_bb_upper,
            'bb_lower': current_bb_lower,
            'rsi_oversold': rsi_oversold,
            'rsi_overbought': rsi_overbought,
            'macd_bullish': macd_bullish,
            'macd_bearish': macd_bearish,
            'price_near_bb_lower': price_near_bb_lower,
            'price_near_bb_upper': price_near_bb_upper
        }
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess market data and add momentum indicators."""
        try:
            # Use parent class preprocessing
            processed_data = super().preprocess_data(data)
            
            # Add RSI if not already present
            if 'rsi' not in processed_data.columns:
                delta = processed_data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                processed_data['rsi'] = 100 - (100 / (1 + rs))
            
            # Add MACD if not already present
            if 'macd' not in processed_data.columns:
                ema_12 = processed_data['close'].ewm(span=12).mean()
                ema_26 = processed_data['close'].ewm(span=26).mean()
                processed_data['macd'] = ema_12 - ema_26
                processed_data['macd_signal'] = processed_data['macd'].ewm(span=9).mean()
            
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
            
            # Get current indicators
            current_price = processed_data['close'].iloc[-1]
            current_rsi = processed_data['rsi'].iloc[-1]
            current_macd = processed_data['macd'].iloc[-1]
            current_macd_signal = processed_data['macd_signal'].iloc[-1]
            current_volume = processed_data['volume'].iloc[-1]
            avg_volume = processed_data['volume'].rolling(20).mean().iloc[-1]
            
            self.logger.info(f"Price: ${current_price:.2f}, RSI: {current_rsi:.2f}, MACD: {current_macd:.4f}")
            
            # Advanced momentum logic
            action = None
            signal_strength = 0
            
            # RSI conditions
            if current_rsi < 30:
                signal_strength += 2  # Strong oversold
            elif current_rsi < 40:
                signal_strength += 1  # Mild oversold
            elif current_rsi > 70:
                signal_strength -= 2  # Strong overbought
            elif current_rsi > 60:
                signal_strength -= 1  # Mild overbought
            
            # MACD conditions
            if current_macd > current_macd_signal and current_macd > 0:
                signal_strength += 1  # Positive momentum
            elif current_macd < current_macd_signal and current_macd < 0:
                signal_strength -= 1  # Negative momentum
            
            # Volume confirmation
            volume_boost = 1 if current_volume > avg_volume * 1.5 else 0
            
            # Determine action based on signal strength
            if signal_strength >= 2 and self.position != "LONG":
                action = {
                    "action": "BUY",
                    "amount": self.trade_amount,
                    "reason": f"Strong momentum signal (strength: {signal_strength}, RSI: {current_rsi:.1f}, MACD: {current_macd:.4f})",
                    "confidence": min(0.9, 0.5 + signal_strength * 0.1),
                    "price": current_price
                }
                self.position = "LONG"
                self.logger.info(f"BUY signal: Strong momentum (strength: {signal_strength})")
                
            elif signal_strength <= -2 and self.position != "SHORT":
                action = {
                    "action": "SELL",
                    "amount": self.trade_amount,
                    "reason": f"Strong negative momentum (strength: {signal_strength}, RSI: {current_rsi:.1f}, MACD: {current_macd:.4f})",
                    "confidence": min(0.9, 0.5 + abs(signal_strength) * 0.1),
                    "price": current_price
                }
                self.position = "SHORT"
                self.logger.info(f"SELL signal: Strong negative momentum (strength: {signal_strength})")
            
            else:
                action = {
                    "action": "HOLD",
                    "amount": 0,
                    "reason": f"Neutral momentum (strength: {signal_strength}, RSI: {current_rsi:.1f}, MACD: {current_macd:.4f})",
                    "confidence": 0.5,
                    "price": current_price
                }
                self.logger.info(f"HOLD: Neutral momentum (strength: {signal_strength})")
            
            # Store last action for next iteration
            self.last_action = action
            
            return action
            
        except Exception as e:
            self.logger.error(f"Error in execute_algorithm: {e}")
            return None
    
    def get_configuration_schema(self) -> Dict[str, Any]:
        """Get configuration schema for this bot"""
        return {
            "type": "object",
            "properties": {
                "rsi_period": {
                    "type": "number",
                    "minimum": 5,
                    "maximum": 50,
                    "default": 14,
                    "description": "RSI calculation period"
                },
                "rsi_oversold": {
                    "type": "number",
                    "minimum": 10,
                    "maximum": 40,
                    "default": 30,
                    "description": "RSI oversold threshold"
                },
                "rsi_overbought": {
                    "type": "number",
                    "minimum": 60,
                    "maximum": 90,
                    "default": 70,
                    "description": "RSI overbought threshold"
                },
                "macd_fast": {
                    "type": "number",
                    "minimum": 5,
                    "maximum": 20,
                    "default": 12,
                    "description": "MACD fast period"
                },
                "macd_slow": {
                    "type": "number",
                    "minimum": 20,
                    "maximum": 50,
                    "default": 26,
                    "description": "MACD slow period"
                },
                "macd_signal": {
                    "type": "number",
                    "minimum": 5,
                    "maximum": 20,
                    "default": 9,
                    "description": "MACD signal period"
                },
                "bb_period": {
                    "type": "number",
                    "minimum": 10,
                    "maximum": 50,
                    "default": 20,
                    "description": "Bollinger Bands period"
                },
                "bb_std": {
                    "type": "number",
                    "minimum": 1,
                    "maximum": 5,
                    "default": 2,
                    "description": "Bollinger Bands standard deviation"
                },
                "allocation_percentage": {
                    "type": "number",
                    "minimum": 1.0,
                    "maximum": 50.0,
                    "default": 15.0,
                    "description": "Percentage of balance to trade"
                },
                "use_limit_orders": {
                    "type": "boolean",
                    "default": True,
                    "description": "Use limit orders instead of market orders"
                },
                "limit_order_offset": {
                    "type": "number",
                    "minimum": 0.1,
                    "maximum": 5.0,
                    "default": 0.5,
                    "description": "Limit order price offset percentage"
                },
                "enable_stop_loss": {
                    "type": "boolean",
                    "default": True,
                    "description": "Enable stop loss orders"
                },
                "stop_loss_percentage": {
                    "type": "number",
                    "minimum": 0.5,
                    "maximum": 10.0,
                    "default": 2.0,
                    "description": "Stop loss percentage"
                },
                "enable_take_profit": {
                    "type": "boolean",
                    "default": True,
                    "description": "Enable take profit orders"
                },
                "take_profit_percentage": {
                    "type": "number",
                    "minimum": 1.0,
                    "maximum": 20.0,
                    "default": 5.0,
                    "description": "Take profit percentage"
                }
            },
            "required": ["rsi_period", "rsi_oversold", "rsi_overbought"],
            "additionalProperties": True
        }
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get detailed strategy information"""
        return {
            "name": self.bot_name,
            "version": self.version,
            "description": self.description,
            "strategy_type": "Advanced Technical Analysis",
            "indicators_used": ["RSI", "MACD", "Bollinger Bands"],
            "suitable_for": ["Experienced traders", "Medium-term trading", "Trending markets"],
            "risk_level": "Medium to High",
            "recommended_timeframes": ["1h", "4h", "1d"],
            "features": [
                "Multiple indicator confirmation",
                "Limit orders with price optimization",
                "Stop loss and take profit automation",
                "Advanced risk management",
                "Configurable order types"
            ],
            "warnings": [
                "Complex strategy requiring understanding of technical analysis",
                "Higher risk due to advanced order types",
                "Requires proper backtesting before live trading",
                "May generate more orders than simple strategies"
            ]
        }
    
    def add_custom_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the data"""
        # RSI
        data['rsi'] = self.calculate_rsi(data, self.rsi_period)
        
        # MACD
        macd_line, signal_line, histogram = self.calculate_macd(data, self.macd_fast, self.macd_slow, self.macd_signal)
        data['macd'] = macd_line
        data['macd_signal'] = signal_line
        data['macd_histogram'] = histogram
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(data, self.bb_period, self.bb_std)
        data['bb_upper'] = bb_upper
        data['bb_middle'] = bb_middle
        data['bb_lower'] = bb_lower
        
        return data 