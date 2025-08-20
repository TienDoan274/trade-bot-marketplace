"""
Fixed Prediction Cycle Bot
This bot demonstrates a fixed prediction cycle approach where:
- prediction_cycle is fixed and cannot be changed by users
- Algorithm is optimized for the specific cycle
- No configuration options for prediction_cycle
"""

from bots.bot_sdk.CustomBot import CustomBot
from bots.bot_sdk.Action import SimpleAction
import pandas as pd
import numpy as np

class FixedCycleBot(CustomBot):
    """
    Bot với prediction_cycle cố định.
    
    Key Features:
    - prediction_cycle: Cố định "15m" - không thể thay đổi
    - Thuật toán được tối ưu cho chu kỳ cụ thể
    - Không có tùy chọn cấu hình prediction_cycle
    """
    
    def initialize(self):
        """Initialize bot-specific parameters."""
        # Bot configuration - FIXED
        self.prediction_cycle = "15m"  # Cố định 15 phút
        self.prediction_cycle_configurable = False  # Không cho phép thay đổi
        
        self.symbol = "BTC/USDT"
        self.data_fetch_timeframe = "5m"      # Tối ưu cho 15m cycle
        self.data_fetch_limit = 200           # 200 nến 5m = 16.7 giờ dữ liệu
        
        # Strategy parameters
        self.rsi_period = 14
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.trade_amount = 150
        
        # Strategy state
        self.last_action = None
        
        self.logger.info(f"FixedCycleBot: Chu kỳ cố định: {self.prediction_cycle}")
        self.logger.info(f"FixedCycleBot: Không thể thay đổi chu kỳ")
    
    def execute_algorithm(self, current_time, market_context=None):
        """
        Thuật toán được tối ưu cho chu kỳ 15 phút.
        """
        try:
            # Lấy dữ liệu tối ưu cho 15m cycle
            data = self.get_market_data(
                symbol=self.symbol,
                timeframe=self.data_fetch_timeframe,
                limit=self.data_fetch_limit
            )
            
            if data.empty:
                return None
            
            # Preprocess data
            processed_data = self.preprocess_data(data)
            
            # Tính RSI
            current_rsi = processed_data['rsi'].iloc[-1]
            current_price = processed_data['close'].iloc[-1]
            
            # Logic tối ưu cho 15m cycle
            if current_rsi < self.rsi_oversold:
                return {
                    "action": "BUY",
                    "amount": self.trade_amount,
                    "reason": f"RSI oversold ({current_rsi:.1f} < {self.rsi_oversold}) - 15m cycle optimized",
                    "confidence": 0.85,
                    "price": current_price
                }
                
            elif current_rsi > self.rsi_overbought:
                return {
                    "action": "SELL",
                    "amount": self.trade_amount,
                    "reason": f"RSI overbought ({current_rsi:.1f} > {self.rsi_overbought}) - 15m cycle optimized",
                    "confidence": 0.85,
                    "price": current_price
                }
            
            return {
                "action": "HOLD",
                "amount": 0,
                "reason": f"RSI neutral ({current_rsi:.1f}) - 15m cycle optimized",
                "confidence": 0.6,
                "price": current_price
            }
            
        except Exception as e:
            self.logger.error(f"Error in execute_algorithm: {e}")
            return None
    
    def get_configuration_schema(self):
        """
        Schema KHÔNG có prediction_cycle - cố định.
        """
        return {
            # KHÔNG có prediction_cycle trong schema
            "trade_amount": {
                "type": "number",
                "label": "Số tiền giao dịch (USDT)",
                "min": 50,
                "max": 1000,
                "default": 150,
                "description": "Số tiền USDT cho mỗi lệnh giao dịch"
            },
            "rsi_oversold": {
                "type": "number",
                "label": "RSI Oversold",
                "min": 20,
                "max": 40,
                "default": 30,
                "description": "Ngưỡng RSI oversold"
            },
            "rsi_overbought": {
                "type": "number",
                "label": "RSI Overbought",
                "min": 60,
                "max": 80,
                "default": 70,
                "description": "Ngưỡng RSI overbought"
            }
        }
    
    def get_action_sample(self):
        """Mẫu action."""
        return {
            "action": "BUY",
            "amount": 150,
            "reason": "RSI oversold (28.5 < 30) - 15m cycle optimized",
            "confidence": 0.85,
            "price": 45000.0
        } 