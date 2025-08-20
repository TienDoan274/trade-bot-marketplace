"""
Adaptive Prediction Cycle Bot
This bot demonstrates adaptive prediction cycle approach where:
- Users can choose from multiple prediction cycles
- Algorithm adapts to different cycles
- Developer provides recommendations and validation
"""

from bots.bot_sdk.CustomBot import CustomBot
from bots.bot_sdk.Action import SimpleAction
import pandas as pd
import numpy as np
from typing import List

class AdaptiveCycleBot(CustomBot):
    """
    Bot với prediction_cycle thích ứng.
    
    Key Features:
    - prediction_cycle: Có thể thay đổi (5m, 15m, 1h, 4h)
    - Thuật toán thích ứng với chu kỳ khác nhau
    - Developer cung cấp khuyến nghị và validation
    """
    
    def initialize(self):
        """Initialize bot-specific parameters."""
        self.symbol = "BTC/USDT"
        self.trade_amount = 200
        
        # Tham số thích ứng theo prediction_cycle
        self.adaptive_params = self.get_adaptive_parameters()
        
        self.logger.info(f"AdaptiveCycleBot: Chu kỳ hiện tại: {self.prediction_cycle}")
    
    def get_adaptive_parameters(self):
        """Lấy tham số thích ứng theo prediction_cycle."""
        cycle_params = {
            "5m": {
                "data_timeframe": "1m",
                "data_limit": 300,
                "rsi_oversold": 25,
                "rsi_overbought": 75,
                "description": "Day Trading"
            },
            "15m": {
                "data_timeframe": "5m", 
                "data_limit": 200,
                "rsi_oversold": 30,
                "rsi_overbought": 70,
                "description": "Swing Trading"
            },
            "1h": {
                "data_timeframe": "15m",
                "data_limit": 100,
                "rsi_oversold": 35,
                "rsi_overbought": 65,
                "description": "Position Trading"
            },
            "4h": {
                "data_timeframe": "1h",
                "data_limit": 50,
                "rsi_oversold": 40,
                "rsi_overbought": 60,
                "description": "Long-term"
            }
        }
        
        return cycle_params.get(self.prediction_cycle, cycle_params["15m"])
    
    def execute_algorithm(self, current_time, market_context=None):
        """Thuật toán thích ứng với prediction_cycle."""
        try:
            params = self.adaptive_params
            
            data = self.get_market_data(
                symbol=self.symbol,
                timeframe=params["data_timeframe"],
                limit=params["data_limit"]
            )
            
            if data.empty:
                return None
            
            processed_data = self.preprocess_data(data)
            current_rsi = processed_data['rsi'].iloc[-1]
            current_price = processed_data['close'].iloc[-1]
            
            if current_rsi < params["rsi_oversold"]:
                return {
                    "action": "BUY",
                    "amount": self.trade_amount,
                    "reason": f"RSI oversold ({current_rsi:.1f} < {params['rsi_oversold']}) - {params['description']}",
                    "confidence": 0.85,
                    "price": current_price
                }
                
            elif current_rsi > params["rsi_overbought"]:
                return {
                    "action": "SELL",
                    "amount": self.trade_amount,
                    "reason": f"RSI overbought ({current_rsi:.1f} > {params['rsi_overbought']}) - {params['description']}",
                    "confidence": 0.85,
                    "price": current_price
                }
            
            return {
                "action": "HOLD",
                "amount": 0,
                "reason": f"RSI neutral ({current_rsi:.1f}) - {params['description']}",
                "confidence": 0.6,
                "price": current_price
            }
            
        except Exception as e:
            self.logger.error(f"Error in execute_algorithm: {e}")
            return None
    
    def get_supported_prediction_cycles(self) -> List[str]:
        """Danh sách chu kỳ được hỗ trợ."""
        return ["5m", "15m", "1h", "4h"]
    
    def get_recommended_prediction_cycle(self) -> str:
        """Chu kỳ được khuyến nghị."""
        return "15m"
    
    def validate_prediction_cycle(self, new_cycle: str) -> bool:
        """Validation cho chu kỳ mới."""
        supported = self.get_supported_prediction_cycles()
        return new_cycle in supported
    
    def get_configuration_schema(self):
        """Schema với prediction_cycle có thể thay đổi."""
        return {
            "prediction_cycle": {
                "type": "select",
                "label": "Chu kỳ dự đoán",
                "options": [
                    {"value": "5m", "label": "5 phút (Day Trading)"},
                    {"value": "15m", "label": "15 phút (Swing Trading) - Khuyến nghị"},
                    {"value": "1h", "label": "1 giờ (Position Trading)"},
                    {"value": "4h", "label": "4 giờ (Long-term)"}
                ],
                "default": "15m",
                "description": "Chu kỳ bot sẽ dự đoán actions. 15 phút được khuyến nghị."
            },
            "trade_amount": {
                "type": "number",
                "label": "Số tiền giao dịch (USDT)",
                "min": 50,
                "max": 5000,
                "default": 200,
                "description": "Số tiền USDT cho mỗi lệnh giao dịch"
            }
        }
    
    def get_action_sample(self):
        """Mẫu action."""
        return {
            "action": "BUY",
            "amount": 200,
            "reason": "RSI oversold (28.5 < 30) - Swing Trading",
            "confidence": 0.85,
            "price": 45000.0
        } 