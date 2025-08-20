import pandas as pd
import numpy as np
from typing import Dict, Any, List
from bots.bot_sdk.CustomBot import CustomBot
from bots.bot_sdk.Action import SimpleAction, AmountType
import logging

logger = logging.getLogger(__name__)

class GoldenCrossBot(CustomBot):
    """
    Golden Cross Trading Bot
    
    This bot uses SMA (Simple Moving Average) crossover strategy.
    - Buy when short SMA crosses above long SMA (Golden Cross)
    - Sell when short SMA crosses below long SMA (Death Cross)
    """
    
    def __init__(self, config: Dict[str, Any], api_keys: Dict[str, str]):
        super().__init__(config, api_keys)
        self.name = "Golden Cross Bot"
        self.version = "1.0.0"
        self.description = "SMA crossover strategy with configurable parameters"
        self.author = "Bot Marketplace"
        
        # Apply configuration
        self.short_period = config.get("short_period", 10)
        self.long_period = config.get("long_period", 20)
        self.use_rsi_filter = config.get("use_rsi_filter", True)
        self.rsi_period = config.get("rsi_period", 14)
        self.rsi_overbought = config.get("rsi_overbought", 70.0)
        self.rsi_oversold = config.get("rsi_oversold", 30.0)
        self.stop_loss_pct = config.get("stop_loss_pct", 3.0)
        self.take_profit_pct = config.get("take_profit_pct", 6.0)
        self.allocation_percentage = config.get("allocation_percentage", 10.0)

    def initialize(self):
        """Initialize bot-specific parameters."""
        # Bot configuration
        self.symbol = "BTC/USDT"
        
        # Get adaptive parameters based on prediction cycle
        self.adaptive_params = self.get_adaptive_parameters()
        self.data_fetch_timeframe = self.adaptive_params.get("data_timeframe", "5m")
        self.data_fetch_limit = self.adaptive_params.get("data_limit", 100)
        
        # Moving average parameters
        self.short_ma_period = 20
        self.long_ma_period = 50
        self.trade_amount = 100
        
        # Strategy state
        self.last_signal = None
        
        self.logger.info(f"GoldenCrossBot initialized with prediction_cycle: {self.prediction_cycle}")
        self.logger.info(f"Data fetch: {self.data_fetch_limit} candles on {self.data_fetch_timeframe}")
        self.logger.info(f"Strategy: {self.adaptive_params.get('description', 'Golden Cross')}")
    
    def get_supported_prediction_cycles(self) -> List[str]:
        """Danh sách chu kỳ được hỗ trợ."""
        return ["15m", "1h", "4h", "1d"]
    
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

    def get_configuration_schema(self) -> Dict[str, Any]:
        """Flat config schema used by UI to render inputs"""
        return {
            "short_period": {"type": "int", "default": 10, "min": 5, "max": 50, "description": "Short SMA period"},
            "long_period": {"type": "int", "default": 20, "min": 10, "max": 100, "description": "Long SMA period"},
            "use_rsi_filter": {"type": "bool", "default": True, "description": "Use RSI filter to avoid overbought/oversold"},
            "rsi_period": {"type": "int", "default": 14, "min": 5, "max": 30, "description": "RSI period for filter"},
            "rsi_overbought": {"type": "float", "default": 70.0, "min": 50.0, "max": 90.0, "description": "RSI overbought threshold"},
            "rsi_oversold": {"type": "float", "default": 30.0, "min": 10.0, "max": 50.0, "description": "RSI oversold threshold"},
            "stop_loss_pct": {"type": "float", "default": 3.0, "min": 0.5, "max": 10.0, "description": "Stop loss percentage (for info)"},
            "take_profit_pct": {"type": "float", "default": 6.0, "min": 1.0, "max": 20.0, "description": "Take profit percentage (for info)"},
            "allocation_percentage": {"type": "float", "default": 10.0, "min": 1.0, "max": 50.0, "description": "Percentage of balance per trade"},
        }

    def calculate_sma(self, prices: List[float], period: int) -> float:
        if len(prices) < period:
            return None
        return float(sum(prices[-period:]) / period)
    
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
        rsi = 100 - (100 / (1 + rs))
        return float(rsi)

    def execute_algorithm(self, data: pd.DataFrame, timeframe: str, subscription_config: Dict[str, Any] = None) -> SimpleAction:
        """Return SimpleAction BUY/SELL/HOLD based on SMA cross with optional RSI filter"""
        try:
            close_prices = data['close'].astype(float).tolist()
            if len(close_prices) < max(self.short_period, self.long_period):
                return SimpleAction.hold("Insufficient data for SMAs")

            short_sma = self.calculate_sma(close_prices, self.short_period)
            long_sma = self.calculate_sma(close_prices, self.long_period)
            if short_sma is None or long_sma is None:
                return SimpleAction.hold("Unable to calculate SMAs")

            rsi_ok = True
            rsi_val = None
            if self.use_rsi_filter:
                rsi_val = self.calculate_rsi(close_prices, self.rsi_period)
                if rsi_val is not None:
                    # Avoid buy if overbought; avoid sell if oversold
                    rsi_ok = True
            
            # Determine signal
            if short_sma > long_sma:
                if not self.use_rsi_filter or (rsi_val is None or rsi_val < self.rsi_overbought):
                    return SimpleAction.buy(
                        amount_type=AmountType.PERCENTAGE,
                        value=self.allocation_percentage,
                        reason=f"Golden cross SMA{self.short_period}>{self.long_period}" + (f", RSI={rsi_val:.1f}" if rsi_val is not None else "")
                    )
                return SimpleAction.hold(f"Golden cross but RSI overbought ({rsi_val:.1f})")
            elif short_sma < long_sma:
                if not self.use_rsi_filter or (rsi_val is None or rsi_val > self.rsi_oversold):
                    return SimpleAction.sell(
                        amount_type=AmountType.PERCENTAGE,
                        value=self.allocation_percentage,
                        reason=f"Death cross SMA{self.short_period}<{self.long_period}" + (f", RSI={rsi_val:.1f}" if rsi_val is not None else "")
                    )
                return SimpleAction.hold(f"Death cross but RSI oversold ({rsi_val:.1f})")
            else:
                return SimpleAction.hold("SMAs equal, waiting for crossover")
        except Exception as e:
            logger.error(f"GoldenCross execute_algorithm error: {e}")
            return SimpleAction.hold(f"Algorithm error: {str(e)}") 