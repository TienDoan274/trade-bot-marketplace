from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json

class CustomBot(ABC):
    """
    Base class for all trading bots.
    All bots must inherit from this class and implement required methods.
    """
    
    def __init__(self):
        self.name = "Custom Bot"
        self.version = "1.0.0"
        self.description = "A custom trading bot"
        self.author = "Developer"
        self.config_schema   = {}
        self.is_initialized = False
        self.backtest_results = None
        self.live_results = None
    
    @abstractmethod
    def get_bot_info(self) -> Dict[str, Any]:
        """Return bot information including name, version, description, author, and config schema"""
        pass
    
    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate bot configuration"""
        pass
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]):
        """Initialize bot with configuration"""
        pass
    
    @abstractmethod
    def generate_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signal based on market data
        
        Args:
            market_data: Dictionary containing:
                - symbol: Trading pair (e.g., "BTC/USDT")
                - timeframe: Timeframe (e.g., "5m", "1h")
                - candles: List of OHLCV data [[timestamp, open, high, low, close, volume], ...]
                - current_price: Current price
                - timestamp: Current timestamp
        
        Returns:
            Dictionary with signal information:
                - action: "BUY", "SELL", or "HOLD"
                - confidence: Confidence level (0-10)
                - reason: Reason for the signal
                - price: Target price (optional)
                - stop_loss: Stop loss price (optional)
                - take_profit: Take profit price (optional)
        """
        pass
    
    def get_config_schema(self) -> Dict[str, Any]:
        """Get configuration schema for the bot"""
        return self.config_schema
    
    def set_config_schema(self, schema: Dict[str, Any]):
        """Set configuration schema for the bot"""
        self.config_schema = schema
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values"""
        default_config = {}
        for key, config in self.config_schema.items():
            if "default" in config:
                default_config[key] = config["default"]
        return default_config 