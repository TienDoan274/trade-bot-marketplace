"""
Exchange Factory Module
Manages multiple cryptocurrency exchanges and provides unified interface
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal
import pandas as pd
from datetime import datetime
from dataclasses import dataclass
import ccxt

logger = logging.getLogger(__name__)

class ExchangeCapabilities:
    """Exchange capabilities information"""
    
    def __init__(self, exchange_name: str):
        self.exchange_name = exchange_name
        self.spot_trading = True
        self.futures_trading = False
        self.margin_trading = False
        self.stop_loss_orders = True
        self.take_profit_orders = True
        self.advanced_orders = False
        self.api_key_permissions = []

class BaseExchange(ABC):
    """Base class for exchange implementations"""
    
    def __init__(self, api_key: str = None, secret: str = None, use_testnet: bool = True):
        self.api_key = api_key
        self.secret = secret
        self.use_testnet = use_testnet
        self.exchange = None
        self.is_connected = False
    
    @abstractmethod
    def connect(self):
        """Connect to exchange"""
        pass
    
    @abstractmethod
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> List[List]:
        """Fetch OHLCV data"""
        pass
    
    @abstractmethod
    def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """Fetch current ticker"""
        pass
    
    @abstractmethod
    def get_balance(self, currency: str = None) -> Dict[str, float]:
        """Get account balance"""
        pass
    
    @abstractmethod
    def place_order(self, symbol: str, side: str, amount: float, price: float = None) -> Dict[str, Any]:
        """Place order"""
        pass

class BinanceExchange(BaseExchange):
    """Binance exchange implementation"""
    
    def __init__(self, api_key: str = None, secret: str = None, use_testnet: bool = True):
        super().__init__(api_key, secret, use_testnet)
        self.exchange_name = "binance"
    
    def connect(self):
        """Connect to Binance"""
        try:
            config = {
                'apiKey': self.api_key,
                'secret': self.secret,
                'sandbox': self.use_testnet,
                'enableRateLimit': True
            }
            
            self.exchange = ccxt.binance(config)
            self.exchange.load_markets()
            self.is_connected = True
            return True
        except Exception as e:
            print(f"Failed to connect to Binance: {e}")
            return False
    
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> List[List]:
        """Fetch OHLCV data from Binance"""
        if not self.is_connected:
            self.connect()
        
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            return ohlcv
        except Exception as e:
            print(f"Failed to fetch OHLCV: {e}")
            return []
    
    def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """Fetch current ticker from Binance"""
        if not self.is_connected:
            self.connect()
        
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker
        except Exception as e:
            print(f"Failed to fetch ticker: {e}")
            return {}
    
    def get_balance(self, currency: str = None) -> Dict[str, float]:
        """Get account balance from Binance"""
        if not self.is_connected:
            self.connect()
        
        try:
            balance = self.exchange.fetch_balance()
            if currency:
                return {currency: balance.get(currency, {}).get('free', 0)}
            return balance
        except Exception as e:
            print(f"Failed to fetch balance: {e}")
            return {}
    
    def place_order(self, symbol: str, side: str, amount: float, price: float = None) -> Dict[str, Any]:
        """Place order on Binance"""
        if not self.is_connected:
            self.connect()
        
        try:
            order_type = 'limit' if price else 'market'
            order = self.exchange.create_order(symbol, order_type, side, amount, price)
            return order
        except Exception as e:
            print(f"Failed to place order: {e}")
            return {}

class CoinbaseExchange(BaseExchange):
    """Coinbase exchange implementation"""
    
    def __init__(self, api_key: str = None, secret: str = None, use_testnet: bool = True):
        super().__init__(api_key, secret, use_testnet)
        self.exchange_name = "coinbase"
    
    def connect(self):
        """Connect to Coinbase"""
        try:
            config = {
                'apiKey': self.api_key,
                'secret': self.secret,
                'sandbox': self.use_testnet,
                'enableRateLimit': True
            }
            
            self.exchange = ccxt.coinbase(config)
            self.exchange.load_markets()
            self.is_connected = True
            return True
        except Exception as e:
            print(f"Failed to connect to Coinbase: {e}")
            return False
    
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> List[List]:
        """Fetch OHLCV data from Coinbase"""
        if not self.is_connected:
            self.connect()
        
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            return ohlcv
        except Exception as e:
            print(f"Failed to fetch OHLCV: {e}")
            return []
    
    def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """Fetch current ticker from Coinbase"""
        if not self.is_connected:
            self.connect()
        
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker
        except Exception as e:
            print(f"Failed to fetch ticker: {e}")
            return {}
    
    def get_balance(self, currency: str = None) -> Dict[str, float]:
        """Get account balance from Coinbase"""
        if not self.is_connected:
            self.connect()
        
        try:
            balance = self.exchange.fetch_balance()
            if currency:
                return {currency: balance.get(currency, {}).get('free', 0)}
            return balance
        except Exception as e:
            print(f"Failed to fetch balance: {e}")
            return {}
    
    def place_order(self, symbol: str, side: str, amount: float, price: float = None) -> Dict[str, Any]:
        """Place order on Coinbase"""
        if not self.is_connected:
            self.connect()
        
        try:
            order_type = 'limit' if price else 'market'
            order = self.exchange.create_order(symbol, order_type, side, amount, price)
            return order
        except Exception as e:
            print(f"Failed to place order: {e}")
            return {}

class ExchangeFactory:
    """Factory for creating exchange instances"""
    
    @staticmethod
    def create_exchange(exchange_type: str, api_key: str = None, secret: str = None, use_testnet: bool = True) -> BaseExchange:
        """Create exchange instance based on type"""
        
        exchange_map = {
            "BINANCE": BinanceExchange,
            "COINBASE": CoinbaseExchange,
        }
        
        if exchange_type not in exchange_map:
            raise ValueError(f"Unsupported exchange type: {exchange_type}")
        
        exchange_class = exchange_map[exchange_type]
        return exchange_class(api_key, secret, use_testnet)
    
    @staticmethod
    def get_supported_exchanges() -> List[str]:
        """Get list of supported exchanges"""
        return ["BINANCE", "COINBASE"]
    
    @staticmethod
    def get_exchange_capabilities(exchange_name: str) -> ExchangeCapabilities:
        """Get capabilities for specific exchange"""
        exchange_name = exchange_name.upper()
        
        if exchange_name == "BINANCE":
            capabilities = ExchangeCapabilities("BINANCE")
            capabilities.futures_trading = True
            capabilities.margin_trading = True
            capabilities.advanced_orders = True
            capabilities.api_key_permissions = ["spot", "futures", "margin"]
            return capabilities
        elif exchange_name == "COINBASE":
            capabilities = ExchangeCapabilities("COINBASE")
            capabilities.api_key_permissions = ["spot"]
            return capabilities
        else:
            raise ValueError(f"Unsupported exchange: {exchange_name}")
        
    @staticmethod
    def validate_exchange_credentials(exchange_name: str, api_key: str, api_secret: str, testnet: bool = True) -> tuple[bool, str]:
        """Validate exchange API credentials"""
        try:
            # Create exchange instance
            exchange = ExchangeFactory.create_exchange(
                exchange_name.upper(), 
                api_key, 
                api_secret, 
                testnet
            )
            
            # Try to connect
            if exchange.connect():
                # Try to fetch account info to verify credentials
                try:
                    balance = exchange.get_balance()
                    return True, "Credentials validated successfully"
                except Exception as e:
                    return False, f"Failed to fetch account info: {str(e)}"
            else:
                return False, "Failed to connect to exchange"
        
        except Exception as e:
                return False, f"Validation failed: {str(e)}" 