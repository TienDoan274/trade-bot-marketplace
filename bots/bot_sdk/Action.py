"""
Enhanced Action System for Trading Bots
Supports both SimpleAction and AdvancedAction with various order types
"""

from enum import Enum
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

class AmountType(Enum):
    """Amount types for trading actions"""
    BASE_AMOUNT = "BASE_AMOUNT"        # Amount in base currency (BTC)
    QUOTE_AMOUNT = "QUOTE_AMOUNT"      # Amount in quote currency (USDT)
    PERCENTAGE = "PERCENTAGE"          # Percentage of available balance
    ALL = "ALL"                        # Use all available balance

class OrderType(Enum):
    """Order types supported by exchanges"""
    MARKET = "MARKET"                  # Market order (immediate execution)
    LIMIT = "LIMIT"                    # Limit order (specified price)
    STOP_LOSS = "STOP_LOSS"           # Stop loss order
    STOP_LOSS_LIMIT = "STOP_LOSS_LIMIT"  # Stop loss limit order
    TAKE_PROFIT = "TAKE_PROFIT"       # Take profit order
    TAKE_PROFIT_LIMIT = "TAKE_PROFIT_LIMIT"  # Take profit limit order
    LIMIT_MAKER = "LIMIT_MAKER"       # Limit maker order (post-only)

class ActionType(Enum):
    """Action types for trading"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

@dataclass
class SimpleAction:
    """
    Simple Action for basic trading strategies
    - Uses market orders only
    - Simple amount types (PERCENTAGE, ALL)
    - Easy to use for beginners
    """
    action: ActionType
    amount_type: AmountType
    value: float
    reason: str
    
    @classmethod
    def buy(cls, amount_type: AmountType = AmountType.PERCENTAGE, value: float = 5.0, reason: str = "Buy signal"):
        """Create a BUY action"""
        return cls(ActionType.BUY, amount_type, value, reason)
    
    @classmethod
    def sell(cls, amount_type: AmountType = AmountType.PERCENTAGE, value: float = 5.0, reason: str = "Sell signal"):
        """Create a SELL action"""
        return cls(ActionType.SELL, amount_type, value, reason)
    
    @classmethod
    def hold(cls, reason: str = "No signal"):
        """Create a HOLD action"""
        return cls(ActionType.HOLD, AmountType.PERCENTAGE, 0.0, reason)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "action": self.action.value,
            "amount_type": self.amount_type.value,
            "value": self.value,
            "reason": self.reason,
            "type": "SIMPLE"
        }

@dataclass
class AdvancedAction:
    """
    Advanced Action for complex trading strategies
    - Supports all order types
    - All amount types
    - Price specification for limit orders
    - Stop loss and take profit levels
    """
    action: ActionType
    order_type: OrderType
    amount_type: AmountType
    value: float
    reason: str
    price: Optional[float] = None          # Price for limit orders
    stop_price: Optional[float] = None     # Stop price for stop orders
    take_profit_price: Optional[float] = None  # Take profit price
    stop_loss_price: Optional[float] = None    # Stop loss price
    time_in_force: str = "GTC"            # GTC, IOC, FOK
    reduce_only: bool = False             # Reduce only flag
    post_only: bool = False               # Post only flag
    
    @classmethod
    def market_buy(cls, amount_type: AmountType = AmountType.PERCENTAGE, value: float = 5.0, reason: str = "Market buy"):
        """Create a market BUY action"""
        return cls(ActionType.BUY, OrderType.MARKET, amount_type, value, reason)
    
    @classmethod
    def market_sell(cls, amount_type: AmountType = AmountType.PERCENTAGE, value: float = 5.0, reason: str = "Market sell"):
        """Create a market SELL action"""
        return cls(ActionType.SELL, OrderType.MARKET, amount_type, value, reason)
    
    @classmethod
    def limit_buy(cls, price: float, amount_type: AmountType = AmountType.PERCENTAGE, value: float = 5.0, reason: str = "Limit buy"):
        """Create a limit BUY action"""
        return cls(ActionType.BUY, OrderType.LIMIT, amount_type, value, reason, price=price)
    
    @classmethod
    def limit_sell(cls, price: float, amount_type: AmountType = AmountType.PERCENTAGE, value: float = 5.0, reason: str = "Limit sell"):
        """Create a limit SELL action"""
        return cls(ActionType.SELL, OrderType.LIMIT, amount_type, value, reason, price=price)
    
    @classmethod
    def stop_loss(cls, stop_price: float, amount_type: AmountType = AmountType.PERCENTAGE, value: float = 5.0, reason: str = "Stop loss"):
        """Create a stop loss action"""
        return cls(ActionType.SELL, OrderType.STOP_LOSS, amount_type, value, reason, stop_price=stop_price)
    
    @classmethod
    def take_profit(cls, take_profit_price: float, amount_type: AmountType = AmountType.PERCENTAGE, value: float = 5.0, reason: str = "Take profit"):
        """Create a take profit action"""
        return cls(ActionType.SELL, OrderType.TAKE_PROFIT, amount_type, value, reason, take_profit_price=take_profit_price)
    
    @classmethod
    def hold(cls, reason: str = "No signal"):
        """Create a HOLD action"""
        return cls(ActionType.HOLD, OrderType.MARKET, AmountType.PERCENTAGE, 0.0, reason)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "action": self.action.value,
            "order_type": self.order_type.value,
            "amount_type": self.amount_type.value,
            "value": self.value,
            "reason": self.reason,
            "price": self.price,
            "stop_price": self.stop_price,
            "take_profit_price": self.take_profit_price,
            "stop_loss_price": self.stop_loss_price,
            "time_in_force": self.time_in_force,
            "reduce_only": self.reduce_only,
            "post_only": self.post_only,
            "type": "ADVANCED"
        }

# Backward compatibility - keep the old Action class
class Action:
    """
    Legacy Action class for backward compatibility
    Maps to SimpleAction internally
    """
    
    def __init__(self, action: str, value: float, reason: str):
        self.action = action
        self.value = value
        self.reason = reason
        self.type = "PERCENTAGE"  # Default to percentage for backward compatibility
    
    @classmethod
    def buy(cls, type: str = "PERCENTAGE", value: float = 5.0, reason: str = "Buy signal"):
        """Create a BUY action (legacy)"""
        return cls("BUY", value, reason)
    
    @classmethod
    def sell(cls, type: str = "PERCENTAGE", value: float = 5.0, reason: str = "Sell signal"):
        """Create a SELL action (legacy)"""
        return cls("SELL", value, reason)
    
    @classmethod
    def hold(cls, reason: str = "No signal"):
        """Create a HOLD action (legacy)"""
        return cls("HOLD", 0.0, reason)
    
    def to_simple_action(self) -> SimpleAction:
        """Convert to SimpleAction"""
        amount_type = AmountType.PERCENTAGE
        if self.type == "BASE_AMOUNT":
            amount_type = AmountType.BASE_AMOUNT
        elif self.type == "QUOTE_AMOUNT":
            amount_type = AmountType.QUOTE_AMOUNT
        elif self.type == "ALL":
            amount_type = AmountType.ALL
        
        return SimpleAction(
            action=ActionType(self.action),
            amount_type=amount_type,
            value=self.value,
            reason=self.reason
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (legacy format)"""
        return {
            "action": self.action,
            "type": self.type,
            "value": self.value,
            "reason": self.reason
        } 