from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from decimal import Decimal
from enum import Enum

# Enums for API responses
class UserRole(str, Enum):
    USER = "USER"
    DEVELOPER = "DEVELOPER"
    ADMIN = "ADMIN"

class BotStatus(str, Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    ARCHIVED = "ARCHIVED"

class BotType(str, Enum):
    TECHNICAL = "TECHNICAL"  # Traditional technical analysis
    ML = "ML"               # Machine Learning
    DL = "DL"               # Deep Learning
    LLM = "LLM"             # Large Language Model

class FileType(str, Enum):
    CODE = "CODE"
    MODEL = "MODEL"
    WEIGHTS = "WEIGHTS"
    CONFIG = "CONFIG"
    DATA = "DATA"

class ExchangeType(str, Enum):
    BINANCE = "BINANCE"
    COINBASE = "COINBASE"
    KRAKEN = "KRAKEN"
    BYBIT = "BYBIT"
    HUOBI = "HUOBI"

class SubscriptionStatus(str, Enum):
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"
    ERROR = "ERROR"

class TradeStatus(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"

# --- User Schemas ---
class UserBase(BaseModel):
    email: EmailStr
    role: UserRole = UserRole.USER
    is_active: bool = True
    developer_name: Optional[str] = None
    developer_bio: Optional[str] = None
    developer_website: Optional[str] = None

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    role: UserRole = UserRole.USER
    developer_name: Optional[str] = None
    developer_bio: Optional[str] = None
    developer_website: Optional[str] = None

class UserUpdate(BaseModel):
    developer_name: Optional[str] = None
    developer_bio: Optional[str] = None
    developer_website: Optional[str] = None
    api_key: Optional[str] = None
    api_secret: Optional[str] = None

class UserInDB(UserBase):
    id: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class UserProfile(UserInDB):
    total_subscriptions: int = 0
    active_subscriptions: int = 0
    total_developed_bots: int = 0
    approved_bots: int = 0

# --- Exchange Credentials Schemas ---
class ExchangeCredentialsBase(BaseModel):
    exchange: ExchangeType
    api_key: str
    api_secret: str
    api_passphrase: Optional[str] = None  # For Coinbase
    is_testnet: bool = True  # Default to testnet for safety

class ExchangeCredentialsCreate(ExchangeCredentialsBase):
    pass

class ExchangeCredentialsUpdate(BaseModel):
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    api_passphrase: Optional[str] = None
    is_testnet: Optional[bool] = None
    is_active: Optional[bool] = None

class ExchangeCredentialsInDB(ExchangeCredentialsBase):
    id: int
    user_id: int
    is_active: bool
    created_at: datetime
    updated_at: datetime
    last_validated: Optional[datetime] = None
    validation_status: str
    
    class Config:
        from_attributes = True

class ExchangeCredentialsPublic(BaseModel):
    """Public view without sensitive data"""
    id: int
    exchange: ExchangeType
    is_testnet: bool
    is_active: bool
    validation_status: str
    last_validated: Optional[datetime] = None
    api_key_preview: str  # Only show first 8 characters
    
    class Config:
        from_attributes = True

class ExchangeCredentialsValidation(BaseModel):
    """Response for credential validation"""
    valid: bool
    message: str
    exchange: ExchangeType
    is_testnet: bool

class Token(BaseModel):
    access_token: str
    token_type: str

# --- Bot Category Schemas ---
class BotCategoryBase(BaseModel):
    name: str
    description: Optional[str] = None

class BotCategoryCreate(BotCategoryBase):
    pass

class BotCategoryInDB(BotCategoryBase):
    id: int
    
    class Config:
        from_attributes = True

# --- Bot Schemas ---
class BotBase(BaseModel):
    name: str
    description: str
    category_id: Optional[int] = None
    version: str = "1.0.0"
    bot_type: BotType = BotType.TECHNICAL
    price_per_month: Decimal = Decimal('0.00')
    is_free: bool = True
    config_schema: Optional[Dict[str, Any]] = None
    default_config: Optional[Dict[str, Any]] = None
    model_metadata: Optional[Dict[str, Any]] = None

class BotCreate(BotBase):
    pass

class BotUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    category_id: Optional[int] = None
    version: Optional[str] = None
    bot_type: Optional[BotType] = None
    price_per_month: Optional[Decimal] = None
    is_free: Optional[bool] = None
    config_schema: Optional[Dict[str, Any]] = None
    default_config: Optional[Dict[str, Any]] = None
    model_metadata: Optional[Dict[str, Any]] = None

class BotInDB(BotBase):
    id: int
    developer_id: int
    status: BotStatus
    code_path: Optional[str] = None
    model_path: Optional[str] = None
    total_subscribers: int = 0
    average_rating: float = 0.0
    total_reviews: int = 0
    created_at: datetime
    updated_at: datetime
    approved_at: Optional[datetime] = None
    approved_by: Optional[int] = None
    
    class Config:
        from_attributes = True

class BotPublic(BaseModel):
    id: int
    name: str
    description: str
    developer_id: int
    category_id: Optional[int] = None
    version: str
    price_per_month: Decimal
    is_free: bool
    total_subscribers: int
    average_rating: float
    total_reviews: int
    default_config: Optional[Dict[str, Any]] = None
    created_at: datetime
    
    class Config:
        from_attributes = True

class BotWithDeveloper(BotPublic):
    developer: UserInDB

# --- Bot File Schemas ---
class BotFileBase(BaseModel):
    file_type: FileType
    file_name: str
    description: Optional[str] = None
    version: str = "1.0.0"
    model_framework: Optional[str] = None
    model_type: Optional[str] = None
    input_shape: Optional[Dict[str, Any]] = None
    output_shape: Optional[Dict[str, Any]] = None

class BotFileCreate(BotFileBase):
    bot_id: int

class BotFileInDB(BotFileBase):
    id: int
    bot_id: int
    file_path: str
    file_size: int
    file_hash: str
    upload_date: datetime
    is_active: bool
    
    class Config:
        from_attributes = True

class BotFileUpload(BaseModel):
    file_type: FileType
    description: Optional[str] = None
    model_framework: Optional[str] = None
    model_type: Optional[str] = None

class MLModelInfo(BaseModel):
    framework: str  # tensorflow, pytorch, sklearn, etc.
    model_type: str  # classification, regression, etc.
    input_features: List[str]
    output_format: Dict[str, Any]
    requirements: List[str]  # Python packages needed
    preprocessing_steps: Optional[List[str]] = None

# --- Binance Account Schemas ---
class BinanceAccount(BaseModel):
    account_type: str
    can_trade: bool
    can_withdraw: bool
    can_deposit: bool
    update_time: datetime
    balances: List[Dict[str, str]]
    permissions: List[str]

class BinanceBalance(BaseModel):
    asset: str
    free: str
    locked: str

class BinanceOrder(BaseModel):
    symbol: str
    order_id: int
    client_order_id: str
    price: str
    orig_qty: str
    executed_qty: str
    status: str
    side: str
    type: str
    time_in_force: str

class CreateOrderRequest(BaseModel):
    symbol: str
    side: str  # BUY or SELL
    type: str  # MARKET, LIMIT, etc.
    quantity: Optional[str] = None
    price: Optional[str] = None
    time_in_force: Optional[str] = "GTC"

# --- Bot Review Schemas ---
class BotReviewBase(BaseModel):
    rating: int = Field(..., ge=1, le=5)
    review_text: Optional[str] = None

class BotReviewCreate(BotReviewBase):
    bot_id: int

class BotReviewInDB(BotReviewBase):
    id: int
    user_id: int
    bot_id: int
    created_at: datetime
    
    class Config:
        from_attributes = True
        
class BotReviewWithUser(BotReviewInDB):
    user: UserInDB

# --- Subscription Schemas ---
class ExecutionConfig(BaseModel):
    buy_order_type: str = Field(..., pattern="^(PERCENTAGE|USDT_AMOUNT)$")
    buy_order_value: float = Field(..., gt=0)
    sell_order_type: str = Field(..., pattern="^(PERCENTAGE|ALL)$")
    sell_order_value: float = Field(..., gt=0)

class RiskConfig(BaseModel):
    stop_loss_percent: Optional[float] = Field(None, gt=0, le=100)
    take_profit_percent: Optional[float] = Field(None, gt=0)
    max_position_size: Optional[float] = Field(None, gt=0)

class SubscriptionBase(BaseModel):
    instance_name: str
    bot_id: int
    exchange_type: ExchangeType = ExchangeType.BINANCE
    trading_pair: str
    timeframe: str = Field(..., pattern="^(1m|5m|15m|1h|4h|1d|1w)$")
    strategy_config: Dict[str, Any] = {}
    execution_config: ExecutionConfig
    risk_config: RiskConfig
    is_testnet: bool = False  # Whether to run on testnet
    is_trial: bool = False    # Whether this is a trial subscription

class SubscriptionCreate(SubscriptionBase):
    pass

class SubscriptionTrialCreate(BaseModel):
    """Schema for creating a trial subscription with simplified setup"""
    instance_name: str
    bot_id: int
    exchange_type: ExchangeType = ExchangeType.BINANCE
    trading_pair: str = "BTC/USDT"
    timeframe: str = "1h"
    trial_duration_hours: float = Field(default=24.0, ge=0.1, le=168.0)  # 0.1 hours (6 minutes) to 1 week

class SubscriptionUpdate(BaseModel):
    instance_name: Optional[str] = None
    exchange_type: Optional[ExchangeType] = None
    trading_pair: Optional[str] = None
    timeframe: Optional[str] = None
    strategy_config: Optional[Dict[str, Any]] = None
    execution_config: Optional[ExecutionConfig] = None
    risk_config: Optional[RiskConfig] = None
    is_testnet: Optional[bool] = None

class SubscriptionInDB(SubscriptionBase):
    id: int
    user_id: int
    status: SubscriptionStatus
    started_at: datetime
    expires_at: Optional[datetime] = None
    trial_expires_at: Optional[datetime] = None
    last_run_at: Optional[datetime] = None
    next_run_at: Optional[datetime] = None
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: Decimal = Decimal('0.0')
    
    class Config:
        from_attributes = True

class SubscriptionWithBot(SubscriptionInDB):
    bot: BotInDB

class SubscriptionResponse(BaseModel):
    subscription_id: int
    status: str
    message: str

# --- Trade Schemas ---
class TradeBase(BaseModel):
    side: str = Field(..., pattern="^(BUY|SELL)$")
    entry_price: Decimal
    quantity: Decimal
    stop_loss_price: Optional[Decimal] = None
    take_profit_price: Optional[Decimal] = None

class TradeCreate(TradeBase):
    subscription_id: int

class TradeInDB(TradeBase):
    id: int
    subscription_id: int
    status: TradeStatus
    exit_price: Optional[Decimal] = None
    entry_time: datetime
    exit_time: Optional[datetime] = None
    pnl: Optional[Decimal] = None
    pnl_percentage: Optional[float] = None
    exchange_order_id: Optional[str] = None
    exchange_trade_id: Optional[str] = None
    
    class Config:
        from_attributes = True

class TradeWithSubscription(TradeInDB):
    subscription: SubscriptionInDB

# --- Performance Schemas ---
class PerformanceMetrics(BaseModel):
    total_trades: int = 0
    winning_trades: int = 0
    win_rate: float = 0.0
    average_pnl: Decimal = Decimal('0.0')
    total_pnl: Decimal = Decimal('0.0')
    max_drawdown: float = 0.0
    sharpe_ratio: Optional[float] = None

class BotPerformanceInDB(PerformanceMetrics):
    id: int
    bot_id: int
    period_start: datetime
    period_end: datetime
    total_subscribers: int = 0
    active_subscribers: int = 0
    
    class Config:
        from_attributes = True

class PerformanceLogBase(BaseModel):
    action: str = Field(..., pattern="^(BUY|SELL|HOLD|STOP_LOSS|TAKE_PROFIT)$")
    price: Decimal
    quantity: Decimal
    balance: Decimal
    signal_data: Optional[Dict[str, Any]] = None

class PerformanceLogCreate(PerformanceLogBase):
    subscription_id: int

class PerformanceLogInDB(PerformanceLogBase):
    id: int
    subscription_id: int
    timestamp: datetime
    
    class Config:
        from_attributes = True

# --- Exchange Schemas ---
class ExchangeCredentials(BaseModel):
    exchange_name: str = Field(..., description="Exchange name (BINANCE, COINBASE, etc.)")
    api_key: str = Field(..., description="Exchange API key")
    api_secret: str = Field(..., description="Exchange API secret")
    testnet: bool = Field(default=True, description="Use testnet/sandbox environment")

class ExchangeValidationResponse(BaseModel):
    valid: bool
    message: str
    exchange: str
    testnet: bool
    account_info: Optional[Dict[str, Any]] = None

# --- Payment Schemas ---
class PaymentBase(BaseModel):
    amount: Decimal
    currency: str = "USD"
    payment_method: str
    payment_provider: str = "STRIPE"

class PaymentCreate(PaymentBase):
    subscription_id: int

class PaymentInDB(PaymentBase):
    id: int
    user_id: int
    subscription_id: int
    payment_status: str
    external_payment_id: Optional[str] = None
    created_at: datetime
    processed_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

# --- Response Schemas ---
class BotListResponse(BaseModel):
    bots: List[BotPublic]
    total: int
    page: int
    per_page: int

class SubscriptionListResponse(BaseModel):
    subscriptions: List[SubscriptionWithBot]
    total: int
    page: int
    per_page: int

class TradeListResponse(BaseModel):
    trades: List[TradeInDB]
    total: int
    page: int
    per_page: int

class PerformanceResponse(BaseModel):
    performance: PerformanceMetrics
    recent_trades: List[TradeInDB]
    logs: List[PerformanceLogInDB]

# --- Error Schemas ---
class ErrorResponse(BaseModel):
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None

# --- Admin Schemas ---
class AdminBotApproval(BaseModel):
    bot_id: int
    status: BotStatus
    approval_notes: Optional[str] = None

class AdminUserUpdate(BaseModel):
    user_id: int
    is_active: bool
    role: Optional[UserRole] = None

class AdminStats(BaseModel):
    total_users: int
    total_developers: int
    total_bots: int
    approved_bots: int
    pending_bots: int
    total_subscriptions: int
    active_subscriptions: int
    total_trades: int
    total_revenue: Decimal

# --- Pricing Plan Schemas ---
class PricingPlanBase(BaseModel):
    plan_name: str
    plan_description: str
    price_per_month: Decimal
    price_per_year: Optional[Decimal] = None
    price_per_quarter: Optional[Decimal] = None
    max_trading_pairs: int = 1
    max_daily_trades: int = 10
    max_position_size: Decimal = Decimal('0.10')
    advanced_features: Optional[Dict[str, Any]] = None
    trial_days: int = 0
    trial_trades_limit: int = 5
    is_popular: bool = False

class PricingPlanCreate(PricingPlanBase):
    pass

class PricingPlanUpdate(BaseModel):
    plan_name: Optional[str] = None
    plan_description: Optional[str] = None
    price_per_month: Optional[Decimal] = None
    price_per_year: Optional[Decimal] = None
    price_per_quarter: Optional[Decimal] = None
    max_trading_pairs: Optional[int] = None
    max_daily_trades: Optional[int] = None
    max_position_size: Optional[Decimal] = None
    advanced_features: Optional[Dict[str, Any]] = None
    trial_days: Optional[int] = None
    trial_trades_limit: Optional[int] = None
    is_popular: Optional[bool] = None
    is_active: Optional[bool] = None

class PricingPlanInDB(PricingPlanBase):
    id: int
    bot_id: int
    is_active: bool
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

# --- Promotion Schemas ---
class PromotionBase(BaseModel):
    promotion_code: str
    promotion_name: str
    promotion_description: str
    discount_type: str  # "PERCENTAGE", "FIXED_AMOUNT", "FREE_TRIAL"
    discount_value: Decimal
    max_uses: int = 100
    valid_from: datetime
    valid_until: datetime
    min_subscription_months: int = 1
    applicable_plans: Optional[List[int]] = None

class PromotionCreate(PromotionBase):
    pass

class PromotionUpdate(BaseModel):
    promotion_name: Optional[str] = None
    promotion_description: Optional[str] = None
    discount_value: Optional[Decimal] = None
    max_uses: Optional[int] = None
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    min_subscription_months: Optional[int] = None
    applicable_plans: Optional[List[int]] = None
    is_active: Optional[bool] = None

class PromotionInDB(PromotionBase):
    id: int
    bot_id: int
    used_count: int
    is_active: bool
    created_at: datetime
    created_by: int
    
    class Config:
        from_attributes = True

# --- Invoice Schemas ---
class InvoiceBase(BaseModel):
    amount: Decimal
    currency: str = "USD"
    base_price: Decimal
    discount_amount: Decimal = Decimal('0.00')
    tax_amount: Decimal = Decimal('0.00')
    final_amount: Decimal
    billing_period_start: datetime
    billing_period_end: datetime
    payment_method: Optional[str] = None
    promotion_code: Optional[str] = None
    promotion_discount: Decimal = Decimal('0.00')

class InvoiceCreate(InvoiceBase):
    subscription_id: int
    user_id: int

class InvoiceUpdate(BaseModel):
    status: Optional[str] = None
    payment_date: Optional[datetime] = None
    payment_method: Optional[str] = None

class InvoiceInDB(InvoiceBase):
    id: int
    subscription_id: int
    user_id: int
    invoice_number: str
    status: str
    payment_date: Optional[datetime] = None
    created_at: datetime
    due_date: datetime
    
    class Config:
        from_attributes = True

# --- Enhanced Bot Schemas ---
class BotWithPricing(BotPublic):
    pricing_plans: List[PricingPlanInDB] = []
    active_promotions: List[PromotionInDB] = []
    
    class Config:
        from_attributes = True

# --- Subscription with Pricing ---
class SubscriptionWithPricing(SubscriptionResponse):
    pricing_plan: Optional[PricingPlanInDB] = None
    next_billing_date: Optional[datetime] = None
    billing_cycle: str = "MONTHLY"
    auto_renew: bool = True
    
    class Config:
        from_attributes = True

# Bot Draft schemas
class BotDraftBase(BaseModel):
    name: str
    description: Optional[str] = None
    bot_code: str
    category_id: Optional[int] = None

class BotDraftCreate(BotDraftBase):
    pass

class BotDraftUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    bot_code: Optional[str] = None
    category_id: Optional[int] = None

class BotDraft(BotDraftBase):
    id: int
    user_id: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True

class BotDraftList(BaseModel):
    drafts: List[BotDraft]
    total: int