from sqlalchemy import Column, Integer, String, Text, JSON, ForeignKey, DateTime, func, Float, Boolean, Enum, DECIMAL, Index
from sqlalchemy.orm import relationship
from core.database import Base
import enum
from datetime import datetime

class UserRole(enum.Enum):
    USER = "USER"
    DEVELOPER = "DEVELOPER"
    ADMIN = "ADMIN"

class BotStatus(enum.Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    ARCHIVED = "ARCHIVED"

class SubscriptionStatus(enum.Enum):
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"
    ERROR = "ERROR"

class TradeStatus(enum.Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"

class ExchangeType(enum.Enum):
    BINANCE = "BINANCE"
    COINBASE = "COINBASE"
    KRAKEN = "KRAKEN"

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True)
    hashed_password = Column(String(255))
    role = Column(Enum(UserRole), default=UserRole.USER)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Developer profile fields
    developer_name = Column(String(255))
    developer_bio = Column(Text)
    developer_website = Column(String(255))
    
    # API credentials for trading
    api_key = Column(String(255))
    api_secret = Column(String(255))
    
    # Relationships
    subscriptions = relationship("Subscription", back_populates="user")
    exchange_credentials = relationship("ExchangeCredentials", back_populates="user")
    bot_drafts = relationship("BotDraft", back_populates="user")
    developed_bots = relationship("Bot", back_populates="developer", foreign_keys="Bot.developer_id")
    approved_bots = relationship("Bot", back_populates="approved_by_user", foreign_keys="Bot.approved_by")

class ExchangeCredentials(Base):
    """User's API credentials for different exchanges"""
    __tablename__ = "exchange_credentials"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    exchange = Column(Enum(ExchangeType), nullable=False)
    
    # Encrypted API credentials
    api_key = Column(String(255), nullable=False)
    api_secret = Column(String(255), nullable=False)
    api_passphrase = Column(String(255))  # For exchanges like Coinbase that need passphrase
    
    # Configuration
    is_testnet = Column(Boolean, default=True)  # Default to testnet for safety
    is_active = Column(Boolean, default=True)
    
    # Metadata
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    last_validated = Column(DateTime)  # When credentials were last validated
    validation_status = Column(String(50), default="pending")  # pending, valid, invalid
    
    # Relationships
    user = relationship("User", back_populates="exchange_credentials")
    
    # Unique constraint: one credential set per user per exchange per testnet/mainnet
    __table_args__ = (
        Index('idx_unique_user_exchange_testnet', 'user_id', 'exchange', 'is_testnet', unique=True),
    )

class BotCategory(Base):
    __tablename__ = "bot_categories"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True)
    description = Column(Text)
    
    # Relationships
    bots = relationship("Bot", back_populates="category")
    drafts = relationship("BotDraft", back_populates="category")

class Bot(Base):
    __tablename__ = "bots"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255))
    description = Column(Text)
    developer_id = Column(Integer, ForeignKey("users.id"))
    category_id = Column(Integer, ForeignKey("bot_categories.id"))
    status = Column(Enum(BotStatus), default=BotStatus.PENDING)
    
    # Bot files and metadata
    code_path = Column(String(500))  # Path to uploaded bot code
    version = Column(String(50), default="1.0.0")
    
    # Bot type and ML model support
    bot_type = Column(String(50), default="TECHNICAL")  # TECHNICAL, ML, DL, LLM
    model_path = Column(String(500))  # Path to ML model files
    model_metadata = Column(JSON)  # Model info, requirements, etc.
    
    # Legacy pricing (keep for backward compatibility)
    price_per_month = Column(DECIMAL(10, 2), default=0.00)
    is_free = Column(Boolean, default=True)
    
    # Performance metrics
    total_subscribers = Column(Integer, default=0)
    average_rating = Column(Float, default=0.0)
    total_reviews = Column(Integer, default=0)
    
    # Bot configuration schema (JSON schema for validation)
    config_schema = Column(JSON)
    default_config = Column(JSON)
    
    # Audit fields
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    approved_at = Column(DateTime)
    approved_by = Column(Integer, ForeignKey("users.id"))
    
    # Relationships
    developer = relationship("User", back_populates="developed_bots", foreign_keys=[developer_id])
    approved_by_user = relationship("User", back_populates="approved_bots", foreign_keys=[approved_by])
    category = relationship("BotCategory", back_populates="bots")
    subscriptions = relationship("Subscription", back_populates="bot")
    reviews = relationship("BotReview", back_populates="bot")
    performance_metrics = relationship("BotPerformance", back_populates="bot")
    bot_files = relationship("BotFile", back_populates="bot")
    pricing_plans = relationship("BotPricingPlan", back_populates="bot", cascade="all, delete-orphan")
    promotions = relationship("BotPromotion", back_populates="bot", cascade="all, delete-orphan")

class BotFile(Base):
    """Storage for bot files including ML models, weights, etc."""
    __tablename__ = "bot_files"
    id = Column(Integer, primary_key=True)
    bot_id = Column(Integer, ForeignKey("bots.id"))
    file_type = Column(String(50))  # CODE, MODEL, WEIGHTS, CONFIG, DATA
    file_name = Column(String(255))
    file_path = Column(String(500))
    file_size = Column(Integer)  # Size in bytes
    file_hash = Column(String(64))  # SHA256 hash for integrity
    version = Column(String(50), default="1.0.0")
    description = Column(Text)
    
    # ML Model specific fields
    model_framework = Column(String(50))  # tensorflow, pytorch, sklearn, etc.
    model_type = Column(String(50))  # classification, regression, transformer, etc.
    input_shape = Column(JSON)  # Model input requirements
    output_shape = Column(JSON)  # Model output format
    
    # File metadata
    upload_date = Column(DateTime, server_default=func.now())
    is_active = Column(Boolean, default=True)
    
    # Relationships
    bot = relationship("Bot", back_populates="bot_files")

class Subscription(Base):
    __tablename__ = "subscriptions"
    id = Column(Integer, primary_key=True)
    instance_name = Column(String(255))  # User's custom name for this bot instance
    user_id = Column(Integer, ForeignKey("users.id"))
    bot_id = Column(Integer, ForeignKey("bots.id"))
    status = Column(Enum(SubscriptionStatus), default=SubscriptionStatus.ACTIVE)
    
    # Pricing plan reference
    pricing_plan_id = Column(Integer, ForeignKey("bot_pricing_plans.id"))
    
    # Testnet and Trial support
    is_testnet = Column(Boolean, default=False)  # True for testnet subscriptions
    is_trial = Column(Boolean, default=False)    # True for trial subscriptions
    trial_expires_at = Column(DateTime)          # Trial expiration (can be different from expires_at)
    
    # Exchange and trading configuration
    exchange_type = Column(Enum(ExchangeType), default=ExchangeType.BINANCE)  # BINANCE, COINBASE, KRAKEN
    trading_pair = Column(String(20))
    timeframe = Column(String(10))  # "1h", "4h", "1d"
    
    # Strategy configuration (overrides defaults)
    strategy_config = Column(JSON)

    # Execution configuration
    execution_config = Column(JSON)

    # Risk management
    risk_config = Column(JSON)
    
    # Subscription management
    started_at = Column(DateTime, server_default=func.now())
    expires_at = Column(DateTime)
    last_run_at = Column(DateTime)
    next_run_at = Column(DateTime)
    
    # Performance tracking
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    total_pnl = Column(DECIMAL(15, 8), default=0.0)
    
    # Billing information
    billing_cycle = Column(String(20), default="MONTHLY")  # MONTHLY, QUARTERLY, YEARLY
    next_billing_date = Column(DateTime)
    auto_renew = Column(Boolean, default=True)
    
    # Relationships
    user = relationship("User", back_populates="subscriptions")
    bot = relationship("Bot", back_populates="subscriptions")
    pricing_plan = relationship("BotPricingPlan", back_populates="subscriptions")
    trades = relationship("Trade", back_populates="subscription")
    performance_logs = relationship("PerformanceLog", back_populates="subscription")
    invoices = relationship("SubscriptionInvoice", back_populates="subscription")

class Trade(Base):
    """Table to track individual trades"""
    __tablename__ = "trades"
    id = Column(Integer, primary_key=True)
    subscription_id = Column(Integer, ForeignKey("subscriptions.id"))
    
    # Trade details
    side = Column(String(10))  # "BUY" or "SELL"
    status = Column(Enum(TradeStatus), default=TradeStatus.OPEN)
    entry_price = Column(DECIMAL(15, 8))
    exit_price = Column(DECIMAL(15, 8))
    quantity = Column(DECIMAL(15, 8))
    
    # Risk management
    stop_loss_price = Column(DECIMAL(15, 8))
    take_profit_price = Column(DECIMAL(15, 8))
    
    # Timing
    entry_time = Column(DateTime, server_default=func.now())
    exit_time = Column(DateTime)
    
    # P&L calculation
    pnl = Column(DECIMAL(15, 8))
    pnl_percentage = Column(Float)
    
    # Exchange information
    exchange_order_id = Column(String(100))
    exchange_trade_id = Column(String(100))
    
    # Relationships
    subscription = relationship("Subscription", back_populates="trades")

class BotReview(Base):
    """User reviews for bots"""
    __tablename__ = "bot_reviews"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    bot_id = Column(Integer, ForeignKey("bots.id"))
    rating = Column(Integer)  # 1-5 stars
    review_text = Column(Text)
    created_at = Column(DateTime, server_default=func.now())
    
    # Relationships
    user = relationship("User")
    bot = relationship("Bot", back_populates="reviews")

class BotPerformance(Base):
    """Aggregate performance metrics for bots"""
    __tablename__ = "bot_performance"
    id = Column(Integer, primary_key=True)
    bot_id = Column(Integer, ForeignKey("bots.id"))
    
    # Time period
    period_start = Column(DateTime)
    period_end = Column(DateTime)
    
    # Performance metrics
    total_subscribers = Column(Integer, default=0)
    active_subscribers = Column(Integer, default=0)
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    win_rate = Column(Float, default=0.0)
    average_pnl = Column(DECIMAL(15, 8), default=0.0)
    total_pnl = Column(DECIMAL(15, 8), default=0.0)
    max_drawdown = Column(Float, default=0.0)
    sharpe_ratio = Column(Float)
    
    # Relationships
    bot = relationship("Bot", back_populates="performance_metrics")

class PerformanceLog(Base):
    """Individual performance logs for subscription tracking"""
    __tablename__ = "performance_logs"
    id = Column(Integer, primary_key=True)
    subscription_id = Column(Integer, ForeignKey("subscriptions.id"))
    
    # Log data
    timestamp = Column(DateTime, server_default=func.now())
    action = Column(String(50))  # "BUY", "SELL", "HOLD", "STOP_LOSS", "TAKE_PROFIT"
    price = Column(DECIMAL(15, 8))
    quantity = Column(DECIMAL(15, 8))
    balance = Column(DECIMAL(15, 8))
    
    # Bot signal details
    signal_data = Column(JSON)
    
    # Relationships
    subscription = relationship("Subscription", back_populates="performance_logs")

class Payment(Base):
    """Payment records for subscriptions"""
    __tablename__ = "payments"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    subscription_id = Column(Integer, ForeignKey("subscriptions.id"))
    
    # Payment details
    amount = Column(DECIMAL(10, 2))
    currency = Column(String(10), default="USD")
    payment_method = Column(String(50))
    payment_status = Column(String(50))  # "PENDING", "COMPLETED", "FAILED", "REFUNDED"
    
    # External payment info
    payment_provider = Column(String(50))  # "STRIPE", "PAYPAL", etc.
    external_payment_id = Column(String(255))
    
    # Timing
    created_at = Column(DateTime, server_default=func.now())
    processed_at = Column(DateTime)
    
    # Relationships
    user = relationship("User")
    subscription = relationship("Subscription")

class BotPricingPlan(Base):
    """Pricing plans for bots (Free, Basic, Pro, Enterprise)"""
    __tablename__ = "bot_pricing_plans"
    
    id = Column(Integer, primary_key=True)
    bot_id = Column(Integer, ForeignKey("bots.id"))
    plan_name = Column(String(100))  # "Free", "Basic", "Pro", "Enterprise"
    plan_description = Column(Text)
    
    # Pricing
    price_per_month = Column(DECIMAL(10, 2), default=0.00)
    price_per_year = Column(DECIMAL(10, 2), default=0.00)  # Annual discount
    price_per_quarter = Column(DECIMAL(10, 2), default=0.00)  # Quarterly discount
    
    # Features and limits
    max_trading_pairs = Column(Integer, default=1)  # Number of trading pairs allowed
    max_daily_trades = Column(Integer, default=10)  # Daily trade limit
    max_position_size = Column(DECIMAL(5, 2), default=0.10)  # Max 10% of balance
    advanced_features = Column(JSON)  # Custom features for this plan
    
    # Trial settings
    trial_days = Column(Integer, default=0)  # Free trial period
    trial_trades_limit = Column(Integer, default=5)  # Trial trade limit
    
    # Status
    is_active = Column(Boolean, default=True)
    is_popular = Column(Boolean, default=False)  # Highlight this plan
    
    # Metadata
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    bot = relationship("Bot", back_populates="pricing_plans")
    subscriptions = relationship("Subscription", back_populates="pricing_plan")

class BotPromotion(Base):
    """Promotional offers and discounts for bots"""
    __tablename__ = "bot_promotions"
    
    id = Column(Integer, primary_key=True)
    bot_id = Column(Integer, ForeignKey("bots.id"))
    promotion_code = Column(String(50), unique=True)
    promotion_name = Column(String(100))
    promotion_description = Column(Text)
    
    # Discount settings
    discount_type = Column(String(20))  # "PERCENTAGE", "FIXED_AMOUNT", "FREE_TRIAL"
    discount_value = Column(DECIMAL(10, 2))  # 20.00 for 20% or $20
    max_uses = Column(Integer, default=100)  # Maximum uses
    used_count = Column(Integer, default=0)  # Current usage count
    
    # Validity period
    valid_from = Column(DateTime)
    valid_until = Column(DateTime)
    
    # Eligibility
    min_subscription_months = Column(Integer, default=1)  # Minimum subscription period
    applicable_plans = Column(JSON)  # Which plans this applies to
    
    # Status
    is_active = Column(Boolean, default=True)
    
    # Metadata
    created_at = Column(DateTime, server_default=func.now())
    created_by = Column(Integer, ForeignKey("users.id"))
    
    # Relationships
    bot = relationship("Bot", back_populates="promotions")
    creator = relationship("User")

class SubscriptionInvoice(Base):
    """Detailed invoice tracking for subscriptions"""
    __tablename__ = "subscription_invoices"
    
    id = Column(Integer, primary_key=True)
    subscription_id = Column(Integer, ForeignKey("subscriptions.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    
    # Invoice details
    invoice_number = Column(String(50), unique=True)
    amount = Column(DECIMAL(10, 2))
    currency = Column(String(10), default="USD")
    
    # Pricing breakdown
    base_price = Column(DECIMAL(10, 2))
    discount_amount = Column(DECIMAL(10, 2), default=0.00)
    tax_amount = Column(DECIMAL(10, 2), default=0.00)
    final_amount = Column(DECIMAL(10, 2))
    
    # Billing period
    billing_period_start = Column(DateTime)
    billing_period_end = Column(DateTime)
    
    # Payment status
    status = Column(String(20), default="PENDING")  # PENDING, PAID, OVERDUE, CANCELLED
    payment_method = Column(String(50))
    payment_date = Column(DateTime)
    
    # Promotional info
    promotion_code = Column(String(50))
    promotion_discount = Column(DECIMAL(10, 2), default=0.00)
    
    # Metadata
    created_at = Column(DateTime, server_default=func.now())
    due_date = Column(DateTime)
    
    # Relationships
    subscription = relationship("Subscription", back_populates="invoices")
    user = relationship("User")

class BotDraft(Base):
    """Model for storing bot code drafts"""
    __tablename__ = "bot_drafts"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    bot_code = Column(Text, nullable=False)  # The actual bot code
    category_id = Column(Integer, ForeignKey("bot_categories.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="bot_drafts")
    category = relationship("BotCategory", back_populates="drafts")
    
    class Config:
        orm_mode = True