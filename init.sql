-- Bot Marketplace Database Initialization Script
-- This script creates all necessary tables and initial data

-- Create database if not exists
CREATE DATABASE IF NOT EXISTS bot_marketplace;
USE bot_marketplace;

-- Drop existing tables if they exist (for clean initialization)
DROP TABLE IF EXISTS subscription_invoices;
DROP TABLE IF EXISTS bot_promotions;
DROP TABLE IF EXISTS bot_pricing_plans;
DROP TABLE IF EXISTS performance_logs;
DROP TABLE IF EXISTS trades;
DROP TABLE IF EXISTS bot_reviews;
DROP TABLE IF EXISTS bot_performance;
DROP TABLE IF EXISTS bot_files;
DROP TABLE IF EXISTS bot_drafts;
DROP TABLE IF EXISTS payments;
DROP TABLE IF EXISTS subscriptions;
DROP TABLE IF EXISTS exchange_credentials;
DROP TABLE IF EXISTS bots;
DROP TABLE IF EXISTS bot_categories;
DROP TABLE IF EXISTS users;

-- Create users table
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    role ENUM('USER', 'DEVELOPER', 'ADMIN') DEFAULT 'USER',
    is_active BOOLEAN DEFAULT TRUE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    -- Developer profile fields
    developer_name VARCHAR(255),
    developer_bio TEXT,
    developer_website VARCHAR(255),
    
    -- API credentials for trading (legacy - now in exchange_credentials)
    api_key VARCHAR(255),
    api_secret VARCHAR(255)
);

-- Create bot_categories table
CREATE TABLE bot_categories (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT
);

-- Create bots table
CREATE TABLE bots (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    developer_id INT NOT NULL,
    category_id INT,
    status ENUM('PENDING', 'APPROVED', 'REJECTED', 'ARCHIVED') DEFAULT 'PENDING',
    
    -- Bot files and metadata
    code_path VARCHAR(500),
    version VARCHAR(50) DEFAULT '1.0.0',
    
    -- Bot type and ML model support
    bot_type VARCHAR(50) DEFAULT 'TECHNICAL',
    model_path VARCHAR(500),
    model_metadata JSON,
    
    -- Legacy pricing (keep for backward compatibility)
    price_per_month DECIMAL(10, 2) DEFAULT 0.00,
    is_free BOOLEAN DEFAULT TRUE,
    
    -- Performance metrics
    total_subscribers INT DEFAULT 0,
    average_rating FLOAT DEFAULT 0.0,
    total_reviews INT DEFAULT 0,
    
    -- Bot configuration schema
    config_schema JSON,
    default_config JSON,
    
    -- Audit fields
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    approved_at DATETIME,
    approved_by INT,
    
    FOREIGN KEY (developer_id) REFERENCES users(id),
    FOREIGN KEY (category_id) REFERENCES bot_categories(id),
    FOREIGN KEY (approved_by) REFERENCES users(id)
);

-- Create bot_categories data
INSERT INTO bot_categories (name, description) VALUES
('Technical Analysis', 'Bots based on technical indicators and chart patterns'),
('Machine Learning', 'AI-powered bots using machine learning algorithms'),
('Arbitrage', 'Bots that exploit price differences across exchanges'),
('Trend Following', 'Bots that follow market trends'),
('Mean Reversion', 'Bots that trade on price reversals'),
('High Frequency', 'Ultra-fast trading bots'),
('Portfolio Management', 'Bots that manage multiple positions');

-- Create bot_pricing_plans table
CREATE TABLE bot_pricing_plans (
    id INT AUTO_INCREMENT PRIMARY KEY,
    bot_id INT NOT NULL,
    plan_name VARCHAR(100) NOT NULL,
    plan_description TEXT,
    price_per_month DECIMAL(10,2) DEFAULT 0.00,
    price_per_year DECIMAL(10,2) DEFAULT 0.00,
    price_per_quarter DECIMAL(10,2) DEFAULT 0.00,
    max_trading_pairs INT DEFAULT 1,
    max_daily_trades INT DEFAULT 10,
    max_position_size DECIMAL(5,2) DEFAULT 0.10,
    advanced_features JSON,
    trial_days INT DEFAULT 0,
    trial_trades_limit INT DEFAULT 5,
    is_active BOOLEAN DEFAULT TRUE,
    is_popular BOOLEAN DEFAULT FALSE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (bot_id) REFERENCES bots(id) ON DELETE CASCADE
);

-- Create bot_promotions table
CREATE TABLE bot_promotions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    bot_id INT NOT NULL,
    promotion_code VARCHAR(50) UNIQUE NOT NULL,
    promotion_name VARCHAR(100) NOT NULL,
    promotion_description TEXT,
    discount_type VARCHAR(20) NOT NULL,
    discount_value DECIMAL(10,2) NOT NULL,
    max_uses INT DEFAULT 100,
    used_count INT DEFAULT 0,
    valid_from DATETIME,
    valid_until DATETIME,
    min_subscription_months INT DEFAULT 1,
    applicable_plans JSON,
    is_active BOOLEAN DEFAULT TRUE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    created_by INT,
    FOREIGN KEY (bot_id) REFERENCES bots(id) ON DELETE CASCADE,
    FOREIGN KEY (created_by) REFERENCES users(id) ON DELETE SET NULL
);

-- Create exchange_credentials table
CREATE TABLE exchange_credentials (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    exchange ENUM('BINANCE', 'COINBASE', 'KRAKEN') NOT NULL,
    api_key VARCHAR(255) NOT NULL,
    api_secret VARCHAR(255) NOT NULL,
    api_passphrase VARCHAR(255),
    is_testnet BOOLEAN DEFAULT TRUE,
    is_active BOOLEAN DEFAULT TRUE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    last_validated DATETIME,
    validation_status VARCHAR(50) DEFAULT 'pending',
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    UNIQUE KEY unique_user_exchange (user_id, exchange, is_testnet)
);

-- Create subscriptions table
CREATE TABLE subscriptions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    instance_name VARCHAR(255) NOT NULL,
    user_id INT NOT NULL,
    bot_id INT NOT NULL,
    pricing_plan_id INT,
    status ENUM('ACTIVE', 'PAUSED', 'CANCELLED', 'EXPIRED', 'ERROR') DEFAULT 'ACTIVE',
    
    -- Testnet and Trial support
    is_testnet BOOLEAN DEFAULT FALSE,
    is_trial BOOLEAN DEFAULT FALSE,
    trial_expires_at DATETIME,
    
    -- Exchange and trading configuration
    exchange_type ENUM('BINANCE', 'COINBASE', 'KRAKEN') DEFAULT 'BINANCE',
    trading_pair VARCHAR(20),
    timeframe VARCHAR(10),
    
    -- Configuration
    strategy_config JSON,
    execution_config JSON,
    risk_config JSON,
    
    -- Subscription management
    started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    expires_at DATETIME,
    last_run_at DATETIME,
    next_run_at DATETIME,
    
    -- Performance tracking
    total_trades INT DEFAULT 0,
    winning_trades INT DEFAULT 0,
    total_pnl DECIMAL(15, 8) DEFAULT 0.0,
    
    -- Billing information
    billing_cycle VARCHAR(20) DEFAULT 'MONTHLY',
    next_billing_date DATETIME,
    auto_renew BOOLEAN DEFAULT TRUE,
    
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (bot_id) REFERENCES bots(id),
    FOREIGN KEY (pricing_plan_id) REFERENCES bot_pricing_plans(id) ON DELETE SET NULL
);

-- Create trades table
CREATE TABLE trades (
    id INT AUTO_INCREMENT PRIMARY KEY,
    subscription_id INT NOT NULL,
    side VARCHAR(10) NOT NULL,
    status ENUM('OPEN', 'CLOSED') DEFAULT 'OPEN',
    entry_price DECIMAL(15, 8),
    exit_price DECIMAL(15, 8),
    quantity DECIMAL(15, 8),
    stop_loss_price DECIMAL(15, 8),
    take_profit_price DECIMAL(15, 8),
    entry_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    exit_time DATETIME,
    pnl DECIMAL(15, 8),
    pnl_percentage FLOAT,
    exchange_order_id VARCHAR(100),
    exchange_trade_id VARCHAR(100),
    FOREIGN KEY (subscription_id) REFERENCES subscriptions(id) ON DELETE CASCADE
);

-- Create bot_reviews table
CREATE TABLE bot_reviews (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    bot_id INT NOT NULL,
    rating INT NOT NULL,
    review_text TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (bot_id) REFERENCES bots(id)
);

-- Create bot_performance table
CREATE TABLE bot_performance (
    id INT AUTO_INCREMENT PRIMARY KEY,
    bot_id INT NOT NULL,
    period_start DATETIME,
    period_end DATETIME,
    total_subscribers INT DEFAULT 0,
    active_subscribers INT DEFAULT 0,
    total_trades INT DEFAULT 0,
    winning_trades INT DEFAULT 0,
    win_rate FLOAT DEFAULT 0.0,
    average_pnl DECIMAL(15, 8) DEFAULT 0.0,
    total_pnl DECIMAL(15, 8) DEFAULT 0.0,
    max_drawdown FLOAT DEFAULT 0.0,
    sharpe_ratio FLOAT,
    FOREIGN KEY (bot_id) REFERENCES bots(id) ON DELETE CASCADE
);

-- Create bot_files table
CREATE TABLE bot_files (
    id INT AUTO_INCREMENT PRIMARY KEY,
    bot_id INT NOT NULL,
    file_type VARCHAR(50),
    file_name VARCHAR(255),
    file_path VARCHAR(500),
    file_size INT,
    file_hash VARCHAR(64),
    version VARCHAR(50) DEFAULT '1.0.0',
    description TEXT,
    model_framework VARCHAR(50),
    model_type VARCHAR(50),
    input_shape JSON,
    output_shape JSON,
    upload_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    FOREIGN KEY (bot_id) REFERENCES bots(id) ON DELETE CASCADE
);

-- Create bot_drafts table
CREATE TABLE bot_drafts (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    bot_code TEXT NOT NULL,
    category_id INT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    -- Foreign key constraints
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (category_id) REFERENCES bot_categories(id) ON DELETE SET NULL,
    
    -- Indexes for better performance
    INDEX idx_user_id (user_id),
    INDEX idx_category_id (category_id),
    INDEX idx_created_at (created_at),
    INDEX idx_updated_at (updated_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Create performance_logs table
CREATE TABLE performance_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    subscription_id INT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    action VARCHAR(50),
    price DECIMAL(15, 8),
    quantity DECIMAL(15, 8),
    balance DECIMAL(15, 8),
    signal_data JSON,
    FOREIGN KEY (subscription_id) REFERENCES subscriptions(id) ON DELETE CASCADE
);

-- Create subscription_invoices table
CREATE TABLE subscription_invoices (
    id INT AUTO_INCREMENT PRIMARY KEY,
    subscription_id INT NOT NULL,
    user_id INT NOT NULL,
    invoice_number VARCHAR(50) UNIQUE NOT NULL,
    amount DECIMAL(10,2) NOT NULL,
    currency VARCHAR(10) DEFAULT 'USD',
    base_price DECIMAL(10,2) NOT NULL,
    discount_amount DECIMAL(10,2) DEFAULT 0.00,
    tax_amount DECIMAL(10,2) DEFAULT 0.00,
    final_amount DECIMAL(10,2) NOT NULL,
    billing_period_start DATETIME,
    billing_period_end DATETIME,
    status VARCHAR(20) DEFAULT 'PENDING',
    payment_method VARCHAR(50),
    payment_date DATETIME,
    promotion_code VARCHAR(50),
    promotion_discount DECIMAL(10,2) DEFAULT 0.00,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    due_date DATETIME,
    FOREIGN KEY (subscription_id) REFERENCES subscriptions(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Create payments table
CREATE TABLE payments (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    subscription_id INT NOT NULL,
    amount DECIMAL(10, 2),
    currency VARCHAR(10) DEFAULT 'USD',
    payment_method VARCHAR(50),
    payment_status VARCHAR(50),
    payment_provider VARCHAR(50),
    external_payment_id VARCHAR(255),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    processed_at DATETIME,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (subscription_id) REFERENCES subscriptions(id)
);

-- Create indexes for better performance
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_bots_developer_id ON bots(developer_id);
CREATE INDEX idx_bots_status ON bots(status);
CREATE INDEX idx_subscriptions_user_id ON subscriptions(user_id);
CREATE INDEX idx_subscriptions_bot_id ON subscriptions(bot_id);
CREATE INDEX idx_subscriptions_status ON subscriptions(status);
CREATE INDEX idx_subscriptions_pricing_plan ON subscriptions(pricing_plan_id);
CREATE INDEX idx_trades_subscription_id ON trades(subscription_id);
CREATE INDEX idx_exchange_credentials_user_id ON exchange_credentials(user_id);
CREATE INDEX idx_bot_pricing_plans_bot_id ON bot_pricing_plans(bot_id);
CREATE INDEX idx_bot_promotions_bot_id ON bot_promotions(bot_id);
CREATE INDEX idx_subscription_invoices_subscription_id ON subscription_invoices(subscription_id);
CREATE INDEX idx_payments_subscription_id ON payments(subscription_id);

-- Insert default admin user
INSERT INTO users (email, hashed_password, role, developer_name) VALUES
('admin@botmarketplace.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj3bp.gS.Oi2', 'ADMIN', 'System Admin');

-- Insert default system user for bot creation
INSERT INTO users (email, hashed_password, role, developer_name) VALUES
('system@botmarketplace.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj3bp.gS.Oi2', 'ADMIN', 'System Bot Creator');

-- Insert sample bots
INSERT INTO bots (name, description, developer_id, category_id, status, bot_type, version, is_free, total_subscribers, average_rating, total_reviews) VALUES
('Golden Cross Strategy', 'Simple moving average crossover strategy for trend following', 2, 1, 'APPROVED', 'TECHNICAL', '1.0.0', TRUE, 0, 0.0, 0),
('RSI Divergence Bot', 'RSI divergence detection for reversal trading', 2, 1, 'APPROVED', 'TECHNICAL', '1.0.0', TRUE, 0, 0.0, 0),
('MACD Signal Bot', 'MACD signal generation for momentum trading', 2, 1, 'APPROVED', 'TECHNICAL', '1.0.0', TRUE, 0, 0.0, 0);

-- Insert default pricing plans for sample bots
INSERT INTO bot_pricing_plans (bot_id, plan_name, plan_description, price_per_month, is_active) VALUES
(1, 'Free Plan', 'Basic trading features with limited functionality', 0.00, TRUE),
(1, 'Pro Plan', 'Advanced features with higher limits', 9.99, TRUE),
(2, 'Free Plan', 'Basic trading features with limited functionality', 0.00, TRUE),
(2, 'Pro Plan', 'Advanced features with higher limits', 9.99, TRUE),
(3, 'Free Plan', 'Basic trading features with limited functionality', 0.00, TRUE),
(3, 'Pro Plan', 'Advanced features with higher limits', 9.99, TRUE);

-- Insert sample exchange credentials for testing
INSERT INTO exchange_credentials (user_id, exchange, api_key, api_secret, is_testnet, is_active, validation_status) VALUES
(1, 'BINANCE', 'test_api_key_1', 'test_api_secret_1', TRUE, TRUE, 'valid'),
(1, 'BINANCE', 'test_api_key_2', 'test_api_secret_2', FALSE, TRUE, 'valid');

-- Insert sample subscriptions
INSERT INTO subscriptions (instance_name, user_id, bot_id, pricing_plan_id, status, is_testnet, exchange_type, trading_pair, timeframe, strategy_config, execution_config, risk_config, started_at, expires_at) VALUES
('My Golden Cross Bot', 1, 1, 1, 'ACTIVE', TRUE, 'BINANCE', 'BTC/USDT', '1h', '{"short_window": 10, "long_window": 30}', '{"allocation_percentage": 5.0}', '{"max_drawdown": 0.1, "stop_loss": 0.05}', NOW(), DATE_ADD(NOW(), INTERVAL 30 DAY)),
('My RSI Bot', 1, 2, 3, 'ACTIVE', TRUE, 'BINANCE', 'ETH/USDT', '4h', '{"rsi_period": 14, "oversold": 30, "overbought": 70}', '{"allocation_percentage": 3.0}', '{"max_drawdown": 0.08, "stop_loss": 0.03}', NOW(), DATE_ADD(NOW(), INTERVAL 30 DAY));

-- Update bot subscriber counts
UPDATE bots SET total_subscribers = (
    SELECT COUNT(*) FROM subscriptions 
    WHERE subscriptions.bot_id = bots.id 
    AND subscriptions.status = 'ACTIVE'
);

-- Create sample performance logs
INSERT INTO performance_logs (subscription_id, action, price, quantity, balance, signal_data) VALUES
(1, 'BUY', 50000.00, 0.001, 1000.00, '{"signal_strength": 0.8, "confidence": 0.75}'),
(1, 'SELL', 51000.00, 0.001, 1010.00, '{"signal_strength": 0.6, "confidence": 0.65}'),
(2, 'BUY', 3000.00, 0.1, 500.00, '{"signal_strength": 0.7, "confidence": 0.70}');

-- Create sample trades
INSERT INTO trades (subscription_id, side, status, entry_price, quantity, entry_time) VALUES
(1, 'BUY', 'OPEN', 50000.00, 0.001, NOW()),
(2, 'BUY', 'OPEN', 3000.00, 0.1, NOW());

-- Update subscription trade counts
UPDATE subscriptions SET total_trades = (
    SELECT COUNT(*) FROM trades 
    WHERE trades.subscription_id = subscriptions.id
);

-- Grant permissions
GRANT ALL PRIVILEGES ON bot_marketplace.* TO 'botuser'@'%';
FLUSH PRIVILEGES;

-- Show final status
SELECT 'Database initialization completed successfully!' as status;
SELECT COUNT(*) as total_users FROM users;
SELECT COUNT(*) as total_bots FROM bots;
SELECT COUNT(*) as total_subscriptions FROM subscriptions; 