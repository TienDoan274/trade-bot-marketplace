-- Add bot_drafts table for storing developer bot drafts
CREATE TABLE IF NOT EXISTS bot_drafts (
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

-- Add some sample data for testing (optional)
-- INSERT INTO bot_drafts (user_id, name, description, bot_code, category_id) VALUES
-- (1, 'My First Bot', 'A simple RSI bot', 'class MyBot(CustomBot):\n    def execute_algorithm(self):\n        return {"action": "HOLD", "reason": "Testing"}', 1),
-- (1, 'Momentum Bot', 'A momentum-based trading bot', 'class MomentumBot(CustomBot):\n    def execute_algorithm(self):\n        return {"action": "BUY", "reason": "Momentum detected"}', 2); 