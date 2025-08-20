# 🤖 Bot Marketplace

Advanced Trading Bot Platform

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/TienDoan274/trade-bot-marketplace
cd bot_marketplace
cp .env.example .env
nano .env  # Configure required settings
```

### 2. Configure Required Settings

#### A. Generate Secret Key

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

#### B. Configure Email (Choose One)

**SendGrid (Recommended):**

```
SENDGRID_API_KEY=your-sendgrid-api-key
FROM_EMAIL=noreply@yourdomain.com
```

**Gmail SMTP:**

```
GMAIL_USER=your-gmail@gmail.com
GMAIL_PASSWORD=your-app-password
```

#### C. AWS S3 (Optional - for bot storage)

```
AWS_ACCESS_KEY_ID=your-aws-key
AWS_SECRET_ACCESS_KEY=your-aws-secret
AWS_REGION=us-east-1
S3_BUCKET_NAME=your-bucket
```

### 3. Start Application

```bash
docker-compose up -d
```

### 4. Access Application

* **GitHub Repository**: https://github.com/TienDoan274/trade-bot-marketplace
* **Health Check**: http://localhost:8000/health
* **Developer Sandbox**: http://localhost:8000/static/dev_sandbox_v2.html

## 📧 Email Setup

### SendGrid

1. Create account at sendgrid.com
2. Generate API key
3. Verify sender email
4. Add to `.env`:  
   ```
   SENDGRID_API_KEY=your-api-key  
   FROM_EMAIL=verified@yourdomain.com
   ```

### Gmail

1. Enable 2FA
2. Generate app password
3. Add to `.env`:  
   ```
   GMAIL_USER=your-gmail@gmail.com  
   GMAIL_PASSWORD=your-app-password
   ```

## 🔧 Troubleshooting

### Reset Database

```bash
./reset_database.sh
```

### View Logs

```bash
docker-compose logs -f
```

### Common Issues

* **Port in use**: Change `API_PORT` in `.env`
* **Email not working**: Check API key/credentials
* **Database error**: Run `./reset_database.sh`

## 📚 Documentation

* **GitHub Repository**: https://github.com/TienDoan274/trade-bot-marketplace
* **Detailed Setup**: See `SETUP_GUIDE.md`

## 🤖 Bot Development

### Simple Actions

For beginners, use `SimpleAction` with basic order types:

```python
from bots.bot_sdk.Action import SimpleAction, AmountType

# Market buy with 10% allocation
action = SimpleAction.buy(
    amount_type=AmountType.PERCENTAGE,
    value=10.0,
    reason="RSI oversold signal"
)

# Market sell with 5% allocation
action = SimpleAction.sell(
    amount_type=AmountType.PERCENTAGE,
    value=5.0,
    reason="Take profit"
)
```

### Advanced Actions

For experienced developers, use `AdvancedAction` with various order types:

```python
from bots.bot_sdk.Action import AdvancedAction, AmountType, OrderType

# Limit buy order
action = AdvancedAction.limit_buy(
    price=50000.0,
    amount_type=AmountType.PERCENTAGE,
    value=15.0,
    reason="Limit buy at support level"
)

# Stop loss order
action = AdvancedAction.stop_loss(
    stop_price=48000.0,
    amount_type=AmountType.PERCENTAGE,
    value=100.0,  # Sell all position
    reason="Stop loss triggered"
)

# Take profit order
action = AdvancedAction.take_profit(
    take_profit_price=55000.0,
    amount_type=AmountType.PERCENTAGE,
    value=50.0,  # Sell half position
    reason="Take profit at resistance"
)
```

### Amount Types

- `BASE_AMOUNT`: Amount in base currency (BTC)
- `QUOTE_AMOUNT`: Amount in quote currency (USDT)
- `PERCENTAGE`: Percentage of available balance
- `ALL`: Use all available balance

### Order Types

- `MARKET`: Market order (immediate execution)
- `LIMIT`: Limit order (specified price)
- `STOP_LOSS`: Stop loss order
- `STOP_LOSS_LIMIT`: Stop loss limit order
- `TAKE_PROFIT`: Take profit order
- `TAKE_PROFIT_LIMIT`: Take profit limit order
- `LIMIT_MAKER`: Limit maker order (post-only)

## About

Advanced trading bot platform with support for both simple and advanced trading strategies. Perfect for developers learning algorithmic trading or building sophisticated trading systems.
