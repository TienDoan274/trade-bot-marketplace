# 🚀 Developer Sandbox Guide

## 📋 Mục lục
- [Truy cập Sandbox](#truy-cập-sandbox)
- [Giao diện chính](#giao-diện-chính)
- [Tính năng chính](#tính-năng-chính)
- [Bot Control Settings](#bot-control-settings)
- [Template Bots](#template-bots)
- [Testing & Debugging](#testing--debugging)
- [Draft Management](#draft-management)
- [Bot Submission](#bot-submission)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Truy cập Sandbox

### URL chính
```
http://localhost:8000/static/dev_sandbox_v2.html
```

### Yêu cầu
- **Đăng nhập**: Phải có tài khoản Developer
- **Environment**: `DEVELOPMENT_MODE=true` trong file `.env`
- **Database**: MySQL với bảng `bot_drafts` đã được tạo

### Đăng nhập
1. Truy cập: `http://localhost:8000/static/login.html`
2. Nhập email và password của Developer
3. Hệ thống sẽ lưu `user_id` và `access_token`
4. Chuyển hướng đến Sandbox

---

## 🖥️ Giao diện chính

### Navigation Bar
- **Dev Sandbox**: Trang chính sandbox
- **Drafts**: Quản lý bản nháp
- **Submit Bot**: Gửi bot lên marketplace
- **GitHub**: Link repository

### Main Sections
1. **Code Editor**: Viết và chỉnh sửa bot code
2. **Bot Control Settings**: Cấu hình bot
3. **Testing Panel**: Quick Test, Backtest, Live Test
4. **Results Display**: Hiển thị kết quả test

---

## ⚙️ Tính năng chính

### 1. Code Editor
- **Syntax Highlighting**: Hỗ trợ Python
- **Auto-completion**: Gợi ý code
- **Error Detection**: Phát hiện lỗi syntax
- **Line Numbers**: Đánh số dòng

### 2. Template Bots
```python
# Load template bot
- RSI Bot: Chiến lược RSI cơ bản
- Bollinger Bands Bot: Chiến lược Bollinger Bands
- Advanced Momentum Bot: Chiến lược momentum nâng cao
```

### 3. Quick Test
- **Kiểm tra syntax**: Phát hiện lỗi cú pháp
- **Validate bot structure**: Kiểm tra cấu trúc bot
- **Test basic execution**: Chạy thử nghiệm cơ bản
- **Config schema**: Tạo form cấu hình động

### 4. Backtest
- **Historical data**: Dữ liệu lịch sử
- **Performance metrics**: Chỉ số hiệu suất
- **Trade history**: Lịch sử giao dịch
- **Visualization**: Biểu đồ kết quả

### 5. Live Test
- **Real-time simulation**: Mô phỏng thời gian thực
- **WebSocket connection**: Kết nối real-time
- **Live charts**: Biểu đồ trực tiếp
- **Trade signals**: Tín hiệu giao dịch

---

## 🎛️ Bot Control Settings

### Data Fetch Timeframe
```
1s, 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
```
- **Mô tả**: Chu kỳ lấy dữ liệu thị trường
- **Mặc định**: 5m
- **Ảnh hưởng**: Tần suất cập nhật dữ liệu

### Prediction Cycle
```
1s, 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
```
- **Mô tả**: Chu kỳ đưa ra quyết định giao dịch
- **Mặc định**: 5m
- **Ảnh hưởng**: Tần suất thực hiện giao dịch

### Data Fetch Limit
- **Mô tả**: Số lượng nến dữ liệu lấy về
- **Range**: 50 - 1000
- **Mặc định**: 100
- **Ảnh hưởng**: Độ sâu dữ liệu phân tích

### Prediction Cycle Configurable
- **Yes**: Người dùng có thể thay đổi
- **No**: Cố định theo developer
- **Mặc định**: Yes

### Validation Rules
```
Prediction Cycle ≥ Data Fetch Timeframe
```
- **Cảnh báo**: Hiển thị khi không hợp lý
- **Gợi ý**: Sử dụng PC ≥ DFT để tối ưu

---

## 🤖 Template Bots

### RSI Bot
```python
"""
RSI Trading Bot using new prediction_cycle logic
This bot demonstrates RSI-based trading with flexible data fetching
"""

class RSIBot(CustomBot):
    def __init__(self):
        super().__init__()
        self.name = "RSI Trading Bot"
        self.description = "RSI-based trading strategy"
        
    def get_configuration_schema(self):
        return {
            "rsi_period": {"type": "int", "default": 14, "min": 5, "max": 50},
            "oversold_threshold": {"type": "float", "default": 30, "min": 10, "max": 40},
            "overbought_threshold": {"type": "float", "default": 70, "min": 60, "max": 90}
        }
```

### Bollinger Bands Bot
```python
class BollingerBandsBot(CustomBot):
    def __init__(self):
        super().__init__()
        self.name = "Bollinger Bands Bot"
        self.description = "Bollinger Bands strategy"
        
    def get_configuration_schema(self):
        return {
            "bb_period": {"type": "int", "default": 20, "min": 10, "max": 50},
            "bb_std": {"type": "float", "default": 2.0, "min": 1.0, "max": 3.0}
        }
```

### Advanced Momentum Bot
```python
class AdvancedMomentumBot(CustomBot):
    def __init__(self):
        super().__init__()
        self.name = "Advanced Momentum Bot"
        self.description = "Multi-indicator momentum strategy"
        
    def get_configuration_schema(self):
        return {
            "macd_fast": {"type": "int", "default": 12, "min": 5, "max": 20},
            "macd_slow": {"type": "int", "default": 26, "min": 20, "max": 50},
            "macd_signal": {"type": "int", "default": 9, "min": 5, "max": 15}
        }
```

---

## 🧪 Testing & Debugging

### Quick Test Process
1. **Syntax Check**: Kiểm tra cú pháp Python
2. **Import Validation**: Kiểm tra import modules
3. **Class Detection**: Tìm class kế thừa CustomBot
4. **Method Validation**: Kiểm tra các method bắt buộc
5. **Config Schema**: Tạo form cấu hình
6. **Basic Execution**: Chạy thử nghiệm cơ bản

### Error Reporting
```python
# Syntax Error
❌ Syntax Error: invalid syntax (bot.py, line 15)
💡 Suggestion: Check parentheses, brackets, and indentation

# Import Error
❌ Import Error: No module named 'pandas'
💡 Suggestion: Add required imports

# Runtime Error
❌ Runtime Error: 'Bot' object has no attribute 'execute_algorithm'
💡 Suggestion: Implement required methods
```

### Backtest Configuration
```python
# Exchange Settings
- Exchange: BINANCE
- Symbol: BTC/USDT
- Timeframe: 5m
- Initial Balance: 10000 USDT
- Fee Rate: 0.001

# Position Sizing
- Type: PERCENT_BALANCE
- Value: 10%
```

### Live Test Features
- **Real-time data**: Dữ liệu thị trường thời gian thực
- **Signal generation**: Tạo tín hiệu giao dịch
- **Trade execution**: Thực hiện giao dịch mô phỏng
- **Performance tracking**: Theo dõi hiệu suất
- **Visual feedback**: Phản hồi trực quan

---

## 📁 Draft Management

### Save Draft
1. **Nhập tên**: Tên bot draft
2. **Mô tả**: Mô tả ngắn gọn
3. **Code**: Bot code hoàn chỉnh
4. **Category**: Phân loại bot (tùy chọn)
5. **Click Save**: Lưu vào database

### Load Draft
1. **Truy cập**: `/static/drafts.html`
2. **Chọn draft**: Click vào draft muốn load
3. **Open in Sandbox**: Mở trong sandbox
4. **Edit**: Chỉnh sửa code
5. **Save**: Lưu thay đổi

### Draft Operations
- **View**: Xem danh sách drafts
- **Edit**: Chỉnh sửa draft
- **Delete**: Xóa draft
- **Submit**: Gửi lên marketplace
- **Duplicate**: Tạo bản sao

---

## 📤 Bot Submission

### Submit from Sandbox
1. **Test bot**: Chạy Quick Test thành công
2. **Click Submit**: Nút Submit trong sandbox
3. **Fill form**: Điền thông tin bot
4. **Upload**: Gửi lên marketplace

### Submit from Draft
1. **Truy cập**: `/static/drafts.html`
2. **Chọn draft**: Draft muốn submit
3. **Click Submit**: Chuyển đến trang submit
4. **Complete form**: Hoàn thành form
5. **Submit**: Gửi bot

### Submission Form
```python
# Required Fields
- Bot Name: Tên bot
- Description: Mô tả chi tiết
- Category: Phân loại
- Pricing: Giá thuê
- Features: Tính năng nổi bật

# Optional Fields
- Documentation: Tài liệu hướng dẫn
- Screenshots: Hình ảnh demo
- Video: Video demo
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. 404 Not Found
```
Error: Failed to load resource: 404 Not Found
Solution: 
- Kiểm tra DEVELOPMENT_MODE=true
- Restart server
- Check router registration
```

#### 2. Authentication Error
```
Error: 401 Unauthorized
Solution:
- Login lại
- Check access_token
- Verify user role
```

#### 3. Import Error
```
Error: ModuleNotFoundError
Solution:
- Check required imports
- Install missing packages
- Verify module paths
```

#### 4. Syntax Error
```
Error: SyntaxError
Solution:
- Check Python syntax
- Verify indentation
- Check parentheses/brackets
```

#### 5. Draft Not Loading
```
Error: Drafts not showing
Solution:
- Check user_id in localStorage
- Verify database connection
- Check user permissions
```

### Debug Tools

#### Console Logs
```javascript
// Check authentication
console.log('Token:', localStorage.getItem('access_token'));
console.log('User ID:', localStorage.getItem('user_id'));

// Test API endpoints
fetch('/api/dev-sandbox/debug-user')
  .then(r => r.json())
  .then(console.log);
```

#### API Endpoints
```
GET /api/dev-sandbox/debug-user
GET /api/dev-sandbox/test-token
GET /api/dev-sandbox/decode-token
GET /api/dev-sandbox/list-saved-bots
```

#### Database Queries
```sql
-- Check user drafts
SELECT * FROM bot_drafts WHERE user_id = 4;

-- Check user info
SELECT * FROM users WHERE role = 'DEVELOPER';
```

---

## 📚 Best Practices

### Code Structure
```python
class MyBot(CustomBot):
    def __init__(self):
        super().__init__()
        self.name = "My Trading Bot"
        self.description = "Description here"
    
    def get_configuration_schema(self):
        return {
            "param1": {"type": "int", "default": 10, "min": 1, "max": 100},
            "param2": {"type": "float", "default": 0.5, "min": 0.1, "max": 1.0}
        }
    
    def execute_algorithm(self, market_data, timeframe=None, config=None):
        # Your trading logic here
        return {"action": "BUY", "reason": "Signal detected"}
```

### Testing Strategy
1. **Start with Quick Test**: Kiểm tra cơ bản
2. **Run Backtest**: Test với dữ liệu lịch sử
3. **Live Test**: Mô phỏng thời gian thực
4. **Save Draft**: Lưu phiên bản ổn định
5. **Submit**: Gửi lên marketplace

### Performance Optimization
- **Efficient data processing**: Xử lý dữ liệu hiệu quả
- **Memory management**: Quản lý bộ nhớ
- **Error handling**: Xử lý lỗi tốt
- **Logging**: Ghi log chi tiết

---

## 🎯 Next Steps

### Development Roadmap
- [ ] Advanced charting tools
- [ ] Multi-timeframe analysis
- [ ] Risk management features
- [ ] Performance analytics
- [ ] Community features

### Support
- **Documentation**: [GitHub Wiki](https://github.com/TienDoan274/trade-bot-marketplace)
- **Issues**: [GitHub Issues](https://github.com/TienDoan274/trade-bot-marketplace/issues)
- **Discussions**: [GitHub Discussions](https://github.com/TienDoan274/trade-bot-marketplace/discussions)

---

*Last updated: August 2025*
*Version: 2.0.0* 