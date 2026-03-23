# Locked Project Rules

## 1) Base Model
ใช้ Price Action / Market Structure เป็นแกนหลัก
Indicator ใช้เป็น filter เท่านั้น

### Core Base
- Market Structure
- BOS
- CHOCH
- Swing High / Swing Low
- Pullback Zone

### Filter Layer
- ATR
- ADX
- EMA20 / EMA50

## 2) EMA Research
- Fast EMA = 1..50
- Slow EMA = 20..100
- fast < slow
- ใช้เพื่อหา filter layer ที่ดีที่สุดเท่านั้น
- ห้ามยก EMA เป็น base หลัก

## 3) Dataset
- canonical symbol = XAUUSD
- ทุก timeframe มาจาก M1 source เดียวกัน
- canonical dataset path ต้องอยู่ใน dataset/tf/
- ห้าม strategy ใช้ dataset ภายนอกโดยไม่ผ่าน canonical pipeline

## 4) Validation
ทุก strategy ต้องผ่าน:
1. in-sample
2. out-of-sample
3. walk-forward

## 5) Evaluation
ต้องวัดอย่างน้อย:
- net profit
- profit factor
- expectancy
- max drawdown
- total trades
- win rate
- avg win
- avg loss

ห้ามตัดสินจาก profit อย่างเดียว

## 6) Broker Symbol Mapping
- วิจัย/backtest ใช้ XAUUSD
- execution layer ค่อย map เป็น GOLD / XAUUSDm / XAUUSD ตาม broker
