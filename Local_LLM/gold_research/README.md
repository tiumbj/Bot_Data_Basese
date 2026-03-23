# gold_research

Version: v1.0.0

## Purpose
โปรเจกต์วิจัย/backtest กลางสำหรับ XAUUSD/GOLD โดยยึด Locked Model ที่ตกลงกันไว้

## Locked Base
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

## Locked EMA Research Rule
- Fast EMA: 1 ถึง 50
- Slow EMA: 20 ถึง 100
- fast < slow
- EMA ใช้เป็น filter only

## Locked Dataset Rule
- canonical symbol = XAUUSD
- dataset ทุก TF มาจาก M1 source เดียวกัน
- backtest ทุกตัวต้องใช้ข้อมูลจาก `dataset/tf/` เท่านั้น

## Locked Validation Rule
- in-sample
- out-of-sample
- walk-forward

## Locked Evaluation Rule
- net profit
- profit factor
- expectancy
- max drawdown
- total trades
- win rate
- avg win
- avg loss

## Folder Root
`C:\Data\Bot\Local_LLM\gold_research`
