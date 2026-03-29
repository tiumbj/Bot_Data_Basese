markdown
# Intelligent Database Platform – Comprehensive Design Specification

**Version:** 1.0  
**Date:** 2026-03-29  
**Purpose:** Single source of truth for AI‑assisted development of the gold trading research system.  
**Scope:** Architecture, toolchain, pipeline, decision rules, and experimental results.

---

## 1. System Overview

The Intelligent Database Platform is a **standalone research & backtest system** that:

- Ingests historical and real‑time market data (MT5)
- Builds a **canonical dataset** once per timeframe
- Pre‑computes **base features** (indicators, regimes) once per timeframe
- Executes **large‑scale strategy screening** using vectorized methods
- **Validates promising strategies** with full backtest metrics
- **Promotes** only the strategies that pass strict quality gates
- Provides **centralized data services** (Python SDK + FastAPI) for all trading bots

All components are designed for **long‑running, unattended research campaigns** with checkpoint/resume capability.

---

## 2. Locked Toolchain (Free & Open Source)

| Layer                | Tool(s)                                 | Responsibility                                                                               |
|----------------------|-----------------------------------------|----------------------------------------------------------------------------------------------|
| Data ingestion       | Python, Polars, MetaTrader5 API         | Pull raw OHLCV from MT5, normalize, deduplicate, validate                                    |
| Storage format       | Parquet                                 | Columnar, compressed, fast for analytics                                                    |
| Data quality         | Pandera + custom checks                 | Schema validation, row‑level contract gate, quarantine                                       |
| Analytical DB        | DuckDB                                  | Registry metadata, run manifests, aggregated results                                         |
| Fast processing      | Polars, NumPy, Numba                    | Feature calculation, signal screening, pre‑computed masks                                    |
| Vectorized backtest  | **VectorBT** (open‑source)              | Portfolio simulation, metric calculation (PnL, PF, DD, win rate, expectancy)                |
| Parallel execution   | Ray                                     | Distribute shards, resume, state management                                                  |
| Similar‑case search  | Qdrant                                  | Vector DB for retrieving historical cases similar to real‑time context                       |
| Local AI analysis    | Ollama + DeepSeek‑R1:7B                 | Multi‑TF market summary, entry confirm, exit guardian (optional layer)                       |
| Service layer        | Python SDK, FastAPI                     | Unified access for bots; no direct file/DB access from bots                                  |

---

## 3. Layered Architecture

### 3.1 Data Layer
- **Input:** MT5 symbols (XAUUSD, GOLD, XAUUSDm – mapped at execution layer)
- **Raw storage:** `C:\Data\Bot\central_market_data\raw\` (historical CSV)
- **Canonical parquet:** `C:\Data\Bot\central_market_data\parquet\` – one file per timeframe, normalized schema:
  `time, open, high, low, close, volume`
- **Duplicate prevention & validation:**  
  - Raw batch fingerprint  
  - Row hash  
  - Timestamp uniqueness  
  - Continuity validator  
  - Resample consistency validator  

### 3.2 Feature Cache Layer
- **Builder:** `build_base_feature_cache.py` (v1.0.2)
- **Output:** `C:\Data\Bot\central_feature_cache\XAUUSD_<TF>_base_features.parquet`
- **Features pre‑computed once per timeframe:**
  - EMA 9, 20, 50, 200  
  - ATR 14, ATR% , ATR% SMA 200  
  - ADX 14  
  - RSI 14  
  - Swing high/low (5, 10)  
  - Bull / bear stack  
  - Volatility bucket (LOW/MID/HIGH)  
  - Trend bucket (WEAK/MID/STRONG)  
  - Price location bucket (ABOVE/BELOW/NEAR EMA stack)  
  - Return, range, body, upper/lower wick  

### 3.3 Signal Screening Layer (Fast)
- **Runner:** `run_signal_debug_shard_fast.py` (v2.0.0)
- **Input:** Feature cache (once per shard), manifest subset (e.g., 1000 jobs)
- **Method:** Pre‑compute NumPy boolean masks, then combine per job using `&` and `np.count_nonzero`
- **Output:** `job_debug_rows.jsonl`, `layer_failure_counts.json`, `stage_summary.json`
- **Throughput:** ≈ 7–20 jobs/sec (depends on TF)

### 3.4 Validation & Ranking Layer (VectorBT)
- **Runner:** `run_vectorbt_seed_validation*.py` (family‑specific)
- **Input:** Feature cache (once), manifest subset
- **Method:** Use `vbt.Portfolio.from_signals()` for each job
- **Metrics calculated per job:**
  - Trade count  
  - Total return (%)  
  - Max drawdown (%)  
  - Win rate (%)  
  - Profit factor  
  - Expectancy  
  - Average trade return (%)  
  - Total profit (cash)
- **Output:** `vectorbt_seed_results.csv`, `vectorbt_seed_top20.csv`, `vectorbt_seed_summary.json`
- **Throughput:** ≈ 20–25 jobs/sec (M30)

### 3.5 Promotion Gate
- **Rules (must satisfy all):**
  - `trade_count >= 30`  
  - `profit_factor > 1.10`  
  - `total_return_pct > 0`  
  - `max_drawdown_pct <= 25`  
  - `expectancy > 0`  
- **Score formula used for ranking (example):**
score = profit_factor * 20

total_return_pct * 1

expectancy * 10

win_rate_pct * 0.15

max_drawdown_pct * 1.5

min(150, trade_count) * 0.04

text
- **Promoted jobs** become **production baseline candidates**  
- **Alpha jobs** (high profit but higher risk) are kept as **parallel research tracks**

### 3.6 Central Service Layer
- **Python SDK** – used by bots running on the same machine  
- **FastAPI** – REST API for cross‑language / remote access  
- **Endpoints:** health, market snapshot, active packages, similar cases, trade feedback  
- **Qdrant** – accessed via REST/gRPC for similarity search  
- **No bot may read raw Parquet or DuckDB directly** – only through service layer

---

## 4. Research Pipeline in Practice

### 4.1 Full Discovery Manifest
- **Builder:** `build_intelligent_backtest_master_manifest.py` (v1.0.2)
- **Search space:** 12 TFs × 5 strategy families × 5 entry logics × 4 micro exits × 4 regimes × 4 cooldowns × 3 side policies × 3 volatility filters × 3 trend strength filters = **518,400 jobs**
- **Manifest:** JSONL, written streaming to avoid RAM explosion

### 4.2 Shard Runner
- **Runner:** `run_intelligent_backtest_batch.py` – orchestrates shard execution
- **State:** `job_state.jsonl` (append‑only) allows resume after crash/stop
- **Shard size:** typically 1000 jobs per run

### 4.3 Aggregation
- **Aggregator:** `aggregate_research_results.py` – produces leaderboards and rejection summaries
- **Output:** `leaderboard_done_only.jsonl`, `progress_summary.json`, `reject_summary.json`

---

## 5. Experimental Results (M30, Breakout Family)

| Family / Version             | Mean Trade Count | Mean Profit Factor | Mean Total Return (%) | Mean Max DD (%) | Promoted Jobs | Throughput (jobs/sec) |
|------------------------------|------------------|--------------------|------------------------|-----------------|---------------|------------------------|
| Pullback (M30)               | 7118             | 0.353              | -99.96                 | 99.97           | 0             | 13.41                  |
| Pullback Strict (M30)        | 595              | 0.352              | -50.10                 | 51.45           | 0             | 22.51                  |
| **Breakout v1.0 (M30)**      | 19.7             | 0.630              | -2.01                  | 2.89            | 0             | 23.99                  |
| **Breakout v1.1 (M30)**      | – (similar)      | – (slightly improved)| – (close to zero)      | – (low)         | 0             | –                      |

**Key observation:** Breakout family dramatically reduced overtrading and drawdown, but still lacks positive expectancy. It is the first family that is close to promotion and will be the target for further tuning.

---

## 6. Role of VectorBT

- **VectorBT is used ONLY in the validation/ranking phase** (after fast screening).
- It is **not** used for:
- Data ingestion
- Feature calculation
- Signal screening (NumPy masks)
- Live execution
- Why VectorBT?  
- It provides **fast, vectorized** backtest of many strategies simultaneously.  
- It computes a rich set of metrics (PnL, PF, DD, win rate, expectancy) that are essential for promotion decisions.  
- It is **open‑source and free**.

**Important:** The main bottleneck of the old system was **per‑job Python loops**.  
The new architecture separates concerns:
1. **Feature cache** (once per TF)
2. **NumPy screening** (fast, filters out dead strategies)
3. **VectorBT validation** (heavy but applied only to surviving candidates)

---

## 7. Current Project Status & Next Steps

### 7.1 Completed
- [x] Canonical parquet datasets for all 12 timeframes
- [x] Base feature cache (Polars + Numba) – 15.9 million rows in 9 seconds
- [x] Fast screening runner (NumPy masks) – 7–20 jobs/sec
- [x] VectorBT validation runners (pullback, strict, breakout families)
- [x] Promotion gate implementation (Gate 5)
- [x] Resume-safe execution runner (Gate 4 acceptance test passed)
- [x] Central service design (SDK + FastAPI + Qdrant)

### 7.2 Next Steps (Locked Gate Sequence)
The project follows a strict gate sequence. Current position:

**Gate 1–5: CLOSED ✅**
**Gate 6 (Feature/Label Runtime): NEXT TARGET**
**Gate 7 (Intelligent DB): NOT STARTED**
**Gate 8 (Downstream Bot): NOT STARTED**

#### Gate 6 – Feature/Label Runtime Integration
Gate 6 must satisfy:
- Shared features are queryable by job_id + timeframe
- Regime/label generation is recurring and reproducible
- Trade outcome labels are attached to source intelligence

Required deliverables:
- [ ] Feature store query interface (by job_id + timeframe)
- [ ] Regime label generation logic (reproducible, recurring)
- [ ] Trade outcome label attachment to source intelligence
- [ ] Feature cache versioning and invalidation strategy
- [ ] Label schema lock in operational form

#### Gate 7 – Intelligent Database Deployment
Prerequisites: Gate 6 must be closed first.

Required deliverables:
- [ ] Retrieval contract exists
- [ ] Human and bot consume same truth
- [ ] Execution feedback can be reattached to source intelligence
- [ ] DuckDB layer for aggregated queries
- [ ] Qdrant for similar-case retrieval

#### After Gates 6 & 7 – Live Bot Integration
- [ ] FastAPI service layer operational
- [ ] Python SDK for bot access
- [ ] MT5 closed-loop feedback integration
- [ ] Local LLM (DeepSeek-R1) for real-time entry confirmation and exit guardian

**Note:** Breakout Family v1.1 tuning (PF > 1.10, positive return) remains a valid research task but is not the gate-critical path. It can proceed in parallel after Gate 6 is closed.

---

## 8. Rules for AI‑Assisted Development

When extending the system, **strictly follow**:

1. **Never change the canonical data layer** without updating the feature cache and all downstream scripts.
2. **Always use the shared logic core** for indicator, entry, exit, and regime definitions.
3. **Keep VectorBT only in validation/ranking phase** – do not use it in fast screening or live code.
4. **Do not bypass the service layer** – bots must use SDK or REST API.
5. **Respect the promotion gate** – no manual override unless documented.
6. **When adding a new strategy family**, create a dedicated VectorBT runner (like `run_vectorbt_seed_validation_breakout_family.py`).
7. **Always include progress logging** in long‑running scripts to avoid “frozen” appearance.
8. **Document any new tuning parameter** in the README or this specification.

---

## 9. File & Folder Structure (Simplified)
C:\Data\Bot
├── central_market_data\                # raw and canonical OHLC (Gate 1)
│   ├── raw\                            # original CSV (kept for reference)
│   └── parquet\                        # XAUUSD_.parquet (canonical)
├── central_feature_cache\               # XAUUSD_base_features.parquet (Gate 6)
├── central_backtest_results\
│   ├── research_evidence_truth_gate_v1_0_0\   # Gate 2 evidence
│   ├── truth_gate_seed_manifest_v1_0_0\        # Gate 3 evidence
│   ├── gate4_*\                        # Gate 4 execution evidence
│   ├── winner_registry\                # Gate 5 promotion output
│   │   ├── registry_candidates.csv
│   │   ├── active_winner_registry.csv
│   │   ├── database_ready_strategy_packages.csv
│   │   ├── promotion_criteria_v1_0_0.json
│   │   ├── promotion_audit_log.csv
│   │   └── gate5_acceptance_summary.json
│   ├── research_jobs_full_discovery
│   │   ├── research_job_manifest_full_discovery.jsonl
│   │   ├── research_job_manifest_full_discovery_summary.json
│   │   └── debug_manifest.jsonl
│   ├── signal_debug\                   # outputs of fast screening
│   ├── vectorbt_seed_validation\        # results of VectorBT validation
│   ├── research_reports\                # aggregated leaderboards
│   └── research_shards\                 # per‑shard job_specs, logs, results
├── central_strategy_registry\            # candidate & approved strategy packages
│   ├── candidates
│   └── approved
├── gold_research\                      # all Python jobs, scripts, configs
│   ├── jobs
│   │   ├── build_manifest.py
│   │   ├── runshard.py
│   │   ├── run_vectorbt_.py
│   │   └── summarize_.py
│   ├── core_logic\                     # shared logic (indicators, entry, exit, regimes)
│   └── configs\                         # YAML manifests for search spaces
└── Local_LLM\                          # (optional) older research work, kept for reference

text

---

## 10. Conclusion

The Intelligent Database Platform is a **production‑grade research system** that enables exhaustive strategy discovery with minimal human supervision. It achieves **speed through vectorization** (Polars, NumPy, Numba, VectorBT) and **reliability through strict data contracts, checkpointing, and promotion gates**.

The platform follows a strict gate sequence (Gate 1–5 CLOSED ✅, **Gate 6 NEXT TARGET**, Gate 7–8 NOT STARTED). All new work must move an OPEN gate toward CLOSED before proceeding.

**VectorBT** is a **critical component** in the validation phase, but it is **one piece of a larger pipeline** that also includes fast screening, feature caching, and centralised services. Future work will focus on **tuning the breakout family to reach positive expectancy** and **integrating the local LLM** for real‑time decision support.

*This document serves as the master reference for AI‑assisted development. All code changes should align with the principles and constraints defined above.*