# Resume-Safe Execution Runner v1.0.0

Purpose: close Gate 4 only.

This package contains:

1. `execution_state_store_v1_0_0.py`
2. `standardized_result_pack_v1_0_0.py`
3. `resume_safe_execution_runner_v1_0_0.py`
4. `acceptance_test_resume_safe_v1_0_0.py`

## What this package does
- same-outdir restart
- persistent execution state
- skip completed work
- no duplicate outputs
- explicit status / reject / error / missing-feature fields
- recovery metrics
- deterministic continuation

## Intended integration
This is a production-oriented Class 2 execution/backtest runner support layer for Gate 4.
It is not a final trading engine and does not replace registry or Intelligent DB.

## How to integrate
Step 1:
Place these files under your project, for example:

- `C:\Data\Bot\Local_LLM\gold_research\jobs\execution_state_store_v1_0_0.py`
- `C:\Data\Bot\Local_LLM\gold_research\jobs\standardized_result_pack_v1_0_0.py`
- `C:\Data\Bot\Local_LLM\gold_research\jobs\resume_safe_execution_runner_v1_0_0.py`
- `C:\Data\Bot\Local_LLM\gold_research\jobs\acceptance_test_resume_safe_v1_0_0.py`

Step 2:
Connect your real execution logic by implementing a callable:
`module:function`

Example:
`my_real_executor:run_job`

The callable must receive one dict job payload and return a dict like:

```python
{
    "status": "success",
    "metrics": {"pnl_sum": 12.3, "trade_count": 42},
    "reject_reason": "",
    "error_reason": "",
    "missing_feature_reason": "",
}
```

Step 3:
Run the promoted runner with the same outdir on restart.

## Acceptance rule
Use:
`acceptance_test_resume_safe_v1_0_0.py`

It simulates:
- partial run
- intentional stop
- restart with same outdir
- verification that progress resumes, completed jobs are skipped, and results are not duplicated

If acceptance test fails, Gate 4 remains OPEN.
