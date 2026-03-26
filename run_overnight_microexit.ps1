$python = "C:\Users\Teera\AppData\Local\Programs\Python\Python311\python.exe"
$script = "C:\Data\Bot\Local_LLM\gold_research\jobs\run_vectorbt_micro_exit_coverage_batched_v1_0_3.py"
$manifest = "C:\Data\Bot\central_backtest_results\research_coverage_master_v1_0_0\research_coverage_master_manifest.csv"
$dataRoot = "C:\Data\Bot\central_market_data\parquet"
$featureRoot = "C:\Data\Bot\central_feature_cache"
$outdir = "C:\Data\Bot\central_backtest_results\coverage_master_runs_v1_0_0\micro_exit_expansion_batched_v1_0_3\run_overnight"
$log = "C:\Data\Bot\central_backtest_results\coverage_master_runs_v1_0_0\micro_exit_expansion_batched_v1_0_3\run_overnight\overnight.log"

New-Item -ItemType Directory -Force $outdir | Out-Null

& $python $script `
  --manifest $manifest `
  --data-root $dataRoot `
  --feature-root $featureRoot `
  --outdir $outdir `
  --phase micro_exit_expansion `
  --portfolio-chunk-size 24 `
  --preflight-flush-size 5000 `
  --progress-every-groups 1 `
  --continue-on-error *> $log
