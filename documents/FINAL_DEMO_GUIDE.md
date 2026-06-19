# Fracture Final Demo Guide

This guide is the clean Phase 6 demo path for Fracture. Use it when presenting
or checking the project from a fresh local clone.

## 1. Demo Goal

Fracture demonstrates process mining for data pipelines.

The demo should answer five questions:

```text
1. What process was expected?
2. What process actually happened?
3. Did actual execution conform to the contract?
4. Where did producer and consumer differ?
5. What should the owner do next?
```

The main demo pipeline is:

```text
trade_positions_sftp
```

It is useful because it shows the flagship issue: the producer can look healthy
while the consumer still waits too long for data.

## 2. Fresh Setup

Run from the project root:

```bash
pip install -r requirements.txt
pip install -e .
```

If Graphviz is missing on macOS:

```bash
brew install graphviz
```

Graphviz is needed for Petri net PNG export.

## 3. Rebuild Demo Data

Generate active demo contracts and local input files:

```bash
python scripts/06_generate_team_contracts.py --clean --days 30
```

Backfill conformance history so drift, prediction, heatmap, and gap trends have
enough rows:

```bash
python scripts/08_backfill_conformance_history.py --start-date 20260519 --end-date 20260618
```

This creates local runtime files:

```text
contracts/
inputs/
conformance_log.csv
pipeline_registry.csv
outputs/
```

These files are generated artifacts and are intentionally ignored by git.

## 4. One Clean Demo Flow

Run the whole demo flow from one command:

```bash
python scripts/09_run_final_demo.py
```

Open the dashboard at the end:

```bash
python scripts/09_run_final_demo.py --with-dashboard
```

If you already generated contracts, inputs, and conformance history, skip setup:

```bash
python scripts/09_run_final_demo.py --skip-setup
```

The script runs the following commands in order:

```bash
python -m fracture.cli status --pipeline-id trade_positions_sftp
python -m fracture.cli discover --pipeline-id trade_positions_sftp --date 20260618
python -m fracture.cli compare --pipeline-id trade_positions_sftp --mode producer-consumer --date 20260618
python -m fracture.cli compare --pipeline-id trade_positions_sftp --mode contract-actual --date 20260618
python -m fracture.cli compare --pipeline-id trade_positions_sftp --mode period
python -m fracture.cli performance --pipeline-id trade_positions_sftp --date 20260618
python -m fracture.cli predict --pipeline-id trade_positions_sftp
python -m fracture.cli recommend --pipeline-id trade_positions_sftp
python -m fracture.cli visualize --pipeline-id trade_positions_sftp --kind all --all-dates
python -m fracture.cli dashboard
```

If Streamlit port `8501` is busy:

```bash
python -m fracture.cli dashboard --port 8502
```

## 5. What Each Command Proves

| Command | Process mining capability | What to say in the demo |
| --- | --- | --- |
| `status` | Conformance summary | Latest health is stored in `conformance_log.csv`. |
| `discover` | Process discovery | Actual event logs produce discovered variants and DFGs. |
| `compare --mode producer-consumer` | Comparative mining | Producer and consumer handoff times differ. |
| `compare --mode contract-actual` | Contract vs actual | Dominant actual path can be checked against the contract path. |
| `compare --mode period` | Temporal comparison | Current period is compared with previous history. |
| `performance` | Performance mining | Slow arcs and execution-time drift are visible. |
| `predict` | Predictive mining | Score and gap trends are projected when history is sufficient. |
| `recommend` | Action-oriented mining | Diagnostics become owner, severity, probable cause, and next command. |
| `visualize` | Static reporting | PNG and JSON artifacts are exported under `outputs/visualizations/`. |
| `dashboard` | Interactive reporting | The same artifacts are available in Streamlit. |

## 6. Expected Main Findings

For `trade_positions_sftp`, the intended demo story is:

```text
Process discovery:
  Dominant actual path matches the expected contract path.

Conformance:
  The producer-side process is broadly conformant.

Comparative mining:
  Producer DATA_AVAILABLE and consumer DATA_AVAILABLE are separated by a large
  bilateral gap.

Performance:
  The handoff and consumer-side availability are the operational bottleneck.

Prediction:
  Gap history is sufficient to identify gap risk and widening-gap behavior.

Recommendation:
  The owner should review producer-consumer handoff, pickup schedule, and
  downstream acknowledgement.
```

## 7. Output Artifacts

After `visualize --kind all --all-dates`, inspect:

```text
outputs/visualizations/trade_positions_sftp/bilateral_gap_timeline.png
outputs/visualizations/trade_positions_sftp/bilateral_gap_analysis.png
outputs/visualizations/trade_positions_sftp/drift_chart.png
outputs/visualizations/trade_positions_sftp/contract_petri_net.png
outputs/visualizations/trade_positions_sftp/discovered_dfg_producer.png
outputs/visualizations/trade_positions_sftp/discovered_dfg_consumer.png
outputs/visualizations/trade_positions_sftp/performance_dfg_producer.png
outputs/visualizations/trade_positions_sftp/performance_dfg_consumer.png
outputs/visualizations/trade_positions_sftp/execution_time_drift_producer.png
outputs/visualizations/trade_positions_sftp/execution_time_drift_consumer.png
outputs/visualizations/trade_positions_sftp/prediction.json
outputs/visualizations/trade_positions_sftp/recommendations.json
outputs/visualizations/fleet_heatmap.png
```

## 8. How To Explain The Dashboard

Use the dashboard in this order:

```text
Fleet Overview
  Show all pipelines, high-gap pipelines, timing zones, and filters.

Pipeline Detail
  Show final score, timing zone, confidence, formula breakdown, and contract.

Visualizations -> Bilateral Gap
  Show the hidden producer-consumer waiting time.

Visualizations -> Drift
  Show final_score trend, risk bands, and changepoints when available.

Visualizations -> Petri Net
  Show the expected formal process model from the contract.

Visualizations -> Discovered DFG
  Show the actual discovered process from event logs.

Visualizations -> Performance DFG
  Show bottleneck arcs and execution-time drift.

Visualizations -> Prediction & Actions
  Show projected risk and recommended action.

Visualizations -> Fleet Heatmap
  Show weekday/fleet patterns.
```

## 9. Final Technical QA

Run these before final submission:

```bash
python tests/test_phase1_core_readiness.py
python tests/test_visualization_data_layer.py
python tests/test_dashboard.py
python tests/test_discovery.py
python tests/test_performance.py
python tests/test_comparison.py
python tests/test_prediction.py
python tests/test_recommendation.py
python tests/test_cli_learning.py
python tests/test_e2e_saturday.py
```

Optional broader regression tests:

```bash
python tests/test_optional_activities.py
python tests/test_analytical_extensions.py
python tests/test_grain_level.py
python tests/test_middle_pipeline.py
python tests/test_hard_sla.py
python tests/test_coverage_gaps.py
python tests/test_sla_breach_scenarios.py
```

The large dataset test is useful but slower:

```bash
python tests/test_citi_large_dataset.py
```

## 10. Current Limitations

V1 intentionally stays local and file-based:

```text
CSV/YAML/parquet storage
Streamlit dashboard
Rule-based recommendation engine
Deterministic prediction
Static PNG exports
```

Future work:

```text
Apache Iceberg storage
RBAC-backed access control
XGBoost/SHAP prediction after enough real historical data exists
Token-replay overlay on Petri net visualization
Cloud deployment and scheduled monitoring
```
