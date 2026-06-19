# Fracture

Fracture is a local process-mining tool for data pipelines. It checks whether a
pipeline followed its contracted process, whether it met its timing SLA, whether
the consumer actually received data when the producer said it was available,
and what the owner should do next.

The project is built for a file-based demo workflow:

```text
contracts/*.yaml
inputs/{pipeline_id}/producer_YYYYMMDD.parquet
inputs/{pipeline_id}/consumer_YYYYMMDD.parquet
conformance_log.csv
outputs/visualizations/
dashboard.py
```

## Why This Project

Traditional monitoring answers:

```text
Did the job finish?
Did it finish before a threshold?
Did an error alert fire?
```

Fracture answers process-mining questions:

```text
What was the expected process?
What actually happened?
Where did producer and consumer differ?
Is the process drifting toward risk?
What action should the owner take next?
```

The flagship finding is the bilateral gap:

```text
Producer DATA_AVAILABLE: 06:47
Consumer DATA_AVAILABLE: 07:21
Bilateral gap:          34 minutes
```

The producer can look healthy while the consumer still waits too long. Fracture
measures that hidden handoff delay.

## Quick Start

Use this from the project root in VSCode terminal:

```bash
pip install -r requirements.txt
pip install -e .
```

Check the current demo pipeline:

```bash
python scripts/08_backfill_conformance_history.py --start-date 20260519 --end-date 20260618
python -m fracture.cli status --pipeline-id trade_positions_sftp
```

Run the full Phase 5 demo flow:

```bash
python -m fracture.cli discover --pipeline-id trade_positions_sftp --date 20260618
python -m fracture.cli compare --pipeline-id trade_positions_sftp --mode producer-consumer --date 20260618
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

## Current Demo Result

For `trade_positions_sftp`, the current demo data shows:

```text
Discovery:    dominant actual path matches the contract path
Comparison:   producer-consumer p95 bilateral gap is severe
Performance:  consumer side has large COMPLETED -> DATA_AVAILABLE delay
Prediction:   31 score points and 31 gap points are available
Recommendation: URGENT handoff review, WATCH timing AMBER
```

The important real command is:

```bash
python -m fracture.cli recommend --pipeline-id trade_positions_sftp
```

Expected high-level output:

```text
[URGENT] Bilateral gap is severe at 33.6 minutes.
owner: market-risk@bank.com
next_command: python -m fracture.cli compare --pipeline-id trade_positions_sftp --mode producer-consumer
```

## Process Mining Capabilities

| Capability | Command | What it proves |
| --- | --- | --- |
| Process Discovery | `discover` | Finds actual variants and dominant actual path from event logs |
| Conformance Checking | `run`, `status`, `explain` | Replays logs against the contract-derived Petri net |
| Performance Analysis | `performance` | Computes arc durations, run duration, and bottleneck arcs |
| Comparative Process Mining | `compare` | Compares producer vs consumer, contract vs actual, and current vs previous period |
| Predictive Process Mining | `predict` | Forecasts score decline and widening bilateral gap using deterministic trends |
| Action-Oriented Mining | `recommend` | Converts diagnostics into owner, severity, cause, action, and next command |
| Visualization | `visualize`, `dashboard` | Exports PNGs and opens a Streamlit dashboard |

## Event File Format

Every event file needs four columns:

```text
pipeline_run_id | activity        | timestamp            | team
run_001         | SCHEDULED       | 2026-06-18T06:00:00  | producer
run_001         | STARTED         | 2026-06-18T06:01:00  | producer
run_001         | COMPLETED       | 2026-06-18T06:43:00  | producer
run_001         | DATA_AVAILABLE  | 2026-06-18T06:46:00  | producer
run_001         | DATA_AVAILABLE  | 2026-06-18T07:20:00  | consumer
```

File naming convention:

```text
inputs/{pipeline_id}/producer_YYYYMMDD.parquet
inputs/{pipeline_id}/consumer_YYYYMMDD.parquet
```

Consumer files are optional. Without consumer files, Fracture can still run
producer-side conformance, discovery, and performance analysis, but it cannot
measure the bilateral gap.

## Contract Model

Each pipeline has one YAML contract:

```yaml
pipeline_id: trade_positions_sftp
owner: market-risk@bank.com
producer_team: market-risk-quant
consumer_team: grid-scheduler
expected_start: "06:00"
expected_end: "08:30"
p50_minutes: 41
p95_minutes: 43
p99_minutes: 44
grace_minutes: 30
status: active

log_contract:
  source_path: inputs/trade_positions_sftp/
  required_events:
    - SCHEDULED
    - STARTED
    - COMPLETED
    - DATA_AVAILABLE
  terminal_event: COMPLETED
  upstream_producer_event: DATA_AVAILABLE
  upstream_consumer_event: DATA_AVAILABLE
  optional_activities: []
  activity_name_map: {}
  deduplicate_retries: false
  grain: pipeline
```

The contract is converted into a normative Petri net. Token replay checks
whether actual traces conform to that model.

## Scoring Formula

```text
final_score = sequence_fitness * 0.35
            + timing_score * 0.50
            + completeness_score * 0.15
```

Meaning:

```text
sequence_fitness   Did activities fire in the expected order?
timing_score       Did the run complete within p50/p95/p99/grace zones?
completeness_score Did expected runs finish instead of silently disappearing?
```

Timing zones:

```text
GREEN   before p95
AMBER   between p95 and p99
RED     after p99 but inside grace
BREACH  after p99 + grace
```

## Static Visualization Outputs

Generate all visuals:

```bash
python -m fracture.cli visualize --pipeline-id trade_positions_sftp --kind all --all-dates
```

Expected files:

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

Short interpretation:

| Output | Purpose |
| --- | --- |
| `bilateral_gap_timeline.png` | Shows producer and consumer handoff timestamps per run |
| `bilateral_gap_analysis.png` | Shows gap distribution and widening/narrowing trend |
| `drift_chart.png` | Shows final_score over time, risk zones, trend, and changepoints |
| `contract_petri_net.png` | Shows expected process model derived from the contract |
| `discovered_dfg_*.png` | Shows actual directly-follows graph from event logs |
| `performance_dfg_*.png` | Shows process arcs weighted by timing/frequency |
| `execution_time_drift_*.png` | Shows run-duration trend against contract p95 and deadline |
| `prediction.json` | Stores score/gap forecast results for dashboard/report use |
| `recommendations.json` | Stores owner, severity, probable cause, recommended action, and next command |
| `fleet_heatmap.png` | Shows weekday/fleet patterns across pipelines |

More detail is in `VISUALIZATION_GUIDE.md`.

## Dashboard

Start the dashboard:

```bash
python -m fracture.cli dashboard
```

The dashboard reads:

```text
conformance_log.csv
contracts/
inputs/
outputs/visualizations/
```

Current pages:

```text
Fleet Overview     Fleet health, filters, high-gap pipelines
Pipeline Detail    Latest score, timing zone, formula breakdown, contract summary
Visualizations     Gap, drift, Petri net, DFG, performance, prediction/actions, heatmap views
```

The dashboard can regenerate missing PNG and JSON artifacts from the UI.

## Tests

Phase 5 focused tests:

```bash
python tests/test_discovery.py
python tests/test_performance.py
python tests/test_comparison.py
python tests/test_prediction.py
python tests/test_recommendation.py
```

Visualization and dashboard tests:

```bash
python tests/test_visualization_data_layer.py
python tests/test_dashboard.py
```

Core readiness tests:

```bash
python tests/test_phase1_core_readiness.py
python tests/test_optional_activities.py
python tests/test_analytical_extensions.py
```

Learning walkthrough:

```bash
python tests/test_cli_learning.py
```

## Project Structure

```text
fracture/
  cli.py              CLI commands
  schema.py           Contract schema and validators
  ingest.py           Input loading and event normalization
  conformance.py      Token replay, scoring, diagnostics
  petri.py            Contract to Petri net
  discovery.py        Process discovery and variants
  performance.py      Performance mining and bottlenecks
  comparison.py       Comparative process mining
  prediction.py       Trend-based predictive mining
  recommendation.py   Action-oriented recommendation rules
  visualization.py    Static PNG export helpers
  engine.py           Orchestration layer
  config.py           Runtime config

dashboard.py          Streamlit dashboard
tests/                Executable documentation and regression tests
scripts/              Demo data and setup helpers
documents/            Longer report and research notes
```

## Git Hygiene

Runtime/demo artifacts are ignored:

```text
inputs/
contracts/
outputs/
conformance_log.csv
pipeline_registry.csv
cluster_assignments.csv
```

Commit source code and documentation, not generated local demo data.

## Research Context

Fracture combines:

```text
process discovery
conformance checking
performance analysis
comparative process mining
predictive process mining
action-oriented process mining
```

The project contribution is a bilateral view of pipeline conformance: the
producer and consumer are measured together, so the handoff gap becomes a
first-class process-mining signal.

Business Information Technology, Data Science specialisation.
University of Twente, 2026.
