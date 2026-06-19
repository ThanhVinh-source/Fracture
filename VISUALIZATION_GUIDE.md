# Fracture Visualization Guide

This guide explains the static PNG exports and dashboard views used in the
Fracture demo.

## Generate Visuals

Generate every visualization for the main demo pipeline:

```bash
python -m fracture.cli visualize --pipeline-id trade_positions_sftp --kind all --all-dates
```

Generate one visual family:

```bash
python -m fracture.cli visualize --pipeline-id trade_positions_sftp --kind gap --date 20260618
python -m fracture.cli visualize --pipeline-id trade_positions_sftp --kind drift --date 20260618
python -m fracture.cli visualize --pipeline-id trade_positions_sftp --kind petri --date 20260618
python -m fracture.cli visualize --pipeline-id trade_positions_sftp --kind dfg --date 20260618
python -m fracture.cli visualize --pipeline-id trade_positions_sftp --kind performance --date 20260618
python -m fracture.cli visualize --kind heatmap
python -m fracture.cli predict --pipeline-id trade_positions_sftp
python -m fracture.cli recommend --pipeline-id trade_positions_sftp
```

Output directory:

```text
outputs/visualizations/
outputs/visualizations/{pipeline_id}/
```

## Output Files

### `bilateral_gap_timeline.png`

Purpose:

```text
Shows producer DATA_AVAILABLE and consumer DATA_AVAILABLE timestamps.
```

What it answers:

```text
How long did the consumer wait after the producer said data was available?
```

How to read:

```text
Blue marker  = producer handoff timestamp
Red marker   = consumer handoff timestamp
Line/gap     = waiting time between producer and consumer
Large gap    = downstream team receives data later than producer believes
```

Why it matters:

```text
This is the flagship bilateral process-mining view.
The producer can be GREEN while the consumer still waits too long.
```

### `bilateral_gap_analysis.png`

Purpose:

```text
Shows bilateral gap distribution and trend.
```

What it answers:

```text
Is the handoff gap stable, widening, or narrowing?
```

How to read:

```text
Left panel   = gap per run
Mean line    = average consumer delay
p95 line     = tail delay experienced on worst runs
Right panel  = trend over chronological runs
Positive slope = widening handoff gap
```

Related CSV diagnostic:

```text
gap_drift_per_day = day-level slope of bilateral_gap_minutes
```

This value is computed from historical bilateral gaps in `conformance_log.csv`
plus the current run gap. It is useful when a gap is not just large, but getting
worse over time.

For `trade_positions_sftp`, this view should show severe handoff delay.

### `drift_chart.png`

Purpose:

```text
Shows final_score history over time.
```

What it answers:

```text
Is the pipeline drifting toward lower conformance?
```

How to read:

```text
Blue points/line = final_score over time
Green band       = healthy final_score zone
Amber band       = warning final_score zone
Red band         = weak final_score zone
Vertical marker  = changepoint, if detected
Trend note       = projected risk if history is meaningful
```

Important distinction:

```text
The drift chart color bands are final_score bands.
They are not the same as timing_zone.
```

Example:

```text
Pipeline can have final_score in GREEN while timing_zone is AMBER.
That means the overall weighted score is still healthy,
but the latest runtime is approaching the SLA boundary.
```

### `contract_petri_net.png`

Purpose:

```text
Shows the expected process model derived from the contract.
```

What it answers:

```text
What process does Fracture measure against?
```

How to read:

```text
Rounded boxes = contract activities
Circles       = Petri net places
Start/end     = initial and final marking
Tau/silent    = silent transition used by Petri net construction
Orange border = optional activity, when configured
```

Why it matters:

```text
This is not a decorative flowchart.
It is the formal process model used by token replay.
```

### `discovered_dfg_producer.png` and `discovered_dfg_consumer.png`

Purpose:

```text
Shows the actual directly-follows graph discovered from event logs.
```

What it answers:

```text
What process actually happened in the logs?
```

How to read:

```text
Nodes = observed activities
Arcs  = observed directly-follows relationships
Arc label/count = how often one activity followed another
```

Use with:

```bash
python -m fracture.cli discover --pipeline-id trade_positions_sftp --date 20260618
python -m fracture.cli compare --pipeline-id trade_positions_sftp --mode contract-actual --date 20260618
```

### `performance_dfg_producer.png` and `performance_dfg_consumer.png`

Purpose:

```text
Shows process arcs with performance timing.
```

What it answers:

```text
Which part of the process is slow?
```

How to read:

```text
Nodes = activities
Arcs  = consecutive activity pairs
Arc labels = mean/p95 duration depending on export
Thicker or highlighted arcs = more important or slower arcs
```

Use with:

```bash
python -m fracture.cli performance --pipeline-id trade_positions_sftp --date 20260618
```

### `execution_time_drift_producer.png` and `execution_time_drift_consumer.png`

Purpose:

```text
Shows runtime duration over historical runs.
```

What it answers:

```text
Is execution time drifting toward the SLA deadline?
```

How to read:

```text
Points       = actual run duration
Trend line   = duration trend
p95 line     = contract p95
Deadline     = p99 + grace
Breach marker = projected breach point, if trend reaches deadline
```

For `trade_positions_sftp`, consumer execution can look slower because the
consumer-side `DATA_AVAILABLE` includes the handoff wait.

### `fleet_heatmap.png`

Purpose:

```text
Shows score patterns across pipelines and weekdays.
```

What it answers:

```text
Are failures clustered on certain days or across certain pipelines?
```

How to read:

```text
Rows    = pipelines
Columns = weekdays
Color   = mean final_score
Grey    = no data for that weekday
```

Why grey is not red:

```text
Missing days are unknown, not failures.
Fracture avoids turning missing data into false red alerts.
```

### `prediction.json`

Purpose:

```text
Stores deterministic prediction results for the dashboard.
```

What it answers:

```text
Is the score or bilateral gap trending toward a future risk threshold?
```

How to read:

```text
status       = whether prediction could run
score points = number of conformance rows available for score trend
gap points   = number of gap rows available for gap trend
slope        = trend direction per day
confidence   = HIGH/MEDIUM/LOW based on history and trend strength
```

Use with:

```bash
python -m fracture.cli predict --pipeline-id trade_positions_sftp
```

### `recommendations.json`

Purpose:

```text
Stores action-oriented recommendations for the dashboard.
```

What it answers:

```text
Who should act, how urgent is it, what likely caused the issue, and what command
should be run next?
```

How to read:

```text
INFO     = no action needed
WATCH    = monitor
ACTION   = fix this sprint
URGENT   = severe issue or SLA risk
BLOCKED  = logging/contract issue prevents reliable measurement
```

Use with:

```bash
python -m fracture.cli recommend --pipeline-id trade_positions_sftp
```

## Dashboard

Start the dashboard:

```bash
python -m fracture.cli dashboard
```

If port `8501` is busy:

```bash
python -m fracture.cli dashboard --port 8502
```

Pages:

```text
Fleet Overview
Pipeline Detail
Visualizations
```

The dashboard reads:

```text
conformance_log.csv
contracts/
inputs/
outputs/visualizations/
```

If a visual or JSON action artifact is missing, the dashboard can regenerate it
from local inputs and `conformance_log.csv`.

## Recommended Demo Order

Use this order when presenting the project:

```bash
python scripts/08_backfill_conformance_history.py --start-date 20260519 --end-date 20260618
python -m fracture.cli status --pipeline-id trade_positions_sftp
python -m fracture.cli discover --pipeline-id trade_positions_sftp --date 20260618
python -m fracture.cli compare --pipeline-id trade_positions_sftp --mode producer-consumer --date 20260618
python -m fracture.cli compare --pipeline-id trade_positions_sftp --mode period
python -m fracture.cli performance --pipeline-id trade_positions_sftp --date 20260618
python -m fracture.cli predict --pipeline-id trade_positions_sftp
python -m fracture.cli recommend --pipeline-id trade_positions_sftp
python -m fracture.cli visualize --pipeline-id trade_positions_sftp --kind all --all-dates
python -m fracture.cli dashboard
```

Narrative:

```text
1. Status shows the latest health.
2. Discovery shows actual path.
3. Compare shows the hidden producer-consumer gap.
4. Performance shows where time is spent.
5. Predict shows whether the risk is widening.
6. Recommend turns diagnostics into next action.
7. Visualize/dashboard make it explainable for non-technical viewers.
```

For the complete Phase 6 script, including setup, expected findings, QA tests,
and limitations, see:

```text
documents/FINAL_DEMO_GUIDE.md
```
