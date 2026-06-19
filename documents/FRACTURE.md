# Fracture

**Process conformance linter for data pipelines.**

Fracture measures whether data pipelines honour the bilateral contract between the team that produces data and the team that consumes it. It uses process mining — specifically token-based replay against a normative Petri net — to detect sequence violations, timing drift, and the handoff gap between producer and consumer that no threshold monitoring tool can see.

---

## What Fracture does

Most monitoring tools answer one question: did the pipeline finish before the deadline? Fracture answers three questions that threshold monitoring cannot:

**1. Did the pipeline execute the contracted sequence?**
A pipeline that completes at 07:25 (within SLA) but fired `COMPLETED` before `STARTED`, or never fired `DATA_AVAILABLE`, has violated its process contract. Token replay detects this. A threshold monitor does not.

**2. Did the consumer actually receive data when the producer said it was ready?**
The producer marks `DATA_AVAILABLE` at 07:10. The consumer experiences data at 07:38. That 28-minute gap is invisible to any tool that only monitors the producer. Fracture replays both sides of the contract against the same Petri net and measures the temporal distance between the two replay outcomes.

**3. Is execution time drifting toward breach?**
A batch that took 44 minutes in January and takes 72 minutes in April is drifting. The SLA says 90 minutes — still within it today. Fracture detects the slope and projects when the breach will happen.

---

## The formula

```
final_score = sequence × 0.35 + timing × 0.50 + completeness × 0.15
```

**Sequence fitness** (0.0–1.0): Did activities fire in the contracted order?
Computed by PM4PY token-based replay against the normative Petri net derived from the contract.

**Timing score** (0.10–1.05): Did the pipeline complete within the SLA window?
Computed by comparing actual duration to contracted p50/p95/p99. Early completion gets a small bonus (max 1.05). Breach past grace gets a heavy penalty (0.10).

**Completeness score** (0.0–1.0): Did all expected runs complete?
Computed as `completed_runs / scheduled_runs`. Silent pipelines — where Monday runs simply never fire — score here, not in sequence.

**Timing is weighted 0.50** because banks have hard SLA penalty clauses. A pipeline that sequences correctly but breaches its deadline is operationally worse than a pipeline with a minor sequence deviation that completes on time.

---

## What Fracture is not

- **Not a data quality tool.** Use Great Expectations, dbt tests, or Soda for data validation. Fracture measures whether the process happened correctly, not whether the data content is correct.
- **Not an infrastructure monitor.** Use Datadog or Prometheus for CPU, memory, and network metrics. Fracture measures process conformance.
- **Not a log aggregator.** Use Splunk or Elasticsearch to collect and store events. Fracture reads from files that your existing tooling produces.
- **Not a real-time alerting system.** Fracture runs as a daily batch job after pipelines complete. For real-time alerts during execution, use your scheduler's built-in SLA callbacks.

---

## Requirements

### Python environment

```bash
Python >= 3.10
pip install pm4py pydantic pyyaml pandas numpy scipy scikit-learn pyarrow ruptures
pip install -e .   # installs fracture CLI
```

### Event file format

Every pipeline needs a 4-column parquet or CSV file per day:

```
pipeline_run_id | activity        | timestamp                | team
batch_001       | SCHEDULED       | 2026-05-03T06:00:00Z    | producer
batch_001       | STARTED         | 2026-05-03T06:01:14Z    | producer
batch_001       | COMPLETED       | 2026-05-03T06:47:22Z    | producer
batch_001       | DATA_AVAILABLE  | 2026-05-03T06:50:05Z    | producer
```

**`pipeline_run_id`** — unique identifier for one pipeline execution. One run = one trace for token replay. At batch grain: one batch ID per day. At trade grain: one trade ID per transaction.

**`activity`** — the business milestone that just occurred. Maps directly to a transition in the Petri net. Must match `required_events` in the contract (or be mapped via `activity_name_map`).

**`timestamp`** — ISO 8601 with timezone. UTC strongly recommended to avoid bilateral gap clock skew errors.

**`team`** — `producer` or `consumer`. Producer events are used for conformance scoring. Consumer events are used for bilateral gap computation.

**File location convention:**
```
inputs/{pipeline_id}/producer_YYYYMMDD.parquet
inputs/{pipeline_id}/consumer_YYYYMMDD.parquet   (optional)
```

### Bilateral gap requirement

Both producer and consumer must emit the same activity name for the handoff event. The bilateral gap is measured between:

```
producer DATA_AVAILABLE timestamp
↕
consumer DATA_AVAILABLE timestamp
= the handoff gap
```

If your systems use different names internally, use `activity_name_map` in the contract to normalise both to the same vocabulary before Fracture processes the logs.

If your consumer cannot emit events yet, run in producer-only mode. You get sequence conformance, timing health, and drift detection. You lose the bilateral gap measurement. Add consumer instrumentation when your team is ready.

---

## The contract

Every pipeline needs one YAML contract file. Register it with:

```bash
fracture register \
  --name "VaR Batch Processing" \
  --owner risk-tech@bank.com \
  --producer-team market-risk-quant \
  --consumer-team grid-scheduler \
  --expected-start 06:00 \
  --expected-end 07:30 \
  --criticality high \
  --slack "#risk-alerts"
```

This creates `contracts/{pipeline_id}.yaml`:

```yaml
pipeline_id:    var_batch_processing
owner:          risk-tech@bank.com
producer_team:  market-risk-quant
consumer_team:  grid-scheduler
criticality:    high
status:         draft            # must activate before conformance runs

expected_start: "06:00"
expected_end:   "07:30"
grace_minutes:  10

# Bootstrapped from historical data — never guess these
p50_minutes:    45
p95_minutes:    71
p99_minutes:    84

notifications:
  - channel: slack
    target:  "#risk-alerts"

log_contract:
  transport:     parquet
  source_path:   inputs/var_batch_processing/
  required_events:
    - SCHEDULED
    - STARTED
    - COMPLETED
    - DATA_AVAILABLE
  terminal_event: COMPLETED
  grain: pipeline           # pipeline | trade | record
```

### Contract field reference

| Field | Required | Description |
|-------|----------|-------------|
| `pipeline_id` | yes | Lowercase, hyphens/underscores only |
| `owner` | yes | Email address of the responsible person |
| `producer_team` | yes | Team that produces the data |
| `consumer_team` | yes | Team that consumes the data — must differ from producer |
| `expected_start` | yes | When the pipeline is scheduled to start (HH:MM) |
| `expected_end` | yes | SLA deadline (HH:MM) |
| `grace_minutes` | yes | Buffer after p99 before breach is declared |
| `p50_minutes` | yes | Median execution time — use `fracture bootstrap` |
| `p95_minutes` | yes | 95th percentile execution time |
| `p99_minutes` | yes | 99th percentile execution time |
| `criticality` | yes | `low` / `medium` / `high` — HIGH requires notifications |
| `status` | yes | `draft` / `active` / `deprecated` |
| `required_events` | yes | Ordered activity sequence — becomes transitions in the Petri net |
| `terminal_event` | yes | Activity that marks successful completion |
| `grain` | no | `pipeline` (default) / `trade` / `record` |
| `activity_name_map` | no | Maps your system names to required_events vocabulary |
| `optional_activities` | no | Activities that may legitimately be absent (bypass arcs) |
| `deduplicate_retries` | no | Collapse duplicate activities per run (for Airflow retry=True) |

### Schema validators (enforced at registration time)

- `p50 ≤ p95 ≤ p99` — percentiles must be ordered
- `p99 + grace_minutes < window` — SLA must be achievable at p99
- `producer_team ≠ consumer_team` — bilateral comparison requires two distinct teams
- `HIGH criticality` requires at least one notification target
- `terminal_event` must be in `required_events`

---

## Onboarding a pipeline

Four steps. In order.

**Step 1: Register**

```bash
fracture register --name "Payment Settlements" \
  --owner payments@bank.com \
  --producer-team payments-platform \
  --consumer-team settlement-ops \
  --expected-start 06:00 --expected-end 08:30 \
  --criticality high --slack "#payments-alerts"
```

Creates the contract YAML with `status: draft`. Draft contracts do not run conformance — this is intentional. A human must review the percentile values before measurements begin.

**Step 2: Produce event files**

Drop your 4-column parquet into `inputs/{pipeline_id}/producer_YYYYMMDD.parquet`. At least 14 days of history is recommended for reliable p99 estimation.

**Step 3: Bootstrap**

```bash
fracture bootstrap --key FRC-xxxxxxxx --no-activate
```

Reads your event files, computes empirical p50/p95/p99 from actual execution history, writes them to the contract YAML. The `--no-activate` flag leaves status as `draft` — the human reviews the values before activating.

Review the contract file. Verify the percentiles make sense for your pipeline. Check that the SLA window gives you enough buffer at p99.

**Step 4: Activate and run**

```bash
fracture activate --key FRC-xxxxxxxx
fracture run --key FRC-xxxxxxxx
```

Activation changes status from `draft` to `active`. This is the human gate — once activated, conformance runs daily and results appear in `conformance_log.csv`.

---

## Daily operation

```bash
# Run all active pipelines
fracture run-all

# Run a specific team's pipelines
fracture run-all --team market-risk-quant

# Check fleet health
fracture status

# Explain a specific pipeline (reads from CSV — fast)
fracture explain --pipeline-id var_batch_processing

# Full diagnostic detail (re-runs conformance — slower)
fracture explain --pipeline-id var_batch_processing --verbose
```

**`fracture status` output:**

```
PIPELINE                           SCORE  ZONE    CONF   SUMMARY
──────────────────────────────────────────────────────────────────
v payment_settlements               102%  GREEN   HIGH   Conformant
! var_batch_processing               94%  AMBER   HIGH   Drifting at -0.8%/wk
x customer_risk_features             91%  GREEN   HIGH   Fails Mon+Thu — infra
```

---

## When a contract changes

If a team changes their event structure — new activities added, different sequence, renamed activities — they must deregister and re-register. Updating the YAML without deregistering leaves a stale Petri net on disk that produces silently wrong scores.

```bash
# Wrong — do not edit the YAML directly
# The cached Petri net does not rebuild automatically

# Right — full clean slate
fracture deregister --key FRC-xxxxxxxx \
  --reason "adding VALIDATED activity to event sequence"

# Then re-register with new event structure
fracture register ...
fracture bootstrap --key FRC-new --no-activate
fracture activate --key FRC-new
```

`fracture deregister` deletes both the contract YAML and the cached `.pnml` file. Historical conformance data in `conformance_log.csv` is preserved.

`fracture deprecate` is for pipelines that are shutting down permanently. It keeps the PNML for audit purposes and skips the pipeline in future `run-all` calls.

---

## Grain-level analysis

For pipelines where the unit of conformance is a transaction rather than a batch run, declare `grain: trade` in the contract:

```yaml
log_contract:
  required_events: [TRADE_STARTED, TRADE_COMPLETED]
  terminal_event:  TRADE_COMPLETED
  grain:           trade          # pipeline_run_id = one trade
  parent_grain:    var_batch_processing  # links to batch contract
```

**Sampling requirement:** when sampling from a large trade population, sample by `pipeline_run_id` (trade IDs), not by event rows. Row sampling splits traces across the sample boundary, producing artificially low completeness scores.

```python
# Wrong — splits traces
sample = events.sample(n=50_000)

# Correct — complete traces
trade_ids = events['pipeline_run_id'].unique()
sampled   = pd.Series(trade_ids).sample(n=25_000)
sample    = events[events['pipeline_run_id'].isin(sampled)]
```

---

## Pipeline chains (A → B → C)

For middle pipelines that act as both consumer of an upstream process and producer for a downstream process:

```yaml
log_contract:
  # B's producer role — standard events
  required_events: [SCHEDULED, STARTED, COMPLETED, DATA_AVAILABLE]
  terminal_event:  COMPLETED

  # B's consumer role — different activities
  consumer_required_events: [DATA_RECEIVED, VALIDATION_DONE]
  consumer_terminal_event:  VALIDATION_DONE

  # How to measure the A→B gap
  upstream_producer_event: DATA_AVAILABLE  # what A fires when done
  upstream_consumer_event: DATA_RECEIVED   # what B fires when A's data arrives

  # How to measure the B→C gap
  downstream_producer_event: DATA_AVAILABLE  # what B fires when done
  downstream_consumer_event: DATA_RECEIVED   # what C fires when B's data arrives
```

When running B's conformance, pass A's events tagged as `upstream_producer`:

```python
b_consumer_events = pd.concat([
    a_events.assign(team='upstream_producer'),  # A's DATA_AVAILABLE for upstream gap
    b_consumer_events,                           # B's DATA_RECEIVED, VALIDATION_DONE
    c_events.assign(team='downstream_consumer'), # C's DATA_RECEIVED for downstream gap
])
```

**Two gaps appear in diagnostics:**

```
upstream_gap_minutes   = 18.3  # A→B: how long after A finishes does B receive?
downstream_gap_minutes = 12.1  # B→C: how long after B finishes does C receive?
```

These tell you which part of the chain is the bottleneck without investigating all three systems.

---

## Airflow integration

Airflow DAGs map naturally to Fracture contracts:

| Airflow concept | Fracture handling |
|----------------|------------------|
| Task state change | Activity event via callback |
| `BranchPythonOperator` | Optional activity (bypass arc) |
| `ShortCircuitOperator` | Optional activity |
| `retry=True` | `deduplicate_retries: true` |
| Task timeout | AMBER preflight check |
| DAG SLA callback | `expected_end` in contract |

**Activity name mapping:**

```yaml
log_contract:
  activity_name_map:
    queued:   SCHEDULED
    running:  STARTED
    success:  COMPLETED
  optional_activities:
    - VALIDATED   # BranchPythonOperator — fires on full loads only
  deduplicate_retries: true   # retry=3 configured on extract task
```

**Splunk SPL to produce the 4-column format:**

```spl
index=pipeline_logs earliest=-1d
| eval pipeline_run_id = dag_run_id
| eval activity = case(
    state="queued",    "SCHEDULED",
    state="running",   "STARTED",
    state="success",   "COMPLETED",
    state="failed",    "FAILED",
    true(),            state)
| eval team = "producer"
| table pipeline_run_id, activity, _time, team
| rename _time AS timestamp
| outputcsv inputs/my_dag/producer_20260503.csv
```

---

## Large-scale datasets

For pipelines with millions of records daily:

**Use batch-grain for overall SLA monitoring:**

One trace per batch run. Token replay on 100 traces takes 7 seconds. Detects overall drift and SLA breach.

**Use trade-grain for transaction-level conformance:**

Build a Spark/SQL job that samples representative trades daily:

```python
# Stratified sample — oversample problem types
sample_ids = (
    events.groupby('instrument_type')['pipeline_run_id']
    .apply(lambda ids: ids.sample(min(5_000, len(ids))))
    .reset_index(drop=True)
)
sample = events[events['pipeline_run_id'].isin(sample_ids)]
# 20,000 trades → ~40,000 rows → Fracture runs in 30 seconds
```

**Always sample traces, not rows.** Row sampling splits traces across the sample boundary and produces artificially low completeness scores.

---

## Output fields

`conformance_log.csv` schema:

| Field | Description |
|-------|-------------|
| `pipeline_id` | Pipeline identifier |
| `run_date` | Date of conformance run (YYYYMMDD) |
| `final_score` | Weighted conformance score (0.0–1.05) |
| `timing_zone` | GREEN / AMBER / RED / BREACH / SUSPICIOUS_EARLY |
| `confidence_level` | HIGH / MEDIUM / LOW / UNRELIABLE |
| `pattern` | STABLE / DRIFTING / INTERMITTENT / RECOVERING |
| `bilateral_gap_minutes` | Mean gap between producer and consumer DATA_AVAILABLE |
| `sequence_fitness` | Token replay fitness (0.0–1.0) |
| `timing_score` | Timing zone score (0.10–1.05) |
| `completeness_score` | Completed runs / scheduled runs |
| `variance_cv` | Coefficient of variation of daily scores |

**Timing zones:**

| Zone | Meaning | Action |
|------|---------|--------|
| GREEN | Completed before p95 | None |
| AMBER | Between p95 and p99 | Watch closely |
| RED | Between p99 and p99+grace | Act now |
| BREACH | Past p99+grace | SLA breached |
| SUSPICIOUS_EARLY | Before p50/2 | Verify output completeness |

**Confidence levels:**

| Level | Meaning |
|-------|---------|
| HIGH | 30+ days of clean history |
| MEDIUM | 14–30 days, or minor data quality issues |
| LOW | Fewer than 14 days |
| UNRELIABLE | Broken log extraction (all same timestamp, future timestamps) |

---

## Process mining layer

Fracture uses four process mining operations:

**1. Petri net construction** (`fracture/petri.py`)
The contract's `required_events` list becomes a Workflow net. Each activity is a labelled transition. Each inter-activity state is a place. Optional activities get bypass arcs. Woflan soundness verification runs on any non-linear net.

**2. Token-based replay** (`fracture/conformance.py` Step 3)
PM4PY replays each trace through the Petri net. Missing tokens = activities that should have fired but did not. Remaining tokens = activities that fired out of sequence. Fitness formula: `0.5×(1 - missing/consumed) + 0.5×(1 - remaining/produced)`.

**3. Bilateral replay** (`fracture/conformance.py` Step 11)
The same Petri net is replayed against both producer and consumer logs independently. The temporal gap between the two replay outcomes — measured in minutes, not score units — is the bilateral gap. This is Fracture's novel contribution. Prior work (C2D2) is unilateral.

**4. Process variant comparison** (`fracture/conformance.py` Step 12b)
When sequence fitness drops below 0.90, PM4PY's `get_variants()` discovers the actual execution paths and compares them to the normative contract path. This explains why fitness is low: retries detected, activity missing, wrong order, high path diversity.

---

## Project structure

```
fracture/
  fracture/
    schema.py          Contract definition (14 fields, 4 validators)
    config.py          Conformance weights (seq=0.35, time=0.50, comp=0.15)
    petri.py           Contract → Petri net, woflan soundness
    preflight.py       8 traffic-light checks before token replay
    conformance.py     11-step conformance engine
    converter.py       DataFrame ↔ PM4PY EventLog (XES format)
    ingest.py          4-column format validation
    generator.py       Synthetic event generation (5 archetypes)
    factory.py         Fleet + demo contract generation
    bootstrap.py       Compute p50/p95/p99 from real event history
    engine.py          Dependency injection hub
    cli.py             All CLI commands

  scripts/
    team_config.py     10 teams, 25 pipeline configurations
    01_setup.py        Generate event files (core / custom markers / dirty logs)
    02_onboard_teams.py  Simulate team onboarding
    03_citi_setup.py   Generate Citi VaR batch data
    04_citi_onboard.py Onboard Citi pipelines
    05_generate_clustering_data.py  Generate 175-row clustering dataset

  tests/
    test_e2e_saturday.py           5 end-to-end scenarios
    test_optional_activities.py    14 Airflow DAG pattern tests
    test_hard_sla.py               19 SLA boundary tests
    test_analytical_extensions.py  10 changepoint + variant tests
    test_coverage_gaps.py          11 architectural guarantee tests
    test_grain_level.py            7 grain-level analysis tests
    test_middle_pipeline.py        8 A→B→C chain tests
    test_sla_breach_scenarios.py   3 breach narrative scenarios
    test_citi_large_dataset.py     Production scale test

  contracts/          Contract YAML files (git-versioned)
  inputs/             Event parquet files (not git-versioned)
  conformance_log.csv Daily conformance results
```

---

## Quick reference

```bash
# Onboard a new pipeline
fracture register --name "..." --owner ... --producer-team ... --consumer-team ...
fracture bootstrap --key FRC-xxx --no-activate
fracture activate  --key FRC-xxx
fracture run       --key FRC-xxx

# Daily operation
fracture run-all
fracture status
fracture explain --all

# Investigation
fracture explain --pipeline-id X --verbose

# Contract lifecycle
fracture list
fracture validate  --contract contracts/X.yaml
fracture deprecate --key FRC-xxx --reason "shutting down"
fracture deregister --key FRC-xxx --reason "changing event structure"

# Verify everything works
python saturday.py
python TEAM_GUIDE.py --run-tests
```

---

## Reference: Prior work

**C2D2** (Yeshchenko et al., 2019) — concept drift detection in process mining. Unilateral: one event log, one model. Fracture extends this to bilateral conformance: two event logs, one shared model, gap between replay outcomes.

**PM4PY** (Berti et al., 2021) — Python process mining library. Fracture uses token-based replay, Inductive Miner, performance DFG, and woflan soundness verification.

**Token-based replay** (Rozinat & van der Aalst, 2007) — the foundational conformance checking algorithm. Reliable for strictly linear workflow nets. Fracture constrains all contracts to sequential event structures to ensure reliable fitness computation.
