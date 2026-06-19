# Fracture — Test Suite Guide

The tests are the best documentation Fracture has.
Each test file explains a concept, then proves it with live code.

Read the file. Run the file. The output is the lesson.

---

## Quick start

```bash
# Verify everything works first
python tests/test_e2e_saturday.py

# Learn the CLI end to end
python tests/test_cli_learning.py

# Run the final high-signal demo tests
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

---

## Test files — what each one teaches

### `test_cli_learning.py` — Start here

**What it teaches:** Every Fracture CLI command, executed live with full explanations.

**Who should read it:** Anyone new to Fracture. Run this before touching anything else.

**Run it:**
```bash
python tests/test_cli_learning.py
```

**What you will see:**

```
Part 1 — register → bootstrap → activate → run
  Why the DRAFT gate exists (prevents measuring against wrong percentiles)
  Why --no-activate is the right default (human review before activation)

Part 2 — Reading the output
  What final_score means (seq×0.35 + timing×0.50 + comp×0.15)
  What timing_zone means (GREEN / AMBER / RED / BREACH)
  What bilateral_gap means (None ≠ 0 — unknown is not zero)
  What confidence means (HIGH needs 30+ days clean history)
  What pattern means (STABLE / DRIFTING / INTERMITTENT / RECOVERING)

Part 3 — The bilateral gap
  Producer scores 100% GREEN
  Consumer waits 28 minutes more
  Both sides replayed against the same Petri net
  The gap only appears when you have both logs

Part 4 — The DRAFT gate in action
  Attempt to run on DRAFT → blocked
  Bootstrap → review → activate → run

Part 5 — deregister vs deprecate
  deregister: contract CHANGES → delete YAML + PNML, start fresh
  deprecate:  pipeline RETIRES → PNML kept for audit, skipped in run-all

Part 6 — run-all and fleet status

Part 7 — Preflight: what bad logs look like
  RED: all same timestamp, future timestamps, empty log
  AMBER: missing terminal event, low coverage
```

---

### `test_e2e_saturday.py` — The five core scenarios

**What it teaches:** The five fundamental conformance failure modes.

**Run it:**
```bash
python tests/test_e2e_saturday.py
```

**Five scenarios:**

| Scenario | Archetype | What Fracture detects |
|----------|-----------|----------------------|
| 1 | Stable mature | Healthy baseline — score ≈102%, GREEN |
| 2 | Assumption asymmetry | Bilateral gap — 25 min between producer and consumer |
| 3 | Silent pipeline | Monday + Thursday never complete — INTERMITTENT |
| 4 | Clock sync outlier | Negative bilateral gap → None (clock skew detected) |
| 5 | Sudden collapse | Score drops from 102% to 53% in one week |

**Key assertions to understand:**

```python
# Scenario 2 — bilateral gap
assert result.bilateral_gap_minutes > 20
# This proves the gap is only visible when BOTH logs are provided.
# Producer scores 100% GREEN. Consumer waits 25 min more.

# Scenario 3 — silent pipeline
assert result.diagnostics.weekday_pattern.has_weekday_clustering
assert result.diagnostics.weekday_pattern.worst_weekday in ('Monday', 'Thursday')
# Mode 2 detector: counts SCHEDULED vs COMPLETED per weekday.
# Mode 1 (score-based) cannot see absent traces. Mode 2 can.

# Scenario 4 — clock skew
assert result.bilateral_gap_minutes is None
# Not zero. Not negative. None.
# >50% of runs show negative gap → clock skew detected → gap discarded.
# Fracture never reports a gap it cannot trust.
```

---

### `test_optional_activities.py` — Airflow DAG patterns

**What it teaches:** How optional activities work in the Petri net.

**Run it:**
```bash
python tests/test_optional_activities.py
```

**14 tests covering:**

- `BranchPythonOperator` — activity that only fires on certain conditions
- `ShortCircuitOperator` — activity that fires on full loads only
- Month-end conditional tasks — fires on last day of month, not otherwise
- Retry=True configurations — `deduplicate_retries: true` in contract
- `activity_name_map` — normalise Airflow task names to Fracture vocabulary

**Key concept — bypass arcs:**

```
Without optional VALIDATED:
  (p2) → [VALIDATED] → (p3)
  Missing VALIDATED → token stranded → fitness penalty

With optional VALIDATED:
  (p2) → [VALIDATED] → (p3)
  (p2) → [t_bypass]  → (p3)   ← bypass arc
  Missing VALIDATED → token takes bypass → no penalty → fitness 1.0
```

**When to use `optional_activities`:**
- Activity fires on full loads but not incremental loads
- Activity fires at month-end but not daily
- BranchPythonOperator task that legitimately skips

**When NOT to use it:**
- Activity that should always fire but sometimes fails
  (that is a genuine sequence violation — do not hide it with optional)

---

### `test_hard_sla.py` — SLA boundary conditions

**What it teaches:** Exactly how the timing zone model works at every boundary.

**Run it:**
```bash
python tests/test_hard_sla.py
```

**19 boundary tests:**

```
Exactly at p50    → timing_score = 1.000 + small bonus
Exactly at p95    → timing_score ≈ 0.875 (AMBER starts)
Exactly at p99    → timing_score ≈ 0.750 (RED starts)
p99 + 1 minute    → timing_score ≈ 0.500 (still in grace)
p99 + grace + 1   → timing_score = 0.100 (BREACH)
grace = 0         → p99 + 1 minute = immediate BREACH
SUSPICIOUS_EARLY  → completed in < p50/2 → flag for review
```

**Why SUSPICIOUS_EARLY exists:**

A pipeline that completes in 2 minutes when the p50 is 45 minutes probably did not actually complete — it failed silently or the terminal event fired too early. Fracture flags it as suspicious rather than rewarding it with a high timing score.

**Key SLA formula:**

```python
window   = expected_end - expected_start   (e.g. 90 min)
deadline = window                          (from expected_start)
# Schema validator enforces: p99 + grace < window
# If violated: contract rejected at registration time
# This prevents circular measurement against an impossible SLA
```

---

### `test_analytical_extensions.py` — Changepoint + variant analysis

**What it teaches:** The two diagnostic layers beyond the conformance score.

**Run it:**
```bash
python tests/test_analytical_extensions.py
```

**10 tests across two analytical capabilities:**

**Changepoint detection (PELT algorithm via `ruptures`):**

```python
# Stable gap — no changepoint
assert not result.diagnostics.changepoint.detected
assert result.diagnostics.changepoint.n_changepoints == 0

# Step change — gap jumped 15 minutes on week 6
assert result.diagnostics.changepoint.detected
assert result.diagnostics.changepoint.magnitude_minutes > 10
assert '2026-' in str(result.diagnostics.changepoint.changepoint_dates[0])
# The date tells you WHEN to look in deployment logs
```

**Variant comparison (PM4PY Inductive Miner):**

```python
# Retries detected
assert vc.compared
assert 'STARTED' in vc.dominant_variant
# When STARTED fires 3x, token replay penalises it.
# Variant comparison explains: "STARTED fires 3x in 100% of runs — retries."
# Fix: add deduplicate_retries: true to contract.

# Missing activity detected
assert not vc.paths_match
assert any('missing' in d.lower() for d in vc.deviations)
```

**When variant comparison fires:**
- Only when `sequence_fitness < 0.90`
- Only when trace completeness > 85% (grain-completeness guard)
- If traces are incomplete (row sampling), guard blocks it with an explanation

---

### `test_grain_level.py` — Grain-level analysis

**What it teaches:** What `grain` means in the contract and why it matters.

**Run it:**
```bash
python tests/test_grain_level.py
```

**Three scenarios:**

**Scenario 1 — batch vs trade grain:**

```
grain=pipeline:  pipeline_run_id = BATCH_20260503
                 one trace = one batch execution
                 completeness = 30/30 batches completed
                 score = 99% GREEN

grain=trade:     pipeline_run_id = TRADE_GB123456
                 one trace = one trade
                 completeness = 98%  (2% of trades never completed)
                 score = 74% RED

Cross-grain finding:
  Batch says GREEN. Trade says RED.
  The batch completed — it just skipped 2% of trades silently.
```

**Scenario 2 — row sampling vs trace sampling:**

```python
# WRONG — splits traces across sample boundary
row_sample = events.sample(n=50_000)
result = compute_conformance(row_sample, contract)
assert result.diagnostics.completeness_score < 0.60  # 49% — wrong!

# CORRECT — samples complete traces
trade_ids    = events['pipeline_run_id'].unique()
sampled_ids  = pd.Series(trade_ids).sample(n=25_000)
trace_sample = events[events['pipeline_run_id'].isin(sampled_ids)]
result = compute_conformance(trace_sample, contract)
assert result.diagnostics.completeness_score > 0.90  # 98% — correct!
```

**Scenario 3 — grain-completeness guard:**

```python
# Row sampling produces <85% trace completeness
# Variant comparison guard fires before Inductive Miner runs
assert not vc.compared
assert 'row sampling' in vc.fitness_explainer.lower()
# Fix message tells you exactly what to do
```

---

### `test_middle_pipeline.py` — A→B→C chain pipelines

**What it teaches:** How pipeline chains work — when B receives from A and produces for C.

**Run it:**
```bash
python tests/test_middle_pipeline.py
```

**Four scenarios:**

**Scenario 1 — healthy chain:**
```
A → B → C
upstream_gap   = 17.5 min  (A→B: A's DATA_AVAILABLE → B's DATA_RECEIVED)
downstream_gap = 12.2 min  (B→C: B's DATA_AVAILABLE → C's DATA_RECEIVED)
Both gaps visible from B's single conformance run.
```

**Scenario 2 — upstream bottleneck:**
```
B score: 102% GREEN   (B's own process is fine)
upstream_gap: 44.5 min   (A is slow — escalate to A's team)

Without this: engineer investigates B. Wastes 2 hours.
With this:    engineer escalates to A's team in 5 minutes.
```

**Scenario 3 — B is the bottleneck:**
```
upstream_gap:   12 min  (A→B is fine)
downstream_gap:  8 min  (B→C is fine)
timing_score:   0.72    (B's processing is slow — B is the problem)
```

**How to pass chain events:**
```python
# B's consumer events include:
b_consumer_events = pd.concat([
    a_events.assign(team='upstream_producer'),   # A's DATA_AVAILABLE
    b_cons_events,                                # B's DATA_RECEIVED
    c_events.assign(team='downstream_consumer'), # C's DATA_RECEIVED
])
result = compute_conformance(b_prod, b_contract, consumer_events=b_consumer_events)
# result.diagnostics.upstream_gap_minutes   = A→B gap
# result.diagnostics.downstream_gap_minutes = B→C gap
```

---

### `test_coverage_gaps.py` — Architectural guarantees

**What it teaches:** The safety guarantees Fracture makes about its own behaviour.

**Run it:**
```bash
python tests/test_coverage_gaps.py
```

**11 tests that prove architectural promises:**

| Guarantee | What the test proves |
|-----------|---------------------|
| AMBER preflight reduces confidence | Missing terminal → confidence drops from HIGH |
| PNML cache written on first build | `.pnml` file exists after `get_or_build()` |
| PNML cache read on second call | Second call faster than first |
| Delete forces rebuild | Delete `.pnml` → next call rebuilds correctly |
| Weekday Mode 1 (score-based) | Monday score=0.3 detected as clustering |
| Weekday Mode 2 (raw events) | Thursday absent traces detected |
| deregister deletes PNML | Both YAML and PNML deleted |
| run-all --team filters | Team B pipelines not in log when --team=team_a |
| deprecated skipped | DEPRECATED pipeline not in conformance_log |

**Why these tests matter:**

Without the PNML cache test: someone refactors the cache, the guarantee disappears silently.

Without the deregister test: someone changes only the YAML without deleting the PNML. Fracture measures conformance against the old process model. Scores appear valid. They are wrong.

These tests catch architecture regressions, not just code bugs.

---

### `test_sla_breach_scenarios.py` — Breach narratives

**What it teaches:** What Fracture finds that no other tool finds — told as stories.

**Run it:**
```bash
python tests/test_sla_breach_scenarios.py
```

**Three full narrative scenarios:**

1. **Gradual drift → breach** — pipeline drifts at 2 min/week. Fracture detects DRIFTING 8 weeks before breach.

2. **Bilateral gap causes consumer breach** — producer scores GREEN. Consumer misses deadline because the 28-minute gap consumes the entire SLA buffer.

3. **Trade-level breach invisible at batch** — batch GREEN. Individual trades breach p99. The cross-grain finding.

---

### `test_citi_large_dataset.py` — Production scale

**What it teaches:** How Fracture handles millions of records without collapsing.

**Run it:** (takes 60–90 seconds)
```bash
python tests/test_citi_large_dataset.py
```

**The scenario:**
- 100 VaR batches
- 10,000–20,000 trades per batch
- ~1.5 million records total

**Two conformance levels:**

```
Level 1 — Batch (100 traces):
  score = 99%  GREEN
  drift = +21.8 min over 20 weeks
  "The batch is healthy today. Breach projected in 7 weeks."

Level 2 — Trade (50,000 sampled):
  score = 53%  RED
  DERIVATIVE trades: p99 = 289s  (SLA = 45s)
  29,000 failed trades (2%) invisible at batch level
```

**The critical sampling fix:**
```python
# WRONG (produces 49% completeness):
sample = trade_events.sample(n=50_000)

# CORRECT (produces 98% completeness):
trade_ids  = trade_events['pipeline_run_id'].unique()
sample_ids = pd.Series(trade_ids).sample(n=25_000)
sample     = trade_events[trade_events['pipeline_run_id'].isin(sample_ids)]
```

---

## Running all tests

```bash
# Final high-signal suite:
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

# Additional regression and research scenario tests:
python tests/test_cli_learning.py           # 7 parts, ~3 min
python tests/test_e2e_saturday.py           # 5 scenarios
python tests/test_optional_activities.py    # 14 tests
python tests/test_hard_sla.py              # 19 boundary tests
python tests/test_analytical_extensions.py # 10 tests
python tests/test_coverage_gaps.py         # 11 tests
python tests/test_grain_level.py           # 7 tests
python tests/test_middle_pipeline.py       # 8 tests
python tests/test_sla_breach_scenarios.py  # 3 narratives
python tests/test_citi_large_dataset.py    # scale test
```

---

## How to read a failing test

When a test fails, read the assertion message. It is written to tell you exactly what went wrong.

```python
AssertionError: upstream gap should be ~18 min: 2.1

# This means: compute_chain_gaps() found an upstream gap of 2.1 min
# when it should be ~18 min.
# Check: did you pass A's events tagged as team='upstream_producer'?
# The upstream gap computation looks for team='upstream_producer' in
# consumer_events. Without the tag it falls back to producer_events
# and finds B's own DATA_AVAILABLE — hence the tiny gap.
```

Every assertion message is a hint, not just a failure.

---

## Coverage map

| Area | Tests | Status |
|------|-------|--------|
| Token replay (sequence fitness) | test_e2e_saturday, test_optional_activities | v |
| Timing zones (all 6 zones) | test_hard_sla | v |
| Completeness | test_hard_sla, test_grain_level | v |
| Bilateral gap | test_e2e_saturday, test_middle_pipeline | v |
| Changepoint detection | test_analytical_extensions | v |
| Variant comparison | test_analytical_extensions, test_grain_level | v |
| Grain-completeness guard | test_grain_level | v |
| Weekday pattern (Mode 1 + 2) | test_coverage_gaps | v |
| Chain gaps (A→B→C) | test_middle_pipeline | v |
| PNML cache | test_coverage_gaps | v |
| DRAFT gate | test_cli_learning | v |
| deregister / deprecate | test_coverage_gaps, test_cli_learning | v |
| run-all team filter | test_coverage_gaps | v |
| Preflight (8 checks) | test_cli_learning | v |
| Large scale (1.5M records) | test_citi_large_dataset | v |
| Breach scenarios | test_sla_breach_scenarios | v |

---

## The test that answers "does it work?"

```bash
python tests/test_e2e_saturday.py
```

Five E2E scenarios. Four PNG visualisations of Petri nets. If this passes, everything works.

---

## May target: RBAC + Iceberg

The next milestone after these tests: replace the 4 CSV functions in `cli.py` with Apache Iceberg.

```python
# Migration boundary — these 4 functions are the only storage layer:
_append_log(row)          # → table.append(pyarrow_row)
_read_log_rows()          # → table.scan().to_arrow().to_pylist()
_read_csv(path)           # → table.scan(row_filter=...).to_pylist()
_write_csv(path, rows)    # → table.overwrite(pyarrow_df)
```

All tests will continue passing after the migration because they test the conformance engine, not the storage layer.

RBAC: partition `conformance_log` by `producer_team`. Each team reads only their partition. `fracture run-all --team X` becomes a genuine access control boundary, not just a filter.
