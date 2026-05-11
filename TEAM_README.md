# Fracture — Team Guide for New Contributors

This document explains Fracture in plain terms.
You do not need a process mining background to contribute.
You need Python and curiosity.

---

## What problem does Fracture solve?

A bank has hundreds of data pipelines. Each pipeline moves data from one team to another.

**The current situation:**
Every team monitors whether their pipeline finished on time. Airflow says "DAG ran successfully". Datadog says "no errors". Everyone is happy.

**The problem nobody sees:**
- Did the pipeline run the *right steps* in the *right order*?
- When the producer said "data is ready", how long did the consumer actually wait before receiving it?
- Is the pipeline slowly getting slower, heading toward a deadline breach?

These three questions require more than a threshold check. They require a *process model*.

**What Fracture does:**
Fracture converts the agreement between two teams (the contract) into a formal process model (a Petri net), then replays the actual execution logs against that model to measure how well reality matched the specification.

---

## The key concept: bilateral conformance

```
Team A (producer)                Team B (consumer)
─────────────────                ────────────────
06:00  SCHEDULED                 
06:01  STARTED                   
06:47  COMPLETED                 
06:50  DATA_AVAILABLE  ────────► 07:18  DATA_AVAILABLE
                                         ↑
                                   28 min gap
                                   This is what Fracture finds
                                   No other tool sees this
```

Team A scores 100% — finished on time. Team B waited 28 extra minutes for data they thought was ready at 06:50. This gap is the bilateral gap.

---

## How Fracture works — step by step

**Step 1: The contract**

Two teams agree on what their pipeline looks like:

```yaml
# contracts/var_batch.yaml
pipeline_id:    var_batch
producer_team:  risk-engine
consumer_team:  grid-scheduler
expected_end:   "07:30"
p99_minutes:    84

log_contract:
  required_events:
    - SCHEDULED    # pipeline was triggered
    - STARTED      # execution began
    - COMPLETED    # finished successfully
    - DATA_AVAILABLE  # consumer can now use the data
  terminal_event: COMPLETED
```

**Step 2: The Petri net**

`fracture/petri.py` converts the contract into a Petri net.

A Petri net is a mathematical model of the process. It looks like this:

```
(p_start) →[SCHEDULED]→ (p1) →[STARTED]→ (p2) →[COMPLETED]→ (p3) →[DATA_AVAIL]→ (p_end)
```

A token starts at `p_start`. Each activity moves the token forward. At the end the token is at `p_end`. If the token gets stuck or goes missing — the process went wrong.

**Step 3: Token replay**

`fracture/conformance.py` takes the actual event logs and replays them through the Petri net.

```
Actual log:
  06:00  SCHEDULED  → token moves to p1  ✓
  06:01  STARTED    → token moves to p2  ✓
  06:47  COMPLETED  → token moves to p3  ✓
  06:50  DATA_AVAIL → token moves to p_end ✓

Result: fitness = 1.0 (perfect)
```

If COMPLETED never fires:
```
  06:00  SCHEDULED  → p1  ✓
  06:01  STARTED    → p2  ✓
  [COMPLETED missing]
  06:50  DATA_AVAIL → needs p3, but token is in p2
                   → create artificial token (missing_token += 1)
                   → token stranded in p2 (remaining_token += 1)

Result: fitness = 0.583  ✗
```

**Step 4: The score**

```python
final_score = sequence × 0.35 + timing × 0.50 + completeness × 0.15

# sequence:     how well the token replay went (from Step 3)
# timing:       did it finish within the agreed deadline?
# completeness: did all expected runs complete? (or did some silently fail?)
```

**Step 5: The bilateral gap**

After scoring, Fracture compares timestamps between producer and consumer logs using the same `pipeline_run_id` as the join key.

```python
gap = consumer_DATA_AVAILABLE_time - producer_DATA_AVAILABLE_time
```

This gap is the measurement no other tool produces.

---

## File map — what each file does

```
fracture/schema.py
  The contract definition. 14 fields, 4 validators.
  Read this first. It defines what a contract is.
  Key class: PipelineContract

fracture/petri.py
  Converts contract → Petri net.
  Uses PM4PY's Petri net objects.
  Key function: contract_to_petri_net(contract) → (net, im, fm)

fracture/preflight.py
  8 checks that run BEFORE conformance.
  RED checks: block conformance entirely (broken logs)
  AMBER checks: reduce confidence level (minor issues)
  Key class: PreflightChain

fracture/conformance.py
  The main engine. 12 steps.
  Steps 1-3:  preflight + token replay
  Steps 4-5:  timing zone + completeness
  Steps 7-9:  weighted score + confidence + pattern
  Steps 11-12: bilateral gap + changepoint + variants
  Key function: compute_conformance(producer_events, contract)

fracture/converter.py
  DataFrame → PM4PY EventLog and back.
  PM4PY needs events in XES format (specific field names).
  This handles the conversion.

fracture/generator.py
  Creates synthetic pipeline events for testing.
  5 archetypes: stable, asymmetry, silent, drifting, fast-drifting
  Key class: StableMatureGenerator, AssumptionAsymmetryGenerator, etc.

fracture/bootstrap.py
  Reads real event files, computes p50/p95/p99 from actual history.
  Why: you should never guess SLA percentiles.
  Key function: compute_percentiles_from_events(events)

fracture/engine.py
  Wires all modules together.
  FractureEngine.run_pipeline(pipeline_id) does the full pipeline.
  This is what the CLI calls.

fracture/cli.py
  All commands: register, bootstrap, activate, run, status, explain...
  If you want to understand the full user workflow, read this.
```

---

## Reading the output

When you run `fracture explain --pipeline-id X`:

```
pipeline_id    : var_batch
final_score    : 94%          ← weighted score (above 95% = healthy)
timing_zone    : AMBER        ← between p95 and p99 (watch closely)
confidence     : HIGH         ← 30+ days of clean history
pattern        : DRIFTING     ← score declining over time
bilateral_gap  : 28.3 min     ← consumer waits 28 min for data
```

**Timing zones:**
- `GREEN` → comfortable, completed before p95
- `AMBER` → between p95 and p99, watch closely
- `RED` → between p99 and grace period, act now
- `BREACH` → past grace period, SLA violated

**Patterns:**
- `STABLE` → consistent performance
- `DRIFTING` → declining over time (breach coming)
- `INTERMITTENT` → specific days fail (Monday, Thursday)
- `RECOVERING` → was declining, now improving

---

## How to contribute

**Adding a test scenario:**

Look at `tests/test_e2e_saturday.py`. Each scenario:
1. Creates a contract with `make_contract()`
2. Generates events with a generator
3. Calls `compute_conformance()`
4. Asserts on the result

Copy the pattern. Change the generator and the assertions.

**Adding a new generator archetype:**

Look at `fracture/generator.py`. Each generator has:
- `generate(contract, days, start_date, pipeline_age, seed)` → `(producer_df, consumer_df, ground_truth)`
- `GroundTruth` with expected cluster, pattern, and drift velocity

**Adding a new preflight check:**

Look at `fracture/preflight.py`. Add a class that inherits from `PreflightCheck` and implement `run(events, contract)`. Add it to `PreflightChain`.

---

## For the Petri net visualisation contributor

The Petri net is built in `fracture/petri.py`:

```python
from fracture.petri import contract_to_petri_net
from fracture.schema import PipelineContract

contract = PipelineContract(...)  # load from YAML
net, im, fm = contract_to_petri_net(contract)

# net   = PM4PY PetriNet object (has .places and .transitions)
# im    = initial marking (token in p_start)
# fm    = final marking (token in p_end)
```

PM4PY has built-in visualisation:

```python
from pm4py.visualization.petri_net import visualizer as pn_viz

gviz = pn_viz.apply(net, im, fm)
pn_viz.save(gviz, 'petri_net.png')     # save as image
pn_viz.view(gviz)                       # open in viewer
```

For the performance-decorated DFG (shows bilateral gap as arc weights):

```python
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.visualization.dfg import visualizer as dfg_viz
from fracture.converter import events_to_pm4py_log

log  = events_to_pm4py_log(producer_events)
dfg, start, end = dfg_discovery.apply(log)
gviz = dfg_viz.apply(dfg, log=log, variant=dfg_viz.Variants.PERFORMANCE)
dfg_viz.view(gviz)  # arc thickness shows time between activities
```

For the bilateral gap visualised on the DFG — produce one DFG for the producer and one for the consumer, overlay them, and colour the DATA_AVAILABLE arc differently in each. The time difference on that arc is the bilateral gap.

See `tests/test_e2e_saturday.py` — it already generates PNGs using PM4PY visualisation.

---

## Running everything

```bash
# Verify setup
python saturday.py

# Run all tests
python TEAM_GUIDE.py --run-tests

# Generate clustering dataset
python scripts/05_generate_clustering_data.py --days 30

# Run clustering
python clustering.py

# Run report
python report.py

# Test the H0
python hypothesis_test.py
```
