# Fracture

> Process conformance linter for data pipelines.

Fracture answers three questions that Airflow, Datadog, and Monte Carlo cannot:

1. **Did the pipeline execute the contracted process** — not just finish on time, but fire the right activities in the right order?
2. **How long did the consumer actually wait** after the producer said data is ready?
3. **Is performance drifting toward breach** before it happens?

---

## The finding no other tool produces

```
Producer marks DATA_AVAILABLE:  06:47
Consumer receives data:         07:15
Bilateral gap:                  28 minutes

Both teams score 100% GREEN in every existing monitoring tool.
Fracture measures the gap.
```

At scale — 1.5 million trades per day:

```
Batch level:  99% GREEN   <- VaR batch completes on time
Trade level:  53% RED     <- derivative trades all breach p99=45s
```

218,000 trades per day missing their SLA. Invisible at batch grain. Visible at trade grain.

---

## Quick start

```bash
pip install -r requirements.txt
pip install -e .

fracture register \
  --name "VaR Batch" --owner risk@bank.com \
  --producer-team market-risk-quant --consumer-team grid-scheduler \
  --expected-start 06:00 --expected-end 07:30 \
  --criticality high --slack "#risk-alerts"

fracture bootstrap --key FRC-xxxx --no-activate
# Review contracts/var_batch.yaml, then:
fracture activate --key FRC-xxxx
fracture run-all
fracture status
python report.py
```

---

## Learn by doing

```bash
python tests/test_cli_learning.py
```

Runs every CLI command live with full explanations. Isolated temp directory. Takes 2 minutes.

---

## Event file format

```
pipeline_run_id | activity        | timestamp                | team
BATCH_001       | SCHEDULED       | 2026-01-03T06:00:00Z     | producer
BATCH_001       | STARTED         | 2026-01-03T06:01:14Z     | producer
BATCH_001       | COMPLETED       | 2026-01-03T06:47:22Z     | producer
BATCH_001       | DATA_AVAILABLE  | 2026-01-03T06:50:05Z     | producer
BATCH_001       | DATA_AVAILABLE  | 2026-01-03T07:18:09Z     | consumer
```

Four columns. pipeline_run_id is the join key between producer and consumer.

---

## The formula

```
final_score = sequence x 0.35 + timing x 0.50 + completeness x 0.15
```

- sequence: did activities fire in the contracted order? (PM4PY token replay)
- timing: did it finish within the SLA window? (GREEN / AMBER / RED / BREACH)
- completeness: did all expected runs complete? (scheduled vs finished)

---

## How it works

```
contract.yaml
     |
     v
petri.py          Contract -> normative Petri net (WF-net)
     |
     v
conformance.py    Token replay -> sequence fitness
                  Zone model   -> timing score
                  Bilateral gap -> consumer wait time
                  Changepoint   -> when did the gap shift?
                  Variant compare -> why is fitness low?
```

---

## Tests as documentation

| Want to understand | Read |
|--------------------|------|
| Complete CLI walkthrough | tests/test_cli_learning.py |
| 5 core scenarios | tests/test_e2e_saturday.py |
| Grain-level analysis | tests/test_grain_level.py |
| A->B->C chains | tests/test_middle_pipeline.py |

```bash
python TEAM_GUIDE.py --run-tests    # all 74 tests
```

---

## Project structure

```
fracture/
  fracture/          13 modules
  tests/             9 test suites
  scripts/           Setup and data generation
  clustering.py      Fleet clustering (k-Means + DBSCAN + ARI)
  report.py          Fleet health report
  hypothesis_test.py H0: pipeline age vs drift velocity
  FRACTURE.md        Full technical documentation
  TEAM_README.md     Guide for new contributors
```

---

## Research context

Novel contribution: bilateral conformance checking.
C2D2 (Yeshchenko et al. 2019) is unilateral. Fracture is bilateral.
Two logs, same Petri net, gap between the two replay outcomes.

Null hypothesis: pipeline age has no effect on drift velocity.
Test: Mann-Whitney U. See hypothesis_test.py.

Upcoming: Apache Iceberg for production storage, RBAC, time travel.

---

Business Information Technology — Data Science specialisation
University of Twente, 2026
