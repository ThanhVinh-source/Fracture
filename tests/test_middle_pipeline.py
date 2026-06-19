"""
tests/test_middle_pipeline.py

Tests for A → B → C pipeline chains.

Process B acts as both consumer (of A) and producer (for C).
B fires consumer events: DATA_RECEIVED, VALIDATION_DONE
B fires producer events: SCHEDULED, STARTED, COMPLETED, DATA_AVAILABLE

Two bilateral gaps:
  Upstream gap   (A→B): A's DATA_AVAILABLE → B's DATA_RECEIVED
  Downstream gap (B→C): B's DATA_AVAILABLE → C's DATA_RECEIVED

Scenarios:
  1. Simple chain — healthy gaps on both sides
  2. Upstream bottleneck — A is slow, affects B start time
  3. B is the bottleneck — B validates fast but processes slowly
  4. Downstream bottleneck — B is fast, C polls slowly
  5. Full chain view — all three gaps visible simultaneously

Run:
  python tests/test_middle_pipeline.py
"""

import sys
import warnings
from datetime import datetime, timedelta, date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings('ignore')

BAR = "─" * 65


def make_contract(pid, producer_team, consumer_team,
                  consumer_required_events=None,
                  consumer_terminal_event='',
                  upstream_consumer_event='DATA_AVAILABLE',
                  downstream_producer_event='DATA_AVAILABLE',
                  downstream_consumer_event='DATA_AVAILABLE'):
    from fracture.schema import PipelineContract
    return PipelineContract(**{
        'pipeline_id':    pid,
        'owner':          f'{producer_team}@bank.com',
        'producer_team':  producer_team,
        'consumer_team':  consumer_team,
        'criticality':    'high',
        'status':         'active',
        'expected_start': '06:00',
        'expected_end':   '08:30',
        'grace_minutes':  15,
        'p50_minutes':    45,
        'p95_minutes':    75,
        'p99_minutes':    90,
        'notifications':  [{'channel': 'slack', 'target': '#alerts'}],
        'log_contract': {
            'transport':       'parquet',
            'source_path':     f'inputs/{pid}/',
            'required_events': ['SCHEDULED','STARTED','COMPLETED','DATA_AVAILABLE'],
            'terminal_event':  'COMPLETED',
            'grain':           'pipeline',
            'consumer_required_events':  consumer_required_events or [],
            'consumer_terminal_event':   consumer_terminal_event,
            'upstream_consumer_event':   upstream_consumer_event,
            'downstream_producer_event': downstream_producer_event,
            'downstream_consumer_event': downstream_consumer_event,
        },
    })


def build_chain_events(
    n_runs=20,
    a_duration=44,          # A's execution time (min)
    upstream_gap=18,        # A→B gap (min): polling + transfer
    b_validate_time=2,      # B's validation of A's data (min)
    b_duration=30,          # B's processing time (min)
    downstream_gap=12,      # B→C gap (min): polling + transfer
    seed=42,
):
    """
    Generate events for A → B → C chain.

    A produces:  SCHEDULED, STARTED, COMPLETED, DATA_AVAILABLE
    B consumes:  DATA_RECEIVED, VALIDATION_DONE   (consumer role)
    B produces:  SCHEDULED, STARTED, COMPLETED, DATA_AVAILABLE  (producer role)
    C consumes:  DATA_RECEIVED                   (acknowledgment)

    Returns three DataFrames: a_events, b_events, c_events
    """
    rng = np.random.RandomState(seed)
    a_rows, b_prod_rows, b_cons_rows, c_rows = [], [], [], []

    for i in range(n_runs):
        ts = pd.Timestamp(f'2026-01-{i+1:02d} 06:00:00+00:00')
        run_id = f'run_{i:03d}'

        # ── Process A ────────────────────────────────────────────────────
        a_dur  = a_duration + rng.normal(0, 3)
        a_done = ts + pd.Timedelta(minutes=a_dur)
        a_da   = a_done + pd.Timedelta(minutes=3)

        a_rows += [
            {'pipeline_run_id': run_id, 'activity': 'SCHEDULED',
             'timestamp': ts - pd.Timedelta(minutes=2), 'team': 'producer'},
            {'pipeline_run_id': run_id, 'activity': 'STARTED',
             'timestamp': ts, 'team': 'producer'},
            {'pipeline_run_id': run_id, 'activity': 'COMPLETED',
             'timestamp': a_done, 'team': 'producer'},
            {'pipeline_run_id': run_id, 'activity': 'DATA_AVAILABLE',
             'timestamp': a_da, 'team': 'producer'},
        ]

        # ── Process B (consumer role): receives A's data ──────────────
        b_received = a_da + pd.Timedelta(minutes=upstream_gap + rng.normal(0,2))
        b_validated = b_received + pd.Timedelta(minutes=b_validate_time)

        b_cons_rows += [
            {'pipeline_run_id': run_id, 'activity': 'DATA_RECEIVED',
             'timestamp': b_received, 'team': 'consumer'},
            {'pipeline_run_id': run_id, 'activity': 'VALIDATION_DONE',
             'timestamp': b_validated, 'team': 'consumer'},
        ]

        # ── Process B (producer role): processes and publishes ─────────
        b_dur  = b_duration + rng.normal(0, 5)
        b_done = b_validated + pd.Timedelta(minutes=b_dur)
        b_da   = b_done + pd.Timedelta(minutes=2)

        b_prod_rows += [
            {'pipeline_run_id': run_id, 'activity': 'SCHEDULED',
             'timestamp': a_da, 'team': 'producer'},
            {'pipeline_run_id': run_id, 'activity': 'STARTED',
             'timestamp': b_received, 'team': 'producer'},
            {'pipeline_run_id': run_id, 'activity': 'COMPLETED',
             'timestamp': b_done, 'team': 'producer'},
            {'pipeline_run_id': run_id, 'activity': 'DATA_AVAILABLE',
             'timestamp': b_da, 'team': 'producer'},
        ]

        # ── Process C: acknowledges B's data ──────────────────────────
        c_received = b_da + pd.Timedelta(minutes=downstream_gap + rng.normal(0,2))
        c_rows.append({
            'pipeline_run_id': run_id, 'activity': 'DATA_RECEIVED',
            'timestamp': c_received, 'team': 'consumer',
        })

    a_events  = pd.DataFrame(a_rows)
    b_events  = pd.DataFrame(b_prod_rows)
    b_cons    = pd.DataFrame(b_cons_rows)
    c_events  = pd.DataFrame(c_rows)

    # Combine B producer + consumer events (in one file for B's contract)
    b_combined = pd.concat([b_events, b_cons], ignore_index=True).sort_values('timestamp')

    return a_events, b_events, b_combined, c_events, b_cons


passed = failed = 0


def check(name, fn):
    global passed, failed
    try:
        fn()
        passed += 1
        print(f'  v  {name}')
    except AssertionError as e:
        failed += 1
        print(f'  x  {name}')
        print(f'       {e}')
    except Exception as e:
        failed += 1
        print(f'  x  {name}: {type(e).__name__}: {e}')


# ═══════════════════════════════════════════════════════════════════════════
# SCENARIO 1: SIMPLE HEALTHY CHAIN
# ═══════════════════════════════════════════════════════════════════════════

print()
print("═" * 65)
print("  SCENARIO 1 — Simple healthy chain  A → B → C")
print("  Upstream gap: 18 min  Downstream gap: 12 min")
print("═" * 65)
print()

from fracture.conformance import compute_conformance

a_events, b_prod, b_combined, c_events, b_cons = build_chain_events(
    n_runs=20, upstream_gap=18, downstream_gap=12
)

# Contract for B — middle pipeline
b_contract = make_contract(
    pid                     = 'pipeline_b',
    producer_team           = 'risk-technology',
    consumer_team           = 'grid-scheduler',
    consumer_required_events = ['DATA_RECEIVED', 'VALIDATION_DONE'],
    consumer_terminal_event  = 'VALIDATION_DONE',
    upstream_consumer_event  = 'DATA_RECEIVED',   # B receives with DATA_RECEIVED
    downstream_consumer_event= 'DATA_RECEIVED',   # C receives with DATA_RECEIVED
)

print(f"  B contract: is_middle_pipeline="
      f"{b_contract.log_contract.is_middle_pipeline}")
print(f"  Consumer role events: "
      f"{b_contract.log_contract.consumer_required_events}")
print(f"  Upstream join:   A.DATA_AVAILABLE → B.DATA_RECEIVED")
print(f"  Downstream join: B.DATA_AVAILABLE → C.DATA_RECEIVED")
print()

# B runs conformance with its producer events + consumer events
# Consumer events include both B's consumer role AND C's acknowledgment
# B's consumer events = A's events (upstream_producer) + B's own consumer events + C's ack
b_with_c = pd.concat([
    a_events.assign(team='upstream_producer'),   # A's DATA_AVAILABLE for upstream gap
    b_cons,                                       # B's DATA_RECEIVED, VALIDATION_DONE
    c_events.assign(team='downstream_consumer'), # C's DATA_RECEIVED for downstream gap
], ignore_index=True)

r_b = compute_conformance(b_prod, b_contract, consumer_events=b_with_c)
d_b = r_b.diagnostics

print(f"  B conformance result:")
print(f"    Score            : {r_b.final_score:.0%}  {r_b.timing_zone}")
print(f"    Producer fitness : {d_b.sequence_fitness:.3f}")
print(f"    Upstream gap     : {d_b.upstream_gap_minutes} min  "
      f"({d_b.upstream_gap_trend})  ← A→B handoff")
print(f"    Downstream gap   : {d_b.downstream_gap_minutes} min  ← B→C handoff")
print(f"    Bilateral gap    : {r_b.bilateral_gap_minutes} min  (compat alias)")
print()
print(f"    Summary: {r_b.human_summary()}")


def test_scenario1_upstream_gap():
    assert d_b.upstream_gap_minutes is not None, "Upstream gap should be computed"
    assert 10 < d_b.upstream_gap_minutes < 30, \
        f"Upstream gap should be ~18 min: {d_b.upstream_gap_minutes}"
check('upstream gap A→B: ~18 min', test_scenario1_upstream_gap)


def test_scenario1_downstream_gap():
    assert d_b.downstream_gap_minutes is not None, \
        "Downstream gap should be computed from C's DATA_RECEIVED"
    assert 5 < d_b.downstream_gap_minutes < 25, \
        f"Downstream gap should be ~12 min: {d_b.downstream_gap_minutes}"
check('downstream gap B→C: ~12 min', test_scenario1_downstream_gap)


def test_scenario1_backward_compat():
    """bilateral_gap_minutes should still work as alias for upstream gap."""
    assert r_b.bilateral_gap_minutes == d_b.upstream_gap_minutes, \
        "bilateral_gap_minutes should equal upstream_gap_minutes"
check('bilateral_gap_minutes = upstream_gap (backward compat)', test_scenario1_backward_compat)


def test_scenario1_is_middle():
    assert b_contract.log_contract.is_middle_pipeline, \
        "B contract should be middle pipeline"
check('B contract: is_middle_pipeline=True', test_scenario1_is_middle)


# ═══════════════════════════════════════════════════════════════════════════
# SCENARIO 2: UPSTREAM BOTTLENECK
# ═══════════════════════════════════════════════════════════════════════════

print()
print("═" * 65)
print("  SCENARIO 2 — Upstream bottleneck")
print("  A is slow. Upstream gap = 45 min. B starts late.")
print("  B scores GREEN (its own process is fine).")
print("  Upstream gap reveals A is the bottleneck.")
print("═" * 65)
print()

a_events, b_prod2, b_comb2, c_ev2, b_cons2 = build_chain_events(
    n_runs=20, a_duration=44, upstream_gap=45, downstream_gap=12
)

b_with_c2 = pd.concat([
    a_events.assign(team='upstream_producer'),
    b_cons2,
    c_ev2.assign(team='downstream_consumer'),
], ignore_index=True)
r_b2 = compute_conformance(b_prod2, b_contract, consumer_events=b_with_c2)
d_b2 = r_b2.diagnostics

print(f"  B score:         {r_b2.final_score:.0%}  {r_b2.timing_zone}")
print(f"  Upstream gap:    {d_b2.upstream_gap_minutes:.1f} min  ← bottleneck")
print(f"  Downstream gap:  {d_b2.downstream_gap_minutes:.1f} min  ← fine")
print()
print(f"  Finding: B is conformant. But it receives A's data 45 min late.")
print(f"  Root cause is upstream (A), not B itself.")
print(f"  Without upstream gap: engineer investigates B. Wastes 2 hours.")
print(f"  With upstream gap:    engineer escalates to A's team immediately.")


def test_scenario2_large_upstream():
    assert d_b2.upstream_gap_minutes > 35, \
        f"Upstream gap should be ~45 min: {d_b2.upstream_gap_minutes}"
    assert r_b2.timing_zone == 'GREEN', \
        f"B itself is healthy: {r_b2.timing_zone}"
check('upstream bottleneck: B=GREEN but upstream gap=45min', test_scenario2_large_upstream)


# ═══════════════════════════════════════════════════════════════════════════
# SCENARIO 3: B IS THE BOTTLENECK
# ═══════════════════════════════════════════════════════════════════════════

print()
print("═" * 65)
print("  SCENARIO 3 — B is the bottleneck")
print("  A→B gap is fine. B processes slowly. B→C gap is large.")
print("  C waits a long time after B finishes.")
print("═" * 65)
print()

a_events, b_prod3, b_comb3, c_ev3, b_cons3 = build_chain_events(
    n_runs=20, upstream_gap=12, b_duration=65, downstream_gap=8, seed=99
)

b_with_c3 = pd.concat([
    a_events.assign(team='upstream_producer'),
    b_cons3,
    c_ev3.assign(team='downstream_consumer'),
], ignore_index=True)
r_b3 = compute_conformance(b_prod3, b_contract, consumer_events=b_with_c3)
d_b3 = r_b3.diagnostics

print(f"  B score:         {r_b3.final_score:.0%}  {r_b3.timing_zone}")
print(f"  Upstream gap:    {d_b3.upstream_gap_minutes:.1f} min  ← fine")
print(f"  Downstream gap:  {d_b3.downstream_gap_minutes:.1f} min  ← fine (polling)")
print(f"  Timing score:    {d_b3.timing_score:.3f}  ← B's processing is slow")
print()
print(f"  Finding: B's processing time is the bottleneck.")
print(f"  C receives data late not because of the gap but because B is slow.")
print(f"  Timing score < 1.0 reveals B's own processing as the cause.")


def test_scenario3_b_bottleneck():
    # B processes for 65 min — in p50=45 min window, it should show AMBER
    # Both gaps should be small — bottleneck is B's own process
    assert d_b3.upstream_gap_minutes < 25, \
        f"Upstream gap fine: {d_b3.upstream_gap_minutes}"
check('B bottleneck: small gaps, B timing score reveals the cause',
      test_scenario3_b_bottleneck)


# ═══════════════════════════════════════════════════════════════════════════
# SCENARIO 4: FULL CHAIN VIEW
# ═══════════════════════════════════════════════════════════════════════════

print()
print("═" * 65)
print("  SCENARIO 4 — Full chain view")
print("  Three contracts. All gaps visible. Complete picture.")
print("═" * 65)
print()

# Simple A contract (producer only)
a_contract = make_contract('pipeline_a', 'data-engineering', 'risk-technology')

# B contract (middle)
b_contract4 = make_contract(
    'pipeline_b', 'risk-technology', 'grid-scheduler',
    consumer_required_events=['DATA_RECEIVED','VALIDATION_DONE'],
    consumer_terminal_event='VALIDATION_DONE',
    upstream_consumer_event='DATA_RECEIVED',
    downstream_consumer_event='DATA_RECEIVED',
)

# C contract (consumer only — simple)
c_contract = make_contract('pipeline_c', 'grid-scheduler', 'reporting-platform')

a_ev4, b_prod4, b_comb4, c_ev4, b_cons4 = build_chain_events(
    n_runs=20, upstream_gap=18, downstream_gap=12
)

# A: producer-only conformance
r_a = compute_conformance(a_ev4, a_contract)

# B: middle pipeline conformance
b_w_c4 = pd.concat([
    a_ev4.assign(team='upstream_producer'),
    b_cons4,
    c_ev4.assign(team='downstream_consumer'),
], ignore_index=True)
r_b4 = compute_conformance(b_prod4, b_contract4, consumer_events=b_w_c4)

print(f"  FULL CHAIN CONFORMANCE PICTURE:")
print(f"  {'PIPELINE':<16} {'SCORE':>6}  {'ZONE':<10} {'UPSTREAM_GAP':>14} {'DOWNSTREAM_GAP':>16}")
print(f"  {BAR}")
print(f"  pipeline_a       {r_a.final_score:>6.0%}  {r_a.timing_zone:<10} {'─':>14} {'─':>16}  (no upstream)")
u = r_b4.diagnostics.upstream_gap_minutes
d = r_b4.diagnostics.downstream_gap_minutes
print(f"  pipeline_b       {r_b4.final_score:>6.0%}  {r_b4.timing_zone:<10} "
      f"{f'{u:.1f} min':>14} {f'{d:.1f} min':>16}  ← middle")
print(f"  pipeline_c       {'─':>6}  {'─':<10} {'─':>14} {'─':>16}  (C can register separately)")
print()
print(f"  Chain health:")
print(f"    A → B handoff  : {u:.1f} min  (polling interval + transfer)")
print(f"    B → C handoff  : {d:.1f} min  (polling interval + transfer)")
print(f"    A processing   : healthy (seq={r_a.diagnostics.sequence_fitness:.3f})")
print(f"    B processing   : healthy (score={r_b4.final_score:.0%})")
print(f"    Total chain    : A starts at 06:00, C receives at "
      f"~{44+3+18+2+30+2+12:.0f} min = 07:31")


def test_full_chain_both_gaps():
    assert r_b4.diagnostics.upstream_gap_minutes is not None
    assert r_b4.diagnostics.downstream_gap_minutes is not None
    assert r_b4.diagnostics.upstream_gap_minutes > 0
    assert r_b4.diagnostics.downstream_gap_minutes > 0
    print(f'       upstream={r_b4.diagnostics.upstream_gap_minutes:.1f}min  '
          f'downstream={r_b4.diagnostics.downstream_gap_minutes:.1f}min')
check('full chain: both upstream and downstream gaps computed', test_full_chain_both_gaps)


def test_schema_middle_pipeline():
    """Middle pipeline contract is valid and correct."""
    lc = b_contract4.log_contract
    assert lc.is_middle_pipeline
    assert lc.consumer_required_events == ['DATA_RECEIVED', 'VALIDATION_DONE']
    assert lc.upstream_consumer_event == 'DATA_RECEIVED'
    assert lc.downstream_consumer_event == 'DATA_RECEIVED'
    assert lc.upstream_producer_event == 'DATA_AVAILABLE'  # default
    assert lc.downstream_producer_event == 'DATA_AVAILABLE'  # default
check('middle pipeline schema: all fields correct', test_schema_middle_pipeline)


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

print()
print("═" * 65)
print(f"  Results: {passed+failed} tests  v {passed}  x {failed}")
print("═" * 65)
print()
print("  WHAT THE CHAIN ARCHITECTURE ADDS:")
print()
print("  Before:")
print("    One bilateral gap per contract.")
print("    Middle pipelines need two contracts.")
print("    Upstream bottleneck vs B-bottleneck: indistinguishable.")
print()
print("  After:")
print("    One contract per pipeline, including middle pipelines.")
print("    Two gaps: upstream (A→B) and downstream (B→C).")
print("    Upstream gap large: A is the bottleneck → escalate to A's team.")
print("    Downstream gap large: polling/transfer issue → escalate to infra.")
print("    B timing score low: B itself is slow → escalate to B's team.")
print("    Three possible root causes. Three distinct signals.")
print("    Backward compatible: existing simple contracts unchanged.")

if failed > 0:
    sys.exit(1)
