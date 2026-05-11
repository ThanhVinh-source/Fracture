"""
tests/test_analytical_extensions.py

Tests for the two analytical extensions beyond process mining:

  1. Changepoint detection on bilateral gap (ruptures PELT algorithm)
     Detects WHEN the gap structure changed, not just THAT it is changing.
     More precise than linear regression for step-change events.

  2. Process variant comparison (PM4PY Inductive Miner)
     When sequence_fitness < 0.90, discovers the actual process model
     and compares it to the normative contract model.
     Tells the engineer WHY fitness is low, not just HOW low it is.

Where these live in the codebase:
  fracture/conformance.py
    detect_gap_changepoint()        ← standalone function, Step 12
    compare_process_variants()      ← standalone function, Step 12b
    ConformanceDiagnostics
      .changepoint                  ← ChangePoint dataclass
      .variant_comparison           ← VariantComparison dataclass

Run:
  python tests/test_analytical_extensions.py
"""

import sys
import warnings
from datetime import datetime, timedelta, date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings('ignore')


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_contract(pid='t'):
    from fracture.schema import PipelineContract
    return PipelineContract(**{
        'pipeline_id': pid, 'owner': 'p@b.com',
        'producer_team': 'risk', 'consumer_team': 'grid',
        'criticality': 'high', 'status': 'active',
        'expected_start': '06:00', 'expected_end': '08:30',
        'grace_minutes': 15, 'p50_minutes': 45,
        'p95_minutes': 75, 'p99_minutes': 90,
        'notifications': [{'channel': 'slack', 'target': '#a'}],
        'log_contract': {'transport': 'parquet', 'source_path': 'x/'},
    })


def run_c(events, contract, consumer=None):
    from fracture.conformance import compute_conformance
    return compute_conformance(events, contract, consumer_events=consumer)


def make_events_with_gap(n_runs, gap_fn, rng=None):
    """Build producer+consumer events where bilateral gap = gap_fn(i)."""
    if rng is None:
        rng = np.random.RandomState(42)
    prod_rows, cons_rows = [], []
    for i in range(n_runs):
        ts   = pd.Timestamp(f'2026-01-{i+1:02d} 06:00:00+00:00')
        dur  = 45 + rng.normal(0, 3)
        gap  = gap_fn(i) + rng.normal(0, 1)
        prod_rows += [
            {'pipeline_run_id': f'r{i}', 'activity': 'SCHEDULED',
             'timestamp': ts - pd.Timedelta(minutes=2), 'team': 'producer'},
            {'pipeline_run_id': f'r{i}', 'activity': 'STARTED',
             'timestamp': ts, 'team': 'producer'},
            {'pipeline_run_id': f'r{i}', 'activity': 'COMPLETED',
             'timestamp': ts + pd.Timedelta(minutes=dur), 'team': 'producer'},
            {'pipeline_run_id': f'r{i}', 'activity': 'DATA_AVAILABLE',
             'timestamp': ts + pd.Timedelta(minutes=dur + 3), 'team': 'producer'},
        ]
        cons_rows += [
            {'pipeline_run_id': f'r{i}', 'activity': 'SCHEDULED',
             'timestamp': ts - pd.Timedelta(minutes=2), 'team': 'consumer'},
            {'pipeline_run_id': f'r{i}', 'activity': 'STARTED',
             'timestamp': ts + pd.Timedelta(minutes=dur + gap), 'team': 'consumer'},
            {'pipeline_run_id': f'r{i}', 'activity': 'COMPLETED',
             'timestamp': ts + pd.Timedelta(minutes=dur + gap + 10), 'team': 'consumer'},
            {'pipeline_run_id': f'r{i}', 'activity': 'DATA_AVAILABLE',
             'timestamp': ts + pd.Timedelta(minutes=dur + gap + 12), 'team': 'consumer'},
        ]
    return pd.DataFrame(prod_rows), pd.DataFrame(cons_rows)


passed = failed = 0


def check(name, fn):
    global passed, failed
    try:
        fn()
        passed += 1
        print(f'  ✓  {name}')
    except AssertionError as e:
        failed += 1
        print(f'  ✗  {name}')
        print(f'       {e}')
    except Exception as e:
        failed += 1
        print(f'  ✗  {name}: {type(e).__name__}: {e}')


# ═══════════════════════════════════════════════════════════════════════════
# CHANGEPOINT DETECTION TESTS
# ═══════════════════════════════════════════════════════════════════════════

print()
print('═' * 65)
print('  ADDITION 1: Changepoint Detection')
print('  Algorithm: PELT (Pruned Exact Linear Time) via ruptures')
print('  Question: WHEN did the gap structure change?')
print('═' * 65)

print()
print('--- Group A: No changepoint (stable gap) ---')


def test_stable_gap_no_changepoint():
    """
    30 runs with stable bilateral gap ~18 min.
    No changepoint should be detected.

    Real-world: SFTP setup has not changed.
    Gap is structural — same configuration for 30 days.
    """
    from fracture.conformance import detect_gap_changepoint

    gaps = [(datetime(2026, 1, i+1), 18.0 + np.random.RandomState(i).normal(0, 1))
            for i in range(30)]
    cp = detect_gap_changepoint(gaps)

    print(f'       stable gap ~18min: detected={cp.detected}')
    print(f'       cause: {cp.probable_cause[:60]}')

    assert not cp.detected, \
        f"Stable gap should not trigger changepoint: {cp.probable_cause}"

check('stable gap: no changepoint detected', test_stable_gap_no_changepoint)


def test_too_few_runs_no_changepoint():
    """
    Only 5 runs — not enough history for reliable changepoint detection.
    Should return detected=False with an informative message.
    """
    from fracture.conformance import detect_gap_changepoint

    gaps = [(datetime(2026, 1, i+1), 20.0) for i in range(5)]
    cp   = detect_gap_changepoint(gaps)

    print(f'       5 runs: detected={cp.detected} n_cp={cp.n_changepoints}')

    assert not cp.detected, \
        f"Too few runs should not trigger changepoint"
    assert 'Insufficient' in cp.probable_cause or 'not' in cp.probable_cause.lower(), \
        f"Should explain insufficient history: {cp.probable_cause}"

check('< 10 runs: insufficient history message', test_too_few_runs_no_changepoint)


print()
print('--- Group B: Clear step changes ---')


def test_gap_increase_step_change():
    """
    First 20 runs: gap ~13 min
    Last 10 runs:  gap ~28 min
    Simulates: new instrument type added, increases handoff time.

    Expected:
      detected = True
      direction = 'increased'
      magnitude ≈ +15 min
      engineer action: what changed on day 20?
    """
    from fracture.conformance import detect_gap_changepoint

    gaps = (
        [(datetime(2026, 1, i+1), 13.0 + np.random.RandomState(i).normal(0, 1))
         for i in range(20)] +
        [(datetime(2026, 1, i+21), 28.0 + np.random.RandomState(i+20).normal(0, 1))
         for i in range(10)]
    )
    cp = detect_gap_changepoint(gaps)

    print(f'       gap 13→28min: detected={cp.detected} '
          f'direction={cp.direction} mag={cp.magnitude_minutes}min')
    print(f'       pre={cp.pre_mean_minutes}min post={cp.post_mean_minutes}min')
    print(f'       cause: {cp.probable_cause[:80]}')

    assert cp.detected, "Step change should be detected"
    assert cp.direction == 'increased', \
        f"Gap increased, direction should be 'increased': {cp.direction}"
    assert cp.magnitude_minutes > 10, \
        f"Magnitude should be ~15 min: {cp.magnitude_minutes}"

check('gap increases by 15 min: changepoint detected, direction=increased',
      test_gap_increase_step_change)


def test_gap_decrease_step_change():
    """
    First 15 runs: gap ~28 min
    Last 15 runs:  gap ~10 min
    Simulates: SFTP polling interval reduced from 15 min to 3 min.

    This is a GOOD change — gap decreased.
    Fracture still reports it because the engineer should know WHY it improved.
    Maybe the improvement can be applied to other pipelines.
    """
    from fracture.conformance import detect_gap_changepoint

    gaps = (
        [(datetime(2026, 1, i+1), 28.0 + np.random.RandomState(i).normal(0, 1))
         for i in range(15)] +
        [(datetime(2026, 1, i+16), 10.0 + np.random.RandomState(i+15).normal(0, 1))
         for i in range(15)]
    )
    cp = detect_gap_changepoint(gaps)

    print(f'       gap 28→10min (SFTP optimised): detected={cp.detected} '
          f'direction={cp.direction} mag={cp.magnitude_minutes}min')
    print(f'       cause: {cp.probable_cause[:80]}')

    assert cp.detected, "Improvement step should also be detected"
    assert cp.direction == 'decreased', \
        f"Gap decreased, direction should be 'decreased': {cp.direction}"
    assert cp.magnitude_minutes < -10, \
        f"Magnitude should be ~-18 min: {cp.magnitude_minutes}"

check('gap decreases by 18 min: changepoint detected, direction=decreased',
      test_gap_decrease_step_change)


def test_changepoint_end_to_end_in_conformance():
    """
    Full conformance run on a pipeline with a step change in bilateral gap.
    Verifies that changepoint appears in ConformanceDiagnostics.
    """
    c = make_contract('cp_e2e')
    # Gap: first 15 runs at 12 min, last 15 runs at 27 min
    prod, cons = make_events_with_gap(
        30,
        lambda i: 12.0 if i < 15 else 27.0
    )

    r  = run_c(prod, c, cons)
    d  = r.diagnostics
    cp = d.changepoint

    print(f'       E2E: gap={r.bilateral_gap_minutes:.1f}min '
          f'changepoint.detected={cp.detected if cp else None}')
    if cp and cp.detected:
        print(f'       magnitude={cp.magnitude_minutes}min '
              f'pre={cp.pre_mean_minutes}min post={cp.post_mean_minutes}min')
        print(f'       cause: {cp.probable_cause[:70]}')

    assert cp is not None, "Changepoint should be in diagnostics"

check('E2E: changepoint in ConformanceDiagnostics',
      test_changepoint_end_to_end_in_conformance)


# ═══════════════════════════════════════════════════════════════════════════
# VARIANT COMPARISON TESTS
# ═══════════════════════════════════════════════════════════════════════════

print()
print('═' * 65)
print('  ADDITION 2: Process Variant Comparison')
print('  Algorithm: PM4PY Inductive Miner + variant analysis')
print('  Question: WHY is sequence fitness low?')
print('═' * 65)

print()
print('--- Group C: Healthy process (no comparison needed) ---')


def test_healthy_variant_comparison_skipped():
    """
    Healthy pipeline with fitness ≥ 0.90.
    Variant comparison should be skipped — no point computing it.
    compared=False, fitness_explainer says it is healthy.
    """
    from fracture.generator import StableMatureGenerator

    c    = make_contract('healthy_vc')
    prod, cons, _ = StableMatureGenerator().generate(
        c, 30, date(2026, 1, 1), 200, 1
    )
    r  = run_c(prod, c)
    d  = r.diagnostics
    vc = d.variant_comparison

    print(f'       fitness={d.sequence_fitness:.3f}: '
          f'compared={vc.compared}')
    print(f'       explainer: {vc.fitness_explainer[:60]}')

    assert not vc.compared, \
        f"Healthy fitness should skip variant comparison: {vc.compared}"
    assert vc.fitness_explainer is not None, \
        "Should have an explainer even when skipped"

check('healthy fitness: variant comparison skipped', test_healthy_variant_comparison_skipped)


print()
print('--- Group D: Retry patterns ---')


def test_retries_detected_as_repeated_activity():
    """
    ALL runs have STARTED firing 3 times (3 retries before success).
    Variant comparison should detect this and recommend deduplicate_retries=true.

    Real-world: Airflow task retrying on transient network failures.
    Fitness is low (0.75) because token replay sees 3x STARTED as violations.
    Variant comparison explains: it is retries, not genuine violations.
    """
    c = make_contract('retry_heavy')

    rows = []
    for i in range(20):
        ts = pd.Timestamp(f'2026-01-{i+1:02d} 06:00:00+00:00')
        rows += [
            {'pipeline_run_id': f'r{i}', 'activity': 'SCHEDULED',
             'timestamp': ts - pd.Timedelta(minutes=2), 'team': 'producer'},
            {'pipeline_run_id': f'r{i}', 'activity': 'STARTED',
             'timestamp': ts, 'team': 'producer'},
            {'pipeline_run_id': f'r{i}', 'activity': 'STARTED',
             'timestamp': ts + pd.Timedelta(minutes=10), 'team': 'producer'},
            {'pipeline_run_id': f'r{i}', 'activity': 'STARTED',
             'timestamp': ts + pd.Timedelta(minutes=20), 'team': 'producer'},
            {'pipeline_run_id': f'r{i}', 'activity': 'COMPLETED',
             'timestamp': ts + pd.Timedelta(minutes=45), 'team': 'producer'},
            {'pipeline_run_id': f'r{i}', 'activity': 'DATA_AVAILABLE',
             'timestamp': ts + pd.Timedelta(minutes=48), 'team': 'producer'},
        ]

    events = pd.DataFrame(rows)
    r  = run_c(events, c)
    d  = r.diagnostics
    vc = d.variant_comparison

    print(f'       fitness={d.sequence_fitness:.3f}  compared={vc.compared}')
    print(f'       dominant_variant={vc.dominant_variant}')
    print(f'       deviations={vc.deviations}')
    print(f'       explainer: {vc.fitness_explainer[:80]}')

    assert vc.compared, \
        f"Low fitness should trigger variant comparison: {d.sequence_fitness:.3f}"
    assert not vc.paths_match, \
        "Retry path should not match normative path"
    assert any('STARTED' in dev for dev in vc.deviations), \
        f"Should detect repeated STARTED: {vc.deviations}"
    assert any('deduplicate_retries' in dev for dev in vc.deviations), \
        f"Should recommend deduplicate_retries: {vc.deviations}"
    assert 'STARTED' in vc.dominant_variant, \
        f"STARTED should appear in dominant variant"

check('3x STARTED = retries: detected, recommends deduplicate_retries',
      test_retries_detected_as_repeated_activity)


print()
print('--- Group E: Missing activities ---')


def test_missing_required_activity_detected():
    """
    DATA_AVAILABLE never fires — contract requires it but process never emits it.

    Real-world: team removed the data publication step without updating the contract.
    Fitness is low (0.75) because token replay penalises missing DATA_AVAILABLE.
    Variant comparison explains: DATA_AVAILABLE is in the contract but never fires.
    """
    c = make_contract('missing_da')

    rows = []
    for i in range(20):
        ts = pd.Timestamp(f'2026-01-{i+1:02d} 06:00:00+00:00')
        rows += [
            {'pipeline_run_id': f'r{i}', 'activity': 'SCHEDULED',
             'timestamp': ts - pd.Timedelta(minutes=2), 'team': 'producer'},
            {'pipeline_run_id': f'r{i}', 'activity': 'STARTED',
             'timestamp': ts, 'team': 'producer'},
            {'pipeline_run_id': f'r{i}', 'activity': 'COMPLETED',
             'timestamp': ts + pd.Timedelta(minutes=45), 'team': 'producer'},
            # DATA_AVAILABLE intentionally absent
        ]

    events = pd.DataFrame(rows)
    r  = run_c(events, c)
    d  = r.diagnostics
    vc = d.variant_comparison

    print(f'       fitness={d.sequence_fitness:.3f}  compared={vc.compared}')
    print(f'       dominant={vc.dominant_variant}')
    print(f'       deviations={vc.deviations}')
    print(f'       explainer: {vc.fitness_explainer[:80]}')

    assert vc.compared, \
        f"Low fitness should trigger comparison: {d.sequence_fitness:.3f}"
    assert any('DATA_AVAILABLE' in dev for dev in vc.deviations), \
        f"Should detect missing DATA_AVAILABLE: {vc.deviations}"

check('DATA_AVAILABLE missing: detected, contract needs update',
      test_missing_required_activity_detected)


print()
print('--- Group F: High variant diversity ---')


def test_high_variant_diversity_detected():
    """
    10 runs, each with a different execution path.
    Process is highly variable — not conforming to any single pattern.
    Variant comparison should flag high diversity.
    """
    c = make_contract('diverse')

    rows = []
    activity_sets = [
        ['SCHEDULED', 'STARTED', 'COMPLETED', 'DATA_AVAILABLE'],
        ['SCHEDULED', 'STARTED', 'DATA_AVAILABLE'],            # missing COMPLETED
        ['SCHEDULED', 'COMPLETED', 'DATA_AVAILABLE'],          # missing STARTED
        ['STARTED', 'COMPLETED', 'DATA_AVAILABLE'],            # missing SCHEDULED
        ['SCHEDULED', 'STARTED', 'STARTED', 'COMPLETED', 'DATA_AVAILABLE'],  # retry
        ['SCHEDULED', 'STARTED', 'COMPLETED'],                 # missing DA
        ['SCHEDULED', 'STARTED', 'COMPLETED', 'DATA_AVAILABLE'],
        ['SCHEDULED', 'STARTED', 'COMPLETED', 'DATA_AVAILABLE'],
        ['SCHEDULED', 'STARTED', 'STARTED', 'COMPLETED', 'DATA_AVAILABLE'],
        ['SCHEDULED', 'COMPLETED', 'STARTED', 'DATA_AVAILABLE'],  # wrong order
    ]

    for i, acts in enumerate(activity_sets):
        ts = pd.Timestamp(f'2026-01-{i+1:02d} 06:00:00+00:00')
        for j, act in enumerate(acts):
            rows.append({
                'pipeline_run_id': f'r{i}',
                'activity': act,
                'timestamp': ts + pd.Timedelta(minutes=j*15),
                'team': 'producer',
            })

    events = pd.DataFrame(rows)
    r  = run_c(events, c)
    d  = r.diagnostics
    vc = d.variant_comparison

    print(f'       fitness={d.sequence_fitness:.3f}  '
          f'n_variants={vc.n_variants_actual}  compared={vc.compared}')
    print(f'       paths_match={vc.paths_match}')
    if vc.deviations:
        for dev in vc.deviations[:2]:
            print(f'       deviation: {dev[:70]}')

    assert vc.compared, \
        f"Low fitness should trigger comparison"
    assert vc.n_variants_actual > 3, \
        f"Should detect multiple variants: {vc.n_variants_actual}"

check('10 different execution paths: high diversity detected',
      test_high_variant_diversity_detected)


# ═══════════════════════════════════════════════════════════════════════════
# COMBINED: Both features together
# ═══════════════════════════════════════════════════════════════════════════

print()
print('═' * 65)
print('  COMBINED: Both additions in one conformance result')
print('═' * 65)


def test_both_features_on_bad_pipeline():
    """
    Pipeline with:
      - Step change in bilateral gap (SFTP optimised on day 15)
      - Retries detected (STARTED fires twice)

    Both features should appear in the same ConformanceDiagnostics.
    This is the realistic scenario: a pipeline that has both
    process violations AND gap changes.
    """
    c = make_contract('bad_pipeline')

    prod_rows, cons_rows = [], []
    for i in range(30):
        ts  = pd.Timestamp(f'2026-01-{i+1:02d} 06:00:00+00:00')
        gap = 12.0 if i >= 15 else 28.0  # step change on day 15

        # Producer — with retries on some runs
        acts = ['SCHEDULED', 'STARTED']
        if i % 4 == 0:  # every 4th run has a retry
            acts.append('STARTED')
        acts += ['COMPLETED', 'DATA_AVAILABLE']

        for j, act in enumerate(acts):
            prod_rows.append({
                'pipeline_run_id': f'r{i}',
                'activity': act,
                'timestamp': ts + pd.Timedelta(minutes=j*12 - (2 if j==0 else 0)),
                'team': 'producer',
            })

        # Consumer
        cons_rows += [
            {'pipeline_run_id': f'r{i}', 'activity': 'SCHEDULED',
             'timestamp': ts - pd.Timedelta(minutes=2), 'team': 'consumer'},
            {'pipeline_run_id': f'r{i}', 'activity': 'STARTED',
             'timestamp': ts + pd.Timedelta(minutes=45 + gap), 'team': 'consumer'},
            {'pipeline_run_id': f'r{i}', 'activity': 'COMPLETED',
             'timestamp': ts + pd.Timedelta(minutes=45 + gap + 10), 'team': 'consumer'},
            {'pipeline_run_id': f'r{i}', 'activity': 'DATA_AVAILABLE',
             'timestamp': ts + pd.Timedelta(minutes=45 + gap + 12), 'team': 'consumer'},
        ]

    prod = pd.DataFrame(prod_rows)
    cons = pd.DataFrame(cons_rows)

    r  = run_c(prod, c, cons)
    d  = r.diagnostics
    cp = d.changepoint
    vc = d.variant_comparison

    print(f'\n       Pipeline with retries + gap step change:')
    print(f'       score={r.final_score:.3f} zone={r.timing_zone}')
    print(f'       gap={r.bilateral_gap_minutes:.1f}min')
    print()
    print(f'       CHANGEPOINT:')
    print(f'         detected={cp.detected if cp else None}')
    if cp and cp.detected:
        print(f'         magnitude={cp.magnitude_minutes}min '
              f'direction={cp.direction}')
        print(f'         cause: {cp.probable_cause[:70]}')
    print()
    print(f'       VARIANT COMPARISON:')
    print(f'         compared={vc.compared}')
    print(f'         n_variants={vc.n_variants_actual}')
    if vc.deviations:
        print(f'         deviations:')
        for dev in vc.deviations:
            print(f'           {dev[:70]}')
    print(f'         explainer: {vc.fitness_explainer[:80]}')

    assert cp is not None, "Changepoint should be in diagnostics"
    assert vc is not None, "Variant comparison should be in diagnostics"

check('both features: combined on a pipeline with retries + gap step change',
      test_both_features_on_bad_pipeline)


# ═══════════════════════════════════════════════════════════════════════════
# HOW IT AIDS PEOPLE — concrete examples
# ═══════════════════════════════════════════════════════════════════════════

print()
print('═' * 65)
print('  HOW THESE AID ENGINEERS — CONCRETE EXAMPLES')
print('═' * 65)
print()

print('  CHANGEPOINT DETECTION:')
print()
print('  Before:')
print('    "Bilateral gap is widening. Trend: +1.5 min/week."')
print('    Engineer: when did this start? checks 30 days of logs manually')
print()
print('  After:')
print('    "Gap increased by 14.7 min around 2026-03-15.')
print('     Before: 13.2 min mean. After: 27.9 min mean.')
print('     Correlate with deployment logs on or before 2026-03-15."')
print('    Engineer: checks deployments on March 15th.')
print('    Finds: SFTP batch size increased from 500MB to 2GB.')
print('    Fix: reduce batch size or switch to streaming.')
print('    Time saved: 2 hours of log archaeology → 5 minutes.')
print()
print('  Citi use case:')
print('    30 days of VaR batch bilateral gap data.')
print('    Gap was 18 min for 3 weeks, jumped to 33 min overnight.')
print('    Changepoint: detected on 2026-03-22.')
print('    Engineer checks that date: new derivatives book added.')
print('    Derivatives require Monte Carlo — slower to publish.')
print('    Fix: separate derivatives into their own Fracture contract.')
print()

print('  VARIANT COMPARISON:')
print()
print('  Before:')
print('    "Sequence fitness: 0.75. Score: 79% RED."')
print('    Engineer: what went wrong? reads token replay output')
print('    token replay says: missing_tokens=1, remaining_tokens=1')
print('    Engineer: which activity? checks all events manually')
print()
print('  After:')
print('    "Sequence fitness 0.75 is low because: STARTED fires')
print('     3x in 100% of runs. Retries or loops detected.')
print('     If retries are expected: set deduplicate_retries=true."')
print('    Engineer: adds deduplicate_retries=true to contract.')
print('    Next run: fitness=1.0, score=102% GREEN.')
print('    Time saved: 30 minutes of trace investigation → 2 minutes.')
print()
print('  Airflow use case:')
print('    Airflow task retries on transient S3 timeout.')
print('    3 retries per run = STARTED fires 4x in the trace.')
print('    Token replay: missing_tokens=3, fitness=0.70.')
print('    Without variant comparison: engineer investigates wrong thing.')
print('    With variant comparison: "STARTED fires 4x — retries detected."')
print('    Engineer adds deduplicate_retries=true. Fixed in 2 minutes.')
print()


# ── Summary ───────────────────────────────────────────────────────────────────

print('═' * 65)
print(f'  Results: {passed+failed} tests  ✓ {passed}  ✗ {failed}')
print('═' * 65)

if failed > 0:
    sys.exit(1)
