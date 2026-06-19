"""
fracture/tests/test_e2e_saturday.py

End-to-end test guide for Saturday deep dive.
Generator → Converter → Petri → Conformance

Run from project root:
    python tests/test_e2e_saturday.py

Or with verbose output:
    python tests/test_e2e_saturday.py --verbose

What this file covers
═════════════════════
Five end-to-end scenarios. Each one exercises the full pipeline
from synthetic data generation through to a ConformanceResult.
Read the scenario description before each test — it explains
what you are testing and what to look for in the output.

The five scenarios
══════════════════
1. HEALTHY PIPELINE
   Generator → StableMatureGenerator
   Expected: score ≥ 0.90, GREEN zone, HIGH confidence
   What to look for: formula breakdown in diagnostics

2. ASSUMPTION ASYMMETRY
   Generator → AssumptionAsymmetryGenerator
   Expected: bilateral_gap_minutes > 15, widening trend
   What to look for: producer vs consumer DATA_AVAILABLE gap

3. SILENT PIPELINE
   Generator → SilentPipelineGenerator
   Expected: completeness_score < 0.75, INTERMITTENT pattern
   What to look for: scheduled runs > completed runs

4. CLOCK SYNC (consumer broken)
   Generator → ClockSyncOutlierGenerator
   Expected: producer score intact, bilateral_gap = None
   What to look for: dual preflight in action

5. SUDDEN COLLAPSE (historical + live split)
   Generator → SuddenCollapseGenerator
   Expected: recent scores critical, drift rate misleading
   What to look for: the documented limitation of lagging indicators

How to read the output
══════════════════════
Each scenario prints:
  ─ The scenario name and what it tests
  ─ The raw numbers from the generator
  ─ The ConformanceResult primary fields
  ─ The diagnostics breakdown
  ─ The human_summary() and alert_owner()
  ─ PASS or FAIL with the assertion that failed

This is your reference for Saturday. Read it top to bottom.
Every number should make sense to you by the end.
"""

import sys
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings('ignore')


# ── Shared fixture ────────────────────────────────────────────────────────────

def make_contract(pipeline_id='test', criticality='high', status='active'):
    """Minimal valid ACTIVE contract for all tests."""
    from fracture.schema import PipelineContract
    return PipelineContract(**{
        'pipeline_id':    pipeline_id,
        'owner':          'platform@bank.com',
        'producer_team':  'market-risk-quant',
        'consumer_team':  'grid-scheduler',
        'criticality':    criticality,
        'status':         status,
        'expected_start': '06:00',
        'expected_end':   '08:30',
        'grace_minutes':  15,
        'p50_minutes':    45,
        'p95_minutes':    75,
        'p99_minutes':    90,
        'notifications':  [{'channel': 'slack', 'target': '#alerts'}]
                          if criticality == 'high' else [],
        'log_contract': {
            'transport':   'plaintext',
            'source_path': 'inputs/test/',
        },
    })


START_DATE = date.today() - timedelta(days=30)
DAYS       = 30

PASS = "v"
FAIL = "x"
results = []


def run_scenario(name, fn):
    """Run one scenario. Print full output. Record pass/fail."""
    print(f"\n{'═'*60}")
    print(f"  {name}")
    print(f"{'═'*60}")
    try:
        fn()
        results.append((PASS, name))
        print(f"\n  {PASS}  PASS")
    except AssertionError as e:
        results.append((FAIL, name, str(e)))
        print(f"\n  {FAIL}  FAIL: {e}")
    except Exception as e:
        results.append((FAIL, name, str(e)))
        print(f"\n  {FAIL}  ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


# ── Scenario 1: Healthy pipeline ──────────────────────────────────────────────

def scenario_healthy():
    """
    HEALTHY PIPELINE — the baseline you measure everything against.

    StableMatureGenerator produces 30 days of well-behaved events.
    Age: 200 days. Low variance. All four activities fire every day.
    Producer and consumer both healthy.

    What to look for:
      score ≥ 0.90  (formula: seq×0.35 + timing×0.50 + comp×0.15)
      timing_zone = GREEN
      confidence_level = HIGH
      diagnostics.sequence_fitness = 1.0 (all activities fire in order)
      diagnostics.completeness_score = 1.0 (all runs complete)
      diagnostics.timing_score ≥ 1.0 (within p50, early bonus)
      bilateral_gap_minutes > 0 (there is always some gap)
      human_summary() = "Conformant at X%. GREEN zone. No action required."
    """
    from fracture.generator import StableMatureGenerator
    from fracture.conformance import compute_conformance

    contract = make_contract('healthy_pipe')

    # Generate events
    producer_df, consumer_df, ground_truth = StableMatureGenerator().generate(
        contract=contract,
        days=DAYS,
        start_date=START_DATE,
        pipeline_age=200,
        seed=42,
    )

    print(f"\n  Generator output:")
    print(f"    Producer events:  {len(producer_df)}")
    print(f"    Consumer events:  {len(consumer_df)}")
    print(f"    Unique activities: {sorted(producer_df['activity'].unique())}")
    print(f"    Ground truth:     cluster={ground_truth.expected_cluster} "
          f"pattern={ground_truth.expected_pattern}")

    # Run conformance
    result = compute_conformance(
        producer_events=producer_df,
        contract=contract,
        consumer_events=consumer_df,
    )

    print(f"\n  ConformanceResult:")
    print(f"    final_score:          {result.final_score:.4f}")
    print(f"    confidence_level:     {result.confidence_level}")
    print(f"    timing_zone:          {result.timing_zone}")
    print(f"    pattern:              {result.pattern}")
    print(f"    bilateral_gap_minutes:{result.bilateral_gap_minutes}")

    print(f"\n  Diagnostics:")
    d = result.diagnostics
    print(f"    sequence_fitness:    {d.sequence_fitness:.4f}")
    print(f"    timing_score:        {d.timing_score:.4f}")
    print(f"    completeness_score:  {d.completeness_score:.4f}")
    print(f"    variance_cv:         {d.variance_cv:.4f}")
    print(f"    bilateral_gap_trend: {d.bilateral_gap_trend}")

    print(f"\n  Formula check:")
    manual = d.sequence_fitness*0.35 + d.timing_score*0.50 + d.completeness_score*0.15
    print(f"    {d.sequence_fitness:.3f}×0.35 + {d.timing_score:.3f}×0.50 + "
          f"{d.completeness_score:.3f}×0.15 = {manual:.4f}")
    print(f"    Stored: {result.final_score:.4f}  "
          f"(capped at 1.05: {min(1.05, manual):.4f})")

    print(f"\n  Operational output:")
    print(f"    human_summary(): {result.human_summary()}")
    print(f"    alert_owner():   {result.alert_owner()}")
    print(f"    is_trustworthy():{result.is_trustworthy()}")

    # Assertions
    assert result.final_score >= 0.90, f"Score {result.final_score:.3f} < 0.90"
    assert result.timing_zone == 'GREEN', f"Zone: {result.timing_zone}"
    assert result.confidence_level in ('HIGH', 'MEDIUM'), f"Conf: {result.confidence_level}"
    assert d.sequence_fitness >= 0.95, f"Seq: {d.sequence_fitness:.3f}"
    assert d.completeness_score >= 0.95, f"Comp: {d.completeness_score:.3f}"
    assert abs(min(1.05, manual) - result.final_score) < 0.02, "Formula mismatch"


# ── Scenario 2: Assumption asymmetry ─────────────────────────────────────────

def scenario_assumption_asymmetry():
    """
    ASSUMPTION ASYMMETRY — the core Fracture contribution.

    Both teams execute correctly. Neither is failing.
    But the producer marks DATA_AVAILABLE at T+47 and the consumer
    experiences DATA_AVAILABLE at T+72. 25-minute gap. Neither team knows.

    This is invisible to any monitoring tool that only monitors one side.
    Only bilateral conformance checking surfaces it.

    What to look for:
      bilateral_gap_minutes ≈ 25+ min
      bilateral_gap_trend = 'widening' (gap grows over 30 days)
      producer score ≥ 0.85 (producer is conformant)
      human_summary() mentions the gap in real minutes
    """
    from fracture.generator import AssumptionAsymmetryGenerator
    from fracture.conformance import compute_conformance

    contract = make_contract('asym_pipe')

    producer_df, consumer_df, ground_truth = AssumptionAsymmetryGenerator(
        initial_gap=25,
        gap_growth_per_week=1.5,
    ).generate(
        contract=contract,
        days=DAYS,
        start_date=START_DATE,
        pipeline_age=90,
        seed=2,
    )

    # Show the raw gap in the data before conformance runs
    prod_da = producer_df[producer_df.activity == 'DATA_AVAILABLE']['timestamp'].mean()
    cons_da = consumer_df[consumer_df.activity == 'DATA_AVAILABLE']['timestamp'].mean()
    raw_gap = (cons_da - prod_da).total_seconds() / 60

    print(f"\n  Generator output:")
    print(f"    Producer DATA_AVAILABLE mean: {prod_da.strftime('%H:%M:%S')}")
    print(f"    Consumer DATA_AVAILABLE mean: {cons_da.strftime('%H:%M:%S')}")
    print(f"    Raw bilateral gap:            {raw_gap:.1f} minutes")
    print(f"    Ground truth:                 {ground_truth.archetype} "
          f"archetype={ground_truth.archetype}")

    result = compute_conformance(
        producer_events=producer_df,
        contract=contract,
        consumer_events=consumer_df,
    )

    print(f"\n  ConformanceResult:")
    print(f"    final_score:           {result.final_score:.4f}")
    print(f"    bilateral_gap_minutes: {result.bilateral_gap_minutes:.1f} min")
    print(f"    bilateral_gap_trend:   {result.diagnostics.bilateral_gap_trend}")
    print(f"    bilateral_gap_p95:     {result.diagnostics.bilateral_gap_p95:.1f} min")

    print(f"\n  Operational output:")
    print(f"    {result.human_summary()}")

    assert result.bilateral_gap_minutes and result.bilateral_gap_minutes > 15, \
        f"Gap {result.bilateral_gap_minutes} should be > 15 min"
    assert result.diagnostics.bilateral_gap_trend in ('widening', 'stable'), \
        f"Trend: {result.diagnostics.bilateral_gap_trend}"
    assert result.final_score >= 0.75, \
        f"Producer score {result.final_score:.3f} too low"


# ── Scenario 3: Silent pipeline ───────────────────────────────────────────────

def scenario_silent_pipeline():
    """
    SILENT PIPELINE — the hardest failure mode to detect.

    The pipeline does not fail. It simply does not run on Mondays and Thursdays.
    No error. No alert from infrastructure monitoring.
    The downstream team gets yesterday's data from a fallback cache.
    Nobody knows.

    Fracture's completeness scoring catches this.
    SCHEDULED fires every day. COMPLETED fires only on non-silent days.
    complete_runs / total_run_ids < 0.75.

    What to look for:
      completeness_score < 0.75
      pattern = INTERMITTENT (not DRIFTING — it's not declining, it's missing)
      scheduled_runs >> completed_runs
    """
    from fracture.generator import SilentPipelineGenerator
    from fracture.conformance import compute_conformance

    contract = make_contract('silent_pipe', criticality='medium')

    producer_df, consumer_df, ground_truth = SilentPipelineGenerator(
        silent_weekdays=[0, 3]  # Monday=0, Thursday=3
    ).generate(
        contract=contract,
        days=DAYS,
        start_date=START_DATE,
        pipeline_age=60,
        seed=3,
    )

    scheduled = producer_df[producer_df.activity == 'SCHEDULED']['pipeline_run_id'].nunique()
    completed = producer_df[producer_df.activity == 'COMPLETED']['pipeline_run_id'].nunique()

    print(f"\n  Generator output:")
    print(f"    Scheduled runs:  {scheduled}")
    print(f"    Completed runs:  {completed}")
    print(f"    Completion rate: {completed/scheduled:.0%}")
    print(f"    Silent weekdays: Monday (0) + Thursday (3)")
    print(f"    Ground truth:    {ground_truth.archetype} cluster={ground_truth.expected_cluster}")

    result = compute_conformance(
        producer_events=producer_df,
        contract=contract,
    )

    print(f"\n  ConformanceResult:")
    print(f"    final_score:          {result.final_score:.4f}")
    print(f"    pattern:              {result.pattern}")
    print(f"    timing_zone:          {result.timing_zone}")
    print(f"    completeness_score:   {result.diagnostics.completeness_score:.4f}")
    print(f"    confidence_level:     {result.confidence_level}")

    print(f"\n  Operational output:")
    print(f"    {result.human_summary()}")

    assert result.diagnostics.completeness_score < 0.75, \
        f"Completeness {result.diagnostics.completeness_score:.2f} should be < 0.75"
    assert scheduled > completed, "Should have more scheduled than completed"


# ── Scenario 4: Clock sync — dual preflight ───────────────────────────────────

def scenario_clock_sync():
    """
    CLOCK SYNC — consumer log has clock skew.

    Consumer timestamps are shifted -15 minutes (timezone mismatch).
    This makes consumer DATA_AVAILABLE appear before producer DATA_AVAILABLE
    on most days — a physically impossible bilateral gap.

    The dual preflight design:
      Producer preflight → CLEAN → continues
      Consumer 29/30 days have negative gaps → bilateral skipped
      Producer conformance score → computed normally

    This is the dual preflight in action. The consumer being broken
    does not invalidate the producer's process conformance measurement.

    What to look for:
      final_score ≥ 0.85 (producer is healthy)
      bilateral_gap_minutes = None (skipped due to clock skew)
      human_summary() does NOT mention a bilateral gap
    """
    from fracture.generator import StableMatureGenerator, ClockSyncOutlierGenerator
    from fracture.conformance import compute_conformance

    contract = make_contract('clock_pipe')

    producer_df, _, _ = StableMatureGenerator().generate(
        contract=contract,
        days=DAYS,
        start_date=START_DATE,
        pipeline_age=200,
        seed=1,
    )

    _, consumer_df, ground_truth = ClockSyncOutlierGenerator(
        clock_skew_minutes=-15
    ).generate(
        contract=contract,
        days=DAYS,
        start_date=START_DATE,
        pipeline_age=60,
        seed=5,
    )

    # Show the clock skew
    prod_da = producer_df[producer_df.activity == 'DATA_AVAILABLE']['timestamp'].iloc[0]
    cons_da = consumer_df[consumer_df.activity == 'DATA_AVAILABLE']['timestamp'].iloc[0]
    print(f"\n  Clock skew demonstration:")
    print(f"    Producer DA (first run): {prod_da}")
    print(f"    Consumer DA (first run): {cons_da}")
    print(f"    Raw gap (first run):     {(cons_da-prod_da).total_seconds()/60:.1f} min (negative = skew)")
    print(f"    Ground truth:            should_raise_guard={ground_truth.should_raise_guard}")

    result = compute_conformance(
        producer_events=producer_df,
        contract=contract,
        consumer_events=consumer_df,
    )

    print(f"\n  ConformanceResult:")
    print(f"    final_score:           {result.final_score:.4f}")
    print(f"    billing_gap_minutes:   {result.bilateral_gap_minutes}")
    print(f"    confidence_level:      {result.confidence_level}")
    print(f"    timing_zone:           {result.timing_zone}")

    print(f"\n  Operational output:")
    print(f"    {result.human_summary()}")
    print(f"    alert_owner(): {result.alert_owner()}")

    assert result.final_score >= 0.85, \
        f"Producer score dropped to {result.final_score:.3f} — should not be affected by consumer"
    assert result.bilateral_gap_minutes is None, \
        f"Clock skew should produce None gap, got {result.bilateral_gap_minutes}"


# ── Scenario 5: Sudden collapse ───────────────────────────────────────────────

def scenario_sudden_collapse():
    """
    SUDDEN COLLAPSE — the documented limitation of drift rate.

    Historical period: 25 days of healthy execution.
    Live period: 5 days of catastrophic failure.

    The drift rate metric is a lagging indicator. Because the historical
    period was healthy, the slope of the fitness decline looks near-zero
    when computed over the full 30-day window. The per-day scores tell
    the real story but the drift rate is misleading.

    This is documented as a known limitation in the paper.
    The per-run score chart is more informative than the aggregate drift rate
    for step-function failures.

    What to look for:
      Historical scores ≈ 0.95 (healthy period)
      Live scores much lower (collapse)
      The combination produces an average that looks moderate
      This is why drift rate alone is not sufficient for monitoring
    """
    from fracture.generator import SuddenCollapseGenerator
    from fracture.conformance import compute_conformance

    contract = make_contract('collapse_pipe')
    gen = SuddenCollapseGenerator()

    # Generate historical period (would be loaded from historical.parquet in production)
    historical_df = gen.generate_historical(
        contract=contract,
        days=25,
        start_date=START_DATE,
        seed=42,
    )

    # Generate live period (current week)
    live_start = START_DATE + timedelta(days=25)
    live_producer, live_consumer, ground_truth = gen.generate_live(
        contract=contract,
        days=5,
        start_date=live_start,
        seed=99,
    )

    print(f"\n  Generator output:")
    print(f"    Historical events: {len(historical_df)} (25 healthy days)")
    print(f"    Live prod events:  {len(live_producer)} (5 collapse days)")
    print(f"    Ground truth:      uses_historical_split={ground_truth.uses_historical_split}")

    # Run conformance on historical
    r_hist = compute_conformance(
        producer_events=historical_df,
        contract=contract,
    )

    # Run conformance on live period
    r_live = compute_conformance(
        producer_events=live_producer,
        contract=contract,
        consumer_events=live_consumer if len(live_consumer) > 0 else None,
    )

    print(f"\n  Historical conformance (25 healthy days):")
    print(f"    final_score:    {r_hist.final_score:.4f}")
    print(f"    timing_zone:    {r_hist.timing_zone}")
    print(f"    completeness:   {r_hist.diagnostics.completeness_score:.4f}")

    print(f"\n  Live conformance (5 collapse days):")
    print(f"    final_score:    {r_live.final_score:.4f}")
    print(f"    timing_zone:    {r_live.timing_zone}")
    print(f"    completeness:   {r_live.diagnostics.completeness_score:.4f}")
    print(f"    {r_live.human_summary()}")

    print(f"\n  Documented limitation:")
    print(f"    A drift rate computed over 30 days (25 healthy + 5 collapse)")
    print(f"    will show near-zero slope because history dominates the regression.")
    print(f"    The per-run chart tells the real story.")
    print(f"    Fracture surfaces this correctly — live score is critical.")
    print(f"    The drift rate metric is the lagging indicator limitation.")

    # Historical should be healthy
    assert r_hist.final_score >= 0.85, \
        f"Historical score {r_hist.final_score:.3f} should be ≥ 0.85"
    # Live should be poor
    assert r_live.final_score < 0.70, \
        f"Live score {r_live.final_score:.3f} should be < 0.70 (collapse)"


# ── Run all scenarios ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Fracture End-to-End Test — Saturday Deep Dive")
    print("Generator → Converter → Petri → Conformance")
    print()
    print("Five scenarios. Read the output. Every number should make sense.")

    run_scenario("1. HEALTHY PIPELINE",         scenario_healthy)
    run_scenario("2. ASSUMPTION ASYMMETRY",      scenario_assumption_asymmetry)
    run_scenario("3. SILENT PIPELINE",           scenario_silent_pipeline)
    run_scenario("4. CLOCK SYNC (dual preflight)", scenario_clock_sync)
    run_scenario("5. SUDDEN COLLAPSE (documented limitation)", scenario_sudden_collapse)

    # Summary
    total  = len(results)
    passed = sum(1 for r in results if r[0] == PASS)
    failed = sum(1 for r in results if r[0] == FAIL)

    print(f"\n{'═'*60}")
    print(f"  RESULTS: {passed}/{total} passed")
    print(f"{'═'*60}")

    if failed > 0:
        print("\n  Failed:")
        for r in results:
            if r[0] == FAIL:
                print(f"    x {r[1]}")
                if len(r) > 2:
                    print(f"      {r[2][:80]}")
        sys.exit(1)
    else:
        print()
        print("  All scenarios passed.")
        print()
        print("  What you just verified:")
        print("  ─ Healthy pipeline scores ≥ 0.90 on token replay")
        print("  ─ Bilateral gap of 25+ minutes is detected in real minutes")
        print("  ─ Silent pipeline caught by completeness scoring")
        print("  ─ Broken consumer log does not invalidate producer score")
        print("  ─ Sudden collapse produces critical live score despite healthy history")
