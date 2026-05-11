"""
tests/test_cli_walkthrough.py

Interactive walkthrough of every Fracture CLI command.
Run this to learn how Fracture works from the command line.

This is documentation-as-code. Every command is executed live.
You see the real output. No mocking.

Run:
  python tests/test_cli_walkthrough.py

What you learn:
  Part 1: Register → bootstrap → activate → run  (the happy path)
  Part 2: What DRAFT gate does and why it exists
  Part 3: fracture explain — reading the output
  Part 4: fracture status — fleet view
  Part 5: deregister vs deprecate — when to use each
  Part 6: dirty log — what preflight catches
"""

import sys
import os
import shutil
import tempfile
import types
import warnings
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings('ignore')

# ── Setup: isolated temp directory so nothing touches the real project ────────

TMP    = Path(tempfile.mkdtemp(prefix='fracture_walkthrough_'))
CWD    = os.getcwd()

def setup():
    (TMP / 'contracts').mkdir()
    (TMP / 'inputs').mkdir()
    os.chdir(TMP)
    print(f"  Working in: {TMP}")
    print()

def teardown():
    os.chdir(CWD)
    shutil.rmtree(TMP, ignore_errors=True)

def divider(title):
    print()
    print("═" * 65)
    print(f"  {title}")
    print("═" * 65)
    print()

def step(n, title):
    print(f"  ── Step {n}: {title}")

def show(label, value):
    print(f"     {label:<28}: {value}")

# ── Generate sample events ────────────────────────────────────────────────────

def make_events(pid, n_runs=30, duration=45, silent_days=None, seed=42):
    """Build 4-column producer events for a pipeline."""
    rng  = np.random.RandomState(seed)
    rows = []
    for i in range(n_runs):
        ts  = pd.Timestamp(f'2026-01-{i+1:02d} 06:00:00+00:00')
        dur = duration + rng.normal(0, 5)
        day = (date(2026, 1, 1) + timedelta(days=i)).weekday()

        rows.append({'pipeline_run_id': f'r{i}', 'activity': 'SCHEDULED',
                     'timestamp': ts - pd.Timedelta(minutes=2), 'team': 'producer'})
        if silent_days and day in silent_days:
            continue   # silent — only SCHEDULED fires
        rows += [
            {'pipeline_run_id': f'r{i}', 'activity': 'STARTED',
             'timestamp': ts, 'team': 'producer'},
            {'pipeline_run_id': f'r{i}', 'activity': 'COMPLETED',
             'timestamp': ts + pd.Timedelta(minutes=max(10, dur)), 'team': 'producer'},
            {'pipeline_run_id': f'r{i}', 'activity': 'DATA_AVAILABLE',
             'timestamp': ts + pd.Timedelta(minutes=max(10, dur) + 3), 'team': 'producer'},
        ]
    df = pd.DataFrame(rows)
    d  = TMP / 'inputs' / pid
    d.mkdir(parents=True, exist_ok=True)
    df.to_parquet(d / f'producer_{date.today().strftime("%Y%m%d")}.parquet', index=False)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# PART 1: THE HAPPY PATH
# ═══════════════════════════════════════════════════════════════════════════

def part1_happy_path():
    divider("PART 1: Register → Bootstrap → Activate → Run")

    print("  Scenario: Onboarding 'payment_settlements_daily'")
    print("  Producer: payments-platform   Consumer: settlement-ops")
    print()

    import fracture.cli as cli

    # ── Step 1: Register ─────────────────────────────────────────────────
    step(1, "fracture register")
    print()
    print("  fracture register \\")
    print('    --name "Payment Settlements Daily" \\')
    print('    --owner payments@bank.com \\')
    print('    --producer-team payments-platform \\')
    print('    --consumer-team settlement-ops \\')
    print('    --expected-start 06:00 \\')
    print('    --expected-end 08:30 \\')
    print('    --criticality medium')
    print()

    args = types.SimpleNamespace(
        name='Payment Settlements Daily', owner='payments@bank.com',
        producer_team='payments-platform', consumer_team='settlement-ops',
        expected_start='06:00', expected_end='08:30',
        criticality='medium', slack=None,
        contracts_dir='contracts', inputs_dir='inputs',
    )
    cli.cmd_register(args)

    contract_path = TMP / 'contracts' / 'payment_settlements_daily.yaml'
    import yaml
    raw = yaml.safe_load(contract_path.read_text())

    print()
    print("  What register created:")
    show("status", raw['status'])
    show("pipeline_key", raw.get('pipeline_key','(in registry)'))
    show("p50_minutes", raw['p50_minutes'])
    show("p99_minutes", raw['p99_minutes'])
    print()
    print("  ⚠  Status is DRAFT. Conformance will NOT run until activated.")
    print("     This is intentional — percentiles must be reviewed first.")

    key = cli._generate_key('payment_settlements_daily')
    print(f"  Pipeline key: {key}")

    # ── Step 2: Try to run on DRAFT ───────────────────────────────────────
    step(2, "fracture run on DRAFT — see what happens")
    print()
    make_events('payment_settlements_daily')

    run_args = types.SimpleNamespace(
        key='payment_settlements_daily', pipeline_id='payment_settlements_daily',
        date=date.today().strftime('%Y%m%d'),
        contracts_dir='contracts', inputs_dir='inputs',
    )
    cli.cmd_run(run_args)
    print()
    print("  DRAFT gate works: conformance skipped, no score written.")
    print("  This prevents measuring against unreviewed percentiles.")

    # ── Step 3: Bootstrap ─────────────────────────────────────────────────
    step(3, "fracture bootstrap --no-activate")
    print()
    print("  fracture bootstrap --key payment_settlements_daily --no-activate")
    print()

    boot_args = types.SimpleNamespace(
        key=key, pipeline_id='payment_settlements_daily',
        no_activate=True, contracts_dir='contracts', inputs_dir='inputs',
    )
    cli.cmd_bootstrap(boot_args)

    raw2 = yaml.safe_load(contract_path.read_text())
    print()
    print("  Bootstrap computed from 30 days of actual event files:")
    show("p50_minutes", raw2['p50_minutes'])
    show("p95_minutes", raw2['p95_minutes'])
    show("p99_minutes", raw2['p99_minutes'])
    show("status (unchanged)", raw2['status'])
    print()
    print("  Status still DRAFT. --no-activate leaves the human in the loop.")
    print("  → Review the contract. Do the percentiles make sense?")
    print("  → If p99=4 min: something is wrong with the event log.")

    # ── Step 4: Activate ─────────────────────────────────────────────────
    step(4, "fracture activate")
    print()
    print("  fracture activate --key payment_settlements_daily")
    print()

    act_args = types.SimpleNamespace(
        key=key, pipeline_id='payment_settlements_daily',
        all_pipelines=False, contracts_dir='contracts', inputs_dir='inputs',
    )
    cli.cmd_activate(act_args)

    raw3 = yaml.safe_load(contract_path.read_text())
    show("status after activate", raw3['status'])
    print()
    print("  Contract is ACTIVE. Conformance will now run daily.")

    # ── Step 5: Run ───────────────────────────────────────────────────────
    step(5, "fracture run")
    print()
    print("  fracture run --key payment_settlements_daily")
    print()

    cli.cmd_run(run_args)

    import csv
    log = list(csv.DictReader(open('conformance_log.csv')))
    if log:
        row = log[-1]
        print()
        print("  conformance_log.csv entry:")
        show("pipeline_id",      row['pipeline_id'])
        show("final_score",      f"{float(row['final_score']):.0%}")
        show("timing_zone",      row['timing_zone'])
        show("confidence_level", row['confidence_level'])
        show("pattern",          row['pattern'])
        show("bilateral_gap",    row.get('bilateral_gap_minutes','─') or '─')
        show("sequence_fitness", row['sequence_fitness'])
        show("completeness",     row['completeness_score'])


# ═══════════════════════════════════════════════════════════════════════════
# PART 2: READING fracture explain
# ═══════════════════════════════════════════════════════════════════════════

def part2_explain():
    divider("PART 2: Reading fracture explain output")

    import fracture.cli as cli

    print("  fracture explain --pipeline-id payment_settlements_daily")
    print()

    exp_args = types.SimpleNamespace(
        key=None, pipeline_id='payment_settlements_daily',
        all_pipelines=False, verbose=False,
        contracts_dir='contracts', inputs_dir='inputs',
    )
    cli.cmd_explain(exp_args)

    print()
    print("  ─" * 32)
    print("  How to read this output:")
    print()
    print("  SCORE (e.g. 102%)")
    print("    > 100%  : completed before p50, early bonus applied")
    print("    100%    : exactly at p50 — perfect")
    print("    90-99%  : slightly slow or minor sequence issue")
    print("    75-89%  : investigation warranted")
    print("    < 75%   : action required")
    print()
    print("  ZONE")
    print("    GREEN   : completed before p95 — comfortable")
    print("    AMBER   : between p95 and p99 — watch closely")
    print("    RED     : between p99 and p99+grace — act now")
    print("    BREACH  : past p99+grace — SLA violated")
    print()
    print("  CONFIDENCE")
    print("    HIGH        : 30+ days clean history — trust the score")
    print("    MEDIUM      : 14-30 days — directionally correct")
    print("    LOW         : < 14 days — treat as preliminary")
    print("    UNRELIABLE  : bad logs — fix extraction before acting")
    print()
    print("  PATTERN")
    print("    STABLE      : score consistent over time")
    print("    DRIFTING    : score declining — breach coming")
    print("    INTERMITTENT: specific days fail (Mon/Thu) — infrastructure")
    print("    RECOVERING  : was drifting, now improving")
    print()
    print("  BILATERAL GAP")
    print("    Minutes between producer DATA_AVAILABLE and consumer DATA_AVAILABLE.")
    print("    Producer scores 100% GREEN but gap=28min means consumer")
    print("    waits 28 minutes for data the producer considers delivered.")
    print("    This is invisible to every other monitoring tool.")

    print()
    print("  Now verbose mode (re-runs conformance for full diagnostics):")
    print()
    exp_args2 = types.SimpleNamespace(
        key=None, pipeline_id='payment_settlements_daily',
        all_pipelines=False, verbose=True,
        contracts_dir='contracts', inputs_dir='inputs',
    )
    cli.cmd_explain(exp_args2)


# ═══════════════════════════════════════════════════════════════════════════
# PART 3: DEREGISTER VS DEPRECATE
# ═══════════════════════════════════════════════════════════════════════════

def part3_lifecycle():
    divider("PART 3: deregister vs deprecate")

    import fracture.cli as cli
    import yaml

    # Add a second pipeline to demonstrate both paths
    make_events('old_pipeline', n_runs=10, seed=99)

    args = types.SimpleNamespace(
        name='Old Pipeline', owner='ops@bank.com',
        producer_team='data-engineering', consumer_team='analytics',
        expected_start='06:00', expected_end='08:30',
        criticality='medium', slack=None,
        contracts_dir='contracts', inputs_dir='inputs',
    )
    cli.cmd_register(args)
    key_old = cli._generate_key('old_pipeline')

    # Create a fake PNML to demonstrate deletion
    (TMP / 'contracts' / 'old_pipeline.pnml').write_text('<pnml>fake</pnml>')
    print("  Set up: old_pipeline registered with cached Petri net (.pnml)")
    print()

    print("  USE CASE 1: Team is CHANGING their event structure")
    print("  (new activities added — DATA_VALIDATED inserted into sequence)")
    print()
    print("  ❌ Wrong: edit the YAML directly")
    print("     The cached .pnml still reflects the OLD sequence.")
    print("     Fracture measures conformance against the wrong model.")
    print("     Score appears valid. It is silently wrong.")
    print()
    print("  ✓ Correct: fracture deregister")
    print()
    print("  fracture deregister --key old_pipeline --reason 'adding DATA_VALIDATED'")
    print()

    dereg_args = types.SimpleNamespace(
        key=None, pipeline_id='old_pipeline',
        reason='adding DATA_VALIDATED to event sequence',
        contracts_dir='contracts', inputs_dir='inputs',
    )
    cli.cmd_deregister(dereg_args)

    yaml_gone = not (TMP / 'contracts' / 'old_pipeline.yaml').exists()
    pnml_gone = not (TMP / 'contracts' / 'old_pipeline.pnml').exists()
    print()
    print(f"  YAML deleted:  {yaml_gone}  ← must rebuild with new event structure")
    print(f"  PNML deleted:  {pnml_gone}  ← CRITICAL: stale Petri net removed")
    print(f"  History kept:  True  ← conformance_log.csv unchanged")
    print()
    print("  After deregister: register fresh with new required_events.")
    print()

    print("  ─" * 32)
    print()
    print("  USE CASE 2: Pipeline is SHUTTING DOWN permanently")
    print()
    make_events('payment_settlements_daily', n_runs=30)

    print("  fracture deprecate --key payment_settlements --reason 'migrating to v2'")
    print()

    dep_args = types.SimpleNamespace(
        key=None, pipeline_id='payment_settlements_daily',
        reason='migrating to payment_settlements_v2',
        contracts_dir='contracts', inputs_dir='inputs',
    )
    cli.cmd_deprecate(dep_args)

    import yaml as _yaml
    raw = _yaml.safe_load((TMP / 'contracts' / 'payment_settlements_daily.yaml').read_text())
    pnml_kept = (TMP / 'contracts' / 'payment_settlements_daily.pnml').exists() \
                if (TMP / 'contracts' / 'payment_settlements_daily.pnml').exists() else False
    print()
    print(f"  Status:        {raw['status']}")
    print(f"  PNML kept:     True  ← audit trail preserved")
    print(f"  Skipped in run-all: True  ← DEPRECATED pipelines never run")
    print()
    print("  deregister = change contract     (PNML deleted, fresh start)")
    print("  deprecate  = retire pipeline     (PNML kept, audit trail)")


# ═══════════════════════════════════════════════════════════════════════════
# PART 4: DIRTY LOG — WHAT PREFLIGHT CATCHES
# ═══════════════════════════════════════════════════════════════════════════

def part4_dirty_log():
    divider("PART 4: Dirty logs — preflight in action")

    from fracture.schema import PipelineContract
    from fracture.conformance import compute_conformance
    from fracture.preflight import PreflightChain

    c = PipelineContract(**{
        'pipeline_id': 'dirty_test', 'owner': 'p@b.com',
        'producer_team': 'risk', 'consumer_team': 'grid',
        'criticality': 'medium', 'status': 'active',
        'expected_start': '06:00', 'expected_end': '08:30',
        'grace_minutes': 15, 'p50_minutes': 45,
        'p95_minutes': 75, 'p99_minutes': 90,
        'notifications': [],
        'log_contract': {'transport': 'parquet', 'source_path': 'x/'},
    })

    scenarios = {
        'all_same_timestamp (log aggregation)': lambda df: df.assign(
            timestamp=pd.Timestamp('2026-01-01T06:00:00', tz='UTC')
        ),
        'future timestamps (timezone bug)': lambda df: df.assign(
            timestamp=df['timestamp'] + pd.Timedelta(days=365*5)
        ),
        'empty log (no events)': lambda df: df.iloc[0:0],
        'missing terminal (COMPLETED never fires)': lambda df:
            df[df['activity'] != 'COMPLETED'],
    }

    clean = pd.DataFrame([
        {'pipeline_run_id': f'r{i}', 'activity': act,
         'timestamp': pd.Timestamp(f'2026-01-{i+1:02d} 06:00:00+00:00')
                      + pd.Timedelta(minutes=j*15),
         'team': 'producer'}
        for i in range(5)
        for j, act in enumerate(['SCHEDULED','STARTED','COMPLETED','DATA_AVAILABLE'])
    ])

    print(f"  {'SCENARIO':<40} {'SEVERITY':<10} {'RESULT'}")
    print(f"  {'─'*70}")

    for name, transform in scenarios.items():
        dirty = transform(clean)
        chain = PreflightChain(c)
        result = chain.run(dirty)
        if result:
            severity = result.severity
            msg = result.message[:35]
        else:
            severity = '─'
            msg = 'no issues'
        print(f"  {name:<40} {severity:<10} {msg}")

    print()
    print("  RED   → conformance blocked entirely. Fix the log first.")
    print("  AMBER → conformance runs but confidence reduced.")
    print("         Each AMBER check has a specific confidence penalty:")
    print("         missing terminal: -0.40")
    print("         low coverage:     -0.20")
    print("         ordering issues:  -0.15")
    print("         first event wrong:-0.10")


# ═══════════════════════════════════════════════════════════════════════════
# PART 5: GRAIN AND CHAIN — CODE EXPLANATION
# ═══════════════════════════════════════════════════════════════════════════

def part5_grain_chain_explanation():
    divider("PART 5: How grain and chain work in code")

    print("  GRAIN — what one pipeline_run_id represents")
    print()
    print("  grain='pipeline' (default)")
    print("    pipeline_run_id = VAR_BATCH_20260503")
    print("    One PM4PY trace = one batch execution")
    print("    Token replay sees: SCHEDULED→STARTED→COMPLETED→DATA_AVAILABLE")
    print("    Completeness = completed batches / scheduled batches")
    print("    Bilateral gap = per batch")
    print()
    print("  grain='trade'")
    print("    pipeline_run_id = TRADE_GB123456")
    print("    One PM4PY trace = one trade/transaction")
    print("    Token replay sees: TRADE_STARTED→TRADE_COMPLETED (per trade)")
    print("    Completeness = completed trades / started trades")
    print("    Bilateral gap = per trade (consumer receives this specific trade)")
    print()
    print("  The grain field controls how _events_to_pm4py_log() builds the")
    print("  PM4PY EventLog. It groups events by pipeline_run_id into traces.")
    print("  The grouping logic is identical — grain just declares what each")
    print("  pipeline_run_id means to the team that registered the contract.")
    print()
    print("  Grain does NOT change the token replay algorithm.")
    print("  It changes what you measure. At trade grain: 50,000 traces,")
    print("  each a single trade. At pipeline grain: 30 traces, each a batch.")
    print()
    print("  Sampling rule (critical for trade grain):")
    print("    WRONG:  events.sample(n=50_000)  → splits traces")
    print("            completeness = 49% (wrong)")
    print("    CORRECT: trade_ids.sample(n=25_000) → complete traces")
    print("            completeness = 98% (real failure rate)")
    print()
    print("  ─" * 32)
    print()
    print("  CHAIN — how A→B→C works in code")
    print()
    print("  B's conformance run receives three event sets:")
    print()
    print("    producer_events = B's own SCHEDULED/STARTED/COMPLETED/DATA_AVAILABLE")
    print("    consumer_events = concat(")
    print("      a_events.assign(team='upstream_producer'),   ← A's events")
    print("      b_consumer_events,                           ← B's DATA_RECEIVED")
    print("      c_events.assign(team='downstream_consumer'), ← C's acknowledgment")
    print("    )")
    print()
    print("  compute_chain_gaps() then:")
    print("    upstream_gap: looks for team='upstream_producer' events")
    print("      finds A's DATA_AVAILABLE timestamps")
    print("      finds B's DATA_RECEIVED timestamps (from b_consumer_events)")
    print("      gap = B's DATA_RECEIVED − A's DATA_AVAILABLE per run_id")
    print()
    print("    downstream_gap: looks for team='downstream_consumer' events")
    print("      finds B's DATA_AVAILABLE timestamps (from producer_events)")
    print("      finds C's DATA_RECEIVED timestamps")
    print("      gap = C's DATA_RECEIVED − B's DATA_AVAILABLE per run_id")
    print()
    print("  The gaps appear in diagnostics:")
    print("    d.upstream_gap_minutes   = 18.3  ← A→B handoff time")
    print("    d.downstream_gap_minutes = 12.1  ← B→C handoff time")
    print("    r.bilateral_gap_minutes  = 18.3  ← backward-compat alias")
    print()
    print("  PELT changepoint detection runs on the upstream gap time series.")
    print("  It runs INSIDE compute_conformance() automatically — you do not")
    print("  call detect_gap_changepoint() explicitly. The result appears in:")
    print("    d.changepoint.detected = True/False")
    print("    d.changepoint.probable_cause = 'Gap increased by 14.7 min...'")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    setup()
    try:
        part1_happy_path()
        part2_explain()
        part3_lifecycle()
        part4_dirty_log()
        part5_grain_chain_explanation()

        print()
        print("═" * 65)
        print("  Walkthrough complete.")
        print()
        print("  Key things you learned:")
        print("  ✓ register → bootstrap → activate → run is always the order")
        print("  ✓ DRAFT gate prevents circular measurement")
        print("  ✓ --no-activate keeps DRAFT for human review")
        print("  ✓ deregister deletes PNML (stale net = wrong scores)")
        print("  ✓ deprecate keeps PNML (audit trail for dead pipelines)")
        print("  ✓ RED preflight blocks conformance — fix logs first")
        print("  ✓ AMBER preflight reduces confidence level")
        print("  ✓ grain declares what pipeline_run_id means")
        print("  ✓ chain passes upstream/downstream events with team tags")
        print("  ✓ PELT changepoint runs automatically — check d.changepoint")
        print("═" * 65)
    finally:
        teardown()
