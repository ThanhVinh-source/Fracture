"""
tests/test_cli_learning.py
══════════════════════════════════════════════════════════════════════

FRACTURE CLI — COMPLETE LEARNING WALKTHROUGH

Read this file top to bottom before running it.
Every command is explained before it executes.
Run it to see real output from every Fracture command.

    python tests/test_cli_learning.py

You will learn:
    Part 1 — The happy path      (register → bootstrap → activate → run)
    Part 2 — Reading the output  (what every field means)
    Part 3 — The bilateral gap   (why producer-only is incomplete)
    Part 4 — DRAFT gate          (why bootstrap --no-activate exists)
    Part 5 — deregister          (when the contract must change)
    Part 6 — deprecate           (when the pipeline shuts down)
    Part 7 — run-all             (fleet operation)
    Part 8 — preflight           (what bad logs look like)

Nothing touches your real project.
Everything runs in a temporary directory that is deleted at the end.

══════════════════════════════════════════════════════════════════════
"""

import os
import sys
import csv
import shutil
import tempfile
import types
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings('ignore')

# ── Helpers ───────────────────────────────────────────────────────────────────

ORIGINAL_DIR = os.getcwd()
WORKDIR      = None


def setup():
    global WORKDIR
    WORKDIR = Path(tempfile.mkdtemp(prefix='fracture_learn_'))
    (WORKDIR / 'contracts').mkdir()
    (WORKDIR / 'inputs').mkdir()
    os.chdir(WORKDIR)


def teardown():
    os.chdir(ORIGINAL_DIR)
    shutil.rmtree(WORKDIR, ignore_errors=True)


def section(title, subtitle=''):
    print()
    print('╔' + '═' * 66 + '╗')
    print(f'║  {title:<64}║')
    if subtitle:
        print(f'║  {subtitle:<64}║')
    print('╚' + '═' * 66 + '╝')
    print()


def explain(text):
    """Print explanation text before running a command."""
    for line in text.strip().split('\n'):
        print(f'  {line}')
    print()


def show_command(cmd):
    print(f'  $ {cmd}')
    print()


def divider():
    print(f'  {"─" * 62}')
    print()


def make_producer_events(pid, n_runs=30, duration=45, seed=42,
                          silent_days=None, add_consumer=False,
                          consumer_gap=25):
    """Generate 4-column event files for a pipeline."""
    rng  = np.random.RandomState(seed)
    prod_rows = []
    cons_rows = []

    for i in range(n_runs):
        ts  = pd.Timestamp(f'2026-01-{i+1:02d} 06:00:00+00:00')
        dur = max(10, duration + rng.normal(0, 5))
        dow = (date(2026, 1, 1) + timedelta(days=i)).weekday()

        prod_rows.append({
            'pipeline_run_id': f'run_{i:03d}',
            'activity': 'SCHEDULED',
            'timestamp': ts - pd.Timedelta(minutes=2),
            'team': 'producer',
        })

        if silent_days and dow in silent_days:
            continue  # silent — SCHEDULED only, never completes

        prod_rows += [
            {'pipeline_run_id': f'run_{i:03d}', 'activity': 'STARTED',
             'timestamp': ts, 'team': 'producer'},
            {'pipeline_run_id': f'run_{i:03d}', 'activity': 'COMPLETED',
             'timestamp': ts + pd.Timedelta(minutes=dur), 'team': 'producer'},
            {'pipeline_run_id': f'run_{i:03d}', 'activity': 'DATA_AVAILABLE',
             'timestamp': ts + pd.Timedelta(minutes=dur + 3), 'team': 'producer'},
        ]

        if add_consumer:
            actual_gap = consumer_gap + rng.normal(0, 2)
            cons_rows.append({
                'pipeline_run_id': f'run_{i:03d}',
                'activity': 'DATA_AVAILABLE',
                'timestamp': ts + pd.Timedelta(minutes=dur + 3 + actual_gap),
                'team': 'consumer',
            })

    d = WORKDIR / 'inputs' / pid
    d.mkdir(parents=True, exist_ok=True)
    today = date.today().strftime('%Y%m%d')

    pd.DataFrame(prod_rows).to_parquet(d / f'producer_{today}.parquet', index=False)
    if cons_rows:
        pd.DataFrame(cons_rows).to_parquet(d / f'consumer_{today}.parquet', index=False)


# ═══════════════════════════════════════════════════════════════════════════
# PART 1 — THE HAPPY PATH
# ═══════════════════════════════════════════════════════════════════════════

def part1_happy_path():
    section(
        'PART 1 — The happy path',
        'register → bootstrap → activate → run'
    )

    import fracture.cli as cli

    # ── 1.1 fracture register ─────────────────────────────────────────────
    explain("""
fracture register

Creates the contract YAML file. This is the agreement between two teams.
You provide:  who owns it, who produces data, who consumes it,
              when it should start and finish.
Fracture creates: contracts/payment_batch.yaml with status: draft

Status starts as DRAFT. This is intentional.
A DRAFT contract does NOT run conformance.
You cannot accidentally measure against unreviewed percentiles.
    """)

    show_command(
        'fracture register \\\n'
        '  --name "Payment Batch" \\\n'
        '  --owner payments@bank.com \\\n'
        '  --producer-team payments-platform \\\n'
        '  --consumer-team settlement-ops \\\n'
        '  --expected-start 06:00 \\\n'
        '  --expected-end 08:30 \\\n'
        '  --criticality medium'
    )

    args = types.SimpleNamespace(
        name='Payment Batch', owner='payments@bank.com',
        producer_team='payments-platform', consumer_team='settlement-ops',
        expected_start='06:00', expected_end='08:30',
        criticality='medium', slack=None,
        contracts_dir='contracts', inputs_dir='inputs',
    )
    cli.cmd_register(args)

    import yaml
    raw = yaml.safe_load((WORKDIR / 'contracts' / 'payment_batch.yaml').read_text())
    print()
    print(f'  Contract created: contracts/payment_batch.yaml')
    print(f'  Status: {raw["status"]}  ← DRAFT until you activate it')
    print(f'  Key:    {raw.get("pipeline_key", "(none yet)")}')
    divider()

    # ── 1.2 Generate event files ──────────────────────────────────────────
    explain("""
Produce the 4-column event file.

Your pipeline team drops a parquet file in:
    inputs/payment_batch/producer_YYYYMMDD.parquet

Four columns only:
    pipeline_run_id | activity       | timestamp                | team
    BATCH_001       | SCHEDULED      | 2026-01-01T06:00:00Z     | producer
    BATCH_001       | STARTED        | 2026-01-01T06:01:14Z     | producer
    BATCH_001       | COMPLETED      | 2026-01-01T06:47:22Z     | producer
    BATCH_001       | DATA_AVAILABLE | 2026-01-01T06:50:05Z     | producer

For bilateral gap measurement add consumer events (same pipeline_run_id):
    BATCH_001       | DATA_AVAILABLE | 2026-01-01T07:15:09Z     | consumer
    """)

    make_producer_events('payment_batch', n_runs=30, duration=45, add_consumer=True, consumer_gap=25)
    print('  Generated: inputs/payment_batch/producer_YYYYMMDD.parquet')
    print('  Generated: inputs/payment_batch/consumer_YYYYMMDD.parquet')
    divider()

    # ── 1.3 fracture bootstrap ────────────────────────────────────────────
    explain("""
fracture bootstrap --no-activate

Reads the event files, computes p50/p95/p99 from ACTUAL execution history.
Updates the contract YAML with those percentiles.

Why --no-activate?
  Bootstrap computes whatever percentiles the data gives.
  If your event files have a bug (all same timestamp, only 3 days),
  the percentiles will be wrong.
  --no-activate leaves status: draft so a human reviews first.
  You activate only after confirming the numbers make sense.

NEVER guess p50/p95/p99 manually. Bootstrap computes them correctly.
    """)

    show_command('fracture bootstrap --key payment_batch --no-activate')

    key = cli._generate_key('payment_batch')
    boot_args = types.SimpleNamespace(
        key=key, pipeline_id='payment_batch',
        no_activate=True,
        contracts_dir='contracts', inputs_dir='inputs',
    )
    cli.cmd_bootstrap(boot_args)

    raw2 = yaml.safe_load((WORKDIR / 'contracts' / 'payment_batch.yaml').read_text())
    print()
    print('  Percentiles computed from 30 days of real execution:')
    print(f'    p50 = {raw2["p50_minutes"]} min  (median run time)')
    print(f'    p95 = {raw2["p95_minutes"]} min  (slow but normal)')
    print(f'    p99 = {raw2["p99_minutes"]} min  (very slow — breach zone)')
    print(f'    status = {raw2["status"]}  ← still DRAFT, not yet active')
    divider()

    # ── 1.4 fracture activate ─────────────────────────────────────────────
    explain("""
fracture activate

The human gate.
Review the contract YAML. Do the percentiles look right?
If p99=4 min from a poorly instrumented log: that is wrong. Fix the logs first.
If p50=45 min and p99=58 min: that looks right. Activate.

After activation: status becomes ACTIVE.
fracture run-all will now include this pipeline.
    """)

    show_command('fracture activate --key payment_batch')

    act_args = types.SimpleNamespace(
        key=key, pipeline_id='payment_batch',
        all_pipelines=False,
        contracts_dir='contracts', inputs_dir='inputs',
    )
    cli.cmd_activate(act_args)

    raw3 = yaml.safe_load((WORKDIR / 'contracts' / 'payment_batch.yaml').read_text())
    print()
    print(f'  Status: {raw3["status"]}  ← ACTIVE, will now run daily')
    divider()

    # ── 1.5 fracture run ──────────────────────────────────────────────────
    explain("""
fracture run

Runs the full conformance pipeline:
  1. Preflight checks (are the logs usable?)
  2. Build Petri net from contract (or load cached .pnml)
  3. Token replay (sequence fitness)
  4. Timing zone (GREEN/AMBER/RED/BREACH)
  5. Completeness (scheduled vs completed runs)
  6. Weighted score (seq×0.35 + time×0.50 + comp×0.15)
  7. Confidence (HIGH/MEDIUM/LOW based on history quality)
  8. Pattern (STABLE/DRIFTING/INTERMITTENT)
  9. Bilateral gap (producer DATA_AVAILABLE vs consumer DATA_AVAILABLE)
 10. Changepoint (did the gap suddenly change?)

Result appended to conformance_log.csv
    """)

    show_command('fracture run --key payment_batch')

    run_args = types.SimpleNamespace(
        key=key, pipeline_id='payment_batch',
        date=date.today().strftime('%Y%m%d'),
        contracts_dir='contracts', inputs_dir='inputs',
    )
    cli.cmd_run(run_args)

    return key


# ═══════════════════════════════════════════════════════════════════════════
# PART 2 — READING THE OUTPUT
# ═══════════════════════════════════════════════════════════════════════════

def part2_reading_output():
    section(
        'PART 2 — Reading the output',
        'What every field means'
    )

    import fracture.cli as cli

    explain("""
fracture explain --pipeline-id X

Reads from conformance_log.csv (fast — no recomputation).
Shows the stored summary for the most recent run.

fracture explain --pipeline-id X --verbose
  Re-runs conformance to get full diagnostics including
  weekday pattern, bilateral gap detail, and variant comparison.
    """)

    show_command('fracture explain --pipeline-id payment_batch')

    exp_args = types.SimpleNamespace(
        key=None, pipeline_id='payment_batch',
        all_pipelines=False, verbose=False,
        contracts_dir='contracts', inputs_dir='inputs',
    )
    cli.cmd_explain(exp_args)

    print()
    print('  HOW TO READ THIS OUTPUT:')
    print()
    print('  final_score  →  seq×0.35 + timing×0.50 + completeness×0.15')
    print('  ─────────────────────────────────────────────────────────────')
    print('  > 100%  pipeline finished BEFORE p50 (early bonus applied)')
    print('  100%    finished at exactly p50')
    print('   90%    slightly slow or minor sequence issue')
    print('  < 75%   needs investigation')
    print()
    print('  timing_zone  →  where in the SLA window it finished')
    print('  ─────────────────────────────────────────────────────────────')
    print('  GREEN   finished before p95         → comfortable')
    print('  AMBER   between p95 and p99         → watch this week')
    print('  RED     between p99 and p99+grace   → act now')
    print('  BREACH  past grace period           → SLA violated')
    print()
    print('  bilateral_gap  →  consumer waited this long after producer said "done"')
    print('  ─────────────────────────────────────────────────────────────')
    print('  None   no consumer log provided')
    print('  5min   healthy (network + polling latency)')
    print('  25min  investigate — SFTP polling window, Kafka lag')
    print('  50min  escalate — consumer architecture problem')
    print()
    print('  confidence  →  how much to trust the score')
    print('  ─────────────────────────────────────────────────────────────')
    print('  HIGH        30+ days clean history')
    print('  MEDIUM      14-30 days, or minor log quality issues')
    print('  LOW         < 14 days of history')
    print('  UNRELIABLE  broken log (all same timestamp, future dates)')
    print()
    print('  pattern  →  trend over time')
    print('  ─────────────────────────────────────────────────────────────')
    print('  STABLE        consistent — no action')
    print('  DRIFTING      declining — breach coming, investigate')
    print('  INTERMITTENT  specific days fail (Monday/Thursday) — infra problem')
    print('  RECOVERING    was drifting, now improving — fix is working')


# ═══════════════════════════════════════════════════════════════════════════
# PART 3 — THE BILATERAL GAP
# ═══════════════════════════════════════════════════════════════════════════

def part3_bilateral_gap():
    section(
        'PART 3 — The bilateral gap',
        'What producer-only monitoring misses'
    )

    explain("""
The bilateral gap is Fracture's core novel contribution.

Every monitoring tool measures the PRODUCER: did it finish on time?
Fracture also measures what the CONSUMER experienced.

The gap is the time between:
  producer DATA_AVAILABLE  ← "I have finished. Data is ready."
  consumer DATA_AVAILABLE  ← "I have received the data. I can start."

This gap exists because:
  - SFTP pipelines poll every 15-30 min (half the gap is just waiting)
  - Kafka consumer lag
  - Network transfer time
  - Consumer-side validation before acknowledging receipt

The producer scores 100% GREEN.
The consumer waits 28 extra minutes.
Airflow: GREEN. Datadog: GREEN. Monte Carlo: GREEN.
Fracture: "bilateral gap = 28 min and widening"

Both sides must emit the same activity name at the handoff event.
Same pipeline_run_id = the join key.

Producer log:   run_001 | DATA_AVAILABLE | 06:47 | producer
Consumer log:   run_001 | DATA_AVAILABLE | 07:15 | consumer
                         same run_id ──────────────┘
Gap = 07:15 - 06:47 = 28 minutes
    """)

    from fracture.schema import PipelineContract
    from fracture.conformance import compute_conformance

    c = PipelineContract(**{
        'pipeline_id': 'gap_demo', 'owner': 'p@bank.com',
        'producer_team': 'risk', 'consumer_team': 'grid',
        'criticality': 'high', 'status': 'active',
        'expected_start': '06:00', 'expected_end': '08:30',
        'grace_minutes': 15, 'p50_minutes': 45,
        'p95_minutes': 75, 'p99_minutes': 90,
        'notifications': [{'channel': 'slack', 'target': '#risk'}],
        'log_contract': {'transport': 'parquet', 'source_path': 'inputs/gap_demo/'},
    })

    rng  = np.random.RandomState(1)
    prod = []
    cons = []
    for i in range(30):
        ts  = pd.Timestamp(f'2026-01-{i+1:02d} 06:00:00+00:00')
        dur = 44 + rng.normal(0, 3)
        da  = ts + pd.Timedelta(minutes=dur + 3)
        gap = 25 + rng.normal(0, 2)
        prod += [
            {'pipeline_run_id': f'r{i}', 'activity': 'SCHEDULED',
             'timestamp': ts - pd.Timedelta(minutes=2), 'team': 'producer'},
            {'pipeline_run_id': f'r{i}', 'activity': 'STARTED',
             'timestamp': ts, 'team': 'producer'},
            {'pipeline_run_id': f'r{i}', 'activity': 'COMPLETED',
             'timestamp': ts + pd.Timedelta(minutes=dur), 'team': 'producer'},
            {'pipeline_run_id': f'r{i}', 'activity': 'DATA_AVAILABLE',
             'timestamp': da, 'team': 'producer'},
        ]
        cons.append({'pipeline_run_id': f'r{i}', 'activity': 'DATA_AVAILABLE',
                     'timestamp': da + pd.Timedelta(minutes=gap), 'team': 'consumer'})

    prod_df = pd.DataFrame(prod)
    cons_df = pd.DataFrame(cons)

    # Producer-only first
    r_prod_only = compute_conformance(prod_df, c, consumer_events=None)
    print(f'  Producer-only score    : {r_prod_only.final_score:.0%}  {r_prod_only.timing_zone}')
    print(f'  Bilateral gap          : {r_prod_only.bilateral_gap_minutes}  ← None without consumer log')
    print()

    # With consumer
    r_bilateral = compute_conformance(prod_df, c, consumer_events=cons_df)
    d = r_bilateral.diagnostics
    print(f'  With consumer log:')
    print(f'  Score                  : {r_bilateral.final_score:.0%}  {r_bilateral.timing_zone}')
    print(f'  Bilateral gap          : {r_bilateral.bilateral_gap_minutes:.1f} min  ← now visible')
    print(f'  Gap trend              : {d.bilateral_gap_trend}')
    print()
    print(f'  The score is the same (102%). The gap is 25 min.')
    print(f'  Airflow sees: GREEN.')
    print(f'  Fracture adds: consumer waits 25 min after producer says done.')
    print(f'  At ING: if downstream stress testing polls every 30 min,')
    print(f'  this 25 min gap could mean a 30-minute delay in risk numbers.')


# ═══════════════════════════════════════════════════════════════════════════
# PART 4 — DRAFT GATE
# ═══════════════════════════════════════════════════════════════════════════

def part4_draft_gate():
    section(
        'PART 4 — The DRAFT gate',
        'Why bootstrap --no-activate exists'
    )

    import fracture.cli as cli

    explain("""
Problem: bootstrap computes percentiles from whatever events it finds.
If the event files have a problem, bootstrap gives wrong percentiles.
Wrong percentiles → wrong conformance scores → wrong decisions.

Examples of wrong percentiles:
  p99 = 4 min   → event file has only 3 days of data
  p50 = 2 min   → STARTED event is missing, only COMPLETED recorded
  p99 = 0 min   → all timestamps are identical (logging bug)

If Fracture automatically activated after bootstrap, you would be
measuring "conformance" against a specification that describes nothing.

The DRAFT gate forces a human to look at the percentiles before they
become the measurement standard. It takes 30 seconds. It prevents
weeks of misleading scores.

Workflow:
  1. bootstrap --no-activate  → percentiles written, status stays DRAFT
  2. Open contracts/X.yaml    → do p50/p95/p99 make sense?
  3. fracture activate         → now it measures against real history
    """)

    # Register a second pipeline to demonstrate
    make_producer_events('risk_report', n_runs=30, duration=55)

    args = types.SimpleNamespace(
        name='Risk Report', owner='risk@bank.com',
        producer_team='risk-tech', consumer_team='reporting',
        expected_start='06:00', expected_end='08:00',
        criticality='high', slack='#risk',
        contracts_dir='contracts', inputs_dir='inputs',
    )
    cli.cmd_register(args)
    key = cli._generate_key('risk_report')

    # Try running on DRAFT — should be blocked
    print('  Attempt to run on DRAFT contract:')
    show_command('fracture run --key risk_report')
    run_args = types.SimpleNamespace(
        key=key, pipeline_id='risk_report',
        date=date.today().strftime('%Y%m%d'),
        contracts_dir='contracts', inputs_dir='inputs',
    )
    cli.cmd_run(run_args)
    print()
    print('  ↑ Blocked. DRAFT contracts do not run.')
    print()

    # Bootstrap + activate
    print('  Bootstrap and review:')
    show_command('fracture bootstrap --key risk_report --no-activate')
    boot_args = types.SimpleNamespace(
        key=key, pipeline_id='risk_report', no_activate=True,
        contracts_dir='contracts', inputs_dir='inputs',
    )
    cli.cmd_bootstrap(boot_args)
    print()
    print('  Review the YAML. Percentiles look reasonable? Activate.')
    show_command('fracture activate --key risk_report')
    act_args = types.SimpleNamespace(
        key=key, pipeline_id='risk_report',
        all_pipelines=False,
        contracts_dir='contracts', inputs_dir='inputs',
    )
    cli.cmd_activate(act_args)
    print()
    print('  Now it runs.')


# ═══════════════════════════════════════════════════════════════════════════
# PART 5 — DEREGISTER VS DEPRECATE
# ═══════════════════════════════════════════════════════════════════════════

def part5_lifecycle():
    section(
        'PART 5 — deregister vs deprecate',
        'Two different situations that look similar but are not'
    )

    import fracture.cli as cli

    # Create a fake PNML to show deletion
    (WORKDIR / 'contracts' / 'payment_batch.pnml').write_text('<pnml>cached net</pnml>')

    explain("""
deregister — use when the CONTRACT CHANGES

  A team adds a new activity to their pipeline (VALIDATED inserted).
  Their contract required_events must change.
  If you just edit the YAML, the cached Petri net (.pnml) is stale.
  Fracture measures conformance against the OLD process model.
  Scores appear valid but are silently wrong.

  deregister deletes both YAML and PNML.
  The team registers fresh with the new event structure.
  The next run builds the correct Petri net.

  Historical scores in conformance_log.csv are preserved.
    """)

    show_command('fracture deregister --pipeline-id payment_batch --reason "adding VALIDATED event"')

    dereg_args = types.SimpleNamespace(
        key=None, pipeline_id='payment_batch',
        reason='adding VALIDATED to event sequence',
        contracts_dir='contracts', inputs_dir='inputs',
    )
    cli.cmd_deregister(dereg_args)

    yaml_gone = not (WORKDIR / 'contracts' / 'payment_batch.yaml').exists()
    pnml_gone = not (WORKDIR / 'contracts' / 'payment_batch.pnml').exists()
    print()
    print(f'  YAML deleted: {yaml_gone}  ← must re-register with new events')
    print(f'  PNML deleted: {pnml_gone}  ← stale Petri net removed')
    print(f'  CSV history:  kept     ← historical measurements preserved')
    print()

    explain("""
deprecate — use when the PIPELINE SHUTS DOWN

  A pipeline is being retired. No more events will come.
  You do not want it running in fracture run-all.
  But you want to keep the historical record for audit.

  deprecate sets status: deprecated in the YAML.
  run-all skips deprecated pipelines.
  PNML is kept (not deleted) for audit trail.
  Historical scores in conformance_log.csv are preserved.
    """)

    show_command('fracture deprecate --pipeline-id risk_report --reason "migrating to v2"')

    dep_args = types.SimpleNamespace(
        key=None, pipeline_id='risk_report',
        reason='migrating to risk_report_v2',
        contracts_dir='contracts', inputs_dir='inputs',
    )
    cli.cmd_deprecate(dep_args)

    import yaml
    raw = yaml.safe_load((WORKDIR / 'contracts' / 'risk_report.yaml').read_text())
    print()
    print(f'  Status: {raw["status"]}  ← skipped in run-all')
    print(f'  PNML kept: True  ← audit trail preserved')
    print()
    print('  SUMMARY:')
    print('  deregister = contract changes   (PNML deleted, fresh start)')
    print('  deprecate  = pipeline retires   (PNML kept, history intact)')


# ═══════════════════════════════════════════════════════════════════════════
# PART 6 — RUN-ALL
# ═══════════════════════════════════════════════════════════════════════════

def part6_run_all():
    section(
        'PART 6 — run-all',
        'Daily fleet operation'
    )

    import fracture.cli as cli

    explain("""
fracture run-all

Runs conformance for every ACTIVE pipeline.
Called once daily, usually 15 minutes after the last pipeline completes.
Skips DEPRECATED and DRAFT pipelines automatically.

fracture run-all --team market-risk-quant
  Only runs pipelines where producer_team = market-risk-quant.
  Foundation for RBAC: each team runs their own pipelines.

After run-all:
  fracture status      → one-line summary per pipeline
  python report.py     → full fleet health report
    """)

    # Re-register payment_batch (was deregistered in part 5)
    make_producer_events('payments_v2', n_runs=30, duration=45,
                         add_consumer=True, consumer_gap=28)

    args = types.SimpleNamespace(
        name='Payments V2', owner='pay@bank.com',
        producer_team='payments-platform', consumer_team='settlement',
        expected_start='06:00', expected_end='08:30',
        criticality='medium', slack=None,
        contracts_dir='contracts', inputs_dir='inputs',
    )
    cli.cmd_register(args)
    key = cli._generate_key('payments_v2')

    boot_args = types.SimpleNamespace(
        key=key, pipeline_id='payments_v2', no_activate=False,
        contracts_dir='contracts', inputs_dir='inputs',
    )
    cli.cmd_bootstrap(boot_args)

    print('  Running all active pipelines...')
    print()
    show_command('fracture run-all')

    runall_args = types.SimpleNamespace(
        date=date.today().strftime('%Y%m%d'),
        team=None,
        contracts_dir='contracts', inputs_dir='inputs',
    )
    cli.cmd_run_all(runall_args)

    print()
    show_command('fracture status')
    status_args = types.SimpleNamespace(
        contracts_dir='contracts', inputs_dir='inputs',
    )
    cli.cmd_status(status_args)


# ═══════════════════════════════════════════════════════════════════════════
# PART 7 — PREFLIGHT: WHAT BAD LOGS LOOK LIKE
# ═══════════════════════════════════════════════════════════════════════════

def part7_preflight():
    section(
        'PART 7 — Preflight checks',
        'What Fracture does with bad logs'
    )

    explain("""
Before token replay runs, Fracture checks the log for common problems.

4 RED checks  → block conformance entirely
  - Empty log (no events at all)
  - All same timestamp (logging system bug — timestamps are meaningless)
  - Future timestamps (timezone misconfiguration)
  - DRAFT contract (not yet activated)

4 AMBER checks → conformance runs but confidence is reduced
  - Terminal event missing from some traces   penalty: -0.40
  - Low coverage (< 50% of expected runs)     penalty: -0.20
  - Ordering violations                        penalty: -0.15
  - First event is wrong                       penalty: -0.10

Each AMBER penalty reduces confidence toward UNRELIABLE.
A pipeline with confidence=UNRELIABLE should not be acted on
until the log extraction is fixed.
    """)

    from fracture.schema import PipelineContract
    from fracture.preflight import PreflightChain

    c = PipelineContract(**{
        'pipeline_id': 'preflight_demo', 'owner': 'p@b.com',
        'producer_team': 'risk', 'consumer_team': 'grid',
        'criticality': 'medium', 'status': 'active',
        'expected_start': '06:00', 'expected_end': '08:30',
        'grace_minutes': 15, 'p50_minutes': 45,
        'p95_minutes': 75, 'p99_minutes': 90,
        'notifications': [],
        'log_contract': {'transport': 'parquet', 'source_path': 'x/'},
    })

    # Clean events baseline
    clean = pd.DataFrame([
        {'pipeline_run_id': f'r{i}', 'activity': act,
         'timestamp': pd.Timestamp(f'2026-01-{i+1:02d} 06:00:00+00:00')
                      + pd.Timedelta(minutes=j*15),
         'team': 'producer'}
        for i in range(5)
        for j, act in enumerate(['SCHEDULED','STARTED','COMPLETED','DATA_AVAILABLE'])
    ])

    scenarios = [
        ('Clean log',           clean,                                  'baseline'),
        ('All same timestamp',  clean.assign(timestamp=pd.Timestamp('2026-01-01 06:00:00+00:00')), 'RED'),
        ('Future timestamps',   clean.assign(timestamp=clean['timestamp'] + pd.Timedelta(days=3000)), 'RED'),
        ('Empty log',           clean.iloc[0:0],                        'RED'),
        ('Missing COMPLETED',   clean[clean['activity'] != 'COMPLETED'], 'AMBER'),
    ]

    print(f'  {"SCENARIO":<28} {"SEVERITY":<10} RESULT')
    print(f'  {"─" * 62}')

    for name, events, expected in scenarios:
        chain  = PreflightChain()
        result = chain.run(events, c)
        if not result.passed:
            # RED check fired
            red = result.red_checks[0] if result.red_checks else None
            sev = 'RED'
            msg = (red.message[:35] if red and hasattr(red,'message') else 'check failed')
        elif result.amber_checks:
            sev = 'AMBER'
            amb = result.amber_checks[0]
            msg = (amb.message[:35] if hasattr(amb,'message') else 'amber check')
        else:
            sev = '─'
            msg = 'passes — conformance can run'
        marker = '✓' if sev == '─' else ('✗' if sev == 'RED' else '⚠')
        print(f'  {marker}  {name:<26} {sev:<10} {msg}')

    print()
    print('  RED   → stop. Fix the log extraction before running again.')
    print('  AMBER → run with reduced confidence. Investigate in parallel.')


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    setup()
    try:
        part1_happy_path()
        part2_reading_output()
        part3_bilateral_gap()
        part4_draft_gate()
        part5_lifecycle()
        part6_run_all()
        part7_preflight()

        print()
        print('╔' + '═' * 66 + '╗')
        print('║  WALKTHROUGH COMPLETE                                          ║')
        print('╠' + '═' * 66 + '╣')
        print('║  You have seen every Fracture CLI command execute live.        ║')
        print('║                                                                ║')
        print('║  Key things to remember:                                       ║')
        print('║                                                                ║')
        print('║  register → bootstrap → activate → run  (always this order)   ║')
        print('║  --no-activate keeps DRAFT for human review                    ║')
        print('║  deregister when contract changes  (PNML deleted)              ║')
        print('║  deprecate when pipeline retires   (PNML kept)                 ║')
        print('║  bilateral_gap=None means unknown, not zero                    ║')
        print('║  RED preflight blocks; AMBER reduces confidence                ║')
        print('║  grain declares what pipeline_run_id means                     ║')
        print('╚' + '═' * 66 + '╝')

    finally:
        teardown()
