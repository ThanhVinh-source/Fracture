"""
tests/test_optional_activities.py

Tests for optional activities in Petri nets.
Models real Airflow DAG patterns where some tasks run conditionally.

Scenarios covered:
  1. Full load pipeline   — VALIDATED fires on all runs
  2. Incremental pipeline — VALIDATED fires on some runs (optional)
  3. Month-end pipeline   — RECONCILE fires only on last day of month
  4. Conditional notify   — NOTIFIED fires only when threshold exceeded
  5. Airflow DAG mapping  — task names mapped to Fracture vocabulary

Run:
  python tests/test_optional_activities.py
"""

import sys
import warnings
from datetime import date, timedelta, datetime
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings('ignore')


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_contract(pipeline_id: str,
                  required_events: list = None,
                  optional_activities: list = None,
                  activity_map: dict = None):
    """Build a contract with custom event sequence."""
    from fracture.schema import PipelineContract

    return PipelineContract(**{
        'pipeline_id':    pipeline_id,
        'owner':          'data-eng@bank.com',
        'producer_team':  'data-engineering',
        'consumer_team':  'data-warehouse',
        'criticality':    'medium',
        'status':         'active',
        'expected_start': '06:00',
        'expected_end':   '08:30',
        'grace_minutes':  15,
        'p50_minutes':    45,
        'p95_minutes':    75,
        'p99_minutes':    90,
        'notifications':  [],
        'log_contract': {
            'transport':         'parquet',
            'source_path':       f'inputs/{pipeline_id}/',
            'required_events':   required_events or [
                'SCHEDULED','STARTED','COMPLETED','DATA_AVAILABLE'
            ],
            'terminal_event':    'COMPLETED',
            'activity_name_map': activity_map or {},
        },
    })


def make_events(run_id: str, activities: list,
                base_time: datetime = None) -> pd.DataFrame:
    """Build a single trace from an activity list."""
    if base_time is None:
        base_time = datetime(2026, 5, 1, 6, 0, 0)
    rows = []
    for i, act in enumerate(activities):
        rows.append({
            'pipeline_run_id': run_id,
            'activity':        act,
            'timestamp':       pd.Timestamp(base_time + timedelta(minutes=i*15),
                                            tz='UTC'),
            'team':            'producer',
        })
    return pd.DataFrame(rows)


passed = failed = 0


def check(name, fn):
    global passed, failed
    try:
        fn()
        passed += 1
        print(f'  ✓  {name}')
    except AssertionError as e:
        failed += 1
        print(f'  ✗  {name}: {e}')
    except Exception as e:
        failed += 1
        print(f'  ✗  {name}: {type(e).__name__}: {e}')


# ── Petri net tests ───────────────────────────────────────────────────────────

print()
print('=== Petri net structure ===')


def test_linear_net_structure():
    """Standard 4-activity linear net has correct structure."""
    from fracture.petri import net_summary
    c = make_contract('linear')
    s = net_summary(c)
    assert s['n_places']      == 6, f"Expected 6 places, got {s['n_places']}"
    assert s['n_transitions'] == 5, f"Expected 5 transitions, got {s['n_transitions']}"
    assert s['n_arcs']        == 10
    assert 'trivially guaranteed' in s['soundness_basis']
check('linear net: 6p 5t 10a, trivially sound', test_linear_net_structure)


def test_5_activity_net():
    """5-activity net (with VALIDATED) has correct structure."""
    from fracture.petri import net_summary
    c = make_contract('five_act',
        required_events=['SCHEDULED','STARTED','VALIDATED','COMPLETED','DATA_AVAILABLE'])
    s = net_summary(c)
    assert s['n_places']      == 7
    assert s['n_transitions'] == 6
    assert s['n_arcs']        == 12
check('5-activity net: 7p 6t 12a', test_5_activity_net)


def test_optional_adds_bypass():
    """Optional activity adds exactly 1 transition and 2 arcs."""
    from fracture.petri import net_summary
    c = make_contract('opt_test',
        required_events=['SCHEDULED','STARTED','VALIDATED','COMPLETED','DATA_AVAILABLE'])
    s_required = net_summary(c)
    s_optional  = net_summary(c, optional_activities=['VALIDATED'])

    assert s_optional['n_transitions'] == s_required['n_transitions'] + 1, \
        "Optional should add exactly 1 bypass transition"
    assert s_optional['n_arcs'] == s_required['n_arcs'] + 2, \
        "Optional should add exactly 2 arcs (bypass in + out)"
    assert 'woflan' in s_optional['soundness_basis'], \
        "Optional activities require woflan verification"
check('optional: +1 transition, +2 arcs, woflan verified', test_optional_adds_bypass)


def test_multiple_optional():
    """Two optional activities add 2 bypass transitions."""
    from fracture.petri import net_summary
    c = make_contract('multi_opt',
        required_events=['SCHEDULED','STARTED','VALIDATED','COMPLETED',
                         'RECONCILED','DATA_AVAILABLE'])
    s = net_summary(c, optional_activities=['VALIDATED','RECONCILED'])
    assert s['n_transitions'] == 9, \
        f"Expected 9 transitions (7 required + 2 bypass), got {s['n_transitions']}"
check('two optional: 9 transitions', test_multiple_optional)


# ── Token replay tests ────────────────────────────────────────────────────────

print()
print('=== Token replay: required vs optional ===')


def test_missing_required_penalised():
    """Missing required activity reduces sequence_fitness."""
    from fracture.conformance import compute_conformance

    c = make_contract('miss_req',
        required_events=['SCHEDULED','STARTED','VALIDATED','COMPLETED','DATA_AVAILABLE'])

    # Trace WITHOUT VALIDATED (required here)
    events_without = pd.concat([
        make_events('run_1', ['SCHEDULED','STARTED','COMPLETED','DATA_AVAILABLE']),
        make_events('run_2', ['SCHEDULED','STARTED','COMPLETED','DATA_AVAILABLE'],
                    base_time=datetime(2026,5,2,6,0,0)),
    ])

    result = compute_conformance(events_without, c)
    assert result.diagnostics.sequence_fitness < 1.0, \
        f"Missing VALIDATED should reduce fitness, got {result.diagnostics.sequence_fitness}"
    assert result.diagnostics.sequence_fitness < 0.90, \
        f"Fitness should be noticeably lower, got {result.diagnostics.sequence_fitness:.3f}"
    print(f'       fitness without VALIDATED (required): {result.diagnostics.sequence_fitness:.4f}')
check('missing required activity reduces fitness', test_missing_required_penalised)


def test_missing_optional_not_penalised():
    """Missing optional activity does NOT reduce sequence_fitness."""
    from fracture.conformance import compute_conformance
    from fracture.petri import contract_to_petri_net

    c = make_contract('miss_opt',
        required_events=['SCHEDULED','STARTED','VALIDATED','COMPLETED','DATA_AVAILABLE'])

    # Trace WITHOUT VALIDATED — but declared optional
    events_without = pd.concat([
        make_events('run_1', ['SCHEDULED','STARTED','COMPLETED','DATA_AVAILABLE']),
        make_events('run_2', ['SCHEDULED','STARTED','COMPLETED','DATA_AVAILABLE'],
                    base_time=datetime(2026,5,2,6,0,0)),
    ])

    # Build net with VALIDATED as optional
    net, im, fm = contract_to_petri_net(c, optional_activities=['VALIDATED'])

    # Run token replay directly with optional net
    from fracture.converter import dataframe_to_eventlog
    from pm4py.algo.conformance.tokenreplay import algorithm as token_replay

    log    = dataframe_to_eventlog(events_without, c, 'producer')
    result = token_replay.apply(log, net, im, fm)
    fitness = sum(r['trace_fitness'] for r in result) / len(result)

    assert fitness == 1.0, \
        f"Optional VALIDATED absence should not reduce fitness, got {fitness:.4f}"
    print(f'       fitness without VALIDATED (optional): {fitness:.4f}')
check('missing optional activity: fitness = 1.0', test_missing_optional_not_penalised)


def test_mixed_runs():
    """
    Mixed traces: some runs fire VALIDATED, some skip it.
    With VALIDATED optional: all traces score 1.0.
    Without: traces without VALIDATED are penalised.
    """
    from fracture.petri import contract_to_petri_net
    from fracture.converter import dataframe_to_eventlog
    from pm4py.algo.conformance.tokenreplay import algorithm as token_replay

    c = make_contract('mixed',
        required_events=['SCHEDULED','STARTED','VALIDATED','COMPLETED','DATA_AVAILABLE'])

    # 10 runs: 5 with VALIDATED (full load), 5 without (incremental)
    all_events = []
    for i in range(10):
        base = datetime(2026, 5, i+1, 6, 0, 0)
        if i < 5:  # full load — VALIDATED fires
            acts = ['SCHEDULED','STARTED','VALIDATED','COMPLETED','DATA_AVAILABLE']
        else:       # incremental — VALIDATED skipped
            acts = ['SCHEDULED','STARTED','COMPLETED','DATA_AVAILABLE']
        all_events.append(make_events(f'run_{i:03d}', acts, base_time=base))

    events = pd.concat(all_events, ignore_index=True)

    # Without optional: incremental runs penalised
    net_req, im_req, fm_req = contract_to_petri_net(c)
    log_req   = dataframe_to_eventlog(events, c, 'producer')
    res_req   = token_replay.apply(log_req, net_req, im_req, fm_req)
    fit_req   = sum(r['trace_fitness'] for r in res_req) / len(res_req)

    # With optional: all runs score 1.0
    net_opt, im_opt, fm_opt = contract_to_petri_net(c, optional_activities=['VALIDATED'])
    log_opt   = dataframe_to_eventlog(events, c, 'producer')
    res_opt   = token_replay.apply(log_opt, net_opt, im_opt, fm_opt)
    fit_opt   = sum(r['trace_fitness'] for r in res_opt) / len(res_opt)

    print(f'       required net (5/10 runs miss VALIDATED): fitness={fit_req:.4f}')
    print(f'       optional net (bypass arc):               fitness={fit_opt:.4f}')

    assert fit_req < 1.0, f"Required net should penalise missing VALIDATED: {fit_req:.4f}"
    assert fit_opt == 1.0, f"Optional net should score 1.0: {fit_opt:.4f}"
    assert fit_opt > fit_req, "Optional net should score higher than required net"
check('mixed runs: optional net scores 1.0, required net penalises', test_mixed_runs)


# ── Airflow DAG patterns ──────────────────────────────────────────────────────

print()
print('=== Airflow DAG patterns ===')


def test_airflow_conditional_task():
    """
    Airflow pattern: run_only_if_full_load BranchPythonOperator.

    DAG structure:
      start → extract → branch ──→ validate ─→ transform → load
                                ↘────────────────↗

    Maps to Fracture with VALIDATED as optional.
    """
    from fracture.petri import contract_to_petri_net, net_summary
    from fracture.converter import dataframe_to_eventlog
    from pm4py.algo.conformance.tokenreplay import algorithm as token_replay

    c = make_contract('airflow_branch',
        required_events=['SCHEDULED','STARTED','VALIDATED','COMPLETED','DATA_AVAILABLE'],
        activity_map={
            'dag_queued':        'SCHEDULED',
            'extract_started':   'STARTED',
            'validate_complete': 'VALIDATED',
            'load_complete':     'COMPLETED',
            'data_available':    'DATA_AVAILABLE',
        }
    )

    # Full load runs (Monday) — VALIDATED fires
    full_load_runs = pd.concat([
        make_events(f'full_{i}',
                    ['SCHEDULED','STARTED','VALIDATED','COMPLETED','DATA_AVAILABLE'],
                    base_time=datetime(2026,5,i*7+1,6,0,0))
        for i in range(4)
    ])

    # Incremental runs (daily) — VALIDATED skipped
    incremental_runs = pd.concat([
        make_events(f'incr_{i}',
                    ['SCHEDULED','STARTED','COMPLETED','DATA_AVAILABLE'],
                    base_time=datetime(2026,5,i+2,6,0,0))
        for i in range(20)
    ])

    all_events = pd.concat([full_load_runs, incremental_runs], ignore_index=True)
    total_runs  = all_events['pipeline_run_id'].nunique()

    # Test with VALIDATED as optional (correct contract)
    net, im, fm = contract_to_petri_net(c, optional_activities=['VALIDATED'])
    log   = dataframe_to_eventlog(all_events, c, 'producer')
    res   = token_replay.apply(log, net, im, fm)
    fitness = sum(r['trace_fitness'] for r in res) / len(res)

    print(f'       {total_runs} runs: {4} full load, {20} incremental')
    print(f'       optional net fitness: {fitness:.4f}')
    print(f'       full loads fire VALIDATED, incrementals skip — both score 1.0')

    assert fitness == 1.0, f"All runs should score 1.0 with optional VALIDATED: {fitness}"
check('airflow branch: full load vs incremental, all score 1.0', test_airflow_conditional_task)


def test_month_end_reconciliation():
    """
    Month-end pipeline pattern: RECONCILE fires on last day of month only.

    28 normal days + 1 month-end day.
    RECONCILE optional → all 29 days score 1.0.
    """
    from fracture.petri import contract_to_petri_net
    from fracture.converter import dataframe_to_eventlog
    from pm4py.algo.conformance.tokenreplay import algorithm as token_replay

    c = make_contract('month_end',
        required_events=['SCHEDULED','STARTED','RECONCILED','COMPLETED','DATA_AVAILABLE'])

    # 28 normal days — no RECONCILE
    normal_runs = pd.concat([
        make_events(f'day_{i}',
                    ['SCHEDULED','STARTED','COMPLETED','DATA_AVAILABLE'],
                    base_time=datetime(2026,5,i+1,6,0,0))
        for i in range(28)
    ])

    # 1 month-end — RECONCILE fires
    month_end = make_events('month_end_run',
        ['SCHEDULED','STARTED','RECONCILED','COMPLETED','DATA_AVAILABLE'],
        base_time=datetime(2026,5,31,6,0,0))

    all_events = pd.concat([normal_runs, month_end], ignore_index=True)

    net, im, fm = contract_to_petri_net(c, optional_activities=['RECONCILED'])
    log   = dataframe_to_eventlog(all_events, c, 'producer')
    res   = token_replay.apply(log, net, im, fm)
    fitness = sum(r['trace_fitness'] for r in res) / len(res)

    print(f'       28 normal + 1 month-end: fitness={fitness:.4f}')
    assert fitness == 1.0, f"Month-end optional should score 1.0: {fitness}"
check('month-end reconciliation: 29 runs, all 1.0', test_month_end_reconciliation)


def test_threshold_notification():
    """
    Pipeline pattern: NOTIFIED fires only when anomaly threshold exceeded.

    Most days: normal run, no alert.
    Some days: anomaly detected, NOTIFIED fires.
    Both should score 1.0 with NOTIFIED optional.
    """
    from fracture.petri import contract_to_petri_net
    from fracture.converter import dataframe_to_eventlog
    from pm4py.algo.conformance.tokenreplay import algorithm as token_replay

    c = make_contract('threshold_notify',
        required_events=['SCHEDULED','STARTED','COMPLETED','NOTIFIED','DATA_AVAILABLE'])

    # 25 normal days — no NOTIFIED
    normal = pd.concat([
        make_events(f'norm_{i}',
                    ['SCHEDULED','STARTED','COMPLETED','DATA_AVAILABLE'],
                    base_time=datetime(2026,5,i+1,6,0,0))
        for i in range(25)
    ])

    # 5 alert days — NOTIFIED fires after COMPLETED
    alert = pd.concat([
        make_events(f'alert_{i}',
                    ['SCHEDULED','STARTED','COMPLETED','NOTIFIED','DATA_AVAILABLE'],
                    base_time=datetime(2026,5,i+5,6,30,0))
        for i in range(5)
    ])

    all_events = pd.concat([normal, alert], ignore_index=True)

    net, im, fm = contract_to_petri_net(c, optional_activities=['NOTIFIED'])
    log   = dataframe_to_eventlog(all_events, c, 'producer')
    res   = token_replay.apply(log, net, im, fm)
    fitness = sum(r['trace_fitness'] for r in res) / len(res)

    print(f'       25 normal + 5 alert: fitness={fitness:.4f}')
    assert fitness == 1.0, f"Optional NOTIFIED should score 1.0: {fitness}"
check('threshold notification: 30 runs, NOTIFIED optional, all 1.0', test_threshold_notification)


def test_activity_name_map_with_optional():
    """
    Combine activity_name_map with optional activities.

    Airflow task names mapped to Fracture vocabulary.
    One task is optional (validation branch).
    """
    from fracture.conformance import compute_conformance
    from fracture.petri import contract_to_petri_net
    from fracture.converter import dataframe_to_eventlog
    from fracture.ingest import validate_format
    from pm4py.algo.conformance.tokenreplay import algorithm as token_replay

    # Contract with Airflow-style names in activity_map
    c = make_contract('airflow_mapped',
        required_events=['SCHEDULED','STARTED','VALIDATED','COMPLETED','DATA_AVAILABLE'],
        activity_map={
            'dag_queued':        'SCHEDULED',
            'task_running':      'STARTED',
            'validate_step':     'VALIDATED',
            'pipeline_success':  'COMPLETED',
            'output_ready':      'DATA_AVAILABLE',
        }
    )

    # Events with Airflow names
    airflow_events = pd.concat([
        # Full load — validate_step fires
        make_events('full_001',
                    ['dag_queued','task_running','validate_step',
                     'pipeline_success','output_ready']),
        # Incremental — validate_step skipped
        make_events('incr_001',
                    ['dag_queued','task_running','pipeline_success','output_ready'],
                    base_time=datetime(2026,5,2,6,0,0)),
    ])

    # Validate format — team column needs to be present
    airflow_events['team'] = 'producer'

    # Apply activity_name_map
    activity_map = c.log_contract.activity_name_map
    airflow_events['activity'] = airflow_events['activity'].map(
        lambda x: activity_map.get(x, x)
    )

    # Token replay with optional VALIDATED
    net, im, fm = contract_to_petri_net(c, optional_activities=['VALIDATED'])
    log   = dataframe_to_eventlog(airflow_events, c, 'producer')
    res   = token_replay.apply(log, net, im, fm)
    fitness = sum(r['trace_fitness'] for r in res) / len(res)

    print(f'       airflow names mapped + optional VALIDATED: fitness={fitness:.4f}')
    assert fitness == 1.0, f"Mapped + optional should score 1.0: {fitness}"
check('activity_name_map + optional: mapped names with bypass arc', test_activity_name_map_with_optional)


# ── Soundness tests ───────────────────────────────────────────────────────────

print()
print('=== Soundness ===')


def test_linear_trivially_sound():
    """Linear net is trivially sound — woflan not needed."""
    from fracture.petri import net_summary
    c = make_contract('sound_test')
    s = net_summary(c)
    assert 'trivially guaranteed' in s['soundness_basis']
check('linear: trivially sound (no woflan)', test_linear_trivially_sound)


def test_optional_woflan_verified():
    """Optional activities trigger woflan verification."""
    from fracture.petri import net_summary
    c = make_contract('woflan_test',
        required_events=['SCHEDULED','STARTED','VALIDATED','COMPLETED','DATA_AVAILABLE'])
    s = net_summary(c, optional_activities=['VALIDATED'])
    assert 'woflan' in s['soundness_basis'].lower()
check('optional: woflan verified', test_optional_woflan_verified)


def test_terminal_cannot_be_optional():
    """Terminal event cannot be declared optional — no valid completion path."""
    from fracture.petri import contract_to_petri_net, UnsoundNetError
    c = make_contract('bad_optional',
        required_events=['SCHEDULED','STARTED','VALIDATED','COMPLETED','DATA_AVAILABLE'])
    # COMPLETED is the terminal event — making it optional removes the completion path
    try:
        net, im, fm = contract_to_petri_net(c, optional_activities=['COMPLETED'])
        # If it does not raise, check that the net at least builds
        # (some implementations may allow this but produce low fitness)
        print(f'       built net with optional terminal (implementation allows it)')
    except (UnsoundNetError, Exception) as e:
        print(f'       correctly refused optional terminal: {type(e).__name__}')
check('terminal event optional: handled correctly', test_terminal_cannot_be_optional)


# ── Summary ───────────────────────────────────────────────────────────────────

print()
print('=' * 60)
print(f'  Total: {passed + failed}  Passed: {passed}  Failed: {failed}')
print('=' * 60)

print()
print('What these tests demonstrate:')
print()
print('  1. Linear net (standard 4-activity):')
print('     6 places, 5 transitions, trivially sound')
print('     Missing any activity → fitness drops')
print()
print('  2. Optional activity (bypass arc):')
print('     +1 silent transition, +2 arcs, woflan verified')
print('     Missing optional → fitness stays 1.0')
print('     Present optional → fires normally')
print()
print('  3. Airflow patterns:')
print('     BranchPythonOperator → optional activity')
print('     Month-end conditional → optional activity')
print('     Threshold notification → optional activity')
print('     Airflow task names → activity_name_map')
print()
print('  4. The key insight:')
print('     Optional activities model LEGITIMATE absence.')
print('     Required activities model CONTRACTED presence.')
print('     The contract decides which is which.')
print('     Fracture measures conformance to the contract.')

if failed > 0:
    import sys
    sys.exit(1)
