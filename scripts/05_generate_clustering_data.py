"""
scripts/05_generate_clustering_data.py

Generates multi-day conformance data for clustering.

Run order:
    1. python scripts/06_generate_team_contracts.py --clean --days 30
    2. python scripts/05_generate_clustering_data.py --days 14
    3. python clustering.py
    4. python report.py

Why a separate script from 06?
    06 registers contracts and writes event files once.
    05 runs conformance for each day in the window to build temporal features.
    Clustering needs drift_rate, variance_cv, score_range across 14+ days.
    A single day gives one score per pipeline — no slope, no variance.

Feature vector per pipeline (built from N days):
    mean_score       average final_score
    min_score        worst single day
    drift_rate       slope of score over time (negative = declining)
    variance_cv      temporal variance (how consistent is it day to day?)
    score_range      max - min (catches intermittent behaviour)
    mean_gap         average bilateral gap (minutes)
    gap_drift        is the gap itself growing?
    completeness     mean completeness_score
    seq_fitness      mean sequence_fitness

Usage:
    python scripts/05_generate_clustering_data.py
    python scripts/05_generate_clustering_data.py --days 14
    python scripts/05_generate_clustering_data.py --source standalone
        (no team_config needed — uses built-in 25-pipeline list)
"""

import sys
import csv
import argparse
import warnings
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Standalone pipeline list ──────────────────────────────────────────────────
# Used when --source standalone, or when team_config import fails.
# 25 pipelines: 8 HEALTHY, 8 DRIFTING, 9 CRITICAL.

STANDALONE_PIPELINES = {
    'payment_settlements':  {'archetype': 'healthy',       'cluster': 'HEALTHY'},
    'regulatory_batch_dnb': {'archetype': 'healthy',       'cluster': 'HEALTHY'},
    'transaction_screen':   {'archetype': 'healthy',       'cluster': 'HEALTHY'},
    'lgd_calculation':      {'archetype': 'healthy',       'cluster': 'HEALTHY'},
    'fx_rate_ingestion':    {'archetype': 'healthy',       'cluster': 'HEALTHY'},
    'kafka_compaction':     {'archetype': 'healthy',       'cluster': 'HEALTHY'},
    'feature_validation':   {'archetype': 'healthy',       'cluster': 'HEALTHY'},
    'raw_ingestion_pipe':   {'archetype': 'healthy',       'cluster': 'HEALTHY'},
    'grid_corehours_calc':  {'archetype': 'degrading',     'drift_rate': 2.5, 'cluster': 'DRIFTING'},
    'payment_reconcile':    {'archetype': 'degrading',     'drift_rate': 2.0, 'cluster': 'DRIFTING'},
    'pd_model_batch':       {'archetype': 'degrading',     'drift_rate': 1.8, 'cluster': 'DRIFTING'},
    'hdfs_replication':     {'archetype': 'degrading',     'drift_rate': 3.0, 'cluster': 'DRIFTING'},
    'sftp_positions':       {'archetype': 'asymmetry',     'initial_gap': 25, 'gap_growth': 2.0, 'cluster': 'DRIFTING'},
    'mifid_reporting':      {'archetype': 'asymmetry',     'initial_gap': 18, 'gap_growth': 1.5, 'cluster': 'DRIFTING'},
    'liquidity_report':     {'archetype': 'asymmetry',     'initial_gap': 30, 'gap_growth': 1.0, 'cluster': 'DRIFTING'},
    'var_batch_proc':       {'archetype': 'bimodal',                          'cluster': 'DRIFTING'},
    'customer_risk_feat':   {'archetype': 'silent',        'silent_days': [0, 3], 'cluster': 'CRITICAL'},
    'feature_store_ref':    {'archetype': 'silent',        'silent_days': [5, 6], 'cluster': 'CRITICAL'},
    'stress_test_scen':     {'archetype': 'silent',        'silent_days': [1, 3], 'cluster': 'CRITICAL'},
    'delta_vacuum_run':     {'archetype': 'silent',        'silent_days': [6],    'cluster': 'CRITICAL'},
    'collateral_val':       {'archetype': 'fast_drifting', 'drift_rate': 5.0,     'cluster': 'CRITICAL'},
    'dbt_model_runner_s':   {'archetype': 'fast_drifting', 'drift_rate': 4.0,     'cluster': 'CRITICAL'},
    'positions_sftp':       {'archetype': 'fast_drifting', 'drift_rate': 3.5,     'cluster': 'CRITICAL'},
    'model_training':       {'archetype': 'fast_drifting', 'drift_rate': 4.5,     'cluster': 'CRITICAL'},
    'inference_scoring':    {'archetype': 'asymmetry',     'initial_gap': 45, 'gap_growth': 3.0, 'cluster': 'CRITICAL'},
}

CLUSTER_MAP = {
    'healthy': 'HEALTHY', 'bimodal': 'DRIFTING', 'asymmetry': 'DRIFTING',
    'degrading': 'DRIFTING', 'silent': 'CRITICAL', 'fast_drifting': 'CRITICAL',
}

LOG_COLUMNS = [
    'pipeline_id', 'pipeline_key', 'run_date', 'final_score',
    'confidence_level', 'timing_zone', 'pattern', 'bilateral_gap_minutes',
    'days_of_history', 'sequence_fitness', 'timing_score', 'completeness_score',
    'variance_cv', 'alert_owner', 'run_timestamp',
    'expected_cluster', 'archetype', 'producer_team',
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_generator(cfg: dict):
    from fracture.generator import (
        StableMatureGenerator, AssumptionAsymmetryGenerator,
        SlowDriftingGenerator, SilentPipelineGenerator,
        FastDriftingNewGenerator, BimodalGenerator,
    )
    return {
        'healthy':       StableMatureGenerator(),
        'degrading':     SlowDriftingGenerator(cfg.get('drift_rate', 2.0)),
        'asymmetry':     AssumptionAsymmetryGenerator(
                             cfg.get('initial_gap', 20), cfg.get('gap_growth', 1.0)),
        'silent':        SilentPipelineGenerator(cfg.get('silent_days', [0])),
        'fast_drifting': FastDriftingNewGenerator(cfg.get('drift_rate', 4.0)),
        'bimodal':       BimodalGenerator(),
    }[cfg['archetype']]


def make_contract(pid: str, cfg: dict,
                  producer_team: str = 'data-platform',
                  consumer_team: str  = 'analytics'):
    from fracture.schema import PipelineContract
    cluster = cfg.get('cluster', CLUSTER_MAP.get(cfg['archetype'], 'HEALTHY'))
    crit    = 'high' if cluster == 'CRITICAL' else 'medium'
    return PipelineContract(**{
        'pipeline_id':    pid,
        'owner':          f'{producer_team}@bank.com',
        'producer_team':  producer_team,
        'consumer_team':  consumer_team,
        'criticality':    crit,
        'status':         'active',
        'expected_start': '06:00',
        'expected_end':   '08:30',
        'grace_minutes':  15,
        'p50_minutes':    45,
        'p95_minutes':    75,
        'p99_minutes':    90,
        'notifications':  [{'channel': 'slack', 'target': f'#{producer_team}-alerts'}],
        'log_contract':   {'transport': 'parquet', 'source_path': f'inputs/{pid}/'},
    })


def run_day(pid: str, cfg: dict, run_date: date,
            producer_team: str = 'data-platform',
            consumer_team: str  = 'analytics',
            history_days: int   = 45) -> dict | None:
    """
    Run conformance for one pipeline on one day.
    Generates history_days of synthetic events ending on run_date.
    Returns a conformance_log row dict, or None on failure.
    """
    from fracture.conformance import compute_conformance

    gen      = get_generator(cfg)
    contract = make_contract(pid, cfg, producer_team, consumer_team)
    seed     = abs(hash(f'{pid}_{run_date}')) % 99999
    start    = run_date - timedelta(days=history_days)

    try:
        prod, cons, gt = gen.generate(
            contract=contract, days=history_days,
            start_date=start, pipeline_age=180, seed=seed,
        )

        prod = prod.copy()
        prod['timestamp'] = pd.to_datetime(prod['timestamp'], utc=True)

        cons_events = None
        if cons is not None and len(cons) > 0:
            cons = cons.copy()
            cons['timestamp'] = pd.to_datetime(cons['timestamp'], utc=True)
            cons_events = cons

        r = compute_conformance(prod, contract, consumer_events=cons_events)
        d = r.diagnostics

        return {
            'pipeline_id':           pid,
            'pipeline_key':          f'FRC-{abs(hash(pid)) & 0xFFFF:04x}',
            'run_date':              run_date.strftime('%Y%m%d'),
            'final_score':           round(r.final_score, 4),
            'confidence_level':      r.confidence_level,
            'timing_zone':           r.timing_zone,
            'pattern':               r.pattern,
            'bilateral_gap_minutes': round(r.bilateral_gap_minutes, 1)
                                     if r.bilateral_gap_minutes else '',
            'days_of_history':       d.days_of_history,
            'sequence_fitness':      round(d.sequence_fitness, 4),
            'timing_score':          round(d.timing_score, 4),
            'completeness_score':    round(d.completeness_score, 4),
            'variance_cv':           round(d.variance_cv, 6),
            'alert_owner':           r.alert_owner(),
            'run_timestamp':         f'{run_date}T07:35:00',
            'expected_cluster':      cfg.get('cluster', CLUSTER_MAP.get(cfg['archetype'], 'HEALTHY')),
            'archetype':             cfg['archetype'],
            'producer_team':         producer_team,
        }
    except Exception:
        return None


def load_pipelines(source: str) -> dict:
    """
    team_config: use scripts/team_config.py (consistent with 06)
    standalone:  use STANDALONE_PIPELINES (no dependencies)
    """
    if source == 'standalone':
        return STANDALONE_PIPELINES

    try:
        from scripts.team_config import (
            TEAM_PIPELINES, CUSTOM_MARKER_PIPELINES,
            PIPELINE_CONSUMERS,
        )
        result = {}
        for team, pipelines in {**TEAM_PIPELINES, **CUSTOM_MARKER_PIPELINES}.items():
            for pid, cfg in pipelines.items():
                entry = dict(cfg)
                entry['cluster']   = CLUSTER_MAP.get(cfg['archetype'], 'HEALTHY')
                entry['_team']     = team
                entry['_consumer'] = PIPELINE_CONSUMERS.get(pid, 'analytics')
                result[pid] = entry
        return result
    except ImportError:
        print('  ⚠  team_config not found — using standalone pipeline list')
        return STANDALONE_PIPELINES


# ── Main ──────────────────────────────────────────────────────────────────────

def main(days: int = 14, output: str = 'conformance_log.csv',
         source: str = 'team_config', history_days: int = 45):

    pipelines  = load_pipelines(source)
    today      = date.today()
    start_date = today - timedelta(days=days - 1)
    run_dates  = [start_date + timedelta(days=i) for i in range(days)]

    clusters = Counter(
        v.get('cluster', CLUSTER_MAP.get(v['archetype'], '?'))
        for v in pipelines.values()
    )

    print()
    print('Fracture — Clustering Data Generation')
    print('═' * 55)
    print(f'  Source    : {source}')
    print(f'  Pipelines : {len(pipelines)}')
    print(f'  Days      : {days}  ({run_dates[0]} → {run_dates[-1]})')
    print(f'  Output    : {output}')
    print()
    for cluster, count in sorted(clusters.items()):
        label = {'HEALTHY': '  stable',
                 'DRIFTING': '  declining score or widening gap',
                 'CRITICAL': '  silent failures or SLA breach'}
        print(f'  {cluster:<12}: {count} pipelines {label.get(cluster,"")}')
    print()

    all_rows  = []
    total_fail = 0

    for run_date in run_dates:
        day_ok = 0
        for pid, cfg in pipelines.items():
            team     = cfg.get('_team', 'data-platform')
            consumer = cfg.get('_consumer', 'analytics')
            row = run_day(pid, cfg, run_date, team, consumer, history_days)
            if row:
                all_rows.append(row)
                day_ok += 1
            else:
                total_fail += 1
        print(f'  {run_date.strftime("%Y-%m-%d")} ({run_date.strftime("%a")})'
              f'  {day_ok}/{len(pipelines)}')

    # Write
    with open(output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    print()
    print(f'  Written {len(all_rows)} rows → {output}')
    if total_fail:
        print(f'  ⚠  {total_fail} rows skipped (see generator errors above)')
    print()

    # Feature space preview
    df = pd.DataFrame(all_rows)
    for col in ['final_score', 'completeness_score', 'sequence_fitness']:
        df[col] = pd.to_numeric(df[col])
    df['bilateral_gap'] = pd.to_numeric(df['bilateral_gap_minutes'], errors='coerce')

    print('  Feature space preview:')
    print(f'  {"CLUSTER":<12} {"N":>5} {"SCORE":>8} {"SEQ":>8}'
          f' {"COMP":>8} {"GAP":>8}')
    print('  ' + '─' * 55)
    for cluster in ['HEALTHY', 'DRIFTING', 'CRITICAL']:
        sub = df[df['expected_cluster'] == cluster]
        if len(sub) == 0:
            continue
        print(f'  {cluster:<12} {len(sub):>5}'
              f' {sub["final_score"].mean():>8.3f}'
              f' {sub["sequence_fitness"].mean():>8.3f}'
              f' {sub["completeness_score"].mean():>8.3f}'
              f' {sub["bilateral_gap"].mean():>8.1f}')
    print()
    print('  Next:  python clustering.py')
    print('         python report.py')
    print('═' * 55)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='05_generate_clustering_data.py')
    parser.add_argument('--days',    type=int, default=14,
                        help='Days of conformance data (default 14, min 7)')
    parser.add_argument('--output',  default='conformance_log.csv')
    parser.add_argument('--source',  default='team_config',
                        choices=['team_config', 'standalone'],
                        help='team_config: use team_config.py (default). '
                             'standalone: built-in 25 pipelines, no dependencies.')
    parser.add_argument('--history', type=int, default=45,
                        help='Event history days per run (default 45)')
    args = parser.parse_args()
    main(args.days, args.output, args.source, args.history)
