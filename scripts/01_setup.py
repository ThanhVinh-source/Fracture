"""
scripts/01_setup.py

Generates synthetic event files for all teams and all scenarios.

Three tiers:
  Tier 1 — Core pipelines (25 pipelines, all archetypes)
  Tier 2 — Custom log markers (2 pipelines, non-standard activity names)
  Tier 3 — Dirty logs (8 scenarios, test preflight checks)

Total: 27 core pipelines + 8 dirty log test files
Across 10 teams + 1 custom marker team

Usage:
  python scripts/01_setup.py                     # all tiers
  python scripts/01_setup.py --team payments-platform
  python scripts/01_setup.py --tier 1            # core only
  python scripts/01_setup.py --tier 2            # custom markers only
  python scripts/01_setup.py --tier 3            # dirty logs only
  python scripts/01_setup.py --days 60           # more history
  python scripts/01_setup.py --list-teams
"""

import argparse
import sys
import warnings
from datetime import date, timedelta, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.team_config import (
    TEAM_PIPELINES, CUSTOM_MARKER_PIPELINES,
    TEAM_OWNERS, PIPELINE_CONSUMERS, DIRTY_LOG_SCENARIOS
)


# ── Generator factory ─────────────────────────────────────────────────────────

def get_generator(cfg: dict):
    """Build the right generator from a pipeline config dict."""
    from fracture.generator import (
        StableMatureGenerator, AssumptionAsymmetryGenerator,
        SlowDriftingGenerator, SilentPipelineGenerator,
        FastDriftingNewGenerator, BimodalGenerator,
    )
    archetype = cfg['archetype']
    return {
        'healthy':      StableMatureGenerator(),
        'asymmetry':    AssumptionAsymmetryGenerator(
                            initial_gap       = cfg.get('initial_gap', 20),
                            gap_growth_per_week = cfg.get('gap_growth', 1.0),
                        ),
        'degrading':    SlowDriftingGenerator(
                            drift_rate_per_week = cfg.get('drift_rate', 2.0),
                        ),
        'silent':       SilentPipelineGenerator(
                            silent_weekdays = cfg.get('silent_days', [0]),
                        ),
        'fast_drifting':FastDriftingNewGenerator(
                            drift_rate_per_week = cfg.get('drift_rate', 3.0),
                        ),
        'bimodal':      BimodalGenerator(),
    }.get(archetype, StableMatureGenerator())


def make_contract(pipeline_id: str, producer_team: str,
                  cfg: dict, status: str = 'active'):
    """Build a minimal contract for event generation."""
    from fracture.schema import PipelineContract
    consumer = PIPELINE_CONSUMERS.get(pipeline_id, 'downstream-team')
    owner    = TEAM_OWNERS.get(producer_team, f'{producer_team}@bank.com')
    crit     = 'high' if producer_team in ('payments-platform', 'aml-platform',
                                             'regulatory-reporting') else 'medium'

    activity_map = cfg.get('activity_map', {})

    return PipelineContract(**{
        'pipeline_id':    pipeline_id,
        'owner':          owner,
        'producer_team':  producer_team,
        'consumer_team':  consumer,
        'criticality':    crit,
        'status':         status,
        'expected_start': '06:00',
        'expected_end':   '08:30',
        'grace_minutes':  15,
        'p50_minutes':    45,
        'p95_minutes':    75,
        'p99_minutes':    90,
        'notifications':  [{'channel': 'slack',
                            'target': f'#{producer_team}-alerts'}]
                          if crit == 'high' else [],
        'log_contract':   {
            'transport':        'parquet',
            'source_path':      f'inputs/{pipeline_id}/',
            'activity_name_map': activity_map,
        },
    })


# ── Tier 1: Core pipelines ────────────────────────────────────────────────────

def generate_tier1(run_date: str, days: int, inputs_dir: Path,
                   team_filter: str = None) -> int:
    """Generate event files for all core pipelines."""
    y, m, d  = int(run_date[:4]), int(run_date[4:6]), int(run_date[6:])
    from datetime import date as dt_
    start    = dt_(y, m, d) - timedelta(days=days)
    total    = 0

    teams = {team_filter: TEAM_PIPELINES[team_filter]} if team_filter else TEAM_PIPELINES

    for team, pipelines in sorted(teams.items()):
        print(f"\n  {team}")
        for pid, cfg in pipelines.items():
            contract  = make_contract(pid, team, cfg)
            generator = get_generator(cfg)

            prod, cons, gt = generator.generate(
                contract     = contract,
                days         = days,
                start_date   = start,
                pipeline_age = 180,
                seed         = hash(pid) % 9999,
            )

            d_ = inputs_dir / pid
            d_.mkdir(parents=True, exist_ok=True)
            prod.to_parquet(d_ / f'producer_{run_date}.parquet', index=False)
            if len(cons) > 0:
                cons.to_parquet(d_ / f'consumer_{run_date}.parquet', index=False)

            gap_str = ''
            if cfg['archetype'] == 'asymmetry' and len(cons) > 0:
                p_da = prod[prod.activity == 'DATA_AVAILABLE']['timestamp'].mean()
                c_da = cons[cons.activity == 'DATA_AVAILABLE']['timestamp'].mean()
                if pd.notna(p_da) and pd.notna(c_da):
                    gap = (c_da - p_da).total_seconds() / 60
                    gap_str = f'  ← {gap:.0f}min gap'

            print(f"    {pid:<40} {gt.archetype:<14} "
                  f"prod={len(prod):4d} cons={len(cons):4d}{gap_str}")
            total += 1

    return total


# ── Tier 2: Custom log markers ────────────────────────────────────────────────

def generate_tier2(run_date: str, days: int, inputs_dir: Path) -> int:
    """
    Generate event files with non-standard activity names.

    Tests the activity_name_map contract feature.
    The events use custom names like 'queued', 'running', 'success'
    instead of Fracture's standard SCHEDULED, STARTED, COMPLETED.

    When the contract has activity_name_map defined, Fracture
    normalises these before token replay. Without the map the
    fitness score would be 0 — all activities are 'missing'.
    """
    from fracture.generator import StableMatureGenerator, SlowDriftingGenerator
    from fracture.schema import PipelineContract

    y, m, d  = int(run_date[:4]), int(run_date[4:6]), int(run_date[6:])
    from datetime import date as dt_
    start    = dt_(y, m, d) - timedelta(days=days)
    total    = 0

    print(f"\n  airflow-pipelines  (custom log markers)")

    for pid, cfg in CUSTOM_MARKER_PIPELINES.get('airflow-pipelines', {}).items():
        contract  = make_contract(pid, 'airflow-pipelines', cfg)
        generator = get_generator(cfg)

        prod_standard, cons, gt = generator.generate(
            contract     = contract,
            days         = days,
            start_date   = start,
            pipeline_age = 90,
            seed         = hash(pid) % 8888,
        )

        # Remap activity names to custom names (simulates Airflow naming)
        activity_map = cfg.get('activity_map', {})
        reverse_map  = {v: k for k, v in activity_map.items()}

        prod_custom = prod_standard.copy()
        prod_custom['activity'] = prod_custom['activity'].map(
            lambda x: reverse_map.get(x, x)
        )

        d_ = inputs_dir / pid
        d_.mkdir(parents=True, exist_ok=True)
        prod_custom.to_parquet(d_ / f'producer_{run_date}.parquet', index=False)

        # Also save a version with standard names for comparison
        prod_standard.to_parquet(d_ / f'producer_{run_date}_standard.parquet',
                                  index=False)

        print(f"    {pid:<40} custom names: "
              f"{sorted(set(prod_custom['activity'].unique()))}")
        print(f"    {'':40} standard:     "
              f"{sorted(set(prod_standard['activity'].unique()))}")
        total += 1

    return total


# ── Tier 3: Dirty logs ────────────────────────────────────────────────────────

def generate_tier3(run_date: str, inputs_dir: Path) -> int:
    """
    Generate dirty log files that test preflight checks.

    Each scenario tests a specific preflight check.
    When run through fracture, each should trigger the documented
    RED or AMBER check and produce the expected result.

    These are written to inputs/dirty_log_{scenario}/ directories.
    """
    from fracture.generator import StableMatureGenerator
    from fracture.schema import PipelineContract, ContractStatus

    base_contract_data = {
        'pipeline_id':    'dirty_log_test',
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
        'log_contract':   {'transport': 'parquet', 'source_path': 'x/'},
    }

    from datetime import date as dt_
    y, m, d = int(run_date[:4]), int(run_date[4:6]), int(run_date[6:])
    today = dt_(y, m, d)

    print(f"\n  dirty log scenarios")
    total = 0

    def base_events(n_runs: int = 5) -> pd.DataFrame:
        """Generate clean base events."""
        rows = []
        for i in range(n_runs):
            base_ts = pd.Timestamp(today, tz='UTC') + timedelta(hours=6, minutes=i*3)
            for j, act in enumerate(['SCHEDULED','STARTED','COMPLETED','DATA_AVAILABLE']):
                rows.append({
                    'pipeline_run_id': f'run_{i:03d}',
                    'activity':        act,
                    'timestamp':       base_ts + timedelta(minutes=j*15),
                    'team':            'producer',
                })
        return pd.DataFrame(rows)

    scenarios = {

        # AMBER checks
        'missing_terminal': lambda df: df[df.activity != 'COMPLETED'],

        'low_coverage': lambda df: df[df.activity.isin(['SCHEDULED', 'STARTED'])],

        'ordering_violation': lambda df: pd.concat([
            df[df.activity.isin(['COMPLETED','DATA_AVAILABLE'])],
            df[df.activity.isin(['SCHEDULED','STARTED'])],
        ], ignore_index=True),

        'first_event_wrong': lambda df: pd.concat([
            df[df.activity == 'STARTED'].head(1),
            df[df.activity != 'STARTED'],
        ], ignore_index=True),

        # RED checks
        'all_same_timestamp': lambda df: df.assign(
            timestamp=pd.Timestamp('2026-01-01T06:00:00', tz='UTC')
        ),

        'future_timestamps': lambda df: df.assign(
            timestamp=df['timestamp'].apply(
                lambda t: t + timedelta(days=365*5)
            )
        ),

        'empty_log': lambda df: df.iloc[0:0],  # empty DataFrame

        # Bilateral
        'consumer_clock_skew': lambda df: df.assign(
            timestamp=df['timestamp'].apply(
                lambda t: t - timedelta(minutes=20)
            ),
            team='consumer',
        ),
    }

    for scenario_name, transform_fn in scenarios.items():
        clean = base_events(n_runs=5)
        dirty = transform_fn(clean)

        d_ = inputs_dir / f'dirty_{scenario_name}'
        d_.mkdir(parents=True, exist_ok=True)

        if scenario_name == 'consumer_clock_skew':
            # Write clean producer + skewed consumer
            clean.to_parquet(d_ / f'producer_{run_date}.parquet', index=False)
            dirty.to_parquet(d_ / f'consumer_{run_date}.parquet', index=False)
            desc = f"consumer has -{20}min clock skew"
        else:
            dirty.to_parquet(d_ / f'producer_{run_date}.parquet', index=False)
            desc = DIRTY_LOG_SCENARIOS.get(scenario_name, '')

        n_rows = len(dirty)
        print(f"    dirty_{scenario_name:<30} {n_rows:3d} rows  {desc[:50]}")
        total += 1

    return total


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='01_setup.py',
        description='Generate synthetic event files for all tiers.',
    )
    parser.add_argument('--date', default=date.today().strftime('%Y%m%d'))
    parser.add_argument('--days', type=int, default=30)
    parser.add_argument('--team', default=None)
    parser.add_argument('--tier', type=int, default=0,
                        choices=[0, 1, 2, 3],
                        help='0=all, 1=core, 2=custom markers, 3=dirty logs')
    parser.add_argument('--inputs-dir', default='inputs')
    parser.add_argument('--list-teams', action='store_true')
    args = parser.parse_args()

    if args.list_teams:
        from scripts.team_config import summary
        summary()
        sys.exit(0)

    inputs_dir = Path(args.inputs_dir)
    inputs_dir.mkdir(exist_ok=True)

    print()
    print("Fracture — generating event files")
    print(f"  Date: {args.date}  Days: {args.days}")

    total = 0

    if args.tier in (0, 1):
        print()
        print("── Tier 1: Core pipelines ──────────────────────────────────")
        n = generate_tier1(args.date, args.days, inputs_dir, args.team)
        print(f"\n  {n} core pipelines generated")
        total += n

    if args.tier in (0, 2) and not args.team:
        print()
        print("── Tier 2: Custom log markers ──────────────────────────────")
        n = generate_tier2(args.date, args.days, inputs_dir)
        print(f"\n  {n} custom marker pipelines generated")
        total += n

    if args.tier in (0, 3) and not args.team:
        print()
        print("── Tier 3: Dirty logs ──────────────────────────────────────")
        n = generate_tier3(args.date, inputs_dir)
        print(f"\n  {n} dirty log scenarios generated")
        total += n

    print()
    print(f"Total: {total} pipeline event files written to {inputs_dir}/")
    print()
    print("Next: onboard teams")
    print("  python scripts/02_onboard_teams.py")
