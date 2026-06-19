"""
scripts/06_generate_team_contracts.py

Director-level fleet setup script.

Reads team_config.py and for every pipeline:
  1. Generates realistic event files (30 days producer + consumer)
  2. Registers the contract via CLI
  3. Bootstraps percentiles from the generated events
  4. Activates the contract

After this script: every pipeline is ACTIVE and ready for fracture run-all.

This is the "as if all 10 teams onboarded simultaneously" scenario.
In reality: each team runs fracture register + bootstrap + activate themselves.

Usage:
  python scripts/06_generate_team_contracts.py
  python scripts/06_generate_team_contracts.py --days 30 --dry-run
  python scripts/06_generate_team_contracts.py --team risk-technology
"""

import sys
import os
import argparse
import warnings
import types
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings('ignore')


# ── Generator map ─────────────────────────────────────────────────────────────

def get_generator(cfg: dict):
    from fracture.generator import (
        StableMatureGenerator, AssumptionAsymmetryGenerator,
        SlowDriftingGenerator, SilentPipelineGenerator,
        FastDriftingNewGenerator, BimodalGenerator,
    )
    a = cfg['archetype']
    return {
        'healthy':      StableMatureGenerator(),
        'asymmetry':    AssumptionAsymmetryGenerator(
                            cfg.get('initial_gap', 20),
                            cfg.get('gap_growth', 1.0)),
        'degrading':    SlowDriftingGenerator(cfg.get('drift_rate', 2.0)),
        'silent':       SilentPipelineGenerator(cfg.get('silent_days', [0])),
        'fast_drifting':FastDriftingNewGenerator(cfg.get('drift_rate', 4.0)),
        'bimodal':      BimodalGenerator(),
    }[a]


# ── Contract builder ──────────────────────────────────────────────────────────

def build_contract_args(pid: str, team: str, cfg: dict,
                        consumers: dict, owners: dict) -> types.SimpleNamespace:
    """Build the args namespace that cmd_register expects."""
    # Map archetype → criticality
    crit_map = {
        'silent':       'high',
        'fast_drifting':'high',
        'degrading':    'medium',
        'asymmetry':    'medium',
        'healthy':      'medium',
        'bimodal':      'medium',
    }
    criticality  = crit_map.get(cfg['archetype'], 'medium')
    consumer_team = consumers.get(pid, 'downstream-ops')
    owner        = owners.get(team, f'{team}@bank.com')
    slack        = f"#{team}-alerts" if criticality == 'high' else None

    return types.SimpleNamespace(
        name         = pid.replace('_', ' ').title(),
        owner        = owner,
        producer_team= team,
        consumer_team= consumer_team,
        expected_start='06:00',
        expected_end ='08:30',
        criticality  = criticality,
        slack        = slack,
        contracts_dir= 'contracts',
        inputs_dir   = 'inputs',
    )


# ── Event file writer ─────────────────────────────────────────────────────────

def write_event_files(pid: str, cfg: dict, days: int,
                      pipeline_age: int, seed: int, today: date):
    """
    Generate and write event files to inputs/{pid}/
    Producer and consumer events in separate parquet files.
    """
    from fracture.schema import PipelineContract
    from fracture.generator import StableMatureGenerator

    # Minimal contract for the generator
    contract = PipelineContract(**{
        'pipeline_id':    pid,
        'owner':          'setup@bank.com',
        'producer_team':  'setup',
        'consumer_team':  'setup-consumer',
        'criticality':    'medium',
        'status':         'draft',
        'expected_start': '06:00',
        'expected_end':   '08:30',
        'grace_minutes':  15,
        'p50_minutes':    45,
        'p95_minutes':    75,
        'p99_minutes':    90,
        'notifications':  [],
        'log_contract': {
            'transport':   'parquet',
            'source_path': f'inputs/{pid}/',
        },
    })

    gen   = get_generator(cfg)
    # Include today's input file so the normal quickstart can run `fracture run-all`
    # without passing --date. For days=30 this creates today plus the prior 29 days.
    start = today - timedelta(days=days - 1)

    try:
        prod, cons, gt = gen.generate(contract, days, start, pipeline_age, seed)
    except Exception as e:
        print(f"    !  Generator failed for {pid}: {e}")
        return False

    # Write files
    d = Path('inputs') / pid
    d.mkdir(parents=True, exist_ok=True)

    # Fracture's file convention is one producer/consumer file per input date.
    # The generator returns all requested history in one DataFrame, so split it
    # by the event date before writing. This keeps dashboard behavior consistent:
    # one input date means one day's runs; all input dates means the full history.
    prod_events = prod[['pipeline_run_id', 'activity', 'timestamp', 'team']].copy()
    prod_events['__input_date'] = pd.to_datetime(prod_events['timestamp']).dt.strftime('%Y%m%d')

    for input_date, day_events in prod_events.groupby('__input_date', sort=True):
        day_events.drop(columns='__input_date').to_parquet(
            d / f'producer_{input_date}.parquet',
            index=False,
        )

    if cons is not None and len(cons) > 0:
        cons_events = cons[['pipeline_run_id', 'activity', 'timestamp', 'team']].copy()
        cons_events['__input_date'] = pd.to_datetime(cons_events['timestamp']).dt.strftime('%Y%m%d')

        for input_date, day_events in cons_events.groupby('__input_date', sort=True):
            day_events.drop(columns='__input_date').to_parquet(
                d / f'consumer_{input_date}.parquet',
                index=False,
            )
        return True, True   # has_consumer=True

    return True, False      # has_consumer=False


# ── Main ──────────────────────────────────────────────────────────────────────

def main(days: int = 30, team_filter: str = None,
         dry_run: bool = False, pipeline_age: int = 180,
         clean: bool = False):

    from scripts.team_config import (
        TEAM_PIPELINES, CUSTOM_MARKER_PIPELINES,
        TEAM_OWNERS, PIPELINE_CONSUMERS
    )
    import fracture.cli as cli
    import shutil

    # ── Clean slate if requested ──────────────────────────────────────────
    if clean:
        print("  Cleaning previous run...")
        for d in ['contracts', 'inputs']:
            if Path(d).exists():
                shutil.rmtree(d)
                print(f"    deleted {d}/")
        for f in ['conformance_log.csv', 'cluster_assignments.csv',
                  'pipeline_registry.csv']:
            if Path(f).exists():
                Path(f).unlink()
                print(f"    deleted {f}")
        print()

    today = date.today()
    rng   = np.random.RandomState(42)

    # Build full pipeline list
    all_pipelines = []
    for team, pipelines in TEAM_PIPELINES.items():
        if team_filter and team != team_filter:
            continue
        for pid, cfg in pipelines.items():
            all_pipelines.append((team, pid, cfg))

    # Add custom marker pipelines
    for team, pipelines in CUSTOM_MARKER_PIPELINES.items():
        if team_filter and team != team_filter:
            continue
        for pid, cfg in pipelines.items():
            all_pipelines.append((team, pid, cfg))

    print()
    print("╔" + "═" * 66 + "╗")
    print(f"║  FRACTURE — Fleet Setup                                          ║")
    print(f"║  Director view: 10 teams, {len(all_pipelines)} pipelines, {days} days of history  ║")
    print("╚" + "═" * 66 + "╝")
    print()
    print(f"  {'─'*62}")
    print(f"  {'TEAM':<28} {'PIPELINE':<30} STATUS")
    print(f"  {'─'*62}")

    success_count = 0
    fail_count    = 0

    for team, pid, cfg in all_pipelines:
        seed = int(rng.randint(0, 99999))

        if dry_run:
            arch = cfg['archetype']
            print(f"  [dry-run] {team:<26} {pid:<30} {arch}")
            continue

        # Step 1: Write event files
        result = write_event_files(pid, cfg, days, pipeline_age, seed, today)
        if result is False:
            print(f"  x  {team:<26} {pid:<30} event generation failed")
            fail_count += 1
            continue

        ok, has_consumer = result

        # Step 2: Register
        try:
            reg_args = build_contract_args(pid, team, cfg, PIPELINE_CONSUMERS, TEAM_OWNERS)
            cli.cmd_register(reg_args)
        except Exception as e:
            # Already registered — skip
            if 'already exists' in str(e).lower() or Path(f'contracts/{pid}.yaml').exists():
                pass
            else:
                print(f"  x  {team:<26} {pid:<30} register failed: {e}")
                fail_count += 1
                continue

        # Step 3: Bootstrap
        key = cli._generate_key(pid)
        try:
            boot_args = types.SimpleNamespace(
                key=key, pipeline_id=pid,
                no_activate=True,
                contracts_dir='contracts', inputs_dir='inputs',
            )
            cli.cmd_bootstrap(boot_args)
        except Exception as e:
            print(f"  !  {team:<26} {pid:<30} bootstrap warning: {e}")

        # Step 4: Activate
        try:
            act_args = types.SimpleNamespace(
                key=key, pipeline_id=pid,
                all_pipelines=False,
                contracts_dir='contracts', inputs_dir='inputs',
            )
            cli.cmd_activate(act_args)
        except Exception as e:
            print(f"  !  {team:<26} {pid:<30} activate warning: {e}")

        arch    = cfg['archetype']
        cons_ok = '+ consumer' if has_consumer else 'producer only'
        print(f"  v  {team:<26} {pid:<30} {arch:<14} {cons_ok}")
        success_count += 1

    if dry_run:
        print()
        print(f"  [dry-run] {len(all_pipelines)} pipelines would be generated")
        return

    print(f"  {'─'*62}")
    print()
    print(f"  v  {success_count} pipelines ready")
    if fail_count:
        print(f"  x  {fail_count} failures")
    print()
    print(f"  Next:")
    print(f"    fracture run-all          → run conformance for all teams")
    print(f"    python report.py          → director-level fleet report")
    print(f"    python clustering.py      → cluster by health pattern")
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='06_generate_team_contracts.py')
    parser.add_argument('--days',     type=int, default=30,
                        help='Days of event history per pipeline (default 30)')
    parser.add_argument('--age',      type=int, default=180,
                        help='Pipeline age in days (default 180)')
    parser.add_argument('--team',     default=None,
                        help='Only generate for one team')
    parser.add_argument('--dry-run',  action='store_true',
                        help='Show what would be generated without running')
    parser.add_argument('--clean',    action='store_true',
                        help='Delete contracts/ inputs/ and CSV files before running. '
                             'Use when re-running from scratch.')
    args = parser.parse_args()
    main(args.days, args.team, args.dry_run, args.age, args.clean)
