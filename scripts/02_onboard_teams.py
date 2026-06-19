"""
scripts/02_onboard_teams.py

Simulates realistic team-wise onboarding to Fracture.

Each team goes through three steps:
  1. fracture register  → writes DRAFT contract, returns key
  2. fracture bootstrap → reads their event files, computes percentiles, ACTIVE
  3. fracture run       → runs conformance, appends to conformance_log.csv

This is the real onboarding flow. The only difference from production:
  - Events come from 01_setup.py (synthetic) not real pipelines
  - All steps run in one script instead of over days/weeks

RBAC story:
  Each team gets key FRC-xxxxxxxx.
  fracture run-all --team X runs only that team's pipelines.
  In future: key becomes access token, stored securely.

Run after 01_setup.py:
  python scripts/01_setup.py      # generate events
  python scripts/02_onboard_teams.py  # register + bootstrap + run

Usage:
  python scripts/02_onboard_teams.py
  python scripts/02_onboard_teams.py --team market-risk-quant
  python scripts/02_onboard_teams.py --skip-run
"""

import sys
import warnings
import types
import argparse
from datetime import date
from pathlib import Path

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.team_config import TEAM_PIPELINES, TEAM_OWNERS, PIPELINE_CONSUMERS


def onboard_team(
    team:          str,
    run_date:      str,
    contracts_dir: str = 'contracts',
    inputs_dir:    str = 'inputs',
    skip_run:      bool = False,
) -> dict:
    """
    Register, bootstrap and run conformance for all pipelines of one team.
    Returns {pipeline_id: key}.
    """
    from fracture.cli import cmd_register, cmd_run, _generate_key
    from fracture.bootstrap import bootstrap_contract

    pipelines = TEAM_PIPELINES.get(team, {})
    owner     = TEAM_OWNERS.get(team, f'{team}@bank.com')

    print(f"\n  {'━'*50}")
    print(f"  Team : {team}")
    print(f"  Owner: {owner}")
    print(f"  {'━'*50}")

    keys = {}

    for pipeline_id in pipelines:
        consumer = PIPELINE_CONSUMERS.get(pipeline_id, 'downstream-team')
        crit     = 'high' if team in ('payments-platform', 'aml-platform') else 'medium'
        key      = _generate_key(pipeline_id)

        print(f"\n  Pipeline: {pipeline_id}")

        # Step 1: Register
        contract_path = Path(contracts_dir) / f'{pipeline_id}.yaml'
        if contract_path.exists():
            print(f"  [1/3] register  : already exists → {key}")
        else:
            reg_args = types.SimpleNamespace(
                name           = pipeline_id.replace('_', ' ').title(),
                owner          = owner,
                producer_team  = team,
                consumer_team  = consumer,
                expected_start = '06:00',
                expected_end   = '08:30',
                criticality    = crit,
                slack          = f'#{team}-alerts' if crit == 'high' else None,
                contracts_dir  = contracts_dir,
                inputs_dir     = inputs_dir,
            )
            rc = cmd_register(reg_args)
            if rc != 0:
                print(f"  [1/3] register  : x failed")
                continue
            print(f"  [1/3] register  : v key={key}")

        keys[pipeline_id] = key

        # Step 2: Bootstrap
        pipeline_dir = Path(inputs_dir) / pipeline_id
        if not pipeline_dir.exists() or not list(pipeline_dir.glob('producer_*.parquet')):
            print(f"  [2/3] bootstrap : x no event files in {pipeline_dir}")
            print(f"        → run scripts/01_setup.py first")
            continue

        try:
            br = bootstrap_contract(
                pipeline_id   = pipeline_id,
                inputs_dir    = inputs_dir,
                contracts_dir = contracts_dir,
            )
            print(
                f"  [2/3] bootstrap : v "
                f"p50={br.p50_minutes}m  "
                f"p95={br.p95_minutes}m  "
                f"p99={br.p99_minutes}m  "
                f"({br.n_runs} runs, {br.coverage_pct:.0%} complete)"
            )
            if br.warning:
                print(f"         ! {br.warning[:75]}")
        except Exception as e:
            print(f"  [2/3] bootstrap : x {e}")
            continue

        # Step 3: Run conformance
        if skip_run:
            print(f"  [3/3] run       : skipped (--skip-run)")
            continue

        run_args = types.SimpleNamespace(
            key           = key,
            pipeline_id   = pipeline_id,
            date          = run_date,
            contracts_dir = contracts_dir,
            inputs_dir    = inputs_dir,
        )
        try:
            rc = cmd_run(run_args)
            print(f"  [3/3] run       : {'v logged' if rc == 0 else '! see above'}")
        except Exception as e:
            print(f"  [3/3] run       : x {e}")

    return keys


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='02_onboard_teams.py',
        description='Simulate team-wise onboarding to Fracture.',
    )
    parser.add_argument('--team',          default=None)
    parser.add_argument('--date',          default=date.today().strftime('%Y%m%d'))
    parser.add_argument('--skip-run',      action='store_true')
    parser.add_argument('--contracts-dir', default='contracts')
    parser.add_argument('--inputs-dir',    default='inputs')
    args = parser.parse_args()

    Path(args.contracts_dir).mkdir(exist_ok=True)

    print()
    print("Fracture — Team-wise onboarding")
    print(f"  Date    : {args.date}")
    print(f"  Steps   : register → bootstrap → {'run' if not args.skip_run else 'skip run'}")
    print(f"  RBAC    : each team gets their own FRC-xxxxxxxx key")

    all_keys = {}
    teams_to_onboard = [args.team] if args.team else sorted(TEAM_PIPELINES.keys())

    for team in teams_to_onboard:
        k = onboard_team(
            team          = team,
            run_date      = args.date,
            contracts_dir = args.contracts_dir,
            inputs_dir    = args.inputs_dir,
            skip_run      = args.skip_run,
        )
        all_keys[team] = k

    # Summary
    print()
    print()
    print("━" * 55)
    print("  ONBOARDING SUMMARY")
    print("━" * 55)
    print()
    print(f"  {'TEAM':<30} {'PIPELINE':<35} KEY")
    print("  " + "─" * 80)
    for team, keys in sorted(all_keys.items()):
        for pid, key in sorted(keys.items()):
            print(f"  {team:<30} {pid:<35} {key}")

    print()
    print("  Commands for each team:")
    print()
    for team in sorted(all_keys.keys()):
        print(f"  # {team}")
        print(f"  python -m fracture.cli run-all --team {team}")
        print()

    print("  Fleet-wide:")
    print("  python -m fracture.cli status")
    print("  python -m fracture.cli log --tail 20")
