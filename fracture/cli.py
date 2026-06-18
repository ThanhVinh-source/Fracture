"""
fracture/cli.py

Fracture CLI — designed around team onboarding.

The three steps every team goes through:

  Step 1: fracture register
    Team provides pipeline name, teams, SLA window.
    Fracture writes the contract and returns a pipeline key.
    Contract starts as DRAFT — not yet measured.

  Step 2: fracture bootstrap --key {key}
    Team brings 30 days of event files.
    Fracture computes real percentiles from actual execution.
    Contract moves from DRAFT → ACTIVE.

  Step 3: fracture run --key {key}
    Daily command. Team runs this after each pipeline execution.
    Fracture reads today's event file, runs conformance,
    appends result to conformance_log.csv.

Fleet commands (no key needed):
  fracture run-all      run all active pipelines for today
  fracture status       show fleet health from CSV log
  fracture log          show recent log entries
  fracture delete       remove a run by primary key (pipeline_id + date)
  fracture demo         run on synthetic data — no setup needed

CSV schema (conformance_log.csv):
  pipeline_id, run_date, final_score, confidence_level,
  timing_zone, pattern, bilateral_gap_minutes, days_of_history,
  sequence_fitness, timing_score, completeness_score,
  variance_cv, alert_owner, run_timestamp

Primary key: (pipeline_id, run_date)

Future additions (not yet built):
  RBAC: key becomes an access token — only key holder runs conformance
  Iceberg: CSV → Iceberg table with time travel queries
  Catalog: role-based schema access per team
"""

import argparse
import csv
import hashlib
import sys
import uuid
import yaml
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path

warnings.filterwarnings('ignore')


# ── CSV conformance log ───────────────────────────────────────────────────────

LOG_COLUMNS = [
    "pipeline_id",
    "pipeline_key",
    "run_date",
    "final_score",
    "confidence_level",
    "timing_zone",
    "pattern",
    "bilateral_gap_minutes",

    # Gap diagnostics for bilateral and chain handoff views.
    # Empty cells mean the gap was unavailable or not computed.
    "bilateral_gap_trend",
    "bilateral_gap_p95",
    "upstream_gap_minutes",
    "upstream_gap_trend",
    "upstream_gap_p95",
    "downstream_gap_minutes",
    "downstream_gap_trend",
    "downstream_gap_p95",

    # Historical score diagnostics for drift charts and fleet analytics.
    # These are populated only when historical scores are available.
    "temporal_variance_cv",
    "drift_rate_per_day",
    "score_range",
    "gap_drift_per_day",
    "mean_score_historical",
    "min_score_historical",

    # Root-cause diagnostics for dashboard annotations.
    "changepoint_detected",
    "changepoint_date",
    "variant_explainer",

    "days_of_history",
    "sequence_fitness",
    "timing_score",
    "completeness_score",
    "variance_cv",
    "alert_owner",
    "run_timestamp",
]

# Registry CSV — maps pipeline_key to pipeline_id
REGISTRY_COLUMNS = [
    "pipeline_key",
    "pipeline_id",
    "owner",
    "producer_team",
    "consumer_team",
    "status",
    "registered_at",
]


def _log_path() -> Path:
    return Path("conformance_log.csv")


def _registry_path() -> Path:
    return Path("pipeline_registry.csv")


def _ensure_csv(path: Path, columns: list):
    if not path.exists():
        with open(path, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=columns).writeheader()


def _read_csv(path: Path) -> list:
    _ensure_csv(path, LOG_COLUMNS if 'final_score' in LOG_COLUMNS else REGISTRY_COLUMNS)
    if not path.exists():
        return []
    with open(path, 'r', newline='') as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list, columns: list):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _append_log(row: dict):
    """Append or update by primary key (pipeline_id, run_date)."""
    _ensure_csv(_log_path(), LOG_COLUMNS)
    rows = _read_csv(_log_path())
    pk = (row['pipeline_id'], row['run_date'])
    updated = False
    for i, r in enumerate(rows):
        if (r.get('pipeline_id'), r.get('run_date')) == pk:
            rows[i] = row
            updated = True
            break
    if not updated:
        rows.append(row)
    _write_csv(_log_path(), rows, LOG_COLUMNS)


def _lookup_key(key: str) -> str:
    """Resolve pipeline_key → pipeline_id from registry."""
    _ensure_csv(_registry_path(), REGISTRY_COLUMNS)
    rows = _read_csv(_registry_path())
    for r in rows:
        if r.get('pipeline_key') == key:
            return r['pipeline_id']
    return None


def _result_to_row(pipeline_id: str, key: str, result, run_date: str) -> dict:
    """Convert engine result to a CSV log row."""
    if result.status != 'SUCCESS' or not result.conformance_result:
        return {
            'pipeline_id':           pipeline_id,
            'pipeline_key':          key or '',
            'run_date':              run_date,
            'final_score':           '',
            'confidence_level':      result.status,
            'timing_zone':           '',
            'pattern':               '',
            'bilateral_gap_minutes': '',

            'bilateral_gap_trend':    '',
            'bilateral_gap_p95':      '',
            'upstream_gap_minutes':   '',
            'upstream_gap_trend':     '',
            'upstream_gap_p95':       '',
            'downstream_gap_minutes': '',
            'downstream_gap_trend':   '',
            'downstream_gap_p95':     '',

            'temporal_variance_cv':   '',
            'drift_rate_per_day':     '',
            'score_range':            '',
            'gap_drift_per_day':      '',
            'mean_score_historical':  '',
            'min_score_historical':   '',

            'changepoint_detected':   '',
            'changepoint_date':       '',
            'variant_explainer':      '',


            'days_of_history':       '',
            'sequence_fitness':      '',
            'timing_score':          '',
            'completeness_score':    '',
            'variance_cv':           '',
            'alert_owner':           '',
            'run_timestamp':         datetime.now().isoformat(),
        }

    r = result.conformance_result
    d = r.diagnostics
    return {
        'pipeline_id':           pipeline_id,
        'pipeline_key':          key or '',
        'run_date':              run_date,
        'final_score':           round(r.final_score, 4),
        'confidence_level':      r.confidence_level,
        'timing_zone':           r.timing_zone,
        'pattern':               r.pattern,
        'bilateral_gap_minutes': round(r.bilateral_gap_minutes, 1)
                                 if r.bilateral_gap_minutes else '',
        
        'bilateral_gap_trend':    d.bilateral_gap_trend if d else '',
        'bilateral_gap_p95':      _fmt_optional_float(d.bilateral_gap_p95, 1) if d else '',
        'upstream_gap_minutes':   _fmt_optional_float(d.upstream_gap_minutes, 1) if d else '',
        'upstream_gap_trend':     d.upstream_gap_trend if d else '',
        'upstream_gap_p95':       _fmt_optional_float(d.upstream_gap_p95, 1) if d else '',
        'downstream_gap_minutes': _fmt_optional_float(d.downstream_gap_minutes, 1) if d else '',
        'downstream_gap_trend':   d.downstream_gap_trend if d else '',
        'downstream_gap_p95':     _fmt_optional_float(d.downstream_gap_p95, 1) if d else '',

        'temporal_variance_cv':   _fmt_optional_float(d.temporal_variance_cv, 4) if d else '',
        'drift_rate_per_day':     _fmt_optional_float(d.drift_rate_per_day, 6) if d else '',
        'score_range':            _fmt_optional_float(d.score_range, 4) if d else '',
        'gap_drift_per_day':      _fmt_optional_float(d.gap_drift_per_day, 4) if d else '',
        'mean_score_historical':  _fmt_optional_float(d.mean_score_historical, 4) if d else '',
        'min_score_historical':   _fmt_optional_float(d.min_score_historical, 4) if d else '',
        
        'changepoint_detected':   bool(getattr(d.changepoint, 'detected', False)) if d else '',
        'changepoint_date':       _first_changepoint_date(d.changepoint) if d else '',
        'variant_explainer':      getattr(d.variant_comparison, 'fitness_explainer', '') if d and d.variant_comparison else '',


        'days_of_history':       r.days_of_history,
        'sequence_fitness':      round(d.sequence_fitness, 4) if d else '',
        'timing_score':          round(d.timing_score, 4) if d else '',
        'completeness_score':    round(d.completeness_score, 4) if d else '',
        'variance_cv':           round(d.variance_cv, 4) if d else '',
        'alert_owner':           r.alert_owner(),
        'run_timestamp':         datetime.now().isoformat(),
    }


def _fmt_optional_float(value, ndigits=4):
    """
    Format optional numeric diagnostics for conformance_log.csv.

    Many dashboard fields are only available after enough history exists.
    Empty string means "not computed"; it should not be confused with 0.
    """
    if value is None:
        return ''
    return round(value, ndigits)


def _first_changepoint_date(changepoint):
    """
    Extract the first detected changepoint date for dashboard timelines.

    Changepoint detection can be skipped when there is insufficient history.
    In that case, write an empty CSV cell instead of a fake date.
    """
    if not changepoint or not getattr(changepoint, 'changepoint_dates', None):
        return ''
    return str(changepoint.changepoint_dates[0])


# ── Key generation ────────────────────────────────────────────────────────────

def _generate_key(pipeline_id: str) -> str:
    """
    Generate a deterministic short key for a pipeline.

    Format: FRC-{8 hex chars}
    Example: FRC-3a7f92b1

    Deterministic so the same pipeline always gets the same key
    (useful for reproducibility). In production this would be a
    random UUID stored securely.
    """
    h = hashlib.sha256(pipeline_id.encode()).hexdigest()[:8]
    return f"FRC-{h}"


def _pipeline_id_from_name(name: str) -> str:
    """Convert a human pipeline name to a valid pipeline_id slug."""
    import re
    slug = name.lower().strip()
    slug = re.sub(r'[^a-z0-9]+', '_', slug)
    slug = slug.strip('_')
    return slug


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_register(args):
    """
    Register a new pipeline contract.

    The team provides:
      - Pipeline name (human readable)
      - Owner email
      - Producer team name
      - Consumer team name
      - Expected start and end time
      - Criticality

    Fracture:
      - Generates a pipeline_id (slug)
      - Generates a pipeline_key (FRC-xxxxxxxx)
      - Writes contracts/{pipeline_id}.yaml (status: draft)
      - Registers in pipeline_registry.csv
      - Returns key to team

    Contract starts as DRAFT. Team must run bootstrap
    with their event logs before conformance runs.
    """
    from fracture.schema import (
        PipelineContract, LogContract, Notification,
        ContractStatus, Criticality, Transport, NotificationChannel
    )

    # Interactive registration if args not fully provided
    print()
    print("  Fracture — Pipeline Registration")
    print("  " + "─" * 40)

    # Collect inputs — use args if provided, otherwise prompt
    name          = args.name or _prompt("  Pipeline name",
                                          "e.g. Trade Batch Grouping")
    owner         = args.owner or _prompt("  Owner email",
                                           "e.g. platform@bank.com")
    producer_team = args.producer_team or _prompt("  Producer team",
                                                    "e.g. market-risk-quant")
    consumer_team = args.consumer_team or _prompt("  Consumer team",
                                                    "e.g. grid-scheduler")
    expected_start = args.expected_start or _prompt("  Expected start (HH:MM)",
                                                      "e.g. 06:00")
    expected_end   = args.expected_end or _prompt("  Expected end   (HH:MM)",
                                                    "e.g. 08:30")
    criticality    = args.criticality or _prompt("  Criticality",
                                                   "high / medium / low",
                                                   default="medium")
    slack_channel  = args.slack if hasattr(args,'slack') else _prompt("  Slack channel (optional)",
                                            "e.g. #data-alerts",
                                            optional=True)

    # Generate IDs
    pipeline_id  = _pipeline_id_from_name(name)
    pipeline_key = _generate_key(pipeline_id)

    # Build notifications
    notifications = []
    if slack_channel:
        notifications.append(
            Notification(channel=NotificationChannel.SLACK, target=slack_channel)
        )
    elif criticality == 'high':
        # HIGH criticality requires notification — use LOG as fallback
        notifications.append(
            Notification(channel=NotificationChannel.LOG, target='fracture.log')
        )

    # Default SLA percentiles — will be overwritten by bootstrap
    # These are intentionally conservative — bootstrap will compute real values
    window_minutes = _window_minutes(expected_start, expected_end)
    p50 = max(10, window_minutes // 3)
    p95 = max(20, window_minutes // 2)
    p99 = max(30, int(window_minutes * 0.7))
    grace = max(5, window_minutes - p99 - 5)

    # Build contract
    try:
        contract = PipelineContract(
            pipeline_id    = pipeline_id,
            owner          = owner,
            producer_team  = producer_team,
            consumer_team  = consumer_team,
            expected_start = expected_start,
            expected_end   = expected_end,
            grace_minutes  = grace,
            p50_minutes    = p50,
            p95_minutes    = p95,
            p99_minutes    = p99,
            criticality    = criticality,
            notifications  = notifications,
            status         = ContractStatus.DRAFT,
            log_contract   = LogContract(
                transport   = Transport.PARQUET,
                source_path = f"inputs/{pipeline_id}/",
            ),
        )
    except Exception as e:
        print(f"\n  ✗ Validation failed: {e}")
        return 1

    # Write contract YAML
    contracts_dir = Path(args.contracts_dir)
    contracts_dir.mkdir(exist_ok=True)
    contract_path = contracts_dir / f"{pipeline_id}.yaml"

    data = contract.model_dump(mode='json')
    data['_pipeline_key'] = pipeline_key  # store key in YAML for reference
    contract_path.write_text(
        yaml.dump(data, default_flow_style=False, allow_unicode=True)
    )

    # Register in registry CSV
    _ensure_csv(_registry_path(), REGISTRY_COLUMNS)
    registry_rows = _read_csv(_registry_path())

    # Check for duplicate
    existing = [r for r in registry_rows if r['pipeline_id'] == pipeline_id]
    if existing:
        print(f"\n  ⚠  Pipeline '{pipeline_id}' already registered.")
        print(f"     Existing key: {existing[0]['pipeline_key']}")
        return 1

    registry_rows.append({
        'pipeline_key':  pipeline_key,
        'pipeline_id':   pipeline_id,
        'owner':         owner,
        'producer_team': producer_team,
        'consumer_team': consumer_team,
        'status':        'DRAFT',
        'registered_at': datetime.now().isoformat(),
    })
    _write_csv(_registry_path(), registry_rows, REGISTRY_COLUMNS)

    # Print result
    print()
    print("  ╔══════════════════════════════════════════════╗")
    print("  ║  Pipeline registered successfully            ║")
    print("  ╚══════════════════════════════════════════════╝")
    print()
    print(f"  Pipeline ID  : {pipeline_id}")
    print(f"  Pipeline Key : {pipeline_key}  ← save this")
    print(f"  Contract     : {contract_path}")
    print(f"  Status       : DRAFT")
    print()
    print("  Next steps:")
    print(f"  1. Collect 30 days of event logs from your pipeline")
    print(f"  2. Format as 4-column parquet:")
    print(f"     pipeline_run_id | activity | timestamp | team")
    print(f"  3. Drop files in: inputs/{pipeline_id}/")
    print(f"     producer_YYYYMMDD.parquet")
    print(f"     consumer_YYYYMMDD.parquet  (if bilateral)")
    print(f"  4. Run: fracture bootstrap --key {pipeline_key}")
    print()
    return 0


def cmd_bootstrap(args):
    """
    Bootstrap a contract from actual event log history.

    Reads event files from inputs/{pipeline_id}/ and computes
    real execution percentiles from actual data.
    Updates contract: DRAFT → ACTIVE with computed p50/p95/p99.

    This is the step that makes the contract meaningful.
    Percentiles from real data, not guesses.
    """
    # Resolve pipeline_id from key or direct ID
    pipeline_id = _resolve_pipeline(args)
    if not pipeline_id:
        return 1

    contracts_dir = Path(args.contracts_dir)
    inputs_dir    = Path(args.inputs_dir)
    contract_path = contracts_dir / f"{pipeline_id}.yaml"

    if not contract_path.exists():
        print(f"  ✗ Contract not found: {contract_path}")
        print(f"    Run: fracture register first")
        return 1

    # Load contract
    from fracture.schema import load_contract
    try:
        contract = load_contract(str(contract_path))
    except Exception as e:
        print(f"  ✗ Contract error: {e}")
        return 1

    print(f"\n  Bootstrapping '{pipeline_id}'...")

    # Find all parquet files for this pipeline
    pipeline_dir = inputs_dir / pipeline_id
    if not pipeline_dir.exists():
        print(f"  ✗ No input files found in {pipeline_dir}")
        print(f"    Drop your event files there first.")
        return 1

    import pandas as pd
    parquet_files = sorted(pipeline_dir.glob("producer_*.parquet"))
    csv_files     = sorted(pipeline_dir.glob("producer_*.csv"))
    all_files     = parquet_files + csv_files

    if not all_files:
        print(f"  ✗ No producer_*.parquet files in {pipeline_dir}")
        return 1

    # Load all event files
    dfs = []
    for f in all_files:
        try:
            if f.suffix == '.parquet':
                dfs.append(pd.read_parquet(f))
            else:
                dfs.append(pd.read_csv(f))
        except Exception:
            pass

    if not dfs:
        print(f"  ✗ Could not read any event files.")
        return 1

    events = pd.concat(dfs, ignore_index=True)
    events['timestamp'] = pd.to_datetime(events['timestamp'], utc=True)

    print(f"  Loaded {len(events)} events from {len(all_files)} files")

    # Compute durations per run (STARTED → COMPLETED)
    durations = _compute_durations(events, contract)

    if len(durations) < 5:
        print(f"  ✗ Only {len(durations)} complete runs found.")
        print(f"    Need at least 5 complete runs for reliable percentiles.")
        return 1

    import numpy as np
    p50 = max(1, int(np.percentile(durations, 50)))
    p95 = max(p50 + 1, int(np.percentile(durations, 95)))
    p99 = max(p95 + 1, int(np.percentile(durations, 99)))

    window = contract.window_minutes()
    grace  = max(5, min(30, window - p99 - 5))

    print(f"\n  Computed from {len(durations)} complete runs:")
    print(f"    p50: {p50} min")
    print(f"    p95: {p95} min")
    print(f"    p99: {p99} min")
    print(f"    grace: {grace} min (computed)")

    no_activate = getattr(args, 'no_activate', False)

    # Update contract YAML — percentiles always written
    # Status only set to active if not --no-activate
    data = yaml.safe_load(contract_path.read_text())
    data['p50_minutes']  = p50
    data['p95_minutes']  = p95
    data['p99_minutes']  = p99
    data['grace_minutes'] = grace
    if not no_activate:
        data['status'] = 'active'
    # else: leave status as draft — human reviews first
    contract_path.write_text(
        yaml.dump(data, default_flow_style=False, allow_unicode=True)
    )

    if no_activate:
        # Leave as DRAFT — human reviews before activating
        print()
        print(f"  ✓ Bootstrap complete: {contract_path}")
        print(f"  Status: DRAFT (--no-activate set)")
        print()
        print(f"  Review the computed values in {contract_path}")
        print(f"  When satisfied, activate:")
        key = _get_key_for_pipeline(pipeline_id)
        print(f"    fracture activate --key {key}")
        print()
        print(f"  Or activate all reviewed pipelines at once:")
        print(f"    fracture activate --all")
    else:
        # Auto-activate (default) — bootstrap trusts its own computation
        # Activate: update both YAML and registry
        import yaml as _yaml2
        raw2 = _yaml2.safe_load(contract_path.read_text())
        raw2['status'] = 'active'
        contract_path.write_text(
            _yaml2.dump(raw2, default_flow_style=False, allow_unicode=True)
        )
        registry_rows = _read_csv(_registry_path())
        for r in registry_rows:
            if r['pipeline_id'] == pipeline_id:
                r['status'] = 'ACTIVE'
        _write_csv(_registry_path(), registry_rows, REGISTRY_COLUMNS)

        print()
        print(f"  ✓ Contract activated: {contract_path}")
        print(f"  Status: DRAFT → ACTIVE")
        print()
        print(f"  Ready. Run daily conformance:")
        key = _get_key_for_pipeline(pipeline_id)
        print(f"    fracture run --key {key}")
        print(f"    or")
        print(f"    fracture run --pipeline-id {pipeline_id}")

    return 0


def cmd_activate(args):
    """
    Activate a DRAFT contract after human review.

    This is the explicit human gate between bootstrap and conformance.
    Use this when bootstrap was run with --no-activate.

    What it does:
      1. Loads the contract
      2. Verifies it is DRAFT (not already active)
      3. Shows the bootstrapped values for final review
      4. Sets status: active in the YAML
      5. Updates the registry

    The human gate:
      DRAFT means "computed but not reviewed"
      ACTIVE means "reviewed and approved by a human"
      Fracture will not run conformance against DRAFT contracts.
      This prevents circular measurement against unreviewed specs.

    Usage:
      fracture activate --key FRC-xxxxxxxx
      fracture activate --pipeline-id trade_batch
      fracture activate --all   (activate all DRAFT contracts)
    """
    # Handle --all flag
    if getattr(args, 'all_pipelines', False):
        return _activate_all(args)

    pipeline_id = _resolve_pipeline(args)
    if not pipeline_id:
        return 1

    contracts_dir = Path(args.contracts_dir)
    contract_path = contracts_dir / f"{pipeline_id}.yaml"

    if not contract_path.exists():
        print(f"\n  ✗ Contract not found: {contract_path}")
        return 1

    import yaml as _yaml
    raw = _yaml.safe_load(contract_path.read_text())

    current_status = raw.get('status', 'draft')

    if current_status == 'active':
        print(f"\n  ○ {pipeline_id} is already ACTIVE")
        print(f"    No action needed.")
        return 0

    if current_status == 'deprecated':
        print(f"\n  ✗ {pipeline_id} is DEPRECATED — cannot activate")
        return 1

    # Show the values for final human review
    print()
    print(f"  ┌─ Review before activating: {pipeline_id}")
    print(f"  │  p50_minutes  : {raw.get('p50_minutes', '?')} min")
    print(f"  │  p95_minutes  : {raw.get('p95_minutes', '?')} min")
    print(f"  │  p99_minutes  : {raw.get('p99_minutes', '?')} min")
    print(f"  │  grace_minutes: {raw.get('grace_minutes', '?')} min")
    print(f"  │  window       : {_window_mins(raw)} min")
    print(f"  │  producer     : {raw.get('producer_team', '?')}")
    print(f"  │  consumer     : {raw.get('consumer_team', '?')}")
    print(f"  └─ status: DRAFT → ACTIVE")
    print()

    # Activate
    raw['status'] = 'active'
    contract_path.write_text(
        _yaml.dump(raw, default_flow_style=False, allow_unicode=True)
    )

    # Update registry
    registry_rows = _read_csv(_registry_path())
    for r in registry_rows:
        if r.get('pipeline_id') == pipeline_id:
            r['status'] = 'ACTIVE'
    _write_csv(_registry_path(), registry_rows, REGISTRY_COLUMNS)

    key = _get_key_for_pipeline(pipeline_id)
    print(f"  ✓ {pipeline_id} is now ACTIVE")
    print()
    print(f"  Run conformance:")
    print(f"    fracture run --key {key}")
    return 0


def _activate_all(args):
    """Activate all DRAFT contracts in the contracts directory."""
    import yaml as _yaml
    contracts_dir = Path(args.contracts_dir)
    draft_contracts = []

    for cf in sorted(contracts_dir.glob('*.yaml')):
        if cf.stem.startswith('_'):
            continue
        raw = _yaml.safe_load(cf.read_text())
        if raw.get('status', 'draft') == 'draft':
            draft_contracts.append(cf.stem)

    if not draft_contracts:
        print("\n  No DRAFT contracts found.")
        return 0

    print(f"\n  Activating {len(draft_contracts)} DRAFT contract(s):")
    print()

    import types
    for pid in draft_contracts:
        mock = types.SimpleNamespace(
            key=None, pipeline_id=pid,
            contracts_dir=args.contracts_dir,
            all_pipelines=False,
        )
        cmd_activate(mock)

    return 0


def _window_mins(raw: dict) -> int:
    """Compute window minutes from raw contract dict."""
    try:
        start = raw.get('expected_start', '06:00')
        end   = raw.get('expected_end',   '08:30')
        sh, sm = map(int, start.split(':'))
        eh, em = map(int, end.split(':'))
        s = sh*60+sm; e = eh*60+em
        if e <= s: e += 24*60
        return e - s
    except Exception:
        return 0



def cmd_deprecate(args):
    """
    Remove a pipeline from active Fracture monitoring.

    Sets contract status: deprecated.
    Updates registry status to DEPRECATED.
    Historical conformance log is preserved — nothing is deleted.
    The pipeline can be re-registered and re-onboarded fresh.

    Use this when:
      - A pipeline is being decommissioned
      - A team is re-onboarding with a new contract
      - You want to demonstrate manual onboarding from scratch

    Usage:
      fracture deprecate --key FRC-xxxxxxxx
      fracture deprecate --pipeline-id payment_settlements_daily
      fracture deprecate --pipeline-id X --reason "replatforming to Kafka"
    """
    import yaml as _yaml

    pipeline_id = _resolve_pipeline(args)
    if not pipeline_id:
        return 1

    contracts_dir = Path(args.contracts_dir)
    contract_path = contracts_dir / f"{pipeline_id}.yaml"

    if not contract_path.exists():
        print(f"\n  ✗ Contract not found: {contract_path}")
        return 1

    raw    = _yaml.safe_load(contract_path.read_text())
    status = raw.get('status', 'draft')

    if status == 'deprecated':
        print(f"\n  ○ {pipeline_id} is already DEPRECATED")
        return 0

    reason = getattr(args, 'reason', 'manually deprecated')

    # Show what is being deprecated
    print()
    print(f"  Deprecating: {pipeline_id}")
    print(f"  Reason     : {reason}")
    print(f"  Current    : {status.upper()}")
    print()

    # Update contract YAML
    raw['status'] = 'deprecated'
    raw['_deprecated_reason'] = reason
    raw['_deprecated_at']     = datetime.now().isoformat()
    contract_path.write_text(
        _yaml.dump(raw, default_flow_style=False, allow_unicode=True)
    )

    # Update registry
    registry_rows = _read_csv(_registry_path())
    for r in registry_rows:
        if r.get('pipeline_id') == pipeline_id:
            r['status'] = 'DEPRECATED'
    _write_csv(_registry_path(), registry_rows, REGISTRY_COLUMNS)

    key = _get_key_for_pipeline(pipeline_id)

    print(f"  ✓ {pipeline_id} → DEPRECATED")
    print(f"  Contract: {contract_path}")
    print(f"  Historical log preserved in conformance_log.csv")
    print()
    print(f"  Fracture will no longer run conformance for this pipeline.")
    print(f"  fracture run-all will skip it.")
    print()
    print(f"  To re-onboard fresh:")
    print(f"    1. Delete: contracts/{pipeline_id}.yaml")
    print(f"    2. Run:    fracture register --name \"{pipeline_id}\" ...")
    print(f"    3. Run:    fracture bootstrap --key <new-key> --no-activate")
    print(f"    4. Review: contracts/{pipeline_id}.yaml")
    print(f"    5. Run:    fracture activate --key <new-key>")
    print(f"    6. Run:    fracture run --key <new-key>")
    return 0


def cmd_deregister(args):
    """
    Full clean slate — team is changing their event structure.

    Use this when:
      - required_events are changing (new activities added or removed)
      - activity_name_map is being restructured
      - terminal_event is changing
      - The pipeline is being replatformed with different event semantics

    What it does:
      1. Deletes contracts/{pipeline_id}.pnml  ← stale Petri net removed
      2. Deletes contracts/{pipeline_id}.yaml  ← stale contract removed
      3. Removes from pipeline_registry.csv
      4. Logs the deregistration to conformance_log.csv as a NOTE row
         so history shows when the contract changed

    What it does NOT do:
      - Does NOT delete historical conformance rows (audit trail preserved)
      - Does NOT delete input event files in inputs/
      - Does NOT affect other pipelines

    After deregistering:
      fracture register   ← fresh registration with new event structure
      fracture bootstrap  ← compute percentiles from new event files
      fracture activate   ← human reviews new contract
      fracture run        ← conformance with correct Petri net

    Why this is different from deprecate:
      deprecate  → pipeline is shutting down, PNML kept for audit
      deregister → team is changing contract, PNML deleted (stale)

    The PNML deletion is the critical difference.
    A stale PNML measures conformance against the old process model.
    Silent wrong scores are worse than no scores.
    """
    import yaml as _yaml
    from datetime import datetime as _dt

    pipeline_id = _resolve_pipeline(args)
    if not pipeline_id:
        return 1

    contracts_dir = Path(args.contracts_dir)
    contract_path = contracts_dir / f"{pipeline_id}.yaml"
    pnml_path     = contracts_dir / f"{pipeline_id}.pnml"
    reason        = getattr(args, 'reason', 'event structure change')
    key           = _get_key_for_pipeline(pipeline_id)

    print()
    print(f"  Deregistering: {pipeline_id}")
    print(f"  Reason       : {reason}")
    print()

    # Show what will be deleted
    print(f"  Files to be removed:")
    if contract_path.exists():
        print(f"    ✗  {contract_path}  (stale contract)")
    if pnml_path.exists():
        print(f"    ✗  {pnml_path}  (stale Petri net — MUST be deleted)")
    else:
        print(f"    ○  {pnml_path}  (no PNML cache found)")
    print()
    print(f"  Preserved:")
    print(f"    ✓  conformance_log.csv  (historical measurements)")
    print(f"    ✓  inputs/{pipeline_id}/  (event files)")
    print()

    # Log the deregistration event to conformance_log.csv
    # So the history shows when the contract changed
    note_row = {
        'pipeline_id':           pipeline_id,
        'pipeline_key':          key or '',
        'run_date':              datetime.now().strftime('%Y%m%d'),
        'final_score':           '',
        'confidence_level':      f'DEREGISTERED',
        'timing_zone':           '',
        'pattern':               '',
        'bilateral_gap_minutes': '',
        'days_of_history':       '',
        'sequence_fitness':      '',
        'timing_score':          '',
        'completeness_score':    '',
        'variance_cv':           '',
        'alert_owner':           '',
        'run_timestamp':         datetime.now().isoformat(),
    }
    _append_log(note_row)

    # Delete PNML (stale Petri net)
    if pnml_path.exists():
        pnml_path.unlink()
        print(f"  ✓ Deleted PNML cache: {pnml_path}")
    
    # Delete contract YAML
    if contract_path.exists():
        contract_path.unlink()
        print(f"  ✓ Deleted contract: {contract_path}")

    # Remove from registry
    registry_rows = _read_csv(_registry_path())
    before = len(registry_rows)
    registry_rows = [r for r in registry_rows
                     if r.get('pipeline_id') != pipeline_id]
    after = len(registry_rows)
    if before > after:
        _write_csv(_registry_path(), registry_rows, REGISTRY_COLUMNS)
        print(f"  ✓ Removed from registry")

    print()
    print(f"  {pipeline_id} deregistered.")
    print(f"  Historical conformance preserved in conformance_log.csv")
    print()
    print(f"  Re-onboard with new event structure:")
    print(f"    1. fracture register --name \"{pipeline_id}\" ...")
    print(f"    2. fracture bootstrap --key <new-key> --no-activate")
    print(f"    3. Review the new contract values")
    print(f"    4. fracture activate --key <new-key>")
    print(f"    5. fracture run --key <new-key>")
    return 0


def _load_historical_scores(pipeline_id: str, grain: str = 'pipeline') -> list:
    """
    Load historical conformance scores from CSV for drift detection.

    Returns list of (datetime, float) tuples — one per past daily run.
    These feed into pattern detection, drift analysis, and weekday clustering.

    Without this, drift detection always returns STABLE and CV=0.0 because
    compute_conformance only sees today's events, not the history.

    For grain='pipeline': one entry per daily batch run.
    For grain='trade': daily aggregates — mean score across all trades per day.
      Raw per-trade timestamps would be meaningless for drift detection
      (50,000 data points per day, all on the same date).
    """
    rows = _read_log_rows()
    pipeline_rows = [r for r in rows if r.get('pipeline_id') == pipeline_id
                     and r.get('final_score')]

    if not pipeline_rows:
        return []

    if grain in ('trade', 'record'):
        # Daily aggregates: group by run_date, take mean final_score
        from collections import defaultdict
        daily = defaultdict(list)
        for r in pipeline_rows:
            try:
                score = float(r['final_score'])
                rd    = datetime.strptime(r['run_date'], '%Y%m%d')
                daily[rd.date()].append(score)
            except (ValueError, KeyError):
                continue
        return sorted(
            [(datetime.combine(d, datetime.min.time()), sum(scores)/len(scores))
             for d, scores in daily.items()],
            key=lambda x: x[0]
        )
    else:
        # Pipeline grain: one entry per run row
        scores = []
        for r in pipeline_rows:
            try:
                score = float(r['final_score'])
                rd    = datetime.strptime(r['run_date'], '%Y%m%d')
                scores.append((rd, score))
            except (ValueError, KeyError):
                continue
        return sorted(scores, key=lambda x: x[0])


def cmd_run(args):
    """
    Run conformance for one pipeline.
    Reads contract + today's event file.
    Loads historical scores from CSV for drift detection.
    Appends result to conformance_log.csv.
    """
    pipeline_id = _resolve_pipeline(args)
    if not pipeline_id:
        return 1

    run_date = args.date or date.today().strftime('%Y%m%d')
    key      = _get_key_for_pipeline(pipeline_id)

    print(f"\n  Running conformance: {pipeline_id}  ({run_date})")

    # Load historical scores for drift detection
    # This is what makes pattern=DRIFTING and variance_cv meaningful
    from fracture.schema import load_contract
    from pathlib import Path as _Path
    contract_path = _Path(getattr(args, 'contracts_dir', 'contracts')) / f'{pipeline_id}.yaml'
    grain = 'pipeline'
    if contract_path.exists():
        try:
            _c = load_contract(str(contract_path))
            grain = _c.log_contract.grain
        except Exception:
            pass

    historical = _load_historical_scores(pipeline_id, grain)

    from fracture.engine import FractureEngine
    engine = FractureEngine(
        contracts_dir = args.contracts_dir,
        inputs_dir    = args.inputs_dir,
    )
    result = engine.run_pipeline(
        pipeline_id,
        date_str         = run_date,
        historical_scores = historical,
    )

    _print_result_full(result)

    row = _result_to_row(pipeline_id, key, result, run_date)
    _append_log(row)

    if result.status == 'SUCCESS':
        print(f"\n  Logged → conformance_log.csv")
    return 0 if result.status == 'SUCCESS' else 1


def cmd_run_all(args):
    """Run conformance for all active pipelines."""
    from fracture.engine import FractureEngine
    from fracture.schema import load_contract, ContractStatus

    run_date      = args.date or date.today().strftime('%Y%m%d')
    contracts_dir = Path(args.contracts_dir)

    contract_files = sorted(
        f for f in contracts_dir.glob('*.yaml')
        if not f.stem.startswith('_')
    )

    if not contract_files:
        print(f"\n  No contracts found in {contracts_dir}")
        print(f"  Run: fracture register")
        return 1

    # Team filter for RBAC — each team runs only their pipelines
    team_filter = getattr(args, 'team', None)
    if team_filter:
        from fracture.schema import load_contract
        filtered = []
        for cf in contract_files:
            try:
                c = load_contract(str(cf))
                if c.producer_team == team_filter or c.consumer_team == team_filter:
                    filtered.append(cf)
            except Exception:
                pass
        if not filtered:
            print(f"\n  No pipelines found for team '{team_filter}'")
            print(f"  Run 'fracture list' to see registered pipelines and teams.")
            return 1
        print(f"\n  Team filter: {team_filter} ({len(filtered)}/{len(contract_files)} pipelines)")
        contract_files = filtered

    print(f"\n  Fleet conformance: {len(contract_files)} pipelines  ({run_date})")
    print()
    print(f"  {'PIPELINE':<38} {'SCORE':<8} {'ZONE':<10} {'CONF':<10} SUMMARY")
    print("  " + "─" * 82)

    engine = FractureEngine(
        contracts_dir = args.contracts_dir,
        inputs_dir    = args.inputs_dir,
    )

    successes = 0
    for cf in contract_files:
        pid = cf.stem

        # Skip DEPRECATED pipelines — they are shut down intentionally.
        # Running conformance on a deprecated pipeline produces misleading
        # NO_INPUT results that pollute the fleet history.
        try:
            import yaml as _yaml
            raw_contract = _yaml.safe_load(cf.read_text())
            if raw_contract.get('status', '').lower() == 'deprecated':
                print(f"  ○ {pid:<36} DEPRECATED — skipped")
                continue
        except Exception:
            pass  # if we cannot read the YAML, let the engine handle it

        hist_scores = _load_historical_scores(pid)
        result = engine.run_pipeline(pid, date_str=run_date,
                                     historical_scores=hist_scores)
        key    = _get_key_for_pipeline(pid)
        row    = _result_to_row(pid, key, result, run_date)
        _append_log(row)

        if result.status == 'SUCCESS' and result.conformance_result:
            r    = result.conformance_result
            icon = '✓' if r.timing_zone == 'GREEN' else '⚠' if 'AMBER' in r.timing_zone else '✗'
            gap  = f" ← {r.bilateral_gap_minutes:.0f}min gap" if r.bilateral_gap_minutes and r.bilateral_gap_minutes > 5 else ""
            print(
                f"  {icon} {pid:<36} {r.final_score:.0%}    "
                f"{r.timing_zone:<10} {r.confidence_level:<10} "
                f"{r.human_summary()[:30]}{gap}"
            )
            successes += 1
        else:
            print(f"  ○ {pid:<36} {result.status}")

    print("  " + "─" * 82)
    print(f"  {successes}/{len(contract_files)} conformant  →  conformance_log.csv")
    return 0


def cmd_validate(args):
    """Validate a contract YAML without running conformance."""
    from fracture.schema import validate_contract_file
    from fracture.petri import net_summary, contract_to_petri_net
    from fracture.schema import load_contract

    result = validate_contract_file(args.contract)
    if result['valid']:
        c = load_contract(args.contract)
        s = net_summary(c)
        print(f"\n  ✓  {args.contract}")
        print(f"     pipeline_id : {result['pipeline_id']}")
        print(f"     status      : {result['status']}")
        print(f"     timing      : {result['timing']}")
        print(f"     petri net   : {s['n_places']}p {s['n_transitions']}t  ({s['soundness_basis'][:40]})")
        if result.get('warnings'):
            for w in result['warnings']:
                print(f"     ⚠  {w}")
    else:
        print(f"\n  ✗  {args.contract}")
        for e in result.get('errors', []):
            print(f"     {e}")
        return 1
    return 0


def cmd_template(args):
    """Print 4-column CSV template for manual event upload."""
    from fracture.ingest import print_csv_template
    print_csv_template(args.pipeline_id, n_days=getattr(args, 'days', 3))
    return 0


def cmd_status(args):
    """Show pipeline conformance history."""
    rows = _read_log_rows()
    if not rows:
        print("\n  No runs logged yet.")
        print("  Run: fracture run-all")
        return 0

    if hasattr(args, 'pipeline_id') and args.pipeline_id:
        pid  = _resolve_pipeline(args) or args.pipeline_id
        rows = [r for r in rows if r['pipeline_id'] == pid]
        if not rows:
            print(f"\n  No history for '{pid}'")
            return 1

    # Latest per pipeline
    latest = {}
    for r in rows:
        latest[r['pipeline_id']] = r

    print(f"\n  {'PIPELINE':<35} {'DATE':<10} {'SCORE':<8} {'ZONE':<10} {'PATTERN':<14} GAP")
    print("  " + "─" * 82)
    for pid, r in sorted(latest.items()):
        score   = r['final_score'] or f"[{r['confidence_level']}]"
        zone    = r['timing_zone'] or '─'
        pattern = r['pattern'] or '─'
        gap     = f"{r['bilateral_gap_minutes']} min" if r['bilateral_gap_minutes'] else '─'
        print(f"  {pid:<35} {r['run_date']:<10} {score:<8} {zone:<10} {pattern:<14} {gap}")

    print()
    print(f"  {len(latest)} pipelines  ·  {len(rows)} total runs  ·  {_log_path()}")
    return 0


def cmd_delete(args):
    """Delete a run entry by primary key (pipeline_id, run_date)."""
    rows   = _read_log_rows()
    before = len(rows)
    rows   = [r for r in rows
              if not (r['pipeline_id'] == args.pipeline_id
                      and r['run_date'] == args.date)]
    after = len(rows)
    if before == after:
        print(f"\n  No entry: pipeline_id={args.pipeline_id} date={args.date}")
        return 1
    _write_csv(_log_path(), rows, LOG_COLUMNS)
    print(f"\n  ✓ Deleted: {args.pipeline_id} / {args.date}")
    print(f"  Log: {before} → {after} rows")
    return 0


def cmd_log(args):
    """Show recent entries from the conformance log."""
    rows = _read_log_rows()
    if not rows:
        print("\n  Log is empty.")
        return 0

    rows = sorted(rows, key=lambda r: r.get('run_timestamp', ''), reverse=True)
    n    = min(args.tail, len(rows))

    print(f"\n  Last {n} entries  ·  {_log_path()}")
    print()
    for r in rows[:n]:
        score = r['final_score'] or f"[{r['confidence_level']}]"
        gap   = f"  gap={r['bilateral_gap_minutes']}m" if r['bilateral_gap_minutes'] else ''
        print(
            f"  {r['run_date']}  {r['pipeline_id']:<32}  "
            f"score={score:<8} {r['timing_zone'] or '─':<8} "
            f"{r['pattern'] or '─'}{gap}"
        )
    return 0


def cmd_demo(args):
    """
    Run Fracture on 6 synthetic pipelines.
    No setup needed. Everything generated internally.
    Results logged to conformance_log.csv.
    """
    from fracture.factory import generate_demo_contracts
    from fracture.generator import (
        StableMatureGenerator, AssumptionAsymmetryGenerator,
        SlowDriftingGenerator, SilentPipelineGenerator,
    )
    from fracture.conformance import compute_conformance
    from fracture.schema import ContractStatus

    print("\n  Fracture demo — 6 synthetic pipelines")
    print()

    contracts, archetypes = generate_demo_contracts()
    contracts = [c.model_copy(update={'status': ContractStatus.ACTIVE})
                 for c in contracts]

    GENERATORS = {
        'payment_settlements_daily': StableMatureGenerator(),
        'trade_positions_sftp':      AssumptionAsymmetryGenerator(initial_gap=25),
        'grid_corehours_calc':       SlowDriftingGenerator(drift_rate_per_week=2.0),
        'customer_risk_features':    SilentPipelineGenerator([0, 3]),
        '_default':                  StableMatureGenerator(),
    }

    today    = date.today().strftime('%Y%m%d')
    start    = date.today() - timedelta(days=30)

    print(f"  {'PIPELINE':<40} {'SCORE':<8} {'ZONE':<10} SUMMARY")
    print("  " + "─" * 78)

    for contract in contracts:
        gen  = GENERATORS.get(contract.pipeline_id, GENERATORS['_default'])
        prod, cons, _ = gen.generate(
            contract     = contract,
            days         = 30,
            start_date   = start,
            pipeline_age = 180,
            seed         = hash(contract.pipeline_id) % 9999,
        )
        try:
            r    = compute_conformance(
                       producer_events = prod,
                       contract        = contract,
                       consumer_events = cons if len(cons) > 0 else None,
                   )
            icon = '✓' if r.timing_zone == 'GREEN' else '⚠' if 'AMBER' in r.timing_zone else '✗'
            key  = _generate_key(contract.pipeline_id)
            print(
                f"  {icon} {contract.pipeline_id:<38} "
                f"{r.final_score:.0%}    {r.timing_zone:<10} "
                f"{r.human_summary()[:40]}"
            )

            # Log it
            mock = type('R', (), {
                'status': 'SUCCESS',
                'conformance_result': r,
                'error_message': None,
                'pipeline_id': contract.pipeline_id,
            })()
            _append_log(_result_to_row(contract.pipeline_id, key, mock, today))

        except Exception as e:
            print(f"  ✗ {contract.pipeline_id:<38} ERROR: {str(e)[:50]}")

    print()
    print(f"  Logged → conformance_log.csv")
    print(f"  Run 'fracture status' to see fleet health")
    return 0


def cmd_explain(args):
    """
    Human summary for the latest conformance run.

    Reads from conformance_log.csv — no re-computation needed.
    Translates the numbers into plain English that a non-engineer
    can read and act on.

    Usage:
      fracture explain --key FRC-xxxxxxxx
      fracture explain --pipeline-id trade_positions_sftp
      fracture explain --all
    """
    rows = _read_log_rows()
    if not rows:
        print("\n  No runs logged yet. Run: fracture run-all")
        return 0

    all_pipelines = getattr(args, 'all_pipelines', False)

    if all_pipelines:
        # Latest row per pipeline
        latest = {}
        for r in rows:
            latest[r['pipeline_id']] = r
        pipelines_to_explain = list(latest.values())
    else:
        pipeline_id = _resolve_pipeline(args)
        if not pipeline_id:
            return 1
        pipeline_rows = [r for r in rows if r['pipeline_id'] == pipeline_id]
        if not pipeline_rows:
            print(f"\n  No runs found for '{pipeline_id}'")
            print(f"  Run: fracture run --pipeline-id {pipeline_id}")
            return 1
        # Latest run
        pipelines_to_explain = [
            sorted(pipeline_rows, key=lambda r: r.get('run_timestamp',''))[-1]
        ]

    print()

    for row in sorted(pipelines_to_explain, key=lambda r: r['pipeline_id']):
        pid   = row['pipeline_id']
        score = row.get('final_score', '')
        zone  = row.get('timing_zone', '')
        conf  = row.get('confidence_level', '')
        pat   = row.get('pattern', '')
        gap   = row.get('bilateral_gap_minutes', '')
        owner = row.get('alert_owner', '')
        dt    = row.get('run_date', '')

        # Build human summary from CSV fields
        # (mirrors ConformanceResult.human_summary() logic)
        summary = _build_human_summary(score, zone, conf, pat, gap, owner)

        # Pick icon
        if not score or score in ('BLOCKED','DRAFT','NO_INPUT','ERROR'):
            icon = '○'
        elif zone == 'GREEN':
            icon = '✓'
        elif zone in ('AMBER', 'RED'):
            icon = '⚠'
        else:
            icon = '✗'

        print(f"  {icon}  {pid}")
        print(f"     Date      : {dt}")

        if score and score not in ('BLOCKED','DRAFT','NO_INPUT','ERROR'):
            score_pct = f"{float(score):.0%}" if score else '?'
            print(f"     Score     : {score_pct}  ({zone})  confidence={conf}")
            print(f"     Pattern   : {pat}")
            if gap:
                print(f"     Gap       : {gap} min bilateral")
            print(f"     Owner     : {owner}")

        print(f"     Summary   : {summary}")

        # Verbose mode — show full diagnostics
        if getattr(args, 'verbose', False):
            seq  = row.get('sequence_fitness', '')
            time = row.get('timing_score', '')
            comp = row.get('completeness_score', '')
            cv   = row.get('variance_cv', '')

            if seq:
                print(f"     ── Diagnostics ──")
                print(f"     seq/time/comp : {float(seq):.3f} / {float(time):.3f} / {float(comp):.3f}")
                print(f"     variance_cv   : {float(cv):.4f}  ({'stable' if float(cv)<0.1 else 'variable'})")

            # Re-run conformance to get weekday pattern (not stored in CSV)
            _explain_weekday(row.get('pipeline_id',''), args)

        print()

    return 0


def _build_human_summary(score, zone, conf, pattern, gap, owner) -> str:
    """
    Reconstruct human_summary from CSV fields.
    Mirrors ConformanceResult.human_summary() without re-running conformance.
    """
    if not score or score in ('BLOCKED', 'DRAFT', 'NO_INPUT', 'ERROR'):
        status_map = {
            'BLOCKED':  'Conformance blocked — preflight check failed. Fix log extraction.',
            'DRAFT':    'Contract is DRAFT. Run: fracture activate --pipeline-id X',
            'NO_INPUT': 'No event file found for this date.',
            'ERROR':    'Conformance engine error. Check the log.',
        }
        return status_map.get(score, f'Status: {score}')

    if conf in ('LOW', 'UNRELIABLE'):
        return (
            'Measurement unreliable — low confidence. '
            'Fix log extraction before acting on this score.'
        )

    if zone == 'BREACH':
        return f'SLA breached. {owner} must investigate immediately.'

    if pattern == 'INTERMITTENT':
        if owner == 'platform-infrastructure':
            return (
                'Fails on specific weekdays — infrastructure problem, '
                'not pipeline problem. Route to platform-infrastructure.'
            )
        return 'Intermittent failures. Investigate specific failure days.'

    if pattern == 'DRIFTING':
        return (
            'Conformance declining. '
            'Schedule contract review with both teams this sprint.'
        )

    if gap and float(gap) > 10:
        return (
            f'Pipeline healthy but bilateral gap is {float(gap):.0f} min. '
            f'Consumer receives data later than producer thinks.'
        )

    try:
        score_pct = f"{float(score):.0%}"
        return f'Conformant at {score_pct}. {zone} zone. No action required.'
    except Exception:
        return f'Score={score} zone={zone}.'


def _explain_weekday(pipeline_id: str, args) -> None:
    """
    Re-run conformance to extract weekday pattern for verbose explain.
    Only called when --verbose is set.
    """
    try:
        from fracture.schema import load_contract
        from fracture.engine import FractureEngine

        contracts_dir = getattr(args, 'contracts_dir', 'contracts')
        inputs_dir    = getattr(args, 'inputs_dir',    'inputs')
        contract_path = Path(contracts_dir) / f'{pipeline_id}.yaml'

        if not contract_path.exists():
            return

        engine = FractureEngine(
            contracts_dir = contracts_dir,
            inputs_dir    = inputs_dir,
        )
        result = engine.run_pipeline(pipeline_id)

        if result.status == 'SUCCESS' and result.conformance_result:
            r  = result.conformance_result
            d  = r.diagnostics
            wp = d.weekday_pattern

            print(f"     ── Weekday pattern ──")
            if wp.has_weekday_clustering:
                print(f"     worst_day     : {wp.worst_weekday} ({wp.worst_weekday_failure_rate:.0%} failure rate)")
                print(f"     affected_days : {', '.join(wp.affected_weekdays)}")
                print(f"     other_days    : {wp.other_days_mean_score:.0%} mean score")
                if wp.infrastructure_probable:
                    print(f"     diagnosis     : INFRASTRUCTURE — not pipeline problem")
                    print(f"     action        : escalate to platform-infrastructure team")
                else:
                    print(f"     diagnosis     : pipeline-specific — investigate {wp.worst_weekday} runs")
            else:
                print(f"     no weekday clustering detected")

            if r.bilateral_gap_minutes:
                print(f"     ── Bilateral gap ──")
                print(f"     gap           : {r.bilateral_gap_minutes:.1f} min")
                print(f"     trend         : {d.bilateral_gap_trend}")
                print(f"     p95           : {d.bilateral_gap_p95:.1f} min (worst case)")

    except Exception:
        pass  # verbose explain never crashes the main output


def cmd_list_pipelines(args):
    """List all registered pipelines."""
    _ensure_csv(_registry_path(), REGISTRY_COLUMNS)
    rows = _read_csv(_registry_path())

    if not rows:
        print("\n  No pipelines registered yet.")
        print("  Run: fracture register")
        return 0

    print(f"\n  {'KEY':<18} {'PIPELINE':<35} {'STATUS':<10} {'OWNER'}")
    print("  " + "─" * 80)
    for r in sorted(rows, key=lambda x: x['pipeline_id']):
        print(
            f"  {r['pipeline_key']:<18} {r['pipeline_id']:<35} "
            f"{r['status']:<10} {r['owner']}"
        )

    print()
    print(f"  {len(rows)} pipelines registered  ·  {_registry_path()}")
    return 0


# ── Helpers ───────────────────────────────────────────────────────────────────

def _prompt(label: str, hint: str = '', default: str = None,
            optional: bool = False) -> str:
    """Interactive prompt. Returns default if provided and user hits enter."""
    hint_str    = f" ({hint})" if hint else ''
    default_str = f" [{default}]" if default else ''
    suffix      = ' (optional, press enter to skip): ' if optional else ': '
    raw = input(f"{label}{hint_str}{default_str}{suffix}").strip()
    if not raw and default:
        return default
    if not raw and optional:
        return ''
    if not raw and not optional:
        raise ValueError(f"{label} is required")
    return raw


def _window_minutes(start: str, end: str) -> int:
    sh, sm = map(int, start.split(':'))
    eh, em = map(int, end.split(':'))
    s = sh * 60 + sm
    e = eh * 60 + em
    if e <= s:
        e += 24 * 60
    return e - s


def _compute_durations(events, contract) -> list:
    """Compute execution duration per run from STARTED → COMPLETED."""
    import pandas as pd
    terminal = contract.log_contract.terminal_event
    start_act = 'STARTED'

    durations = []
    for run_id, group in events.groupby('pipeline_run_id'):
        group = group.sort_values('timestamp')
        acts  = group['activity'].tolist()
        ts    = group['timestamp'].tolist()

        if start_act in acts and terminal in acts:
            t_start = group[group.activity == start_act]['timestamp'].iloc[0]
            t_end   = group[group.activity == terminal]['timestamp'].iloc[-1]
            dur     = (t_end - t_start).total_seconds() / 60
            if dur > 0:
                durations.append(dur)

    return durations


def _resolve_pipeline(args) -> str:
    """Resolve pipeline_id from --key or --pipeline-id."""
    if hasattr(args, 'key') and args.key:
        pid = _lookup_key(args.key)
        if not pid:
            print(f"\n  ✗ Key not found: {args.key}")
            print(f"    Run 'fracture list' to see registered pipelines.")
        return pid
    if hasattr(args, 'pipeline_id') and args.pipeline_id:
        return args.pipeline_id
    print("\n  ✗ Provide --key FRC-xxxxxxxx or --pipeline-id NAME")
    return None


def _get_key_for_pipeline(pipeline_id: str) -> str:
    """Look up key for pipeline_id from registry."""
    _ensure_csv(_registry_path(), REGISTRY_COLUMNS)
    rows = _read_csv(_registry_path())
    for r in rows:
        if r.get('pipeline_id') == pipeline_id:
            return r['pipeline_key']
    return _generate_key(pipeline_id)


def _read_log_rows() -> list:
    _ensure_csv(_log_path(), LOG_COLUMNS)
    return _read_csv(_log_path())


def _print_result_full(result):
    """Print full conformance result to terminal."""
    if result.status == 'SUCCESS' and result.conformance_result:
        r = result.conformance_result
        d = r.diagnostics
        gap = f"{r.bilateral_gap_minutes:.1f} min" if r.bilateral_gap_minutes else "─"
        print()
        print(f"  ┌─ {result.pipeline_id}")
        print(f"  │  Score        : {r.final_score:.4f} ({r.final_score:.0%})")
        print(f"  │  Zone         : {r.timing_zone}")
        print(f"  │  Confidence   : {r.confidence_level}")
        print(f"  │  Pattern      : {r.pattern}")
        print(f"  │  Bilateral gap: {gap}")
        if d:
            print(f"  │  seq/time/comp: {d.sequence_fitness:.3f} / "
                  f"{d.timing_score:.3f} / {d.completeness_score:.3f}")
        print(f"  └─ {r.human_summary()}")
        print(f"     Alert owner  : {r.alert_owner()}")
    else:
        print(f"\n  {result.pipeline_id}: {result.status}")
        if hasattr(result, 'error_message') and result.error_message:
            print(f"  {result.error_message[:100]}")

def _ensure_in_registry(pipeline_id, key, owner, producer_team,
                         consumer_team, contracts_dir):
    """Add pipeline to registry if not already there."""
    _ensure_csv(_registry_path(), REGISTRY_COLUMNS)
    rows = _read_csv(_registry_path())
    if not any(r.get('pipeline_id') == pipeline_id for r in rows):
        rows.append({
            'pipeline_key':  key,
            'pipeline_id':   pipeline_id,
            'owner':         owner,
            'producer_team': producer_team,
            'consumer_team': consumer_team,
            'status':        'ACTIVE',
            'registered_at': datetime.now().isoformat(),
        })
        _write_csv(_registry_path(), rows, REGISTRY_COLUMNS)


def cmd_bootstrap_contract(pipeline_id: str, inputs_dir: str = 'inputs',
                            contracts_dir: str = 'contracts') -> int:
    """Thin CLI wrapper around bootstrap.bootstrap_contract."""
    from fracture.bootstrap import bootstrap_contract
    try:
        r = bootstrap_contract(pipeline_id, inputs_dir, contracts_dir)
        print(f"  bootstrap: {r.summary()}")
        return 0
    except Exception as e:
        print(f"  bootstrap failed: {e}")
        return 1
    
def cmd_visualize(args):
    """
    Export static visualization artifacts for one pipeline.

    Phase 3 supports:
    - gap: producer-consumer bilateral handoff timeline.
    - drift: final_score trend from conformance_log.csv.
    - all: every implemented visual.
    """
    kind = args.kind or "gap"

    # Heatmap is fleet-level, so it does not require --pipeline-id.
    # Pipeline-specific visuals still require a selected pipeline.
    pipeline_id = None
    if kind != "heatmap":
        pipeline_id = _resolve_pipeline(args)
        if not pipeline_id:
            return 1

    # Keep future options in argparse, but only run implemented visuals here.
    if kind not in ("gap", "drift", "heatmap", "all"):
        print()
        print(f"  Visualization kind '{kind}' is not implemented yet.")
        print("  Available in this build: gap, drift, heatmap")
        return 1

    from fracture.visualization import (
        load_conformance_log,
        load_pipeline_events,
        save_bilateral_gap_timeline,
        save_drift_chart,
        save_fleet_heatmap,
    )

    print()
    print(f"  Visualizing: {pipeline_id if pipeline_id else 'fleet'}")
    print(f"  Kind       : {kind}")
    print(f"  Date       : {args.date or 'today'}")

    saved_paths = []
    skipped = []

    if kind in ("gap", "all"):
        producer_event = "DATA_AVAILABLE"
        consumer_event = "DATA_AVAILABLE"

        # If a contract exists, use its configured handoff event names.
        contract_path = Path(args.contracts_dir) / f"{pipeline_id}.yaml"
        if contract_path.exists():
            try:
                from fracture.schema import load_contract
                contract = load_contract(str(contract_path))
                producer_event = contract.log_contract.upstream_producer_event
                consumer_event = contract.log_contract.upstream_consumer_event
            except Exception as e:
                # Bad contract should not block valid raw logs.
                print()
                print(f"  Warning: could not read contract events, using defaults: {e}")

        producer_df, consumer_df, load_status = load_pipeline_events(
            inputs_dir=args.inputs_dir,
            pipeline_id=pipeline_id,
            date_str=args.date,
        )

        if producer_df.empty:
            # Gap visual needs producer timestamps as the anchor.
            skipped.append(f"gap: {load_status}")
        elif consumer_df is None:
            # Bilateral gap cannot be measured from producer-only logs.
            skipped.append("gap: consumer log unavailable")
        else:
            output_path, status = save_bilateral_gap_timeline(
                producer_df=producer_df,
                consumer_df=consumer_df,
                pipeline_id=pipeline_id,
                output_dir=args.output_dir,
                producer_event=producer_event,
                consumer_event=consumer_event,
            )

            if status == "ok":
                saved_paths.append(output_path)
            else:
                skipped.append(f"gap: {status}")

    if kind in ("drift", "all"):
        # Drift visual uses conformance_log.csv, not inputs/.
        log_path = getattr(args, "log_path", "conformance_log.csv")
        conformance_df, log_status = load_conformance_log(log_path)

        if conformance_df.empty:
            skipped.append(f"drift: {log_status}")
        else:
            output_path, status = save_drift_chart(
                conformance_df=conformance_df,
                pipeline_id=pipeline_id,
                output_dir=args.output_dir,
            )

            if status == "ok":
                saved_paths.append(output_path)
            else:
                skipped.append(f"drift: {status}")

    if kind in ("heatmap", "all"):
        # Fleet heatmap uses conformance_log.csv across all pipelines.
        # It does not read producer/consumer input files.
        log_path = getattr(args, "log_path", "conformance_log.csv")
        conformance_df, log_status = load_conformance_log(log_path)

        if conformance_df.empty:
            skipped.append(f"heatmap: {log_status}")
        else:
            output_path, status = save_fleet_heatmap(
                conformance_df=conformance_df,
                output_dir=args.output_dir,
            )

            if status == "ok":
                saved_paths.append(output_path)
            else:
                skipped.append(f"heatmap: {status}")                

    if saved_paths:
        print()
        print("  Created:")
        for path in saved_paths:
            print(f"    {path}")

    if skipped:
        print()
        print("  Skipped:")
        for item in skipped:
            print(f"    {item}")

    # Success means at least one requested visual was created.
    return 0 if saved_paths else 1

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog        = 'fracture',
        description = 'Process conformance linter for data pipelines.',
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog = """
onboarding a new team (3 steps):
  fracture register                              ← step 1: register pipeline
  # drop event files in inputs/{pipeline_id}/
  fracture bootstrap --key FRC-xxxxxxxx          ← step 2: compute percentiles
  fracture run --key FRC-xxxxxxxx                ← step 3: daily conformance

fleet commands:
  fracture run-all                               ← run all active pipelines
  fracture status                                ← fleet health from CSV
  fracture log --tail 20                         ← recent log entries
  fracture list                                  ← all registered pipelines
  fracture delete --pipeline-id X --date DATE    ← remove a bad run
  fracture demo                                  ← try with synthetic data

primary key:
  (pipeline_id, run_date) — unique per pipeline per day
  use --key FRC-xxxxxxxx or --pipeline-id for run/bootstrap
        """
    )

    # Global flags
    parser.add_argument('--contracts-dir', default='contracts')
    parser.add_argument('--inputs-dir',    default='inputs')

    sub = parser.add_subparsers(dest='command')

    # register
    p_reg = sub.add_parser('register', help='Register a new pipeline')
    p_reg.add_argument('--name',           default=None)
    p_reg.add_argument('--owner',          default=None)
    p_reg.add_argument('--producer-team',  default=None, dest='producer_team')
    p_reg.add_argument('--consumer-team',  default=None, dest='consumer_team')
    p_reg.add_argument('--expected-start', default=None, dest='expected_start')
    p_reg.add_argument('--expected-end',   default=None, dest='expected_end')
    p_reg.add_argument('--criticality',    default=None)
    p_reg.add_argument('--slack',          default=None)

    # bootstrap
    p_boot = sub.add_parser('bootstrap', help='Compute percentiles from log history')
    p_boot.add_argument('--key',         default=None, help='Pipeline key FRC-xxxxxxxx')
    p_boot.add_argument('--pipeline-id', default=None, dest='pipeline_id')
    p_boot.add_argument('--no-activate', action='store_true', dest='no_activate',
                        help='Leave contract as DRAFT after bootstrap — human reviews first')

    # activate
    p_act = sub.add_parser('activate', help='Activate a DRAFT contract after review')
    p_act.add_argument('--key',         default=None, help='Pipeline key FRC-xxxxxxxx')
    p_act.add_argument('--pipeline-id', default=None, dest='pipeline_id')
    p_act.add_argument('--all',         action='store_true', dest='all_pipelines',
                        help='Activate all DRAFT contracts')

    # run
    p_run = sub.add_parser('run', help='Run conformance for one pipeline')
    p_run.add_argument('--key',         default=None)
    p_run.add_argument('--pipeline-id', default=None, dest='pipeline_id')
    p_run.add_argument('--date',        default=None, help='YYYYMMDD')

    # run-all
    p_all = sub.add_parser('run-all', help='Run all active pipelines')
    p_all.add_argument('--date', default=None)
    p_all.add_argument('--team', default=None,
                       help='Run only pipelines owned by this team')

    # validate
    p_val = sub.add_parser('validate', help='Validate a contract YAML')
    p_val.add_argument('--contract', required=True)

    # template
    p_tmpl = sub.add_parser('template', help='Print CSV event template')
    p_tmpl.add_argument('--pipeline-id', required=True, dest='pipeline_id')
    p_tmpl.add_argument('--days',        type=int, default=3)

    # status
    p_stat = sub.add_parser('status', help='Show conformance history')
    p_stat.add_argument('--key',         default=None)
    p_stat.add_argument('--pipeline-id', default=None, dest='pipeline_id')

    # visualize
    p_viz = sub.add_parser(
        'visualize',
        help='Export static visualizations for one pipeline'
    )
    p_viz.add_argument('--key', default=None,
                       help='Pipeline key FRC-xxxxxxxx')
    p_viz.add_argument('--pipeline-id', default=None, dest='pipeline_id')
    p_viz.add_argument('--date', default=None,
                       help='YYYYMMDD input date to visualize')
    p_viz.add_argument('--kind', default='gap',
                       choices=['gap', 'all', 'petri', 'drift', 'dfg', 'heatmap'],
                       help='Visualization kind. Phase 3 implements gap, drift, and heatmap.')
    p_viz.add_argument('--output-dir', default='outputs/visualizations',
                       help='Where visualization files are written')

    # Drift chart reads conformance history from this CSV.
    # Keeping this configurable lets tests and demos use temporary log files.
    p_viz.add_argument('--log-path', default='conformance_log.csv',
                       help='Conformance log used by drift visualizations')

    # delete
    p_del = sub.add_parser('delete', help='Delete a run entry')
    p_del.add_argument('--pipeline-id', required=True, dest='pipeline_id')
    p_del.add_argument('--date',        required=True)

    # log
    p_log = sub.add_parser('log', help='Show recent log entries')
    p_log.add_argument('--tail', type=int, default=10)

    # deprecate
    p_dep = sub.add_parser('deprecate',
                            help='Remove a pipeline from active monitoring')
    p_dep.add_argument('--key',         default=None)
    p_dep.add_argument('--pipeline-id', default=None, dest='pipeline_id')
    p_dep.add_argument('--reason',      default='manually deprecated',
                        help='Reason for deprecation (logged)')

    # deregister — team is changing event structure, full clean slate
    p_dereg = sub.add_parser('deregister',
                              help='Full clean slate — team is changing event structure')
    p_dereg.add_argument('--key',         default=None)
    p_dereg.add_argument('--pipeline-id', default=None, dest='pipeline_id')
    p_dereg.add_argument('--reason',      default='event structure change',
                          help='Reason for deregistration (logged to history)')

    # list
    sub.add_parser('list', help='List all registered pipelines')

    # explain
    p_exp = sub.add_parser('explain',
                            help='Human summary for latest run of a pipeline')
    p_exp.add_argument('--key',         default=None)
    p_exp.add_argument('--pipeline-id', default=None, dest='pipeline_id')
    p_exp.add_argument('--all',         action='store_true', dest='all_pipelines',
                        help='Explain all pipelines in the fleet')
    p_exp.add_argument('--verbose', '-v', action='store_true',
                        help='Show full diagnostics including weekday pattern')

    # demo
    sub.add_parser('demo', help='Run on synthetic demo data')

    args = parser.parse_args()

    dispatch = {
        'register':  cmd_register,
        'bootstrap': cmd_bootstrap,
        'activate':  cmd_activate,
        'run':       cmd_run,
        'run-all':   cmd_run_all,
        'validate':  cmd_validate,
        'template':  cmd_template,
        'status':    cmd_status,
        'visualize': cmd_visualize,
        'delete':    cmd_delete,
        'log':       cmd_log,
        'deprecate':   cmd_deprecate,
        'deregister':  cmd_deregister,
        'list':      cmd_list_pipelines,
        'explain':   cmd_explain,
        'demo':      cmd_demo,
    }

    if args.command in dispatch:
        sys.exit(dispatch[args.command](args))
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == '__main__':
    main()
