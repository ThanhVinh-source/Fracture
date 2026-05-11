"""
fracture/bootstrap.py

Computes real SLA percentiles from actual pipeline execution history.

This is the bridge between raw event logs and a valid contract.
Teams provide 30 days of 4-column event files.
Bootstrap reads them, computes p50/p95/p99 from real execution durations,
and writes a contract that reflects how the pipeline actually behaves.

Why this matters:
  A contract written by guessing p99=90 is wrong if the pipeline
  actually runs at p99=134. Token replay against a wrong contract
  produces misleading fitness scores. Bootstrap ensures the contract
  reflects reality before conformance measurement begins.

Design:
  Input:  inputs/{pipeline_id}/producer_*.parquet (or .csv)
  Output: contracts/{pipeline_id}.yaml (status: active)

  Teams never guess percentiles. Bootstrap computes them.
  Humans review and activate. Fracture measures.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml


# ── Bootstrap result ──────────────────────────────────────────────────────────

@dataclass
class BootstrapResult:
    """
    Result of bootstrapping a contract from event history.

    Contains the computed percentiles and diagnostic information
    so teams can review before activating the contract.
    """
    pipeline_id:      str
    n_runs:           int         # complete runs found
    n_files:          int         # event files read
    date_range:       tuple       # (earliest_date, latest_date)

    # Computed percentiles (minutes)
    p50_minutes:      int
    p95_minutes:      int
    p99_minutes:      int
    grace_minutes:    int

    # Distribution insight
    min_minutes:      float
    max_minutes:      float
    mean_minutes:     float
    std_minutes:      float

    # Quality signals
    coverage_pct:     float   # fraction of scheduled runs that completed
    has_outliers:     bool    # any runs > 3 std devs from mean
    warning:          Optional[str] = None

    def summary(self) -> str:
        return (
            f"p50={self.p50_minutes}m  p95={self.p95_minutes}m  "
            f"p99={self.p99_minutes}m  grace={self.grace_minutes}m  "
            f"(from {self.n_runs} runs over {self.n_files} files)"
        )

    def is_reliable(self) -> bool:
        """True if we have enough data to trust the percentiles."""
        return self.n_runs >= 14  # at least 2 weeks of history


# ── Core computation ──────────────────────────────────────────────────────────

def compute_percentiles_from_events(
    events:        pd.DataFrame,
    terminal_event: str = 'COMPLETED',
    start_event:   str = 'STARTED',
) -> dict:
    """
    Compute execution duration percentiles from a 4-column event DataFrame.

    Duration = time from start_event to terminal_event per run.
    Only complete runs (both events present) are included.

    Returns dict with p50, p95, p99, and diagnostic fields.
    """
    events = events.copy()
    events['timestamp'] = pd.to_datetime(events['timestamp'], utc=True)

    # Focus on producer events only for duration computation
    prod = events[events['team'] == 'producer'].copy()

    durations     = []
    complete_runs = 0
    total_runs    = prod['pipeline_run_id'].nunique()

    for run_id, group in prod.groupby('pipeline_run_id'):
        acts = set(group['activity'].tolist())

        if start_event in acts and terminal_event in acts:
            t_start = group[group['activity'] == start_event]['timestamp'].min()
            t_end   = group[group['activity'] == terminal_event]['timestamp'].max()
            dur     = (t_end - t_start).total_seconds() / 60

            if dur > 0:
                durations.append(dur)
                complete_runs += 1

    if len(durations) < 5:
        return {
            'error': f"Only {len(durations)} complete runs found. Need at least 5.",
            'durations': durations,
        }

    durations_arr = np.array(durations)

    p50 = max(1,  int(np.percentile(durations_arr, 50)))
    p95 = max(p50 + 1, int(np.percentile(durations_arr, 95)))
    p99 = max(p95 + 1, int(np.percentile(durations_arr, 99)))

    # Detect outliers (> 3 std devs)
    mean = durations_arr.mean()
    std  = durations_arr.std()
    has_outliers = bool(any(d > mean + 3 * std for d in durations))

    warning = None
    if len(durations) < 14:
        warning = (
            f"Only {len(durations)} complete runs. "
            f"Recommend 14+ for reliable percentiles. "
            f"Run bootstrap again after more history accumulates."
        )
    if has_outliers:
        outlier_count = sum(1 for d in durations if d > mean + 3 * std)
        warning = (
            (warning or "") +
            f" {outlier_count} outlier run(s) detected (>{mean + 3*std:.0f}m). "
            f"Check for maintenance windows or infrastructure events."
        )

    return {
        'p50_minutes':    p50,
        'p95_minutes':    p95,
        'p99_minutes':    p99,
        'n_complete':     complete_runs,
        'total_runs':     total_runs,
        'coverage_pct':   complete_runs / max(total_runs, 1),
        'min_minutes':    round(float(durations_arr.min()), 1),
        'max_minutes':    round(float(durations_arr.max()), 1),
        'mean_minutes':   round(float(mean), 1),
        'std_minutes':    round(float(std), 1),
        'has_outliers':   has_outliers,
        'warning':        warning,
        'durations':      durations,
    }


def bootstrap_contract(
    pipeline_id:    str,
    inputs_dir:     str = 'inputs',
    contracts_dir:  str = 'contracts',
    hard_deadline:  Optional[str] = None,
) -> BootstrapResult:
    """
    Read event files, compute percentiles, update contract.

    If a contract already exists in contracts_dir:
        Updates p50/p95/p99/grace, sets status: active.

    If no contract exists:
        Raises ValueError — register the pipeline first.

    Args:
        pipeline_id:   Which pipeline to bootstrap.
        inputs_dir:    Where event parquet/csv files live.
        contracts_dir: Where contract YAML lives.
        hard_deadline: Optional override for expected_end time (HH:MM).
                       If not provided, uses the contract's expected_end.

    Returns:
        BootstrapResult with computed values and diagnostic info.

    Raises:
        FileNotFoundError: No event files found.
        ValueError:        Contract not found — register first.
        ValueError:        Not enough complete runs.
    """
    contracts_dir_p = Path(contracts_dir)
    inputs_dir_p    = Path(inputs_dir)
    contract_path   = contracts_dir_p / f"{pipeline_id}.yaml"

    # Contract must exist (registered first)
    if not contract_path.exists():
        raise ValueError(
            f"Contract not found: {contract_path}\n"
            f"Register the pipeline first: "
            f"fracture register --name '{pipeline_id}' ..."
        )

    # Find event files
    pipeline_dir = inputs_dir_p / pipeline_id
    if not pipeline_dir.exists():
        raise FileNotFoundError(
            f"No input directory: {pipeline_dir}\n"
            f"Drop event files here: "
            f"{pipeline_dir}/producer_YYYYMMDD.parquet"
        )

    parquet_files = sorted(pipeline_dir.glob("producer_*.parquet"))
    csv_files     = sorted(pipeline_dir.glob("producer_*.csv"))
    all_files     = list(parquet_files) + list(csv_files)

    if not all_files:
        raise FileNotFoundError(
            f"No producer event files in {pipeline_dir}\n"
            f"Expected: producer_YYYYMMDD.parquet or producer_YYYYMMDD.csv"
        )

    # Load all event files
    dfs = []
    for f in all_files:
        try:
            df = pd.read_parquet(f) if f.suffix == '.parquet' else pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            warnings.warn(f"Could not read {f.name}: {e}")

    if not dfs:
        raise FileNotFoundError("Could not read any event files.")

    events = pd.concat(dfs, ignore_index=True)

    # Get terminal event from existing contract
    raw = yaml.safe_load(contract_path.read_text())
    terminal = (
        raw.get('log_contract', {})
           .get('terminal_event', 'COMPLETED')
    )

    # Compute percentiles
    # Detect start_event from contract required_events
    # First event before terminal that is not terminal itself
    required = raw.get('log_contract', {}).get('required_events', [])
    if required:
        # Start event = first event in required_events
        start_event = required[0]
    else:
        start_event = 'STARTED'

    result = compute_percentiles_from_events(
        events         = events,
        terminal_event = terminal,
        start_event    = start_event,
    )

    if 'error' in result:
        raise ValueError(result['error'])

    # Compute grace from window
    expected_start = raw.get('expected_start', '06:00')
    expected_end   = hard_deadline or raw.get('expected_end', '08:30')

    def _window(s, e):
        sh, sm = map(int, s.split(':'))
        eh, em = map(int, e.split(':'))
        start  = sh * 60 + sm
        end    = eh * 60 + em
        if end <= start:
            end += 24 * 60
        return end - start

    window    = _window(expected_start, expected_end)
    p99       = result['p99_minutes']
    grace     = max(5, min(30, window - p99 - 5))

    # Update contract YAML — percentiles only
    # Status is NOT changed here — that is the CLI's responsibility.
    # bootstrap computes the values. Human (or CLI) decides to activate.
    raw['p50_minutes']   = result['p50_minutes']
    raw['p95_minutes']   = result['p95_minutes']
    raw['p99_minutes']   = p99
    raw['grace_minutes'] = grace
    # status left unchanged — still DRAFT after bootstrap
    if hard_deadline:
        raw['expected_end'] = hard_deadline

    contract_path.write_text(
        yaml.dump(raw, default_flow_style=False, allow_unicode=True)
    )

    # Date range from events
    events['timestamp'] = pd.to_datetime(events['timestamp'], utc=True, errors='coerce')
    ts = events['timestamp'].dropna()
    date_range = (
        ts.min().date().isoformat() if len(ts) > 0 else '?',
        ts.max().date().isoformat() if len(ts) > 0 else '?',
    )

    return BootstrapResult(
        pipeline_id   = pipeline_id,
        n_runs        = result['n_complete'],
        n_files       = len(all_files),
        date_range    = date_range,
        p50_minutes   = result['p50_minutes'],
        p95_minutes   = result['p95_minutes'],
        p99_minutes   = p99,
        grace_minutes = grace,
        min_minutes   = result['min_minutes'],
        max_minutes   = result['max_minutes'],
        mean_minutes  = result['mean_minutes'],
        std_minutes   = result['std_minutes'],
        coverage_pct  = result['coverage_pct'],
        has_outliers  = result['has_outliers'],
        warning       = result['warning'],
    )


# ── Convenience ───────────────────────────────────────────────────────────────

def bootstrap_all(
    inputs_dir:    str = 'inputs',
    contracts_dir: str = 'contracts',
) -> list[BootstrapResult]:
    """
    Bootstrap all pipelines that have event files but are still DRAFT.

    Useful for bulk onboarding when all teams have submitted their files.
    """
    contracts_dir_p = Path(contracts_dir)
    inputs_dir_p    = Path(inputs_dir)

    results = []
    for contract_path in sorted(contracts_dir_p.glob('*.yaml')):
        pid = contract_path.stem
        if pid.startswith('_'):
            continue

        raw    = yaml.safe_load(contract_path.read_text())
        status = raw.get('status', 'draft')

        if status != 'draft':
            continue  # already active

        pipeline_dir = inputs_dir_p / pid
        if not pipeline_dir.exists():
            continue  # no events yet

        try:
            result = bootstrap_contract(pid, inputs_dir, contracts_dir)
            results.append(result)
        except Exception as e:
            warnings.warn(f"Bootstrap failed for {pid}: {e}")

    return results
