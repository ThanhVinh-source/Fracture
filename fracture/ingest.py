"""
fracture/ingest.py

Fracture's input contract with the outside world.

Design principle: teams produce the 4-column format.
Fracture validates it and moves on. Fracture is not an ETL tool.
Extraction, transformation, and delivery of log data to this
format is the responsibility of the team that owns the pipeline.

WHY THIS BOUNDARY EXISTS
═════════════════════════
A bank's Splunk instance has security requirements, network
restrictions, and authentication patterns that Fracture should
never need to know about. Building connectors inside Fracture
creates coupling that does not belong here.

The team that owns the pipeline owns the extraction.
Fracture owns the measurement.

THE ONLY FORMAT FRACTURE ACCEPTS
══════════════════════════════════

Four columns. Parquet or CSV. Always.

  pipeline_run_id  |  activity  |  timestamp  |  team
  ─────────────────────────────────────────────────────
  trade_batch_001  │  SCHEDULED │  2026-05-01T05:58:00Z  │  producer
  trade_batch_001  │  STARTED   │  2026-05-01T06:00:12Z  │  producer
  trade_batch_001  │  COMPLETED │  2026-05-01T06:44:51Z  │  producer
  trade_batch_001  │  DATA_AVAIL│  2026-05-01T06:47:02Z  │  producer
  trade_batch_001  │  SCHEDULED │  2026-05-01T05:58:00Z  │  consumer
  trade_batch_001  │  STARTED   │  2026-05-01T06:00:14Z  │  consumer
  trade_batch_001  │  COMPLETED │  2026-05-01T06:44:51Z  │  consumer
  trade_batch_001  │  DATA_AVAIL│  2026-05-01T07:11:44Z  │  consumer
                                                           ↑
                                             25-minute bilateral gap
                                             visible immediately

COLUMN DEFINITIONS
══════════════════
pipeline_run_id
  One unique ID per process execution instance.
  All events from one batch run share this ID.
  Think: one invoice, one trade, one batch, one job.
  This becomes case:concept:name in XES (PM4PY standard).
  Example: "trade_batch_20260501_run_001"

activity
  The business milestone that fired.
  Must match the required_events declared in your contract YAML.
  This is a process state change — not a technical operation.
  RIGHT:  "COMPLETED"           (what the process did)
  WRONG:  "write_to_table()"    (how the system did it)
  This becomes concept:name in XES.

timestamp
  When the event fired.
  UTC strongly recommended — mixed timezones cause Guard 4 to fire.
  ISO 8601 format: 2026-05-01T06:44:51Z
  Fracture guards:
    Guard 3 rejects all-same timestamps (log aggregation problem)
    Guard 4 rejects future timestamps (timezone misconfiguration)

team
  Which side of the contract emitted this event.
  Exactly "producer" or "consumer" — no other values accepted.
  This single field enables bilateral comparison.
  Same pipeline_run_id, different team values, different
  DATA_AVAILABLE timestamps = the bilateral gap.

HOW TO PRODUCE THIS FORMAT
═══════════════════════════

From Splunk (SPL):
  index=pipeline_logs pipeline_id=trade_batch
  | rename run_id AS pipeline_run_id,
           status AS activity,
           _time AS timestamp
  | eval team="producer"
  | table pipeline_run_id, activity, timestamp, team
  Export → CSV → drop in inputs/trade_batch/producer_YYYYMMDD.csv

From Elasticsearch (Python):
  results = es.search(index="pipeline-logs", query={...})
  df = pd.DataFrame([{
      "pipeline_run_id": h["_source"]["run_id"],
      "activity":        h["_source"]["event_type"],
      "timestamp":       h["_source"]["@timestamp"],
      "team":            "producer",
  } for h in results["hits"]["hits"]])
  df.to_parquet("inputs/trade_batch/producer_20260501.parquet")

From any flat file (manual):
  python fracture.py template --pipeline-id trade_batch
  # Fill in the printed CSV template and save to inputs/

ACTIVITY NAME NORMALISATION
════════════════════════════
If your logs use different activity names than your contract,
declare the mapping in your contract YAML — not in code:

  log_contract:
    activity_name_map:
      "pipeline_started":  "STARTED"
      "job_complete":      "COMPLETED"
      "data_ready":        "DATA_AVAILABLE"
      "pipeline_queued":   "SCHEDULED"

Fracture's adapters apply this map during extraction.
The normalised names must match required_events in the contract.

WHERE TO PUT THE FILE
══════════════════════
inputs/
└── trade_batch_grouping/
    ├── producer_20260501.parquet   ← producer events
    └── consumer_20260501.parquet  ← consumer events (optional)

File naming: {team}_{YYYYMMDD}.parquet or {team}_{YYYYMMDD}.csv
Fracture picks up today's file automatically.
Historical files are not reprocessed unless .processed marker deleted.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd


# ── The contract ──────────────────────────────────────────────────────────────

FRACTURE_COLUMNS = [
    "pipeline_run_id",   # case:concept:name in XES
    "activity",          # concept:name in XES
    "timestamp",         # time:timestamp in XES
    "team",              # "producer" or "consumer"
]

VALID_TEAMS = {"producer", "consumer"}


# ── Validation ────────────────────────────────────────────────────────────────

def validate_format(
    df:          pd.DataFrame,
    source_name: str = "",
) -> pd.DataFrame:
    """
    Validate that a DataFrame matches Fracture's universal format.

    Called once when a file is loaded from the inputs directory.
    If the format is wrong Fracture fails loud here with a clear
    message rather than silently passing bad data to conformance.

    Teams should run this manually when first setting up their
    extraction to verify their output is correct before scheduling.

    Returns the validated DataFrame with exactly the four required
    columns in the correct order. Extra columns are stripped.

    Raises ValueError with a human-readable message telling the
    team exactly what to fix.
    """
    source = f" from {source_name}" if source_name else ""

    # ── Rule 1: required columns present ─────────────────────────
    missing = set(FRACTURE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"Fracture input{source} is missing required columns: {missing}\n"
            f"\n"
            f"Required: {FRACTURE_COLUMNS}\n"
            f"Found:    {list(df.columns)}\n"
            f"\n"
            f"Fix: rename your columns or add the missing ones.\n"
            f"Tip: use activity_name_map in your contract to normalise names."
        )

    # ── Rule 2: timestamp parseable ───────────────────────────────
    # Fail here rather than letting Guard 3/4 fire with a less
    # informative error message during conformance checking
    try:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    except Exception as e:
        raise ValueError(
            f"Fracture input{source}: timestamp column cannot be parsed.\n"
            f"Error: {e}\n"
            f"\n"
            f"Expected format: ISO 8601 with timezone\n"
            f"  Correct:   2026-05-01T06:44:51Z\n"
            f"  Correct:   2026-05-01 06:44:51+00:00\n"
            f"  Incorrect: 2026-05-01 06:44:51  (no timezone)\n"
            f"\n"
            f"Tip: ensure your extraction adds timezone info.\n"
            f"  Splunk: | eval timestamp=strftime(_time, '%Y-%m-%dT%H:%M:%SZ')\n"
            f"  ES:     use @timestamp which is already UTC"
        )

    # ── Rule 3: team values valid ─────────────────────────────────
    invalid_teams = set(df["team"].unique()) - VALID_TEAMS
    if invalid_teams:
        raise ValueError(
            f"Fracture input{source}: team column has invalid values: {invalid_teams}\n"
            f"\n"
            f"Allowed values: 'producer', 'consumer' (exact, case-sensitive)\n"
            f"\n"
            f"Fix: add a team column to your extraction with these exact values.\n"
            f"  Splunk: | eval team=if(source_system='risk_engine', 'producer', 'consumer')\n"
            f"  Python: df['team'] = 'producer'"
        )

    # ── Rule 4: no nulls in key columns ──────────────────────────
    for col in ["pipeline_run_id", "activity"]:
        null_count = df[col].isna().sum()
        if null_count > 0:
            raise ValueError(
                f"Fracture input{source}: {null_count} null values in '{col}'.\n"
                f"Every event must have a {col}.\n"
                f"\n"
                f"Fix: check your extraction query for missing {col} values.\n"
                f"  Splunk: | where isnotnull({col})\n"
                f"  Python: df = df.dropna(subset=['{col}'])"
            )

    # ── Rule 5: at least one event ────────────────────────────────
    if len(df) == 0:
        raise ValueError(
            f"Fracture input{source}: DataFrame is empty.\n"
            f"No events found. Check your extraction query and time range."
        )

    # Strip extra columns — return exactly the four required
    # This ensures downstream code always receives the expected schema
    return df[FRACTURE_COLUMNS].copy()

def normalize_events(df: pd.DataFrame, contract) -> pd.DataFrame:
    """
    Normalize validated Fracture events before preflight and token replay.

    This keeps raw extraction flexible while making the conformance engine
    operate on contract vocabulary.
    """
    if df is None:
        return None

    normalized = df.copy()

    # Teams can emit system-specific names, but token replay requires activity
    # labels to match the contract's required_events exactly.
    activity_map = contract.log_contract.activity_name_map or {}
    if activity_map:
        normalized["activity"] = normalized["activity"].replace(activity_map)

    # Retries are operational noise when the contract cares about milestones.
    # Keep the latest timestamp per run/activity so repeated Airflow task events
    # do not look like extra process steps during token replay.
    if contract.log_contract.deduplicate_retries:
        # First group retries by milestone and keep the latest event, then
        # restore timestamp order so PM4PY sees the real trace sequence.
        normalized = (
            normalized
            .sort_values(["pipeline_run_id", "activity", "timestamp"])
            .drop_duplicates(["pipeline_run_id", "activity"], keep="last")
            .sort_values(["pipeline_run_id", "timestamp"])
        )
    return normalized[FRACTURE_COLUMNS].copy()


# ── File loader ───────────────────────────────────────────────────────────────

def load_pipeline_events(
    pipeline_id: str,
    inputs_dir:  str = "inputs",
    date_str:    Optional[str] = None,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Load validated producer and consumer event files for a pipeline.

    This is what FractureEngine calls. It reads whatever file the
    team's extraction job wrote to the inputs directory.
    Fracture never reaches into log systems directly.

    Supports both parquet and CSV input files.
    Validates format before returning — fails loud if wrong.

    Returns
    -------
    (producer_df, consumer_df)
    consumer_df is None if no consumer file exists for this date.
    Bilateral comparison is skipped when consumer_df is None.
    """
    pipeline_dir = Path(inputs_dir) / pipeline_id
    today = date_str or date.today().strftime("%Y%m%d")

    if not pipeline_dir.exists():
        raise FileNotFoundError(
            f"No input directory for pipeline '{pipeline_id}'.\n"
            f"Expected: {pipeline_dir}\n"
            f"\n"
            f"Has your extraction job run today?\n"
            f"Create the directory and drop your event file there:\n"
            f"  inputs/{pipeline_id}/producer_{today}.parquet"
        )

    producer_df  = _load_team_file(pipeline_dir, "producer", today, pipeline_id)
    consumer_df  = _load_team_file(pipeline_dir, "consumer", today, pipeline_id,
                                   required=False)

    msg = f"[ingest] {pipeline_id}: {len(producer_df)} producer events"
    if consumer_df is not None:
        msg += f", {len(consumer_df)} consumer events"
    else:
        msg += " (no consumer file — unilateral mode)"
    print(msg)

    return producer_df, consumer_df


def _load_team_file(
    pipeline_dir: Path,
    team:         str,
    date_str:     str,
    pipeline_id:  str,
    required:     bool = True,
) -> Optional[pd.DataFrame]:
    """
    Load and validate one team's event file.

    Tries parquet first (preferred — faster, typed).
    Falls back to CSV if parquet not found.
    Falls back to historical.parquet for sudden_collapse archetype.
    """
    # Try parquet first
    parquet_path = pipeline_dir / f"{team}_{date_str}.parquet"
    csv_path     = pipeline_dir / f"{team}_{date_str}.csv"
    historical   = pipeline_dir / "historical.parquet"

    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)

    elif csv_path.exists():
        # CSV fallback — teams may export as CSV before switching to parquet
        df = _read_csv_auto(csv_path)

    elif team == "producer" and historical.exists():
        # Historical split for sudden_collapse archetype
        print(f"[ingest] Loading historical events for {pipeline_id}")
        df = pd.read_parquet(historical)

    elif required:
        raise FileNotFoundError(
            f"No {team} events found for '{pipeline_id}' on {date_str}.\n"
            f"\n"
            f"Looked for:\n"
            f"  {parquet_path}\n"
            f"  {csv_path}\n"
            f"\n"
            f"Ensure your extraction job ran today and wrote the file.\n"
            f"Format: {FRACTURE_COLUMNS}"
        )
    else:
        return None

    # Validate — fail loud before anything else runs
    return validate_format(df, source_name=f"{pipeline_id}/{team}")


def _read_csv_auto(path: Path) -> pd.DataFrame:
    """
    Read a CSV file auto-detecting the separator.
    Tries comma, semicolon (European), tab, pipe in order.
    """
    for sep in [",", ";", "\t", "|"]:
        try:
            df = pd.read_csv(path, sep=sep)
            if len(df.columns) >= 4:
                return df
        except Exception:
            continue

    raise ValueError(
        f"Cannot parse {path.name}. "
        f"Tried separators: comma, semicolon, tab, pipe. "
        f"Ensure the file has at least 4 columns."
    )


# ── CSV template generator ────────────────────────────────────────────────────

def print_csv_template(
    pipeline_id: str = "your_pipeline",
    n_days:      int  = 3,
) -> None:
    """
    Print a CSV template a team can fill in manually.

    Zero-infrastructure onboarding path.
    Team fills this in, drops it in inputs/{pipeline_id}/,
    and Fracture processes it.

    The template includes a deliberate 25-minute bilateral gap
    in the DATA_AVAILABLE timestamps — this shows teams exactly
    what bilateral comparison means before they have real data.

    Usage:
        python fracture.py template --pipeline-id trade_batch
        # or:
        from fracture.ingest import print_csv_template
        print_csv_template("trade_batch", n_days=5)
    """
    from datetime import datetime, timedelta

    print(f"# Fracture CSV template for: {pipeline_id}")
    print(f"# Save as: inputs/{pipeline_id}/producer_YYYYMMDD.csv")
    print(f"#          inputs/{pipeline_id}/consumer_YYYYMMDD.csv")
    print(f"#")
    print(f"# Rules:")
    print(f"#   pipeline_run_id — unique ID per batch/process run")
    print(f"#   activity        — must match required_events in your contract")
    print(f"#   timestamp       — ISO 8601, UTC (2026-05-01T06:00:00Z)")
    print(f"#   team            — 'producer' or 'consumer' exactly")
    print(f"#")
    print(f"# NOTE: consumer DATA_AVAILABLE is 25 min later than producer")
    print(f"# This is a bilateral gap — the core thing Fracture measures")
    print()
    print("pipeline_run_id,activity,timestamp,team")

    base = datetime.now().replace(
        hour=6, minute=0, second=0, microsecond=0
    )

    for i in range(n_days):
        run_id   = f"{pipeline_id}_{(base + timedelta(days=i)).strftime('%Y%m%d')}"
        run_base = base + timedelta(days=i)

        # Producer events — process completes at T+45, data available T+48
        for activity, delta in [
            ("SCHEDULED",    -2),
            ("STARTED",       0),
            ("COMPLETED",    45),
            ("DATA_AVAILABLE",48),
        ]:
            ts = (run_base + timedelta(minutes=delta)).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
            print(f"{run_id},{activity},{ts},producer")

        # Consumer events — DATA_AVAILABLE 25 min later (bilateral gap)
        for activity, delta in [
            ("SCHEDULED",    -2),
            ("STARTED",       0),
            ("COMPLETED",    45),
            ("DATA_AVAILABLE",73),  # ← 25 min gap shown explicitly
        ]:
            ts = (run_base + timedelta(minutes=delta)).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
            print(f"{run_id},{activity},{ts},consumer")
