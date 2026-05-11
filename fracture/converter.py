"""
fracture/converter.py

Converts Fracture's internal DataFrame format to PM4PY EventLog objects.

This is the bridge between the data layer and the process mining layer.
Two different worlds with two different vocabularies:

  Fracture world:
    pipeline_run_id | activity | timestamp | team | pipeline_id
    One row per event. Pandas DataFrame. Parquet on disk.

  PM4PY world:
    EventLog → [Trace] → [Event]
    case:concept:name  = pipeline_run_id  (which run this event belongs to)
    concept:name       = activity         (what happened)
    time:timestamp     = timestamp        (when it happened)
    org:resource       = team             (producer or consumer)

XES (eXtensible Event Stream) is the ISO standard event log format
for process mining. PM4PY reads and writes XES natively. The attribute
names above (concept:name, time:timestamp, org:resource) are the
XES standard attribute names defined in IEEE 1849-2016.

Design principle: this module only converts. It does not validate,
filter, or modify events. The preflight checker does validation.
The extractor does normalisation. This module does one thing:
translate between two data structures.

Two conversion paths:
  1. In-memory: DataFrame → PM4PY EventLog (used by conformance engine)
  2. File-based: DataFrame → .xes file on disk (used by notebooks)

The file-based path is important for the course project — you can
load XES files directly in ProM, Disco, or Celonis for visual
inspection and comparison with Fracture's results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.objects.conversion.log import converter as log_converter
import pm4py

from fracture.schema import PipelineContract


# XES standard attribute names
# These are not arbitrary — they are the IEEE 1849-2016 standard.
# PM4PY's token replay expects exactly these names.
XES_CASE_ID    = "case:concept:name"    # which case (pipeline run)
XES_ACTIVITY   = "concept:name"         # what activity happened
XES_TIMESTAMP  = "time:timestamp"       # when it happened
XES_RESOURCE   = "org:resource"         # who/what performed it
XES_PIPELINE   = "concept:instance"     # pipeline identifier


def dataframe_to_eventlog(
    df:       pd.DataFrame,
    contract: PipelineContract,
    team:     str = "producer",
) -> EventLog:
    """
    Convert a Fracture event DataFrame to a PM4PY EventLog.

    This is the primary conversion used by the conformance engine.
    In-memory only — no files written.

    Parameters
    ----------
    df       : DataFrame with columns pipeline_run_id, activity,
               timestamp, team, pipeline_id.
               Must already be normalised — activity names should
               already match the contract's required_events or
               have been mapped via activity_name_map.
    contract : The pipeline contract. Used to attach contract
               metadata to the log for traceability.
    team     : "producer" or "consumer". Used to label the log
               so the conformance engine knows which side it is
               reading. This is how bilateral comparison works —
               same contract, two separately labelled EventLogs.

    Returns
    -------
    PM4PY EventLog with one Trace per pipeline_run_id.
    Events within each trace are sorted by timestamp.

    PM4PY token replay requires:
      1. Events grouped into traces by case ID
      2. Events within each trace sorted by timestamp
      3. Activity names matching the Petri net transition labels
    All three are handled here.
    """
    if df is None or len(df) == 0:
        # Return empty EventLog — the preflight checker will have
        # already flagged this as a RED check before we get here.
        # This defensive return prevents crashes in edge cases.
        return EventLog()

    log = EventLog()

    # Attach log-level attributes for traceability
    # These appear in XES file headers and in PM4PY diagnostics
    log.attributes["concept:name"]      = contract.pipeline_id
    log.attributes["fracture:team"]     = team
    log.attributes["fracture:contract"] = contract.pipeline_id
    log.attributes["fracture:version"]  = "1.0.0"

    # Group events by pipeline run ID — each run becomes one trace
    # Sort groups by the first event timestamp to get chronological traces
    # This ordering matters for the token replay algorithm
    for run_id, run_events in df.groupby("pipeline_run_id"):

        trace = Trace()

        # Trace-level attributes
        trace.attributes[XES_CASE_ID]         = str(run_id)
        trace.attributes["fracture:team"]      = team
        trace.attributes["fracture:pipeline"]  = contract.pipeline_id

        # Sort events by timestamp within this trace
        # Critical: token replay assumes chronological order within a trace.
        # Out-of-order events within a trace cause incorrect fitness scores.
        # The preflight checker warns about ordering violations but does not
        # reorder. We reorder here to give token replay the best possible
        # input while the confidence score reflects the ordering issues.
        sorted_events = run_events.sort_values("timestamp")

        for _, row in sorted_events.iterrows():
            event = Event()

            # XES standard attributes — PM4PY looks for these exact names
            event[XES_ACTIVITY]  = str(row["activity"])
            event[XES_TIMESTAMP] = row["timestamp"]
            event[XES_RESOURCE]  = str(row.get("team", team))

            # Additional Fracture attributes preserved as XES extensions
            # These are visible in ProM and Disco for debugging
            event["fracture:pipeline_id"]  = str(row.get("pipeline_id", ""))
            event["fracture:run_id"]        = str(run_id)

            trace.append(event)

        log.append(trace)

    return log


def dataframe_to_xes_file(
    df:         pd.DataFrame,
    contract:   PipelineContract,
    output_dir: str,
    team:       str = "producer",
    date_str:   Optional[str] = None,
) -> Path:
    """
    Convert DataFrame to XES file and write to disk.

    Used by notebooks for visual inspection in ProM, Disco, Celonis.
    Also useful for the course project submission — the XES files
    are the evidence that Fracture produced valid process mining input.

    File naming: {pipeline_id}_{team}_{date}.xes
    Example:     trade_batch_grouping_producer_20260515.xes

    Parameters
    ----------
    df         : same as dataframe_to_eventlog
    contract   : same as dataframe_to_eventlog
    output_dir : directory to write XES file
    team       : "producer" or "consumer"
    date_str   : YYYYMMDD string for filename, defaults to today

    Returns
    -------
    Path to the written XES file.
    """
    from datetime import date

    if date_str is None:
        date_str = date.today().strftime("%Y%m%d")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{contract.pipeline_id}_{team}_{date_str}.xes"
    filepath = out_dir / filename

    log = dataframe_to_eventlog(df, contract, team)

    # PM4PY's XES exporter — writes IEEE 1849-2016 compliant XES
    pm4py.write_xes(log, str(filepath))

    return filepath


def eventlog_to_dataframe(
    log:      EventLog,
    team:     str = "producer",
) -> pd.DataFrame:
    """
    Convert a PM4PY EventLog back to a Fracture DataFrame.

    Inverse operation — used when loading XES files from disk
    for replay, or when comparing discovered models across runs.

    Useful in notebooks: load an XES, convert back to DataFrame,
    run analysis, compare with the original generator output.
    """
    rows = []
    for trace in log:
        run_id = trace.attributes.get(XES_CASE_ID, "unknown")
        for event in trace:
            rows.append({
                "pipeline_run_id": run_id,
                "activity":        event.get(XES_ACTIVITY, ""),
                "timestamp":       event.get(XES_TIMESTAMP),
                "team":            event.get(XES_RESOURCE, team),
                "pipeline_id":     event.get("fracture:pipeline_id", ""),
            })
    return pd.DataFrame(rows)


def load_xes_file(filepath: str) -> EventLog:
    """
    Load an XES file from disk into a PM4PY EventLog.

    Convenience wrapper around pm4py.read_xes.
    Used in notebooks and in the file-based watcher path.
    """
    return pm4py.read_xes(str(filepath))


def split_by_team(
    df:       pd.DataFrame,
    contract: PipelineContract,
) -> tuple[EventLog, Optional[EventLog]]:
    """
    Split a combined event DataFrame into producer and consumer
    EventLogs, converting each to PM4PY format.

    This is the core setup for bilateral conformance checking.
    The same contract-derived Petri net will be replayed against
    both EventLogs independently. The difference in fitness scores
    between the two sides is the bilateral assumption gap signal.

    Parameters
    ----------
    df       : DataFrame that may contain events from both teams.
               Filtered by the "team" column.
    contract : Used to label the logs and attach metadata.

    Returns
    -------
    (producer_log, consumer_log)
    consumer_log is None if no consumer events exist in df.
    The conformance engine handles the None case — it skips
    bilateral comparison when consumer data is unavailable.
    """
    producer_df = df[df["team"] == "producer"].copy()
    consumer_df = df[df["team"] == "consumer"].copy()

    producer_log = dataframe_to_eventlog(producer_df, contract, "producer")

    consumer_log = None
    if len(consumer_df) > 0:
        consumer_log = dataframe_to_eventlog(consumer_df, contract, "consumer")

    return producer_log, consumer_log


def get_log_statistics(log: EventLog) -> dict:
    """
    Compute basic statistics about an EventLog.

    Useful for debugging and for the preflight checker.
    Returns counts and activity distribution without
    running any process mining — pure descriptive stats.

    Used in notebooks: call this after conversion to verify
    the EventLog has the expected structure before running
    the expensive token replay.
    """
    if not log:
        return {
            "n_traces":     0,
            "n_events":     0,
            "activities":   [],
            "min_trace_len": 0,
            "max_trace_len": 0,
            "avg_trace_len": 0.0,
        }

    trace_lengths = [len(t) for t in log]
    activities    = set()
    for trace in log:
        for event in trace:
            activities.add(event.get(XES_ACTIVITY, ""))

    return {
        "n_traces":      len(log),
        "n_events":      sum(trace_lengths),
        "activities":    sorted(list(activities)),
        "min_trace_len": min(trace_lengths) if trace_lengths else 0,
        "max_trace_len": max(trace_lengths) if trace_lengths else 0,
        "avg_trace_len": round(
            sum(trace_lengths) / len(trace_lengths), 2
        ) if trace_lengths else 0.0,
    }
