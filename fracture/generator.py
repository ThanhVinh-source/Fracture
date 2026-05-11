"""
fracture/generator.py

Synthetic event log generator for Fracture.

Generates a realistic population of 100 pipelines across a three-tier
test taxonomy designed to validate every component of the conformance engine:

  TIER 1 — Normal cases (60 pipelines)
    Stable mature, stable new, slow drifting established.
    These validate that healthy pipelines score correctly
    and that the warmup multiplier reduces confidence
    appropriately for new pipelines.

  TIER 2 — Deliberate failure cases (25 pipelines)
    Fast drifting new, assumption asymmetry, silent, sudden collapse.
    These validate specific failure mode detection.
    The fast drifting new pipelines are the H0 test group:
    H0: pipeline age has no effect on drift velocity
    H1: newer pipelines show significantly higher drift velocity

  TIER 3 — Outlier cases (15 pipelines)
    Self-correcting drift, bimodal, clock sync, reverse drift,
    month-end seasonal, ghost pipeline.
    These test edge cases and unseen situations that validate
    the system's robustness beyond the normal operating envelope.

Design principle: lightweight.
  No database. No infrastructure. No streaming.
  Every archetype writes parquet files to outputs/logs/.
  The sudden_collapse archetype uses historical + live split —
  realistic because Fracture is always deployed on a running system
  that already has history. The historical runs are pre-written once
  as historical.parquet. Live runs are generated fresh per session.

H0 test design:
  pipeline_age_days is stored as contract metadata.
  After generating all pipelines, run Mann-Whitney U test on
  drift_rate_per_week between new group (age < 45) and
  mature group (age > 60). Expected: reject H0 at p < 0.05.
  This is a legitimate statistical hypothesis for the paper.
"""

from __future__ import annotations

import random
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date, time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from fracture.schema import PipelineContract
from fracture.config import FractureConfig, DEFAULT_CONFIG


# ── Event schema ──────────────────────────────────────────────────────────────
# Every generated event has exactly these fields.
# The XES converter maps these to PM4PY's expected attribute names.

REQUIRED_COLUMNS = [
    "pipeline_run_id",   # case:concept:name in XES
    "activity",          # concept:name in XES
    "timestamp",         # time:timestamp in XES
    "team",              # which side: producer or consumer
    "pipeline_id",       # links back to the contract
]


# ── Ground truth label ────────────────────────────────────────────────────────
# Every generated pipeline has expected outputs attached.
# The conformance engine and clustering should produce results
# that match these expectations. The test suite asserts against them.

@dataclass
class GroundTruth:
    """
    Ground truth labels for clustering validation.
    Used for ARI scoring — not passed to the conformance engine.
    The clustering algorithm should recover these from features alone.

    6 fields. Minimum needed for H0 test and ARI validation.
    """
    archetype:               str    # generating archetype name
    tier:                    int    # 1=normal 2=failure 3=outlier

    expected_cluster:        str    # HEALTHY | DRIFTING | CRITICAL
    expected_pattern:        str    # STABLE | DRIFTING | INTERMITTENT

    # H0 test — pipeline age has no effect on drift velocity
    pipeline_age_days:       int    # age when Fracture deployed
    drift_velocity_per_week: float  # 0.0 = stable, negative = declining

    # Guard flags for Tier 3 outliers
    should_raise_guard:      bool = False
    uses_historical_split:   bool = False


# ── Base event builder ────────────────────────────────────────────────────────

def _event(
    pipeline_id:     str,
    run_id:          str,
    activity:        str,
    timestamp:       datetime,
    team:            str,
) -> dict:
    """Single structured event. Always returns all required columns."""
    return {
        "pipeline_run_id": run_id,
        "activity":        activity,
        "timestamp":       pd.Timestamp(timestamp),
        "team":            team,
        "pipeline_id":     pipeline_id,
    }


def _run_id(pipeline_id: str, run_date: date) -> str:
    """Deterministic run ID from pipeline and date."""
    return f"{pipeline_id}_{run_date.strftime('%Y%m%d')}"


def _base_time(run_date: date, expected_start) -> datetime:
    """Absolute start datetime for a pipeline run."""
    if isinstance(expected_start, str):
        from datetime import time as dt_time
        h, m = map(int, expected_start.split(":"))
        expected_start = dt_time(h, m)
    return datetime.combine(run_date, expected_start)


# ── Archetype generators ──────────────────────────────────────────────────────

class StableMatureGenerator:
    """
    TIER 1 — Normal case: stable mature pipeline.

    Age: 180-365 days. Low variance. Drift velocity near zero.
    This is the H0 comparison group — mature pipelines drift slowly.

    Expected: HEALTHY cluster, STABLE pattern, HIGH confidence.
    The 0.05 bonus from early completion zone appears frequently
    because mature pipelines have been tuned to run fast.
    """

    def generate(
        self,
        contract:     PipelineContract,
        days:         int,
        start_date:   date,
        pipeline_age: int,
        seed:         int,
    ) -> tuple[pd.DataFrame, pd.DataFrame, GroundTruth]:

        rng = random.Random(seed)
        producer_events = []
        consumer_events = []

        base_duration = contract.p50_minutes * rng.uniform(0.85, 0.95)

        for i in range(days):
            run_date = start_date + timedelta(days=i)
            base     = _base_time(run_date, contract.expected_start)
            run_id   = _run_id(contract.pipeline_id, run_date)

            # Mature pipelines have low variance and complete early
            # Small daily noise: ±3 minutes
            daily_noise = rng.gauss(0, 1.5)
            duration    = max(10, base_duration + daily_noise)

            # Producer trace
            for act, delta in [
                ("SCHEDULED",    -2),
                ("STARTED",       1),
                ("COMPLETED",    duration),
                ("DATA_AVAILABLE", duration + 3),
            ]:
                producer_events.append(_event(
                    contract.pipeline_id, run_id, act,
                    base + timedelta(minutes=delta
                                     if act == "SCHEDULED"
                                     else duration if act in ("COMPLETED", "DATA_AVAILABLE")
                                     else 1),
                    "producer"
                ))

            # Consumer trace — receives data 4-6 minutes after producer
            consumer_gap = rng.uniform(4, 6)
            for act, delta in [
                ("SCHEDULED",    -2),
                ("STARTED",       1),
                ("COMPLETED",    duration),
                ("DATA_AVAILABLE", duration + consumer_gap),
            ]:
                consumer_events.append(_event(
                    contract.pipeline_id, run_id, act,
                    base + timedelta(minutes=delta
                                     if act == "SCHEDULED"
                                     else duration if act == "COMPLETED"
                                     else duration + consumer_gap if act == "DATA_AVAILABLE"
                                     else 1),
                    "consumer"
                ))

        gt = GroundTruth(
            archetype="stable_mature",
            tier=1,
            expected_pattern="STABLE",
            expected_cluster="HEALTHY",
            pipeline_age_days=pipeline_age,
            drift_velocity_per_week=0.2,  # near zero
        )

        return (
            pd.DataFrame(producer_events),
            pd.DataFrame(consumer_events),
            gt
        )


class StableNewGenerator:
    """
    TIER 1 — Normal case: stable new pipeline.

    Age: 7-30 days. Higher variance because the pipeline is new
    and not yet tuned. Warmup multiplier reduces confidence.

    Expected: HEALTHY or DRIFTING cluster (borderline).
    Pattern: STABLE. Confidence: MEDIUM (warmup active).

    H0 note: higher variance than mature but not higher drift.
    This is the stable-new control — new pipelines can be stable.
    """

    def generate(
        self,
        contract:     PipelineContract,
        days:         int,
        start_date:   date,
        pipeline_age: int,
        seed:         int,
    ) -> tuple[pd.DataFrame, pd.DataFrame, GroundTruth]:

        rng = random.Random(seed)
        producer_events = []
        consumer_events = []

        # New pipelines have higher variance — not tuned yet
        base_duration = contract.p50_minutes * rng.uniform(0.90, 1.05)

        for i in range(days):
            run_date = start_date + timedelta(days=i)
            base     = _base_time(run_date, contract.expected_start)
            run_id   = _run_id(contract.pipeline_id, run_date)

            # Higher noise for new pipelines: ±8 minutes
            daily_noise = rng.gauss(0, 4.0)
            duration    = max(10, base_duration + daily_noise)

            consumer_gap = rng.uniform(5, 12)  # wider gap, not settled yet

            for act in ["SCHEDULED", "STARTED", "COMPLETED", "DATA_AVAILABLE"]:
                ts = (base - timedelta(minutes=2) if act == "SCHEDULED"
                      else base + timedelta(minutes=1) if act == "STARTED"
                      else base + timedelta(minutes=duration) if act == "COMPLETED"
                      else base + timedelta(minutes=duration + 4))
                producer_events.append(_event(
                    contract.pipeline_id, run_id, act, ts, "producer"
                ))

            for act in ["SCHEDULED", "STARTED", "COMPLETED", "DATA_AVAILABLE"]:
                ts = (base - timedelta(minutes=2) if act == "SCHEDULED"
                      else base + timedelta(minutes=1) if act == "STARTED"
                      else base + timedelta(minutes=duration) if act == "COMPLETED"
                      else base + timedelta(minutes=duration + consumer_gap))
                consumer_events.append(_event(
                    contract.pipeline_id, run_id, act, ts, "consumer"
                ))

        gt = GroundTruth(
            archetype="stable_new",
            tier=1,
            expected_pattern="STABLE",
            expected_cluster="HEALTHY",
            pipeline_age_days=pipeline_age,
            drift_velocity_per_week=0.5,  # slightly higher but not alarming
        )

        return pd.DataFrame(producer_events), pd.DataFrame(consumer_events), gt


class SlowDriftingGenerator:
    """
    TIER 1 — Normal case: slow drifting established pipeline.

    Age: 60-180 days. Drift velocity 1.0-2.5 min/week.
    The H0 baseline drifting case — established pipelines drift slowly.

    Expected: DRIFTING cluster, DRIFTING pattern, HIGH confidence.
    Drift rate negative (score declining over time).
    """

    def __init__(self, drift_rate_per_week: float = 1.5):
        """
        drift_rate_per_week: how many minutes per week the pipeline
        adds to its execution time. 1.5 means each week it takes
        1.5 minutes longer — slow, barely noticeable day-to-day.
        """
        self.drift_rate = drift_rate_per_week

    def generate(
        self,
        contract:     PipelineContract,
        days:         int,
        start_date:   date,
        pipeline_age: int,
        seed:         int,
    ) -> tuple[pd.DataFrame, pd.DataFrame, GroundTruth]:

        rng = random.Random(seed)
        producer_events = []
        consumer_events = []

        base_duration = contract.p50_minutes

        for i in range(days):
            run_date = start_date + timedelta(days=i)
            base     = _base_time(run_date, contract.expected_start)
            run_id   = _run_id(contract.pipeline_id, run_date)

            # Drift applied as cumulative weekly increment
            # Week 0: base_duration, Week 1: base + drift, etc.
            weeks_elapsed = i / 7
            drift_added   = self.drift_rate * weeks_elapsed
            daily_noise   = rng.gauss(0, 2.0)
            duration      = max(10, base_duration + drift_added + daily_noise)

            # Consumer gap also grows slightly — both sides degrade
            consumer_gap  = rng.uniform(6, 10) + (weeks_elapsed * 0.3)

            for act in ["SCHEDULED", "STARTED", "COMPLETED", "DATA_AVAILABLE"]:
                ts = (base - timedelta(minutes=2) if act == "SCHEDULED"
                      else base + timedelta(minutes=1) if act == "STARTED"
                      else base + timedelta(minutes=duration) if act == "COMPLETED"
                      else base + timedelta(minutes=duration + 4))
                producer_events.append(_event(
                    contract.pipeline_id, run_id, act, ts, "producer"
                ))

            for act in ["SCHEDULED", "STARTED", "COMPLETED", "DATA_AVAILABLE"]:
                ts = (base - timedelta(minutes=2) if act == "SCHEDULED"
                      else base + timedelta(minutes=1) if act == "STARTED"
                      else base + timedelta(minutes=duration) if act == "COMPLETED"
                      else base + timedelta(minutes=duration + consumer_gap))
                consumer_events.append(_event(
                    contract.pipeline_id, run_id, act, ts, "consumer"
                ))

        gt = GroundTruth(
            archetype="slow_drifting",
            tier=1,
            expected_pattern="DRIFTING",
            expected_cluster="DRIFTING",
            pipeline_age_days=pipeline_age,
            drift_velocity_per_week=self.drift_rate,
        )

        return pd.DataFrame(producer_events), pd.DataFrame(consumer_events), gt


class FastDriftingNewGenerator:
    """
    TIER 2 — Deliberate failure: fast drifting new pipeline.

    Age: 14-45 days. Drift velocity: 3.0-6.0 min/week.
    The H1 test group for H0.

    Why new pipelines drift fast:
    The contract was written optimistically before the pipeline's
    actual behaviour was understood. As real execution data accumulates,
    the gap between contract and reality becomes visible quickly.
    At Citi this was common — deadlines were set by management
    before engineering knew the actual processing time.

    Expected: CRITICAL cluster, DRIFTING pattern, MEDIUM confidence.
    H0 test: this group should have statistically higher drift_velocity
    than the stable_mature and slow_drifting groups (Mann-Whitney p < 0.05).
    """

    def __init__(self, drift_rate_per_week: float = 4.5):
        self.drift_rate = drift_rate_per_week

    def generate(
        self,
        contract:     PipelineContract,
        days:         int,
        start_date:   date,
        pipeline_age: int,
        seed:         int,
    ) -> tuple[pd.DataFrame, pd.DataFrame, GroundTruth]:

        rng = random.Random(seed)
        producer_events = []
        consumer_events = []

        # Contract says p50 minutes but reality starts higher already
        # because the contract was optimistic
        base_duration = contract.p50_minutes * 1.15  # 15% over contract p50

        for i in range(days):
            run_date = start_date + timedelta(days=i)
            base     = _base_time(run_date, contract.expected_start)
            run_id   = _run_id(contract.pipeline_id, run_date)

            weeks_elapsed = i / 7
            drift_added   = self.drift_rate * weeks_elapsed
            daily_noise   = rng.gauss(0, 3.0)  # higher variance too
            duration      = max(10, base_duration + drift_added + daily_noise)

            # Consumer gap grows fast — undocumented dependencies emerging
            consumer_gap = rng.uniform(10, 20) + (weeks_elapsed * 2.0)

            for act in ["SCHEDULED", "STARTED", "COMPLETED", "DATA_AVAILABLE"]:
                ts = (base - timedelta(minutes=2) if act == "SCHEDULED"
                      else base + timedelta(minutes=1) if act == "STARTED"
                      else base + timedelta(minutes=duration) if act == "COMPLETED"
                      else base + timedelta(minutes=duration + 4))
                producer_events.append(_event(
                    contract.pipeline_id, run_id, act, ts, "producer"
                ))

            for act in ["SCHEDULED", "STARTED", "COMPLETED", "DATA_AVAILABLE"]:
                ts = (base - timedelta(minutes=2) if act == "SCHEDULED"
                      else base + timedelta(minutes=1) if act == "STARTED"
                      else base + timedelta(minutes=duration) if act == "COMPLETED"
                      else base + timedelta(minutes=duration + consumer_gap))
                consumer_events.append(_event(
                    contract.pipeline_id, run_id, act, ts, "consumer"
                ))

        gt = GroundTruth(
            archetype="fast_drifting_new",
            tier=2,
            expected_pattern="DRIFTING",
            expected_cluster="CRITICAL",
            pipeline_age_days=pipeline_age,
            drift_velocity_per_week=self.drift_rate,
        )

        return pd.DataFrame(producer_events), pd.DataFrame(consumer_events), gt


class AssumptionAsymmetryGenerator:
    """
    TIER 2 — Deliberate failure: assumption asymmetry.

    The core bilateral failure mode. Both teams think they are conformant.
    Neither is lying. The contract is silent on the gap between them.

    Producer marks DATA_AVAILABLE at T+47 (data written to table).
    Consumer experiences DATA_AVAILABLE at T+72 (data queryable after transforms).
    Contract says both should be done by T+60.

    The gap widens slowly over time as the consumer's transformation
    step grows with data volume — another undocumented assumption.

    Key signature: pattern=STABLE (pipeline is not drifting),
    but producer_score >> consumer_score. This asymmetry is the
    finding that only bilateral conformance checking can surface.
    """

    def __init__(
        self,
        initial_gap:       float = 25.0,   # minutes
        gap_growth_per_week: float = 1.5,  # gap widens over time
    ):
        self.initial_gap    = initial_gap
        self.gap_growth     = gap_growth_per_week

    def generate(
        self,
        contract:     PipelineContract,
        days:         int,
        start_date:   date,
        pipeline_age: int,
        seed:         int,
    ) -> tuple[pd.DataFrame, pd.DataFrame, GroundTruth]:

        rng = random.Random(seed)
        producer_events = []
        consumer_events = []

        base_duration = contract.p50_minutes * 0.95  # producer is efficient

        for i in range(days):
            run_date = start_date + timedelta(days=i)
            base     = _base_time(run_date, contract.expected_start)
            run_id   = _run_id(contract.pipeline_id, run_date)

            weeks_elapsed = i / 7
            current_gap   = self.initial_gap + (self.gap_growth * weeks_elapsed)
            daily_noise   = rng.gauss(0, 2.0)
            duration      = max(10, base_duration + daily_noise)

            # Producer trace — conforms well, marks done early
            for act in ["SCHEDULED", "STARTED", "COMPLETED", "DATA_AVAILABLE"]:
                ts = (base - timedelta(minutes=2) if act == "SCHEDULED"
                      else base + timedelta(minutes=1) if act == "STARTED"
                      else base + timedelta(minutes=duration) if act == "COMPLETED"
                      else base + timedelta(minutes=duration + 3))  # 3 min post-complete
                producer_events.append(_event(
                    contract.pipeline_id, run_id, act, ts, "producer"
                ))

            # Consumer trace — same pipeline, but data_available much later
            # because of undocumented downstream transformation step
            consumer_data_available = duration + 3 + current_gap + rng.gauss(0, 2)
            for act in ["SCHEDULED", "STARTED", "COMPLETED", "DATA_AVAILABLE"]:
                ts = (base - timedelta(minutes=2) if act == "SCHEDULED"
                      else base + timedelta(minutes=1) if act == "STARTED"
                      else base + timedelta(minutes=duration) if act == "COMPLETED"
                      else base + timedelta(minutes=consumer_data_available))
                consumer_events.append(_event(
                    contract.pipeline_id, run_id, act, ts, "consumer"
                ))

        gt = GroundTruth(
            archetype="assumption_asymmetry",
            tier=2,
            expected_pattern="STABLE",     # pipeline itself is not drifting
            expected_cluster="DRIFTING",   # consumer score is low
            pipeline_age_days=pipeline_age,
            drift_velocity_per_week=0.3,   # producer drift near zero
        )

        return pd.DataFrame(producer_events), pd.DataFrame(consumer_events), gt


class SilentPipelineGenerator:
    """
    TIER 2 — Deliberate failure: silent pipeline.

    No events at all on specific days. The hardest failure mode to detect
    because the system does not fail — it simply does not run.
    Downstream fallback uses yesterday's data silently.

    Silent on 2 specific weekdays (configurable).
    On silent days: only SCHEDULED fires, then nothing.
    This is the SILENT breach type from the SLA monitor spec.

    Expected: CRITICAL cluster, INTERMITTENT pattern (day-clustered).
    DBSCAN likely marks as noise point — correct behaviour.
    """

    def __init__(self, silent_weekdays: list[int] = None):
        """
        silent_weekdays: list of weekday indices (0=Monday, 6=Sunday).
        Default: Monday (0) and Thursday (3).
        """
        self.silent_weekdays = silent_weekdays or [0, 3]

    def generate(
        self,
        contract:     PipelineContract,
        days:         int,
        start_date:   date,
        pipeline_age: int,
        seed:         int,
    ) -> tuple[pd.DataFrame, pd.DataFrame, GroundTruth]:

        rng = random.Random(seed)
        producer_events = []
        consumer_events = []

        base_duration = contract.p50_minutes * rng.uniform(0.90, 0.98)

        for i in range(days):
            run_date = start_date + timedelta(days=i)
            base     = _base_time(run_date, contract.expected_start)
            run_id   = _run_id(contract.pipeline_id, run_date)
            is_silent = run_date.weekday() in self.silent_weekdays

            if is_silent:
                # Only SCHEDULED fires — pipeline never starts
                # This is the signature the SLA monitor catches as SILENT breach
                producer_events.append(_event(
                    contract.pipeline_id, run_id, "SCHEDULED",
                    base - timedelta(minutes=2), "producer"
                ))
                # Consumer sees nothing — no events at all
                # (intentionally empty for this run)
            else:
                # Normal healthy execution on non-silent days
                daily_noise = rng.gauss(0, 2.0)
                duration    = max(10, base_duration + daily_noise)
                consumer_gap = rng.uniform(4, 8)

                for act in ["SCHEDULED", "STARTED", "COMPLETED", "DATA_AVAILABLE"]:
                    ts = (base - timedelta(minutes=2) if act == "SCHEDULED"
                          else base + timedelta(minutes=1) if act == "STARTED"
                          else base + timedelta(minutes=duration) if act == "COMPLETED"
                          else base + timedelta(minutes=duration + 4))
                    producer_events.append(_event(
                        contract.pipeline_id, run_id, act, ts, "producer"
                    ))

                for act in ["SCHEDULED", "STARTED", "COMPLETED", "DATA_AVAILABLE"]:
                    ts = (base - timedelta(minutes=2) if act == "SCHEDULED"
                          else base + timedelta(minutes=1) if act == "STARTED"
                          else base + timedelta(minutes=duration) if act == "COMPLETED"
                          else base + timedelta(minutes=duration + consumer_gap))
                    consumer_events.append(_event(
                        contract.pipeline_id, run_id, act, ts, "consumer"
                    ))

        gt = GroundTruth(
            archetype="silent",
            tier=2,
            expected_pattern="INTERMITTENT",
            expected_cluster="CRITICAL",
            pipeline_age_days=pipeline_age,
            drift_velocity_per_week=0.0,     # silence is not drift
        )

        return pd.DataFrame(producer_events), pd.DataFrame(consumer_events), gt


class SuddenCollapseGenerator:
    """
    TIER 2 — Deliberate failure: sudden collapse.

    Historical + live split design.
    Days 1-25: healthy execution (historical.parquet — pre-written)
    Days 26+:  catastrophic failure (fresh generated)

    This tests the drift rate's limitation — a step function collapse
    shows near-zero drift slope (because history was healthy) but the
    most recent scores are critical. The tool should not mislead the
    engineer into thinking this is slow drift when it is sudden failure.

    Implementation:
      historical.parquet contains 25 days of healthy runs.
      Live runs start at day 26 with catastrophic failure.
      The conformance engine receives historical_scores from
      the historical parquet and live events from the fresh parquet.

    Expected: CRITICAL cluster, DRIFTING pattern (misleading on drift rate —
    this is a known limitation documented in the paper).
    The per-run score chart tells the real story even when the slope metric
    is misleading.
    """

    def generate_historical(
        self,
        contract:   PipelineContract,
        days:       int = 25,
        start_date: date = None,
        seed:       int = 42,
    ) -> pd.DataFrame:
        """
        Generate the historical healthy period.
        Written to historical.parquet once.
        The conformance engine loads this as historical_scores context.
        """
        if start_date is None:
            start_date = date.today() - timedelta(days=days + 5)

        rng = random.Random(seed)
        events = []
        base_duration = contract.p50_minutes * 0.90

        for i in range(days):
            run_date = start_date + timedelta(days=i)
            base     = _base_time(run_date, contract.expected_start)
            run_id   = _run_id(contract.pipeline_id, run_date)

            duration    = max(10, base_duration + rng.gauss(0, 1.5))
            consumer_gap = rng.uniform(4, 6)

            for act in ["SCHEDULED", "STARTED", "COMPLETED", "DATA_AVAILABLE"]:
                ts = (base - timedelta(minutes=2) if act == "SCHEDULED"
                      else base + timedelta(minutes=1) if act == "STARTED"
                      else base + timedelta(minutes=duration) if act == "COMPLETED"
                      else base + timedelta(minutes=duration + consumer_gap))
                events.append(_event(
                    contract.pipeline_id, run_id, act, ts, "producer"
                ))

        return pd.DataFrame(events)

    def generate_live(
        self,
        contract:   PipelineContract,
        days:       int = 5,
        start_date: date = None,
        seed:       int = 99,
    ) -> tuple[pd.DataFrame, pd.DataFrame, GroundTruth]:
        """
        Generate the post-collapse live period.
        Catastrophic failure — pipeline times out or produces nothing useful.
        """
        if start_date is None:
            start_date = date.today() - timedelta(days=days)

        rng = random.Random(seed)
        producer_events = []
        consumer_events = []

        for i in range(days):
            run_date = start_date + timedelta(days=i)
            base     = _base_time(run_date, contract.expected_start)
            run_id   = _run_id(contract.pipeline_id, run_date)

            # Catastrophic failure: takes 3-4x p99 and barely completes
            # Or randomly produces no DATA_AVAILABLE (partial failure)
            failure_mode = rng.choice(["timeout", "partial"])

            if failure_mode == "timeout":
                # Runs but exceeds p99+grace badly
                duration = contract.p99_minutes * rng.uniform(2.5, 4.0)
                for act in ["SCHEDULED", "STARTED", "COMPLETED"]:
                    ts = (base - timedelta(minutes=2) if act == "SCHEDULED"
                          else base + timedelta(minutes=1) if act == "STARTED"
                          else base + timedelta(minutes=duration))
                    producer_events.append(_event(
                        contract.pipeline_id, run_id, act, ts, "producer"
                    ))
                # DATA_AVAILABLE never fires (timeout before completion)
            else:
                # Partial: starts and completes but consumer never sees data
                duration = contract.p99_minutes * rng.uniform(1.2, 1.8)
                for act in ["SCHEDULED", "STARTED", "COMPLETED"]:
                    ts = (base - timedelta(minutes=2) if act == "SCHEDULED"
                          else base + timedelta(minutes=1) if act == "STARTED"
                          else base + timedelta(minutes=duration))
                    producer_events.append(_event(
                        contract.pipeline_id, run_id, act, ts, "producer"
                    ))
                # Consumer never sees anything
                consumer_events.append(_event(
                    contract.pipeline_id, run_id, "SCHEDULED",
                    base - timedelta(minutes=2), "consumer"
                ))

        gt = GroundTruth(
            archetype="sudden_collapse",
            tier=2,
            expected_pattern="DRIFTING",    # slope misleads — documented limitation
            expected_cluster="CRITICAL",
            pipeline_age_days=180,          # was mature before collapse
            drift_velocity_per_week=0.5,    # historical slope near zero
            uses_historical_split=True,
        )

        return pd.DataFrame(producer_events), pd.DataFrame(consumer_events), gt


# ── Tier 3: Outlier generators ────────────────────────────────────────────────

class SelfCorrectingGenerator:
    """
    TIER 3 — Outlier: self-correcting drift.

    Days 1-15: healthy
    Days 16-25: drifting (performance problem detected and fixed)
    Days 26-30: recovered

    Tests the drift rate calculation under non-linear patterns.
    Net slope over 30 days should be near zero despite the dip.
    CV should flag it as INTERMITTENT (high variance).
    Outlier handling should weight the dip traces appropriately.
    """

    def generate(
        self,
        contract:     PipelineContract,
        days:         int,
        start_date:   date,
        pipeline_age: int,
        seed:         int,
    ) -> tuple[pd.DataFrame, pd.DataFrame, GroundTruth]:

        rng = random.Random(seed)
        producer_events = []
        consumer_events = []
        base_duration = contract.p50_minutes * 0.92

        for i in range(days):
            run_date = start_date + timedelta(days=i)
            base     = _base_time(run_date, contract.expected_start)
            run_id   = _run_id(contract.pipeline_id, run_date)

            # Phase-based duration
            if i < 15:
                duration = base_duration + rng.gauss(0, 1.5)
            elif i < 25:
                # Degraded: extra 30-40 minutes
                duration = base_duration + 35 + rng.gauss(0, 3.0)
            else:
                # Recovered: back to normal
                duration = base_duration + rng.gauss(0, 2.0)

            duration     = max(10, duration)
            consumer_gap = rng.uniform(4, 8)

            for act in ["SCHEDULED", "STARTED", "COMPLETED", "DATA_AVAILABLE"]:
                ts = (base - timedelta(minutes=2) if act == "SCHEDULED"
                      else base + timedelta(minutes=1) if act == "STARTED"
                      else base + timedelta(minutes=duration) if act == "COMPLETED"
                      else base + timedelta(minutes=duration + 4))
                producer_events.append(_event(
                    contract.pipeline_id, run_id, act, ts, "producer"
                ))

            for act in ["SCHEDULED", "STARTED", "COMPLETED", "DATA_AVAILABLE"]:
                ts = (base - timedelta(minutes=2) if act == "SCHEDULED"
                      else base + timedelta(minutes=1) if act == "STARTED"
                      else base + timedelta(minutes=duration) if act == "COMPLETED"
                      else base + timedelta(minutes=duration + consumer_gap))
                consumer_events.append(_event(
                    contract.pipeline_id, run_id, act, ts, "consumer"
                ))

        gt = GroundTruth(
            archetype="self_correcting",
            tier=3,
            expected_pattern="INTERMITTENT",  # high CV from the dip
            expected_cluster="DRIFTING",       # borderline
            pipeline_age_days=pipeline_age,
            drift_velocity_per_week=0.1,       # near zero net slope
        )

        return pd.DataFrame(producer_events), pd.DataFrame(consumer_events), gt


class BimodalGenerator:
    """
    TIER 3 — Outlier: bimodal performance.

    Weekdays: fast, GREEN zone.
    Weekends + month-end: slow, RED zone.

    This is NOT the Monday infrastructure problem (Guard 7 territory).
    Weekends are legitimately slow due to batch processing load.
    Month-end is slow due to volume spike.

    Tests: weekday pattern detector must NOT flag this as infrastructure.
    infrastructure_probable should be False because multiple days
    (Saturday, Sunday, month-end) are slow — not one specific day.
    """

    def generate(
        self,
        contract:     PipelineContract,
        days:         int,
        start_date:   date,
        pipeline_age: int,
        seed:         int,
    ) -> tuple[pd.DataFrame, pd.DataFrame, GroundTruth]:

        rng = random.Random(seed)
        producer_events = []
        consumer_events = []

        fast_duration = contract.p50_minutes * 0.85  # weekday: fast
        slow_duration = contract.p99_minutes * 0.90  # weekend/month-end: slow

        for i in range(days):
            run_date  = start_date + timedelta(days=i)
            base      = _base_time(run_date, contract.expected_start)
            run_id    = _run_id(contract.pipeline_id, run_date)
            is_weekend = run_date.weekday() >= 5
            is_month_end = run_date.day >= 28

            if is_weekend or is_month_end:
                duration = slow_duration + rng.gauss(0, 5.0)
            else:
                duration = fast_duration + rng.gauss(0, 2.0)

            duration     = max(10, duration)
            consumer_gap = rng.uniform(5, 10)

            for act in ["SCHEDULED", "STARTED", "COMPLETED", "DATA_AVAILABLE"]:
                ts = (base - timedelta(minutes=2) if act == "SCHEDULED"
                      else base + timedelta(minutes=1) if act == "STARTED"
                      else base + timedelta(minutes=duration) if act == "COMPLETED"
                      else base + timedelta(minutes=duration + 4))
                producer_events.append(_event(
                    contract.pipeline_id, run_id, act, ts, "producer"
                ))

            for act in ["SCHEDULED", "STARTED", "COMPLETED", "DATA_AVAILABLE"]:
                ts = (base - timedelta(minutes=2) if act == "SCHEDULED"
                      else base + timedelta(minutes=1) if act == "STARTED"
                      else base + timedelta(minutes=duration) if act == "COMPLETED"
                      else base + timedelta(minutes=duration + consumer_gap))
                consumer_events.append(_event(
                    contract.pipeline_id, run_id, act, ts, "consumer"
                ))

        gt = GroundTruth(
            archetype="bimodal",
            tier=3,
            expected_pattern="INTERMITTENT",
            expected_cluster="DRIFTING",
            pipeline_age_days=pipeline_age,
            drift_velocity_per_week=0.8,  # apparent drift from weekend load
        )

        return pd.DataFrame(producer_events), pd.DataFrame(consumer_events), gt


class ClockSyncOutlierGenerator:
    """
    TIER 3 — Outlier: clock sync violation.

    Consumer DATA_AVAILABLE fires BEFORE producer DATA_AVAILABLE.
    Physically impossible — indicates timezone mismatch or clock skew.

    Expected behaviour: ConformanceGuardError raised immediately.
    Fracture refuses to compute a score and tells the engineer
    exactly what to investigate.

    This is the fail-loud, fail-early design principle.
    A negative bilateral gap is not a bad score — it is a
    fundamental data quality violation that invalidates the measurement.
    """

    def __init__(self, clock_skew_minutes: float = -15.0):
        """
        clock_skew_minutes: how far ahead the consumer's clock is.
        Negative means consumer timestamps are earlier than they should be.
        """
        self.clock_skew = clock_skew_minutes

    def generate(
        self,
        contract:     PipelineContract,
        days:         int,
        start_date:   date,
        pipeline_age: int,
        seed:         int,
    ) -> tuple[pd.DataFrame, pd.DataFrame, GroundTruth]:

        rng = random.Random(seed)
        producer_events = []
        consumer_events = []

        base_duration = contract.p50_minutes * 0.92

        for i in range(days):
            run_date     = start_date + timedelta(days=i)
            base         = _base_time(run_date, contract.expected_start)
            run_id       = _run_id(contract.pipeline_id, run_date)
            duration     = max(10, base_duration + rng.gauss(0, 2.0))
            consumer_gap = rng.uniform(4, 8)

            for act in ["SCHEDULED", "STARTED", "COMPLETED", "DATA_AVAILABLE"]:
                ts = (base - timedelta(minutes=2) if act == "SCHEDULED"
                      else base + timedelta(minutes=1) if act == "STARTED"
                      else base + timedelta(minutes=duration) if act == "COMPLETED"
                      else base + timedelta(minutes=duration + 4))
                producer_events.append(_event(
                    contract.pipeline_id, run_id, act, ts, "producer"
                ))

            # Consumer clock is skewed — all timestamps shifted
            # This makes DATA_AVAILABLE appear before producer's
            for act in ["SCHEDULED", "STARTED", "COMPLETED", "DATA_AVAILABLE"]:
                ts = (base - timedelta(minutes=2) if act == "SCHEDULED"
                      else base + timedelta(minutes=1) if act == "STARTED"
                      else base + timedelta(minutes=duration) if act == "COMPLETED"
                      else base + timedelta(minutes=duration + consumer_gap))
                # Apply clock skew to all consumer events
                skewed_ts = ts + timedelta(minutes=self.clock_skew)
                consumer_events.append(_event(
                    contract.pipeline_id, run_id, act, skewed_ts, "consumer"
                ))

        gt = GroundTruth(
            archetype="clock_sync_outlier",
            tier=3,
            expected_pattern="STABLE",
            expected_cluster="HEALTHY",     # would be healthy if clocks were right
            pipeline_age_days=pipeline_age,
            drift_velocity_per_week=0.0,
            should_raise_guard=True,
        )

        return pd.DataFrame(producer_events), pd.DataFrame(consumer_events), gt


class GhostPipelineGenerator:
    """
    TIER 3 — Outlier: ghost pipeline.

    Only SCHEDULED fires. Nothing else. Ever.
    The pipeline is registered, scheduled, and silently does nothing.

    This is the nuclear outlier case.
    Could mean: the scheduler is broken, the process never starts,
    or Fracture is receiving only the orchestrator logs without
    the actual execution logs.

    Expected: ConformanceGuardError from preflight (no required events).
    Completeness score = 0.0. Confidence = UNRELIABLE.
    """

    def generate(
        self,
        contract:     PipelineContract,
        days:         int,
        start_date:   date,
        pipeline_age: int,
        seed:         int,
    ) -> tuple[pd.DataFrame, pd.DataFrame, GroundTruth]:

        producer_events = []

        for i in range(days):
            run_date = start_date + timedelta(days=i)
            base     = _base_time(run_date, contract.expected_start)
            run_id   = _run_id(contract.pipeline_id, run_date)

            # Only SCHEDULED — nothing else ever fires
            producer_events.append(_event(
                contract.pipeline_id, run_id, "SCHEDULED",
                base - timedelta(minutes=2), "producer"
            ))

        gt = GroundTruth(
            archetype="ghost_pipeline",
            tier=3,
            expected_pattern="INTERMITTENT",
            expected_cluster="CRITICAL",
            pipeline_age_days=pipeline_age,
            drift_velocity_per_week=0.0,
            should_raise_guard=True,
        )

        # Empty consumer — ghost pipeline has no consumer activity
        return (
            pd.DataFrame(producer_events),
            pd.DataFrame(columns=REQUIRED_COLUMNS),
            gt
        )


# ── Population orchestrator ───────────────────────────────────────────────────

# Archetype distribution matching our taxonomy
# tier: (generator_class, count, age_range_days, drift_velocity_range)
ARCHETYPE_SPECS = [
    # TIER 1 — Normal cases (60 total)
    ("stable_mature",      StableMatureGenerator,    20, (180, 365), (0.0,  0.5)),
    ("stable_new",         StableNewGenerator,       15, (7,   30),  (0.0,  1.0)),
    ("slow_drifting",      SlowDriftingGenerator,    15, (60,  180), (1.0,  2.5)),
    # TIER 2 — Deliberate failure cases (25 total)
    ("fast_drifting_new",  FastDriftingNewGenerator,  8, (14,  45),  (3.0,  6.0)),
    ("assumption_asymmetry", AssumptionAsymmetryGenerator, 8, (30, 180), (0.1, 0.4)),
    ("silent",             SilentPipelineGenerator,   5, (30,  180), (0.0,  0.0)),
    ("sudden_collapse",    SuddenCollapseGenerator,   4, (180, 365), (0.0,  0.5)),
    # TIER 3 — Outlier cases (15 total)
    ("self_correcting",    SelfCorrectingGenerator,   3, (60,  180), (0.0,  0.2)),
    ("bimodal",            BimodalGenerator,          3, (90,  270), (0.5,  1.2)),
    ("clock_sync_outlier", ClockSyncOutlierGenerator, 2, (30,  180), (0.0,  0.0)),
    ("ghost_pipeline",     GhostPipelineGenerator,    2, (7,   30),  (0.0,  0.0)),
    ("slow_drifting_old",  SlowDriftingGenerator,     3, (270, 500), (0.1,  0.8)),
]


def generate_population(
    contracts:   list[PipelineContract],
    config:      FractureConfig = DEFAULT_CONFIG,
    output_dir:  str = "outputs/logs",
    history_days: int = 30,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame, GroundTruth]]:
    """
    Generate the full 100-pipeline population.

    Returns a dict mapping pipeline_id to
    (producer_events, consumer_events, ground_truth).

    Also writes parquet files to output_dir for the watcher.

    For sudden_collapse pipelines, also writes historical.parquet
    to support the historical + live split design.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results = {}
    contract_idx = 0
    today = date.today()
    start_date = today - timedelta(days=history_days)

    for archetype_name, gen_class, count, age_range, drift_range in ARCHETYPE_SPECS:
        for j in range(count):
            if contract_idx >= len(contracts):
                break

            contract = contracts[contract_idx]
            contract_idx += 1

            seed = contract_idx * 137 + 42
            rng  = random.Random(seed)

            pipeline_age = rng.randint(*age_range)

            # Instantiate generator with archetype-appropriate parameters
            if archetype_name == "slow_drifting":
                drift_rate = rng.uniform(*drift_range)
                gen = gen_class(drift_rate_per_week=drift_rate)
            elif archetype_name == "slow_drifting_old":
                drift_rate = rng.uniform(*drift_range)
                gen = SlowDriftingGenerator(drift_rate_per_week=drift_rate)
            elif archetype_name == "fast_drifting_new":
                drift_rate = rng.uniform(*drift_range)
                gen = gen_class(drift_rate_per_week=drift_rate)
            elif archetype_name == "assumption_asymmetry":
                initial_gap = rng.uniform(15, 35)
                gen = gen_class(initial_gap=initial_gap)
            elif archetype_name == "clock_sync_outlier":
                skew = rng.uniform(-20, -8)
                gen = gen_class(clock_skew_minutes=skew)
            else:
                gen = gen_class()

            # Generate events
            if archetype_name == "sudden_collapse":
                # Historical + live split
                historical = gen.generate_historical(
                    contract=contract,
                    days=25,
                    start_date=start_date,
                    seed=seed,
                )
                prod, cons, gt = gen.generate_live(
                    contract=contract,
                    days=history_days - 25,
                    start_date=start_date + timedelta(days=25),
                    seed=seed + 1,
                )
                # Write historical parquet separately
                hist_dir = out / contract.pipeline_id
                hist_dir.mkdir(exist_ok=True)
                historical.to_parquet(hist_dir / "historical.parquet", index=False)
            else:
                prod, cons, gt = gen.generate(
                    contract=contract,
                    days=history_days,
                    start_date=start_date,
                    pipeline_age=pipeline_age,
                    seed=seed,
                )

            # Write live parquet files
            pipeline_dir = out / contract.pipeline_id
            pipeline_dir.mkdir(exist_ok=True)

            today_str = today.strftime('%Y%m%d')
            prod.to_parquet(
                pipeline_dir / f"producer_{today_str}.parquet",
                index=False
            )
            if len(cons) > 0:
                cons.to_parquet(
                    pipeline_dir / f"consumer_{today_str}.parquet",
                    index=False
                )

            results[contract.pipeline_id] = (prod, cons, gt)

    print(f"[generator] Generated {len(results)} pipelines")
    print(f"[generator] Tier distribution:")
    for tier, label in [(1, "Normal"), (2, "Failure"), (3, "Outlier")]:
        count = sum(1 for _, _, gt in results.values() if gt.tier == tier)
        print(f"  Tier {tier} ({label}): {count} pipelines")

    return results


def get_ground_truth_dataframe(
    results: dict[str, tuple[pd.DataFrame, pd.DataFrame, GroundTruth]]
) -> pd.DataFrame:
    """
    Convert ground truth labels to a DataFrame for H0 testing
    and ARI validation.

    Returns one row per pipeline with all ground truth fields.
    This is the y_true for the ML experiment.
    """
    rows = []
    for pipeline_id, (_, _, gt) in results.items():
        rows.append({
            "pipeline_id":             pipeline_id,
            "archetype":               gt.archetype,
            "tier":                    gt.tier,
            "expected_pattern":        gt.expected_pattern,
            "expected_cluster":        gt.expected_cluster,
            "expected_confidence":     gt.expected_confidence,
            "bilateral_gap_direction": gt.bilateral_gap_direction,
            "pipeline_age_days":       gt.pipeline_age_days,
            "drift_velocity_per_week": gt.drift_velocity_per_week,
            "should_raise_guard":      gt.should_raise_guard,
            "guard_type":              gt.guard_type or "",
            "uses_historical_split":   gt.uses_historical_split,
        })

    return pd.DataFrame(rows)
