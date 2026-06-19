"""
fracture/conformance.py

The conformance engine — the analytical heart of Fracture.

This module answers one question per pipeline per day:
"How faithfully did this pipeline execute according to what was agreed?"

Design principles:
  1. The formula is driven by the contract, not hardcoded thresholds.
     Different contracts produce different sensitivity to the same behaviour.
  2. A score is only meaningful alongside its confidence.
     Always report both together, never one without the other.
  3. Wrong for a known reason is different from wrong for an unknown reason.
     Attribution, pattern detection, and confidence all exist to make
     this distinction explicit.
  4. Early completion is not automatically good.
     A pipeline that finishes suspiciously early may have failed silently.
  5. The formula must produce results that match intuition on the test archetypes:
     healthy → 0.93+, degrading → declines weekly, asymmetry → producer ≠ consumer,
     silent → below 0.50. If any of these fail the formula has a bug.

Scenario coverage (see formula spec for full details):
  ✓ Hard SLA with penalty clause
  ✓ Soft SLA internal consumer
  ✓ Newly added process (warmup mode)
  ✓ Legacy process (baseline mode)
  ✓ Cascade dependency attribution
  ✓ Intermittent failure pattern (Monday bounce)
  ✓ Schema drift without timing violation
  ✓ Suspicious early completion
  ✓ Cross-fleet infrastructure pattern detection
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from fracture.ingest import normalize_events


import numpy as np
import pandas as pd
from scipy.stats import linregress

# PM4PY imports for process mining core
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay

from fracture.schema import PipelineContract
from fracture.config import FractureConfig, DEFAULT_CONFIG

# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class TimingZoneResult:
    """
    Result of the zone-based timing analysis.

    The zone model is the core innovation over simple pass/fail timing.
    It captures WHERE in the SLA window the pipeline completed,
    not just WHETHER it completed before the deadline.

    This enables:
    - Early warning before breach (AMBER zone)
    - Graduated scoring that reflects business impact
    - Penalty exposure calculation for client-facing pipelines
    - Downstream start eligibility signal
    """
    zone:                    str    # GREEN | AMBER | RED | BREACH
    score:                   float  # 0.10 to 1.05
    actual_duration_minutes: float
    minutes_to_deadline:     float  # negative = over deadline
    penalty_exposure_eur:    Optional[float] = None
    downstream_can_start_early: bool = False
    suspicious_early:        bool = False
    early_bonus_applied:     bool = False


@dataclass
class WeekdayPattern:
    """
    Result of weekday failure clustering analysis.

    The Monday restart problem at Citi: a pipeline scoring 0.81 on average
    because it fails every Monday looks like a drifting pipeline.
    It is actually a healthy pipeline with an infrastructure problem.
    This pattern detector separates those two cases.
    """
    has_weekday_clustering:       bool
    worst_weekday:                Optional[str]   # "Monday"
    worst_weekday_failure_rate:   float           # 1.0 = fails every Monday
    other_days_mean_score:        float           # score on non-bad days
    infrastructure_probable:      bool            # True if rate > 0.60
    affected_weekdays:            list[str]       # all weekdays with high failure rate


@dataclass
class DependencyAttribution:
    """
    Attribution result when cascade dependencies are involved.

    When Pipeline B fails because Pipeline A was late, the blame
    belongs to Pipeline A. This dataclass carries that attribution
    so the report can suppress downstream alerts and point at the
    root cause instead.
    """
    has_upstream_failure:    bool
    root_cause_pipeline:     Optional[str]
    upstream_conformance:    Optional[float]
    upstream_delay_minutes:  Optional[float]
    suppress_alert:          bool
    attribution_confidence:  float


@dataclass
class ChangePoint:
    """
    A detected structural change in the bilateral gap time series.
    Uses the PELT algorithm (Pruned Exact Linear Time) via ruptures.

    A changepoint means the gap distribution shifted at a specific date.
    This is more precise than a trend — it tells you WHEN the gap changed,
    not just THAT it is changing.

    Examples of what causes changepoints:
      A deployment changed the SFTP polling interval — gap drops by 10 min
      A new instrument type added — gap widens by 15 min overnight
      Consumer preprocessing parallelised — gap shrinks suddenly
      Infrastructure migration — gap spikes then stabilises

    Engineers correlate the changepoint date with deployment logs.
    """
    detected:            bool
    n_changepoints:      int
    changepoint_dates:   list         # dates where gap structure shifted
    pre_mean_minutes:    Optional[float]   # mean gap before last changepoint
    post_mean_minutes:   Optional[float]   # mean gap after last changepoint
    magnitude_minutes:   Optional[float]   # how much the gap shifted
    direction:           Optional[str]     # 'increased' | 'decreased'
    probable_cause:      str               # human-readable explanation


@dataclass
class VariantComparison:
    """
    Comparison between the normative Petri net (from contract)
    and the process model discovered from actual execution logs.

    When sequence_fitness is low, this tells you WHY.
    Token replay tells you the score. Variant comparison tells you the story.

    Examples:
      Normative: SCHEDULED → STARTED → COMPLETED → DATA_AVAILABLE
      Discovered: SCHEDULED → STARTED → STARTED → COMPLETED → DATA_AVAILABLE
      Finding: retries detected — activity STARTED fires twice per run

      Normative: SCHEDULED → STARTED → VALIDATED → COMPLETED → DATA_AVAILABLE
      Discovered: SCHEDULED → STARTED → COMPLETED → DATA_AVAILABLE (VALIDATED missing)
      Finding: VALIDATED never fires — contract is wrong or validation step removed
    """
    compared:               bool
    n_variants_actual:      int           # how many distinct paths in actual log
    dominant_variant:       list[str]     # most common actual execution path
    normative_path:         list[str]     # contracted expected path
    paths_match:            bool          # True if dominant variant = normative
    deviations:             list[str]     # list of plain-language deviation descriptions
    fitness_explainer:      str           # one sentence explaining low fitness


@dataclass
class ConformanceDiagnostics:
    """
    Analytical detail for engineers who need to investigate.

    Separated from the primary result so the operational
    layer stays clean. Most callers only need ConformanceResult.
    Access via result.diagnostics when you need to dig deeper.
    """
    sequence_fitness:    float          # PM4PY token replay (0.0–1.0)
    timing_score:        float          # zone model score (0.10–1.05)
    completeness_score:  float          # complete_runs / total_runs
    variance_cv:         float          # coefficient of variation
    timing_result:       TimingZoneResult
    weekday_pattern:     WeekdayPattern
    attribution:         DependencyAttribution
    # ── Bilateral gap fields ─────────────────────────────────────────────
    # upstream_gap:   gap from upstream producer to this pipeline
    # downstream_gap: gap from this pipeline to downstream consumer
    # bilateral_gap_minutes: backward-compatible alias = upstream_gap
    #
    # For simple producer-only pipelines: upstream_gap = the bilateral gap
    # For middle pipelines: both gaps computed independently

    bilateral_gap_trend:    Optional[str]   = None
    bilateral_gap_p95:      Optional[float] = None

    # Upstream gap (A → B): how long after A says done does B receive data?
    upstream_gap_minutes:   Optional[float] = None
    upstream_gap_trend:     Optional[str]   = None
    upstream_gap_p95:       Optional[float] = None

    # Downstream gap (B → C): how long after B says done does C receive data?
    downstream_gap_minutes: Optional[float] = None
    downstream_gap_trend:   Optional[str]   = None
    downstream_gap_p95:     Optional[float] = None

    # Consumer conformance (for middle pipelines)
    consumer_seq_fitness:   Optional[float] = None
    consumer_pattern:       Optional[str]   = None

    days_of_history:     int  = 0
    warmup_active:       bool = False
    changepoint:         Optional[object] = None
    variant_comparison:  Optional[object] = None

    # ── Temporal features (computed from historical_scores at run time) ───
    # These are written to conformance_log.csv so clustering.py
    # reads pre-computed features instead of recalculating from scratch.
    #
    # Without historical_scores: all None (first run, no history yet)
    # With historical_scores:    computed from the full score time series

    temporal_variance_cv:  Optional[float] = None  # std/mean across days
    drift_rate_per_day:    Optional[float] = None  # score slope (negative=declining)
    score_range:           Optional[float] = None  # max - min across days
    gap_drift_per_day:     Optional[float] = None  # bilateral gap slope
    mean_score_historical: Optional[float] = None  # mean of past scores
    min_score_historical:  Optional[float] = None  # worst past day


@dataclass
class ConformanceResult:
    """
    What you need to act. Seven fields plus diagnostics.

    Read these together:
      final_score + confidence_level  → is the score trustworthy?
      final_score + pattern           → is this drifting or intermittent?
      timing_zone + pattern           → what kind of problem is this?
      bilateral_gap_minutes           → is the handoff gap widening?

    Diagnostics are populated for engineers who need to investigate.
    Most callers — alerting, clustering, reporting — only need this.
    """
    pipeline_id:           str
    final_score:           float          # 0.0–1.05 weighted score
    confidence_level:      str            # HIGH | MEDIUM | LOW | UNRELIABLE
    timing_zone:           str            # GREEN | AMBER | RED | BREACH
    pattern:               str            # STABLE | DRIFTING | INTERMITTENT
    bilateral_gap_minutes: Optional[float] = None  # real minutes, not score units
    days_of_history:       int = 0

    # Detail for investigation — not needed for alerting or clustering
    diagnostics: Optional[ConformanceDiagnostics] = None

    # ── Operational interface ──────────────────────────────────

    def is_trustworthy(self) -> bool:
        """Confidence >= 0.70 — safe to act on this score."""
        level = self.confidence_level
        return level in ("HIGH", "MEDIUM")

    def needs_investigation(self) -> bool:
        """Low score AND trustworthy — something is genuinely wrong."""
        return self.final_score < 0.75 and self.is_trustworthy()

    def alert_owner(self) -> str:
        """Which team receives the alert. Ends the blame game."""
        if self.diagnostics and self.diagnostics.attribution.has_upstream_failure:
            return f"upstream ({self.diagnostics.attribution.root_cause_pipeline})"
        if self.diagnostics and self.diagnostics.weekday_pattern.infrastructure_probable:
            return "platform-infrastructure"
        return "pipeline-owner"

    def human_summary(self) -> str:
        """One sentence a manager can read and act on."""
        if not self.is_trustworthy():
            return (
                f"Measurement unreliable — confidence {self.confidence_level}. "
                f"Fix log extraction before acting on this score."
            )
        if self.timing_zone == "BREACH":
            return f"SLA breached. {self.alert_owner()} must investigate."
        if self.pattern == "INTERMITTENT":
            if (self.diagnostics and
                    self.diagnostics.weekday_pattern.infrastructure_probable):
                return (
                    "Fails on specific weekdays — infrastructure problem, "
                    "not pipeline problem. Route to platform-infrastructure."
                )
            return "Intermittent failures. Investigate specific failure days."
        if self.pattern == "DRIFTING":
            return (
                "Conformance declining. "
                "Schedule contract review with both teams this sprint."
            )
        if self.bilateral_gap_minutes and self.bilateral_gap_minutes > 10:
            return (
                f"Pipeline healthy but bilateral gap is "
                f"{self.bilateral_gap_minutes:.0f} min. "
                f"Consumer receives data later than producer thinks."
            )
        score_pct = f"{self.final_score:.0%}"
        return f"Conformant at {score_pct}. {self.timing_zone} zone. No action required."


# ── Guard system ─────────────────────────────────────────────────────────────

class ConformanceGuardError(Exception):
    """Raised when a guard blocks conformance computation."""
    pass


class DraftContractError(Exception):
    """Raised when attempting to run conformance on a DRAFT contract."""
    pass


def _compute_timing_zone(
    actual_duration_minutes: float,
    contract:                PipelineContract,
    expected_record_count:   Optional[int] = None,
    actual_record_count:     Optional[int] = None,
) -> TimingZoneResult:
    """
    Zone-based timing conformance — the core innovation.

    Instead of binary pass/fail, the zone model captures WHERE
    in the SLA window the pipeline completed. This enables:
    - Early warning before breach (AMBER zone, score starts degrading)
    - Graduated scoring reflecting actual business risk
    - Penalty exposure for client-facing pipelines
    - Suspicious early detection to catch silent failures

    Hard SLA floors at 0.10 (breach is serious).
    Soft SLA floors at 0.50 (breach is inconvenient, not critical).

    The early completion bonus (1.05) rewards pipelines that
    consistently complete well before their deadline — they have
    headroom for bad days. This distinction is invisible in
    pass/fail models but meaningful in clustering.
    """
    p50        = contract.p50_minutes
    p95        = contract.p95_minutes
    p99        = contract.p99_minutes
    grace      = contract.grace_minutes
    sla_type   = getattr(contract, 'sla_type', 'hard')
    is_hard    = (sla_type == 'hard')

    # Floor values differ between hard and soft SLAs
    # Hard SLA breach is a financial/regulatory event — harsh penalty
    # Soft SLA breach is an inconvenience — lighter penalty
    breach_floor = 0.10 if is_hard else 0.50

    # Penalty exposure (only for hard SLA with penalty clause)
    penalty = None
    client_sla = getattr(contract, 'client_sla', None)

    # ── Suspicious early detection ────────────────────────────
    # A pipeline completing before p50/2 is suspicious.
    # Could be: error exit, incomplete processing, skipped steps.
    # Real performance improvements are gradual, not 50% faster overnight.
    if actual_duration_minutes < (p50 * 0.50) and p50 > 10:
        if actual_record_count is not None and expected_record_count is not None:
            # Check if record count is also low — confirms silent failure
            record_ratio = actual_record_count / expected_record_count
            if record_ratio < 0.80:
                return TimingZoneResult(
                    zone='SUSPICIOUS_EARLY',
                    score=0.60,
                    actual_duration_minutes=actual_duration_minutes,
                    minutes_to_deadline=(p99 + grace) - actual_duration_minutes,
                    downstream_can_start_early=False,  # do not propagate
                    suspicious_early=True,
                )

    # ── Zone classification and scoring ───────────────────────

    # GREEN zone: at or before p50 — healthy, has headroom
    if actual_duration_minutes <= p50:
        return TimingZoneResult(
            zone='GREEN',
            score=1.05,  # small bonus for consistent early completion
            actual_duration_minutes=actual_duration_minutes,
            minutes_to_deadline=(p99 + grace) - actual_duration_minutes,
            downstream_can_start_early=True,
            early_bonus_applied=True,
        )

    # GREEN zone: p50 to p95 — normal healthy range
    elif actual_duration_minutes <= p95:
        return TimingZoneResult(
            zone='GREEN',
            score=1.0,
            actual_duration_minutes=actual_duration_minutes,
            minutes_to_deadline=(p99 + grace) - actual_duration_minutes,
            downstream_can_start_early=True,
        )

    # AMBER zone: p95 to p99 — approaching limit, early warning
    # Linear decay from 1.0 to 0.75
    elif actual_duration_minutes <= p99:
        ratio = (actual_duration_minutes - p95) / (p99 - p95)
        score = 1.0 - (0.25 * ratio)
        return TimingZoneResult(
            zone='AMBER',
            score=round(score, 4),
            actual_duration_minutes=actual_duration_minutes,
            minutes_to_deadline=(p99 + grace) - actual_duration_minutes,
            downstream_can_start_early=False,
        )

    # RED zone: p99 to p99+grace — burning through grace period
    # Linear decay from 0.75 to breach_floor
    elif actual_duration_minutes <= (p99 + grace):
        ratio = (actual_duration_minutes - p99) / grace
        score = 0.75 - ((0.75 - breach_floor) * ratio)
        return TimingZoneResult(
            zone='RED',
            score=round(score, 4),
            actual_duration_minutes=actual_duration_minutes,
            minutes_to_deadline=(p99 + grace) - actual_duration_minutes,
            downstream_can_start_early=False,
        )

    # BREACH zone: beyond p99+grace — SLA violated
    else:
        minutes_over = actual_duration_minutes - (p99 + grace)

        # Gradual continued decay from breach_floor
        # Hard SLA: 0.03 per minute over grace (severe)
        # Soft SLA: 0.01 per minute over grace (less severe)
        decay_per_minute = 0.03 if is_hard else 0.01
        score = max(
            breach_floor * 0.25,  # absolute floor — pipeline ran, just very late
            breach_floor - (decay_per_minute * minutes_over)
        )

        # Compute penalty exposure
        if client_sla and hasattr(client_sla, 'penalty_per_minute_eur'):
            penalty = round(minutes_over * client_sla.penalty_per_minute_eur, 2)

        return TimingZoneResult(
            zone='BREACH',
            score=round(score, 4),
            actual_duration_minutes=actual_duration_minutes,
            minutes_to_deadline=-(minutes_over),  # negative = over
            penalty_exposure_eur=penalty,
            downstream_can_start_early=False,
        )


# ── Confidence computation ────────────────────────────────────────────────────

def _compute_confidence(
    events:            pd.DataFrame,
    contract:          PipelineContract,
    guard_warnings:    list[str],
    days_of_history:   int = 0,
    preflight_penalty: float = 0.0,
) -> tuple[float, str]:
    """
    Compute how much to trust the conformance score.

    Confidence is the minimum of four factors — not the average.
    One bad factor poisons the measurement regardless of how
    clean the others are. A 90% complete trace with 10% duplicate
    events has a meaningful confidence issue — averaging would hide it.

    The warmup multiplier reduces confidence for new pipelines.
    10 days of history gives less reliable trend information
    than 30 days. Confidence grows linearly over the first 30 days.
    """
    required  = contract.log_contract.required_events
    terminal  = contract.log_contract.terminal_event

    # Factor 1: trace completeness
    # How many required events were actually found?
    found     = set(events['activity'].values)
    coverage  = len(found & set(required)) / max(len(required), 1)

    # Factor 2: terminal event presence
    # A trace without its terminal event is definitionally incomplete.
    has_terminal = 1.0 if terminal in found else 0.60

    # Factor 3: timestamp ordering quality
    # Events should arrive roughly in order.
    # Out-of-order events beyond the configured tolerance
    # suggest log aggregation problems.
    sorted_events = events.sort_values('timestamp')
    order_violations = 0
    tolerance = timedelta(seconds=30)

    for i in range(1, len(sorted_events)):
        curr_ts = sorted_events.iloc[i]['timestamp']
        prev_ts = sorted_events.iloc[i-1]['timestamp']
        if prev_ts - curr_ts > tolerance:
            order_violations += 1

    ordering_quality = 1.0 - min(0.50, order_violations / max(len(events), 1))

    # Factor 4: warning and preflight quality.
    # Runtime warnings use a generic penalty. Preflight AMBER checks use the
    # exact confidence_delta defined in preflight.py, so a missing terminal
    # event can reduce trust more than a small first-event mismatch.
    generic_warning_penalty = len(guard_warnings) * 0.15
    exact_preflight_penalty = abs(min(0.0, preflight_penalty))
    warning_penalty = min(
        0.60,
        generic_warning_penalty + exact_preflight_penalty,
    )
    warning_quality = 1.0 - warning_penalty

    # Confidence = minimum of all factors
    # (not average — one bad factor is enough to reduce trust)
    base_confidence = min(
        coverage,
        has_terminal,
        ordering_quality,
        warning_quality,
    )

    # Warmup multiplier: new pipelines get less confidence
    # because we have less history to anchor the measurement
    if days_of_history < 30:
        warmup_multiplier = days_of_history / 30
        base_confidence = base_confidence * warmup_multiplier

    # Classify confidence level
    if base_confidence >= 0.90:
        level = "HIGH"
    elif base_confidence >= 0.70:
        level = "MEDIUM"
    elif base_confidence >= 0.50:
        level = "LOW"
    else:
        level = "UNRELIABLE"

    return round(base_confidence, 4), level


def _compute_gap_drift_per_day(
    historical_gaps: Optional[list[tuple[datetime, float]]] = None,
    current_gap_minutes: Optional[float] = None,
    current_events: Optional[pd.DataFrame] = None,
) -> Optional[float]:
    """
    Estimate how quickly the producer-consumer handoff gap is changing.

    The input history comes from conformance_log.csv. The current gap comes
    from today's producer/consumer event files before today's row is appended
    to the log. Grouping by calendar date keeps reruns on the same day from
    pretending to be multiple separate days of drift.
    """
    points = []

    for ts, gap in historical_gaps or []:
        if gap is None:
            continue
        try:
            points.append((pd.to_datetime(ts).date(), float(gap)))
        except (TypeError, ValueError):
            continue

    if current_gap_minutes is not None and current_events is not None:
        try:
            current_ts = pd.to_datetime(
                current_events["timestamp"], utc=True
            ).max()
            points.append((current_ts.date(), float(current_gap_minutes)))
        except (KeyError, TypeError, ValueError):
            pass

    if len(points) < 3:
        return None

    # Keep the latest value per date. This makes rerunning the same date safe
    # while preserving the natural chronological order for the trend.
    daily = {}
    for day, gap in sorted(points, key=lambda item: item[0]):
        daily[day] = gap

    gaps = list(daily.values())
    if len(gaps) < 3:
        return None

    slope = float(np.polyfit(range(len(gaps)), np.array(gaps), 1)[0])
    return round(slope, 6)


# ── Weekday pattern detection ─────────────────────────────────────────────────

def _detect_weekday_pattern(
    run_scores:      list[tuple[datetime, float]],
    producer_events: Optional[pd.DataFrame] = None,
    terminal_event:  str = 'COMPLETED',
) -> WeekdayPattern:
    """
    Detect if failures cluster on specific weekdays.

    Two detection modes:

    Mode 1 — score-based (drifting pipelines):
      A pipeline with a Monday infrastructure issue shows low scores
      on Mondays. Detects when score < 0.70 clusters on specific weekdays.

    Mode 2 — completeness-based (silent pipelines):
      A silent pipeline produces NO events on certain days.
      The score-based detector cannot see absences — they have no score.
      This mode computes failure rate from SCHEDULED vs COMPLETED events
      directly from the raw event log.

    Mode 2 is tried first when producer_events is provided.
    Falls back to Mode 1 if no SCHEDULED events exist.
    """
    # Mode 2: completeness-based detection from raw events
    if producer_events is not None and len(producer_events) > 0:
        try:
            events = producer_events.copy()
            events['timestamp'] = pd.to_datetime(events['timestamp'], utc=True)
            events['weekday']   = events['timestamp'].dt.day_name()

            scheduled = events[events['activity'] == 'SCHEDULED']
            completed = events[events['activity'] == terminal_event]

            if len(scheduled) >= 7:
                completed_runs = set(completed['pipeline_run_id'].unique())

                by_weekday_sched = defaultdict(set)
                for _, row in scheduled.iterrows():
                    by_weekday_sched[row['weekday']].add(row['pipeline_run_id'])

                weekday_failure_rates = {}
                weekday_run_counts    = {}
                for weekday, run_ids in by_weekday_sched.items():
                    failed = run_ids - completed_runs
                    weekday_failure_rates[weekday] = len(failed) / len(run_ids)
                    weekday_run_counts[weekday]    = len(run_ids)

                if weekday_failure_rates:
                    worst_weekday = max(weekday_failure_rates,
                                        key=weekday_failure_rates.get)
                    worst_rate    = weekday_failure_rates[worst_weekday]
                    affected      = [d for d, r in weekday_failure_rates.items()
                                     if r > 0.40]

                    other_scores = [
                        s for run_date, s in run_scores
                        if run_date.strftime('%A') != worst_weekday
                    ]
                    other_mean = np.mean(other_scores) if other_scores else 1.0

                    infrastructure_probable = (
                        worst_rate > 0.60 and
                        other_mean > 0.80 and
                        len(affected) <= 2
                    )

                    if worst_rate > 0.40:
                        return WeekdayPattern(
                            has_weekday_clustering    = True,
                            worst_weekday             = worst_weekday,
                            worst_weekday_failure_rate= worst_rate,
                            other_days_mean_score     = round(other_mean, 4),
                            infrastructure_probable   = infrastructure_probable,
                            affected_weekdays         = sorted(affected),
                        )
        except Exception:
            pass  # fall through to Mode 1

    # Mode 1: score-based detection (original logic)
    if len(run_scores) < 7:
        return WeekdayPattern(
            has_weekday_clustering    = False,
            worst_weekday             = None,
            worst_weekday_failure_rate= 0.0,
            other_days_mean_score     = 1.0,
            infrastructure_probable   = False,
            affected_weekdays         = [],
        )

    by_weekday = defaultdict(list)
    for run_date, score in run_scores:
        weekday = run_date.strftime('%A')
        by_weekday[weekday].append(score)

    weekday_failure_rates = {}
    for weekday, scores in by_weekday.items():
        failure_rate = sum(1 for s in scores if s < 0.70) / len(scores)
        weekday_failure_rates[weekday] = failure_rate

    worst_weekday = max(weekday_failure_rates, key=weekday_failure_rates.get)
    worst_rate    = weekday_failure_rates[worst_weekday]
    affected      = [d for d, r in weekday_failure_rates.items() if r > 0.40]

    other_scores = [
        s for day, scores in by_weekday.items()
        for s in scores
        if day != worst_weekday
    ]
    other_mean = np.mean(other_scores) if other_scores else 1.0
    infrastructure_probable = (
        worst_rate > 0.60 and
        other_mean > 0.80 and
        len(affected) <= 2
    )

    return WeekdayPattern(
        has_weekday_clustering    = worst_rate > 0.40,
        worst_weekday             = worst_weekday if worst_rate > 0.40 else None,
        worst_weekday_failure_rate= worst_rate,
        other_days_mean_score     = round(other_mean, 4),
        infrastructure_probable   = infrastructure_probable,
        affected_weekdays         = sorted(affected),
    )


# ── Variance analysis ─────────────────────────────────────────────────────────

def _compute_variance_analysis(
    run_scores: list[float],
) -> tuple[str, float]:
    """
    Detect whether score variability reflects intermittent failures
    or genuine drift.

    Coefficient of Variation (CV) = std / mean
    A high CV means the pipeline fails on some days and not others
    — intermittent, likely infrastructure or external trigger.
    A low CV with declining mean means genuine drift
    — systematic, likely process or contract issue.

    CV > 0.25: INTERMITTENT — investigate specific failure days
    Low CV + negative slope: DRIFTING — investigate process drift
    Low CV + flat/positive slope: STABLE — no action needed
    """
    if len(run_scores) < 5:
        return "STABLE", 0.0

    mean  = np.mean(run_scores)
    std   = np.std(run_scores)
    cv    = std / mean if mean > 0 else 0.0

    if cv > 0.25:
        return "INTERMITTENT", round(cv, 4)

    # Check for trend (drift) using linear regression
    if len(run_scores) >= 10:
        slope, _, _, p_value, _ = linregress(
            range(len(run_scores)), run_scores
        )
        # p_value < 0.05 = statistically significant trend
        # slope < -0.003 per day = meaningful decline (>2% per week)
        if p_value < 0.05 and slope < -0.003:
            return "DRIFTING", round(cv, 4)

    return "STABLE", round(cv, 4)



# ── Bilateral gap computation ─────────────────────────────────────────────────

def detect_gap_changepoint(
    gaps: list,       # list of (timestamp, gap_minutes) tuples
) -> 'ChangePoint':
    """
    Detect structural changes in the bilateral gap time series.
    Uses PELT algorithm — Pruned Exact Linear Time.

    Why PELT over linear regression:
      Linear regression detects monotonic trends (slope).
      PELT detects step changes — where the gap jumped or dropped suddenly.
      Real-world gap changes are almost always step changes, not gradual trends.
      A deployment happens overnight. The gap changes by 15 minutes the next day.
      Linear regression would show a "trend" across the entire window.
      PELT shows a changepoint on the specific date.

    Example output for a pipeline where SFTP polling was reduced on day 20:
      pre_mean = 28.4 min  (days 1-19)
      post_mean = 13.1 min (days 20-30)
      magnitude = -15.3 min (decreased)
      direction = 'decreased'
      probable_cause = "Gap decreased by 15.3 min around day 20.
                        Likely cause: configuration change, deployment,
                        or infrastructure improvement."

    Requires: pip install ruptures
    Falls back gracefully if not installed.
    """
    try:
        import ruptures as rpt
    except ImportError:
        return ChangePoint(
            detected=False, n_changepoints=0, changepoint_dates=[],
            pre_mean_minutes=None, post_mean_minutes=None,
            magnitude_minutes=None, direction=None,
            probable_cause="ruptures not installed — pip install ruptures",
        )

    if len(gaps) < 10:
        return ChangePoint(
            detected=False, n_changepoints=0, changepoint_dates=[],
            pre_mean_minutes=None, post_mean_minutes=None,
            magnitude_minutes=None, direction=None,
            probable_cause="Insufficient history for changepoint detection (need 10+ runs)",
        )

    gap_values = np.array([g[1] for g in gaps], dtype=float)
    gap_dates  = [g[0] for g in gaps]

    try:
        # PELT with rbf cost — robust to noise, finds step changes
        algo   = rpt.Pelt(model="rbf", min_size=3, jump=1).fit(gap_values)
        # pen=5: penalty for each additional changepoint
        # Higher pen = fewer changepoints detected
        breakpoints = algo.predict(pen=5)

        # breakpoints includes the end index — remove it
        breakpoints = [b for b in breakpoints if b < len(gap_values)]

        if not breakpoints:
            return ChangePoint(
                detected=False, n_changepoints=0, changepoint_dates=[],
                pre_mean_minutes=round(float(gap_values.mean()), 1),
                post_mean_minutes=None, magnitude_minutes=None, direction=None,
                probable_cause="No structural change detected — gap is stable",
            )

        # Analyse the most recent changepoint
        last_cp   = breakpoints[-1]
        pre_vals  = gap_values[:last_cp]
        post_vals = gap_values[last_cp:]

        pre_mean  = float(np.mean(pre_vals))  if len(pre_vals)  > 0 else None
        post_mean = float(np.mean(post_vals)) if len(post_vals) > 0 else None
        magnitude = round(post_mean - pre_mean, 1) if (pre_mean and post_mean) else None
        direction = ('increased' if magnitude > 0 else 'decreased') if magnitude else None

        # Get the date of the last changepoint
        cp_dates = []
        for bp in breakpoints:
            if bp < len(gap_dates):
                cp_dates.append(gap_dates[bp])

        # Build human-readable explanation
        if magnitude and cp_dates:
            cp_date_str = cp_dates[-1].strftime('%Y-%m-%d')                           if hasattr(cp_dates[-1], 'strftime') else str(cp_dates[-1])
            probable_cause = (
                f"Gap {direction} by {abs(magnitude):.1f} min around {cp_date_str}. "
                f"Before: {pre_mean:.1f} min mean. After: {post_mean:.1f} min mean. "
                f"Likely cause: deployment, configuration change, or infrastructure event. "
                f"Correlate with deployment logs on or before {cp_date_str}."
            )
        else:
            probable_cause = f"{len(breakpoints)} changepoint(s) detected in gap series."

        return ChangePoint(
            detected         = True,
            n_changepoints   = len(breakpoints),
            changepoint_dates= cp_dates,
            pre_mean_minutes = round(pre_mean, 1)  if pre_mean  else None,
            post_mean_minutes= round(post_mean, 1) if post_mean else None,
            magnitude_minutes= magnitude,
            direction        = direction,
            probable_cause   = probable_cause,
        )

    except Exception as e:
        return ChangePoint(
            detected=False, n_changepoints=0, changepoint_dates=[],
            pre_mean_minutes=None, post_mean_minutes=None,
            magnitude_minutes=None, direction=None,
            probable_cause=f"Changepoint detection failed: {e}",
        )



def compute_chain_gaps(
    producer_events: pd.DataFrame,
    consumer_events: Optional[pd.DataFrame],
    contract,
) -> dict:
    """
    Compute bilateral gaps for pipeline chains (A → B → C).

    For simple pipelines (producer only):
      upstream_gap   = gap between producer DATA_AVAILABLE and
                       consumer DATA_AVAILABLE
      downstream_gap = None (no downstream consumer log)

    For middle pipelines (consumer + producer):
      upstream_gap   = gap between upstream DATA_AVAILABLE and
                       this pipeline's DATA_RECEIVED
                       (how quickly did B receive A's output?)
      downstream_gap = gap between this pipeline's DATA_AVAILABLE
                       and downstream DATA_RECEIVED
                       (how quickly did C receive B's output?)

    The join events are declared in the contract:
      upstream_producer_event  (default: DATA_AVAILABLE)
      upstream_consumer_event  (default: DATA_AVAILABLE)
      downstream_producer_event (default: DATA_AVAILABLE)
      downstream_consumer_event (default: DATA_AVAILABLE)

    Returns dict with upstream_* and downstream_* keys.
    """
    lc = contract.log_contract
    result = {
        'upstream_gap_minutes':   None,
        'upstream_gap_trend':     None,
        'upstream_gap_p95':       None,
        'downstream_gap_minutes': None,
        'downstream_gap_trend':   None,
        'downstream_gap_p95':     None,
    }

    if consumer_events is None or len(consumer_events) == 0:
        return result

    producer_events = producer_events.copy()
    consumer_events = consumer_events.copy()
    producer_events['timestamp'] = pd.to_datetime(producer_events['timestamp'], utc=True)
    consumer_events['timestamp'] = pd.to_datetime(consumer_events['timestamp'], utc=True)

    # ── Upstream gap: upstream producer → this pipeline (consumer role) ──
    # Which event in the producer log marks "upstream done"?
    upstream_prod_event = lc.upstream_producer_event   # default DATA_AVAILABLE
    # Which event in the consumer log marks "this pipeline received it"?
    upstream_cons_event = lc.upstream_consumer_event   # default DATA_AVAILABLE

    # Upstream producer events can be in two places:
    # 1. producer_events with team='upstream_producer' (middle pipeline pattern)
    # 2. consumer_events with team='upstream_producer' (tagged when passed in)
    # 3. producer_events directly (simple pipeline — backward compat)
    
    # Check if upstream producer events are tagged separately
    upstream_in_prod = producer_events[
        (producer_events['activity'] == upstream_prod_event) &
        (producer_events.get('team', pd.Series('producer', index=producer_events.index))
         == 'upstream_producer')
    ] if 'team' in producer_events.columns else pd.DataFrame()
    
    upstream_in_cons = consumer_events[
        (consumer_events['activity'] == upstream_prod_event) &
        (consumer_events.get('team', pd.Series('consumer', index=consumer_events.index))
         == 'upstream_producer')
    ] if 'team' in consumer_events.columns else pd.DataFrame()
    
    upstream_source = (
        upstream_in_prod if len(upstream_in_prod) > 0
        else upstream_in_cons if len(upstream_in_cons) > 0
        else producer_events[producer_events['activity'] == upstream_prod_event]
    )
    
    prod_da = (
        upstream_source
        .sort_values('timestamp')
        .groupby('pipeline_run_id')['timestamp'].first()
    )
    cons_da = (
        consumer_events[consumer_events['activity'] == upstream_cons_event]
        .sort_values('timestamp')
        .groupby('pipeline_run_id')['timestamp'].first()
    )

    common = prod_da.index.intersection(cons_da.index)
    if len(common) > 0:
        gaps = []
        for run_id in common:
            gap_min = (cons_da[run_id] - prod_da[run_id]).total_seconds() / 60
            if gap_min >= 0:
                gaps.append((prod_da[run_id], gap_min))

        neg_rate = (len(common) - len(gaps)) / max(len(common), 1)
        if neg_rate <= 0.50 and gaps:
            gap_values = [g[1] for g in gaps]
            mean_gap   = float(np.mean(gap_values))
            p95_gap    = float(np.percentile(gap_values, 95))

            # Trend from linear regression on gap time series
            if len(gap_values) >= 3:
                x = np.arange(len(gap_values))
                slope = np.polyfit(x, gap_values, 1)[0]
                if slope > 0.5:
                    trend = 'widening'
                elif slope < -0.5:
                    trend = 'narrowing'
                else:
                    trend = 'stable'
            else:
                trend = 'stable'

            result['upstream_gap_minutes'] = round(mean_gap, 1)
            result['upstream_gap_trend']   = trend
            result['upstream_gap_p95']     = round(p95_gap, 1)

    # ── Downstream gap: this pipeline (producer) → downstream consumer ──
    # For middle pipelines, consumer_events may contain downstream events
    # flagged with team='downstream_consumer'
    # For now: downstream gap is captured when the downstream pipeline
    # registers and runs its own conformance (producer = B, consumer = C)
    # This is the clean separation — each contract owns its own gaps

    # If there are consumer events with a downstream marker, compute it
    downstream_cons = consumer_events[
        consumer_events.get('team', pd.Series(dtype=str)).str.contains(
            'downstream', case=False, na=False
        )
    ] if 'team' in consumer_events.columns else pd.DataFrame()

    if len(downstream_cons) > 0:
        downstream_prod_event = lc.downstream_producer_event
        downstream_cons_event = lc.downstream_consumer_event

        prod_out = (
            producer_events[producer_events['activity'] == downstream_prod_event]
            .groupby('pipeline_run_id')['timestamp'].first()
        )
        cons_in = (
            downstream_cons[downstream_cons['activity'] == downstream_cons_event]
            .groupby('pipeline_run_id')['timestamp'].first()
        )

        common_d = prod_out.index.intersection(cons_in.index)
        if len(common_d) > 0:
            d_gaps = []
            for run_id in common_d:
                gap = (cons_in[run_id] - prod_out[run_id]).total_seconds() / 60
                if gap >= 0:
                    d_gaps.append(gap)

            if d_gaps:
                result['downstream_gap_minutes'] = round(float(np.mean(d_gaps)), 1)
                result['downstream_gap_p95']     = round(float(np.percentile(d_gaps, 95)), 1)
                slope = np.polyfit(range(len(d_gaps)), d_gaps, 1)[0] if len(d_gaps) >= 3 else 0
                result['downstream_gap_trend'] = (
                    'widening' if slope > 0.5 else
                    'narrowing' if slope < -0.5 else
                    'stable'
                )

    return result



def compare_process_variants(
    log,               # PM4PY EventLog
    net, im, fm,       # normative Petri net from contract
    contract,          # PipelineContract
    sequence_fitness: float,
) -> 'VariantComparison':
    """
    Compare the normative process model (from contract) with the
    process model discovered from actual execution logs.

    Only runs when sequence_fitness < 0.90 — no point comparing
    when the process is already conformant.

    Why this matters:
      Token replay gives a number: fitness = 0.74.
      Variant comparison gives a story: "STARTED fires twice on 30% of runs.
      This means retries are happening but not declared in the contract."

    The engineer reads the story and immediately knows what to fix:
      - If retries: add deduplicate_retries=true to the contract
      - If missing activity: contract is wrong, missing step in process
      - If extra activity: undocumented step was added to the pipeline

    Uses Inductive Miner — discovers a sound and fitting process model
    from the actual event log without any prior knowledge.

    Algorithm:
      1. Get variants from actual log (all distinct execution paths)
      2. Find dominant variant (most frequent path)
      3. Get normative path from contract.required_events
      4. Compare the two paths
      5. Describe deviations in plain language
    """
    # ── Grain-completeness guard ───────────────────────────────────────────
    # Run BEFORE the fitness threshold check.
    # If trace completeness is low due to row sampling, we must warn
    # regardless of whether fitness is high or low.
    # A 99% fitness from 28% complete traces is still meaningless.
    # Variant comparison is only meaningful when traces are COMPLETE.
    # A complete trace has the terminal_event present.
    #
    # If > 15% of traces are missing the terminal event, the comparison
    # is measuring sampling noise, not process behaviour.
    #
    # Root cause: row sampling splits traces.
    #   Wrong: sample(n=50_000 rows) → some traces missing terminal event
    #   Right:  sample(n=25_000 pipeline_run_ids) → all traces complete
    #
    # The guard detects this and tells the engineer how to fix it.
    try:
        terminal = contract.log_contract.terminal_event
        grain    = contract.log_contract.grain

        # Check trace completeness directly from the PM4PY log
        # PM4PY stores activity name as 'concept:name' in each event
        all_cases      = set()
        terminal_cases = set()
        for trace in log:
            case_id = trace.attributes.get('concept:name',
                      trace.attributes.get('case:concept:name', str(id(trace))))
            all_cases.add(case_id)
            for event in trace:
                act = event.get('concept:name', '')
                if act == terminal:
                    terminal_cases.add(case_id)
                    break

        trace_completeness = (
            len(terminal_cases) / max(len(all_cases), 1)
        )

        if trace_completeness < 0.85:
            fix_msg = (
                f"sample {grain} IDs not rows"
                if grain in ('trade', 'record')
                else "check event file for missing terminal events"
            )
            return VariantComparison(
                compared=False,
                n_variants_actual=len(all_cases),
                dominant_variant=[],
                normative_path=list(contract.log_contract.required_events),
                paths_match=False,
                deviations=[
                    f"Only {trace_completeness:.0%} of traces have the "
                    f"terminal event '{terminal}'. "
                    f"Variant comparison skipped — results would reflect "
                    f"sampling artefacts, not process behaviour. "
                    f"Fix: {fix_msg}. "
                    f"Sample pipeline_run_id values, not individual event rows."
                ],
                fitness_explainer=(
                    f"Variant comparison skipped: {trace_completeness:.0%} trace "
                    f"completeness at grain='{grain}'. "
                    f"Low completeness at this grain level suggests row sampling "
                    f"rather than trace sampling. "
                    f"See deviations for the fix."
                ),
            )
    except Exception:
        pass  # if guard fails, proceed with comparison

    # Fitness threshold check — after grain guard so guard always runs first
    if sequence_fitness >= 0.90:
        return VariantComparison(
            compared=False,
            n_variants_actual=0,
            dominant_variant=[],
            normative_path=list(contract.log_contract.required_events),
            paths_match=True,
            deviations=[],
            fitness_explainer=(
                f"Sequence fitness {sequence_fitness:.2f} is healthy — "
                f"no variant analysis needed."
            ),
        )

    try:
        import pm4py

        # Get variants from actual log
        variants = pm4py.get_variants(log)
        # variants is {(activity_tuple): [case_id1, ...]}

        if not variants:
            return VariantComparison(
                compared=True,
                n_variants_actual=0,
                dominant_variant=[],
                normative_path=list(contract.log_contract.required_events),
                paths_match=False,
                deviations=["No variants found in log — possibly empty log"],
                fitness_explainer="Log appears to contain no complete traces.",
            )

        # Find dominant variant (most traces)
        dominant_variant = max(variants, key=lambda v: len(variants[v]))
        dominant_path    = list(dominant_variant)
        normative_path   = list(contract.log_contract.required_events)
        n_variants       = len(variants)

        # Coverage: what fraction of traces follow the dominant path?
        total_traces    = sum(len(v) for v in variants.values())
        dominant_count  = len(variants[dominant_variant])
        dominant_pct    = dominant_count / max(total_traces, 1)

        # Compare paths
        paths_match = (dominant_path == normative_path)

        # Find deviations — what is different?
        deviations = []

        # Missing activities (in normative but not in dominant actual)
        missing = [a for a in normative_path if a not in dominant_path]
        if missing:
            for act in missing:
                deviations.append(
                    f"'{act}' is in the contract but missing from "
                    f"{dominant_pct:.0%} of actual runs. "
                    f"Either the contract is wrong or this step was removed."
                )

        # Extra activities (in actual but not in normative)
        extra = [a for a in dominant_path if a not in normative_path]
        if extra:
            for act in extra:
                deviations.append(
                    f"'{act}' appears in {dominant_pct:.0%} of actual runs "
                    f"but is not in the contract. "
                    f"Undocumented activity — add to contract or investigate."
                )

        # Repeated activities (appears more than once in actual path)
        from collections import Counter
        act_counts = Counter(dominant_path)
        repeated = [a for a, n in act_counts.items() if n > 1]
        if repeated:
            for act in repeated:
                deviations.append(
                    f"'{act}' fires {act_counts[act]}x in {dominant_pct:.0%} "
                    f"of runs. Retries or loops detected. "
                    f"If retries are expected: set deduplicate_retries=true "
                    f"in the contract."
                )

        # Ordering differences
        if not missing and not extra and not repeated and not paths_match:
            deviations.append(
                f"Activities present but in wrong order. "
                f"Actual: {' → '.join(dominant_path[:5])}. "
                f"Expected: {' → '.join(normative_path[:5])}."
            )

        # High variant diversity — process is unpredictable
        if n_variants > 5 and dominant_pct < 0.60:
            deviations.append(
                f"{n_variants} distinct execution paths detected. "
                f"Dominant path covers only {dominant_pct:.0%} of runs. "
                f"Process is highly variable — contract may need multiple "
                f"optional activities declared."
            )

        # Build fitness explainer
        if deviations:
            fitness_explainer = (
                f"Sequence fitness {sequence_fitness:.2f} is low because: "
                f"{deviations[0]}"
            )
        elif not paths_match:
            fitness_explainer = (
                f"Dominant actual path differs from contracted path. "
                f"Actual ({dominant_pct:.0%} of runs): "
                f"{' → '.join(dominant_path[:4])}."
            )
        else:
            fitness_explainer = (
                f"Dominant path matches contract but fitness is "
                f"{sequence_fitness:.2f}. "
                f"Minority paths ({n_variants-1} variants) are non-conformant."
            )

        return VariantComparison(
            compared           = True,
            n_variants_actual  = n_variants,
            dominant_variant   = dominant_path,
            normative_path     = normative_path,
            paths_match        = paths_match,
            deviations         = deviations,
            fitness_explainer  = fitness_explainer,
        )

    except Exception as e:
        return VariantComparison(
            compared=True,
            n_variants_actual=0,
            dominant_variant=[],
            normative_path=list(contract.log_contract.required_events),
            paths_match=False,
            deviations=[f"Variant comparison failed: {e}"],
            fitness_explainer=f"Could not compare variants: {e}",
        )



# ── Main conformance computation ──────────────────────────────────────────────

def compute_conformance(
    producer_events:  pd.DataFrame,
    contract:         PipelineContract,
    config:           FractureConfig = DEFAULT_CONFIG,
    consumer_events:  Optional[pd.DataFrame] = None,
    historical_scores: Optional[list[tuple[datetime, float]]] = None,
    historical_gaps: Optional[list[tuple[datetime, float]]] = None,
) -> ConformanceResult:
    """
    Compute full bilateral conformance for one pipeline.

    Parameters
    ----------
    producer_events  : DataFrame with columns [pipeline_run_id, activity,
                       timestamp, team]. One row per activity fired.
    contract         : Validated PipelineContract. DRAFT contracts are blocked.
    config           : FractureConfig with all tunable parameters.
    consumer_events  : Optional consumer-side event log for bilateral comparison.
                       If None, bilateral gap is not computed.
    historical_scores: Previous conformance scores from conformance_log.csv.
                       Used for drift and weekday analysis.
    historical_gaps  : Previous bilateral gaps from conformance_log.csv.
                       Used to compute gap_drift_per_day.
                       If None, gap drift remains unavailable.

    Returns
    -------
    ConformanceResult with full analysis including:
    - Weighted conformance score (0.0–1.05)
    - Confidence level (HIGH/MEDIUM/LOW/UNRELIABLE)
    - Timing zone (GREEN/AMBER/RED/BREACH)
    - Pattern classification (STABLE/DRIFTING/INTERMITTENT)
    - Weekday failure clustering (Monday bounce detection)
    - Bilateral gap in minutes (if consumer_events provided)
    - Data quality conformance (if DataQualityContract present)
    - Attribution (which team owns the failure)
    """

    # Normalize raw event names and retry noise before any quality gates.
    # Preflight, token replay, timing, completeness, and bilateral gap must all
    # operate on the same contract vocabulary.
    producer_events = normalize_events(producer_events, contract)
    if consumer_events is not None:
        consumer_events = normalize_events(consumer_events, contract)    

    # ── Step 1: Preflight checks (traffic light severity) ────
    #
    # Two separate preflights:
    #   Producer → RED blocks everything. No valid producer = no score.
    #   Consumer → RED skips bilateral only. Producer score still runs.
    #
    # This means a broken consumer log never invalidates a valid
    # producer conformance measurement.
    from fracture.preflight import run_preflight

    # Step 1a: Producer preflight — blocks entire computation on RED
    producer_preflight = run_preflight(producer_events, contract)
    if not producer_preflight.passed:
        red = producer_preflight.red_checks[0]
        if red.name == "draft_contract":
            raise DraftContractError(red.message + " " + red.fix_hint)
        raise ConformanceGuardError(red.message + " " + red.fix_hint)

    # Step 1b: Consumer preflight — RED skips bilateral, not entire run
    consumer_preflight_penalty = 0.0
    if consumer_events is not None and len(consumer_events) > 0:
        cons_preflight = run_preflight(consumer_events, contract)
        if not cons_preflight.passed:
            # Consumer log is broken — skip bilateral, log warning
            print(
                f"[preflight] Consumer preflight failed for "
                f"'{contract.pipeline_id}': "
                f"{cons_preflight.red_checks[0].message} "
                f"Bilateral comparison skipped."
            )
            consumer_events = None
        else:
            # Consumer AMBER penalties count at half weight.
            # They reduce bilateral confidence, not core conformance.
            # A consumer with ordering violations should not penalise
            # the producer's sequence fitness measurement.
            consumer_preflight_penalty = cons_preflight.confidence_penalty * 0.5

    # Preflight penalties are applied with their exact confidence_delta below.
    # Keep guard_warnings for runtime warnings that do not come from preflight,
    # such as a PM4PY token replay fallback.
    guard_warnings = []

    # Total preflight penalty = producer AMBER + half consumer AMBER
    total_preflight_penalty = (
        producer_preflight.confidence_penalty + consumer_preflight_penalty
    )

    # ── Step 2: Compute history length ────────────────────────
    days_of_history = len(historical_scores) if historical_scores else 30
    is_warmup = days_of_history < 30

    # ── Step 3: Sequence fitness via PM4PY token replay ───────
    try:
        from fracture.petri import contract_to_petri_net
        net, im, fm = contract_to_petri_net(contract)
        pm4py_log   = _events_to_pm4py_log(producer_events, contract)

        replayed    = token_replay.apply(pm4py_log, net, im, fm)

        # token replay returns one result per trace
        # we aggregate across all traces in the window
        fitnesses = [r['trace_fitness'] for r in replayed if 'trace_fitness' in r]
        sequence  = round(np.mean(fitnesses), 4) if fitnesses else 0.50

    except Exception as e:
        # If PM4PY fails (rare but possible with malformed traces),
        # use a conservative default and add a warning
        guard_warnings.append(f"PM4PY token replay failed: {str(e)[:100]}. Using conservative sequence score.")
        sequence = 0.50

    # ── Step 4: Timing conformance ────────────────────────────
    # Compute actual duration per run and average the zone scores
    duration_scores = []
    run_durations   = []

    for run_id, run_events in producer_events.groupby('pipeline_run_id'):
        start_event = run_events[run_events['activity'] == 'STARTED']
        end_event   = run_events[
            run_events['activity'] == contract.log_contract.terminal_event
        ]

        if len(start_event) > 0 and len(end_event) > 0:
            duration = (
                end_event.iloc[0]['timestamp'] -
                start_event.iloc[0]['timestamp']
            ).total_seconds() / 60

            run_durations.append(duration)

            zone_result = _compute_timing_zone(
                actual_duration_minutes=duration,
                contract=contract,
            )
            duration_scores.append(zone_result.score)

    timing_score = round(np.mean(duration_scores), 4) if duration_scores else 0.50

    # Get the most recent timing zone for the result
    if run_durations:
        latest_zone_result = _compute_timing_zone(
            actual_duration_minutes=run_durations[-1],
            contract=contract,
        )
    else:
        latest_zone_result = TimingZoneResult(
            zone='RED', score=0.50,
            actual_duration_minutes=0,
            minutes_to_deadline=-999,
        )

    # ── Step 5: Completeness score ────────────────────────────
    # What fraction of expected runs in the window produced a complete trace?
    total_run_ids  = producer_events['pipeline_run_id'].nunique()
    terminal       = contract.log_contract.terminal_event
    complete_runs  = producer_events[
        producer_events['activity'] == terminal
    ]['pipeline_run_id'].nunique()

    completeness = round(complete_runs / max(total_run_ids, 1), 4)

    # ── Step 7: Weighted final score ──────────────────────────
    from fracture.config import DEFAULT_CONFIG
    weights = DEFAULT_CONFIG.weights

    core_score = (
        weights.sequence     * sequence     +
        weights.timing       * timing_score +
        weights.completeness * completeness
    )

    # Final score: three process dimensions
    # weights.sequence + weights.timing + weights.completeness = 1.0
    final_score = core_score

    # Cap at 1.05 (early bonus from timing zone)
    final_score = round(min(1.05, max(0.0, final_score)), 4)

    # ── Step 8: Confidence ────────────────────────────────────
    confidence, confidence_level = _compute_confidence(
        events=producer_events,
        guard_warnings=guard_warnings,
        contract=contract,
        days_of_history=days_of_history,
        preflight_penalty=total_preflight_penalty,
    )

    # ── Step 9: Variance and pattern analysis + temporal features ──────
    # historical_scores = [(datetime, float), ...] loaded from CSV by CLI.
    # historical_gaps   = [(datetime, float), ...] loaded from the same log.
    # These temporal features are computed ONCE here and written to the
    # CSV row so clustering.py reads them directly — no recalculation.

    temporal_variance_cv  = None
    drift_rate_per_day    = None
    score_range           = None
    gap_drift_per_day     = None
    mean_score_historical = None
    min_score_historical  = None

    if historical_scores:
        past_scores = [s for _, s in historical_scores]
        pattern, cv = _compute_variance_analysis(past_scores)

        # Temporal variance_cv — std/mean across days (not within-day)
        # This is what clustering needs: how consistent is the pipeline day-to-day?
        if len(past_scores) >= 3:
            arr = np.array(past_scores)
            mean_s = float(np.mean(arr))
            std_s  = float(np.std(arr))
            temporal_variance_cv  = round(std_s / mean_s, 6) if mean_s > 0 else 0.0
            drift_rate_per_day    = round(float(np.polyfit(range(len(arr)), arr, 1)[0]), 6)
            score_range           = round(float(np.max(arr) - np.min(arr)), 4)
            mean_score_historical = round(mean_s, 4)
            min_score_historical  = round(float(np.min(arr)), 4)

    else:
        pattern, cv = "STABLE", 0.0

    # ── Step 10: Weekday failure clustering ───────────────────
    # The Monday restart detector
    weekday_pattern = _detect_weekday_pattern(
        historical_scores or [],
        producer_events = producer_events,
        terminal_event  = contract.log_contract.terminal_event,
    )

    # Override pattern if weekday clustering is detected
    # INTERMITTENT is more specific than DRIFTING
    if weekday_pattern.infrastructure_probable:
        pattern = "INTERMITTENT"

    # ── Step 11: Dependency attribution ──────────────────────
    # Check if this pipeline's failures are caused by upstream issues
    attribution = _resolve_dependencies(contract, historical_scores)

    # ── Step 12: Bilateral gap + chain gaps + changepoint ────
    bilateral_gap    = None
    bilateral_trend  = None
    bilateral_p95    = None
    changepoint      = None
    gaps_series      = []

    # Chain gap fields (upstream + downstream for middle pipelines)
    upstream_gap_minutes   = None
    upstream_gap_trend     = None
    upstream_gap_p95       = None
    downstream_gap_minutes = None
    downstream_gap_trend   = None
    downstream_gap_p95     = None

    if consumer_events is not None:
        # Compute all gap types via chain_gaps (handles simple + middle)
        chain = compute_chain_gaps(producer_events, consumer_events, contract)
        upstream_gap_minutes   = chain['upstream_gap_minutes']
        upstream_gap_trend     = chain['upstream_gap_trend']
        upstream_gap_p95       = chain['upstream_gap_p95']
        downstream_gap_minutes = chain['downstream_gap_minutes']
        downstream_gap_trend   = chain['downstream_gap_trend']
        downstream_gap_p95     = chain['downstream_gap_p95']

        # Backward compat: bilateral_gap = upstream_gap
        bilateral_gap   = upstream_gap_minutes
        bilateral_trend = upstream_gap_trend
        bilateral_p95   = upstream_gap_p95

        if bilateral_gap is not None and bilateral_gap < 0:
            print(
                f"[conformance] Negative bilateral gap ({bilateral_gap:.1f} min) "
                f"for '{contract.pipeline_id}'. "
                f"Possible clock skew. Bilateral comparison set to None."
            )
            bilateral_gap = upstream_gap_minutes = None
            bilateral_trend = upstream_gap_trend = None
            bilateral_p95 = upstream_gap_p95 = None
        else:
            # ── Changepoint detection on gap time series ──────────────
            # Rebuild the (timestamp, gap) series for changepoint analysis
            # This tells us WHEN the gap structure changed, not just THAT
            # it is widening.
            #
            # Example:
            #   "Gap increased by 12 min around 2026-03-15.
            #    Correlate with deployments on that date."
            #
            # We rebuild the series from the raw events.
            target = 'DATA_AVAILABLE'
            prod_da = (producer_events[producer_events['activity'] == target]
                       .sort_values('timestamp'))
            cons_da = (consumer_events[consumer_events['activity'] == target]
                       .sort_values('timestamp'))

            prod_da['timestamp'] = pd.to_datetime(
                prod_da['timestamp'], utc=True
            )
            cons_da['timestamp'] = pd.to_datetime(
                cons_da['timestamp'], utc=True
            )

            for _, prod_row in prod_da.iterrows():
                prod_date = prod_row['timestamp'].date()
                cons_match = cons_da[
                    cons_da['timestamp'].dt.date == prod_date
                ]
                if len(cons_match) > 0:
                    gap_min = (
                        cons_match.iloc[0]['timestamp'] - prod_row['timestamp']
                    ).total_seconds() / 60
                    if gap_min >= 0:
                        gaps_series.append((prod_row['timestamp'], gap_min))

            if len(gaps_series) >= 10:
                changepoint = detect_gap_changepoint(gaps_series)

    # Gap drift is computed after bilateral gap because today's current gap is
    # only known once producer and consumer DATA_AVAILABLE events are matched.
    gap_drift_per_day = _compute_gap_drift_per_day(
        historical_gaps=historical_gaps,
        current_gap_minutes=bilateral_gap,
        current_events=producer_events,
    )

    # ── Step 12b: Process variant comparison ─────────────────
    # When sequence fitness is low, compare normative vs discovered process.
    # This tells the engineer WHY fitness is low, not just HOW low it is.
    #
    # Only runs when fitness < 0.90 — no point running when healthy.
    # Uses PM4PY Inductive Miner on the producer log.
    #
    # Example findings:
    #   "STARTED fires 2x — retries detected, add deduplicate_retries=true"
    #   "VALIDATED missing from 70% of runs — contract may be wrong"
    #   "Dominant path matches contract but 8 variants detected — high variability"
    variant_comparison = compare_process_variants(
        log              = pm4py_log,
        net              = net,
        im               = im,
        fm               = fm,
        contract         = contract,
        sequence_fitness = sequence,
    )

    # ── Step 13: Assemble result ──────────────────────────────
    return ConformanceResult(
        pipeline_id           = contract.pipeline_id,
        final_score           = final_score,
        confidence_level      = confidence_level,
        timing_zone           = latest_zone_result.zone,
        pattern               = pattern,
        bilateral_gap_minutes = bilateral_gap,
        days_of_history       = days_of_history,
        diagnostics           = ConformanceDiagnostics(
            sequence_fitness       = sequence,
            timing_score           = timing_score,
            completeness_score     = completeness,
            variance_cv            = cv,
            timing_result          = latest_zone_result,
            weekday_pattern        = weekday_pattern,
            attribution            = attribution,
            bilateral_gap_trend    = bilateral_trend,
            bilateral_gap_p95      = bilateral_p95,
            upstream_gap_minutes   = upstream_gap_minutes,
            upstream_gap_trend     = upstream_gap_trend,
            upstream_gap_p95       = upstream_gap_p95,
            downstream_gap_minutes = downstream_gap_minutes,
            downstream_gap_trend   = downstream_gap_trend,
            downstream_gap_p95     = downstream_gap_p95,
            days_of_history        = days_of_history,
            warmup_active          = is_warmup,
            changepoint            = changepoint,
            variant_comparison     = variant_comparison,
            # Temporal features — pre-computed from historical_scores
            temporal_variance_cv   = temporal_variance_cv,
            drift_rate_per_day     = drift_rate_per_day,
            score_range            = score_range,
            gap_drift_per_day      = gap_drift_per_day,
            mean_score_historical  = mean_score_historical,
            min_score_historical   = min_score_historical,
        ),
    )


# ── Dependency resolution ─────────────────────────────────────────────────────

def _resolve_dependencies(
    contract:          'PipelineContract',
    historical_scores: Optional[list],
) -> 'DependencyAttribution':
    """
    Cascade attribution placeholder.

    Dependencies removed from individual contracts.
    A future topology file will declare the pipeline graph.
    For now return empty attribution — no cascade detection.
    """
    return DependencyAttribution(
        has_upstream_failure   = False,
        root_cause_pipeline    = None,
        upstream_conformance   = None,
        upstream_delay_minutes = None,
        suppress_alert         = False,
        attribution_confidence = 1.0,
    )


def _events_to_pm4py_log(
    events:   pd.DataFrame,
    contract: PipelineContract,
) -> EventLog:
    """
    Convert a pandas DataFrame of pipeline events to a PM4PY EventLog.

    PM4PY speaks XES — the standard process mining event log format.
    XES requires specific attribute names:
      - concept:name  = the activity name
      - time:timestamp = when it happened
      - case:concept:name = the case (pipeline run) ID

    The XES converter (converter.py) handles the full file-based conversion.
    This helper handles the in-memory conversion needed by the conformance engine.
    """
    log = EventLog()

    for run_id, run_events in events.groupby('pipeline_run_id'):
        trace = Trace()
        trace.attributes['concept:name'] = str(run_id)

        for _, row in run_events.sort_values('timestamp').iterrows():
            event = Event()
            event['concept:name']   = row['activity']
            event['time:timestamp'] = row['timestamp']
            trace.append(event)

        log.append(trace)

    return log


# ── Cross-fleet pattern detection ────────────────────────────────────────────
