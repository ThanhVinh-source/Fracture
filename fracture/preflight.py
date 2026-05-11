"""
fracture/preflight.py

Preflight checks before conformance computation.

Traffic light severity model — not guards, not blockers.
Each check has a severity and a clear consequence:

  RED    → hard stop, ConformanceGuardError raised
           computing a score would be meaningless or wrong
           the caller must handle this before proceeding

  AMBER  → reduce confidence, continue
           the score exists but interpret with caution
           confidence is reduced by the delta for each AMBER check

  GREEN  → no issue, full confidence
           this check passed cleanly

Design: Chain of Responsibility.
Each check is an independent object with a single responsibility.
Adding a new check means adding a new class — not modifying existing ones.
RED checks stop the chain. AMBER checks accumulate.

Why not "guards":
  Guards implies all checks block. Most do not.
  Preflight checks is the accurate term — some are advisory,
  some are mandatory. The severity field makes the distinction explicit.

Connection to confidence scoring:
  The preflight result feeds directly into _compute_confidence().
  Each AMBER check reduces confidence by its stated delta.
  RED checks never reach confidence computation — they raise first.
  The minimum-of-four-factors confidence formula then applies
  the accumulated AMBER penalty as one of its four factors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd


# ── Severity ──────────────────────────────────────────────────────────────────

class Severity:
    RED   = "RED"    # hard stop
    AMBER = "AMBER"  # reduce confidence, continue
    GREEN = "GREEN"  # clean


# ── Result objects ────────────────────────────────────────────────────────────

@dataclass
class PreflightCheck:
    """Result of one preflight check."""
    name:               str
    severity:           str            # RED | AMBER | GREEN
    message:            str            # human-readable, actionable
    confidence_delta:   float = 0.0   # 0 for RED/GREEN, negative for AMBER
    fix_hint:           str   = ""     # specific fix instruction


@dataclass
class PreflightResult:
    """
    Aggregated result of all preflight checks.

    Passed means no RED checks fired — conformance can proceed.
    The confidence_penalty is the sum of all AMBER deltas and
    feeds into the confidence scoring step.
    """
    passed:              bool
    checks:              list[PreflightCheck] = field(default_factory=list)

    @property
    def red_checks(self) -> list[PreflightCheck]:
        return [c for c in self.checks if c.severity == Severity.RED]

    @property
    def amber_checks(self) -> list[PreflightCheck]:
        return [c for c in self.checks if c.severity == Severity.AMBER]

    @property
    def confidence_penalty(self) -> float:
        """Sum of all AMBER confidence deltas. Always negative or zero."""
        return sum(c.confidence_delta for c in self.amber_checks)

    def summary(self) -> str:
        if not self.passed:
            return f"BLOCKED: {self.red_checks[0].message}"
        if self.amber_checks:
            return (
                f"{len(self.amber_checks)} warning(s), "
                f"confidence reduced by {abs(self.confidence_penalty):.0%}"
            )
        return "CLEAN"


# ── Individual checks ─────────────────────────────────────────────────────────

class EmptyTraceCheck:
    """
    RED — no events extracted at all.

    No events means no information means no score.
    Most likely cause: log extraction patterns do not match
    the log format, or the pipeline did not run today.
    """
    name = "empty_trace"

    def evaluate(
        self,
        events:   pd.DataFrame,
        contract: object,
    ) -> Optional[PreflightCheck]:
        if len(events) == 0:
            return PreflightCheck(
                name     = self.name,
                severity = Severity.RED,
                message  = "No events extracted from logs.",
                fix_hint = (
                    "Check that your extraction job ran today and wrote "
                    "a file to inputs/{pipeline_id}/. "
                    "Run: fracture template --pipeline-id {pipeline_id} "
                    "to see the expected format."
                ),
            )
        return None


class DraftContractCheck:
    """
    RED — contract is in DRAFT status.

    DRAFT contracts are bootstrapped descriptions of current behaviour.
    Running conformance against them is circular measurement —
    you are measuring reality against a description of itself.
    A human must review and set status: active.
    """
    name = "draft_contract"

    def evaluate(
        self,
        events:   pd.DataFrame,
        contract: object,
    ) -> Optional[PreflightCheck]:
        from fracture.schema import ContractStatus
        if contract.status == ContractStatus.DRAFT:
            return PreflightCheck(
                name     = self.name,
                severity = Severity.RED,
                message  = (
                    f"Contract '{contract.pipeline_id}' is DRAFT. "
                    f"Conformance blocked."
                ),
                fix_hint = (
                    f"Review the bootstrapped values in "
                    f"contracts/{contract.pipeline_id}.yaml "
                    f"and change status to 'active'."
                ),
            )
        return None


class AllSameTimestampCheck:
    """
    RED — all events have identical timestamps.

    Log aggregation collapsed all events to one moment.
    Token replay cannot determine ordering from identical timestamps.
    Any fitness score produced would be meaningless.
    """
    name = "all_same_timestamp"

    def evaluate(
        self,
        events:   pd.DataFrame,
        contract: object,
    ) -> Optional[PreflightCheck]:
        if len(events) > 1 and events['timestamp'].nunique() == 1:
            return PreflightCheck(
                name     = self.name,
                severity = Severity.RED,
                message  = "All events have identical timestamps.",
                fix_hint = (
                    "Log aggregation collapsed timestamps. "
                    "Ensure your extraction preserves full datetime "
                    "including time component. "
                    "Splunk: | eval ts=strftime(_time,'%Y-%m-%dT%H:%M:%SZ') "
                    "ES: use @timestamp not date-only fields."
                ),
            )
        return None


class FutureTimestampCheck:
    """
    RED — events have timestamps in the future.

    Timezone misconfiguration produces events hours or days ahead.
    Timing zone analysis will be completely wrong.
    Two-hour tolerance handles minor clock skew.
    """
    name = "future_timestamps"

    def evaluate(
        self,
        events:   pd.DataFrame,
        contract: object,
    ) -> Optional[PreflightCheck]:
        now     = pd.Timestamp.now(tz='UTC')
        cutoff  = now + pd.Timedelta(hours=2)

        # Handle both tz-aware and tz-naive timestamps
        ts = events['timestamp']
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize('UTC')

        future = (ts > cutoff).sum()
        if future > 0:
            return PreflightCheck(
                name     = self.name,
                severity = Severity.RED,
                message  = f"{future} events have future timestamps.",
                fix_hint = (
                    "Timezone misconfiguration. "
                    "Ensure all timestamps are UTC or include "
                    "timezone offset. "
                    "Mixed timezones between producer and consumer "
                    "also cause negative bilateral gaps."
                ),
            )
        return None


class MissingTerminalEventCheck:
    """
    AMBER — terminal event not found in any trace.
    Confidence -0.40.

    Pipeline may still be running, completion event was dropped,
    or the terminal_event in the contract does not match the
    actual activity name in the logs.
    Score is computed but is partial — completeness will be low.
    """
    name             = "missing_terminal"
    confidence_delta = -0.40

    def evaluate(
        self,
        events:   pd.DataFrame,
        contract: object,
    ) -> Optional[PreflightCheck]:
        terminal = contract.log_contract.terminal_event
        if terminal not in events['activity'].values:
            return PreflightCheck(
                name             = self.name,
                severity         = Severity.AMBER,
                message          = (
                    f"Terminal event '{terminal}' not found. "
                    f"Pipeline may still be running."
                ),
                confidence_delta = self.confidence_delta,
                fix_hint         = (
                    f"Check whether '{terminal}' matches your log's "
                    f"actual activity name. "
                    f"If the pipeline is still running, wait for completion. "
                    f"If events are missing, check your extraction query."
                ),
            )
        return None


class LowEventCoverageCheck:
    """
    AMBER — fewer than 50% of required events found.
    Confidence -0.20 per missing required event, max -0.40.

    Partial trace. Some required milestones were not extracted.
    Score is partial — missing events produce missing tokens in replay.
    """
    name             = "low_event_coverage"

    def evaluate(
        self,
        events:   pd.DataFrame,
        contract: object,
    ) -> Optional[PreflightCheck]:
        required = contract.log_contract.required_events
        found    = set(events['activity'].values)
        missing  = set(required) - found
        coverage = len(found & set(required)) / max(len(required), 1)

        if coverage < 0.50:
            penalty = min(0.40, len(missing) * 0.20)
            return PreflightCheck(
                name             = self.name,
                severity         = Severity.AMBER,
                message          = (
                    f"Only {coverage:.0%} of required events found. "
                    f"Missing: {', '.join(sorted(missing))}."
                ),
                confidence_delta = -penalty,
                fix_hint         = (
                    f"Check activity_name_map in your contract. "
                    f"Your logs may use different names for these events. "
                    f"Run: fracture template --pipeline-id "
                    f"{contract.pipeline_id} to see expected names."
                ),
            )
        return None


class OrderingViolationCheck:
    """
    AMBER — events arrived significantly out of order.
    Confidence -0.15.

    Token replay sorts within traces so the fitness score is not
    directly affected. But ordering violations indicate log
    collection problems that may also affect timestamps.
    30-second tolerance for minor clock skew.
    """
    name             = "ordering_violations"
    confidence_delta = -0.15
    tolerance_seconds = 30

    def evaluate(
        self,
        events:   pd.DataFrame,
        contract: object,
    ) -> Optional[PreflightCheck]:
        if len(events) < 2:
            return None

        sorted_events   = events.sort_values('timestamp')
        ts              = sorted_events['timestamp']
        tolerance       = pd.Timedelta(seconds=self.tolerance_seconds)

        # Count pairs where time difference is suspiciously small
        # (events that arrived at almost the same time but from
        # different activities — suggests batched log shipping)
        diffs           = ts.diff().dropna()
        violations      = (diffs < pd.Timedelta(0)).sum()
        # With tz-aware timestamps the diff might behave differently
        # Just check if any consecutive timestamps went backwards
        backward        = 0
        ts_list         = sorted_events['timestamp'].tolist()
        for i in range(1, len(ts_list)):
            try:
                if ts_list[i] < ts_list[i-1]:
                    backward += 1
            except Exception:
                pass

        violation_rate = backward / max(len(events) - 1, 1)

        if violation_rate > 0.20:
            return PreflightCheck(
                name             = self.name,
                severity         = Severity.AMBER,
                message          = (
                    f"{violation_rate:.0%} of events arrived out of order."
                ),
                confidence_delta = self.confidence_delta,
                fix_hint         = (
                    "Check log shipping configuration. "
                    "Events from different systems may have clock skew. "
                    "Ensure all sources use UTC timestamps."
                ),
            )
        return None


class FirstEventMismatchCheck:
    """
    AMBER — first required event exists but was not the first event seen.
    Confidence -0.10.

    Not hardcoded to SCHEDULED — works for any pipeline type.
    Only fires when the first declared required event IS present
    in the log but was not the first event seen.
    Suggests early events were missed during extraction.
    """
    name             = "first_event_mismatch"
    confidence_delta = -0.10

    def evaluate(
        self,
        events:   pd.DataFrame,
        contract: object,
    ) -> Optional[PreflightCheck]:
        required = contract.log_contract.required_events
        if not required:
            return None

        found          = set(events['activity'].values)
        first_required = required[0]

        # Only fires when the first required event IS present
        # but was not the first event we see
        if first_required not in found:
            return None

        first_seen = (
            events.sort_values('timestamp')
                  .iloc[0]['activity']
        )

        if first_seen != first_required:
            return PreflightCheck(
                name             = self.name,
                severity         = Severity.AMBER,
                message          = (
                    f"First event seen: '{first_seen}'. "
                    f"Expected '{first_required}' to come first."
                ),
                confidence_delta = self.confidence_delta,
                fix_hint         = (
                    "Log extraction may be missing early events. "
                    "Widen your time window or check extraction filters."
                ),
            )
        return None


# ── Chain ─────────────────────────────────────────────────────────────────────

class PreflightChain:
    """
    Runs all preflight checks in order.

    RED checks stop the chain immediately —
    no point running further checks if the data is unusable.

    AMBER checks accumulate — all warnings are collected
    even if multiple fire. The confidence engine reads the
    full list to compute the total confidence penalty.

    Adding a new check: create a new class above with an
    evaluate() method returning Optional[PreflightCheck].
    Add it to self.checks. No other changes needed.
    """

    def __init__(self):
        # Order matters for RED checks — they stop the chain.
        # Most fundamental checks first.
        self.checks = [
            EmptyTraceCheck(),
            DraftContractCheck(),
            AllSameTimestampCheck(),
            FutureTimestampCheck(),
            MissingTerminalEventCheck(),
            LowEventCoverageCheck(),
            OrderingViolationCheck(),
            FirstEventMismatchCheck(),
        ]

    def run(
        self,
        events:   pd.DataFrame,
        contract: object,
    ) -> PreflightResult:
        """
        Run all checks. Stop on first RED.
        Collect all AMBERs.
        Return PreflightResult.
        """
        fired = []

        for check in self.checks:
            try:
                result = check.evaluate(events, contract)
            except Exception:
                # A check that crashes should not crash conformance.
                # Skip it silently — the score will be computed
                # without this check's input.
                continue

            if result is None:
                continue

            fired.append(result)

            if result.severity == Severity.RED:
                # Chain stops here.
                # No partial scores from unusable data.
                return PreflightResult(passed=False, checks=fired)

        return PreflightResult(passed=True, checks=fired)


# ── Convenience ───────────────────────────────────────────────────────────────

# Module-level singleton — same chain for all pipelines in a fleet run.
# Tests can construct a custom PreflightChain with different checks.
DEFAULT_CHAIN = PreflightChain()


def run_preflight(
    events:   pd.DataFrame,
    contract: object,
    chain:    PreflightChain = DEFAULT_CHAIN,
) -> PreflightResult:
    """
    Run preflight checks on events and contract.

    This is the function conformance.py calls.
    Replaces _run_guards() entirely.

    Returns PreflightResult with:
      .passed             → whether conformance can proceed
      .confidence_penalty → how much to reduce confidence (negative)
      .red_checks         → blocking failures
      .amber_checks       → warnings
      .summary()          → one-line human description
    """
    return chain.run(events, contract)
