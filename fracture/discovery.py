"""
fracture/discovery.py

Phase 5 process-discovery helpers.

This file answers the first discovery question:
"What process actually happened in the event log?"

V1 deliberately starts with variant discovery because it is stable, easy to
explain, and directly useful before adding heavier PM4PY DFG/Inductive Miner
exports later.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from fracture.ingest import FRACTURE_COLUMNS, normalize_events


@dataclass
class DiscoverySummary:
    """
    Small structured result returned by discovery.

    Keeping discovery output structured makes it reusable by CLI, dashboard,
    tests, and later static report exports.
    """

    pipeline_id: str
    log_side: str
    status: str
    n_traces: int
    n_variants: int
    dominant_variant: list[str]
    dominant_variant_count: int
    dominant_variant_frequency: float
    contract_path: list[str]
    comparison_path: list[str]
    dominant_matches_contract: bool
    deviations: list[str]

    def as_dict(self) -> dict:
        """Return plain Python values for JSON/report/dashboard use."""
        return {
            "pipeline_id": self.pipeline_id,
            "log_side": self.log_side,
            "status": self.status,
            "n_traces": self.n_traces,
            "n_variants": self.n_variants,
            "dominant_variant": self.dominant_variant,
            "dominant_variant_count": self.dominant_variant_count,
            "dominant_variant_frequency": self.dominant_variant_frequency,
            "contract_path": self.contract_path,
            "comparison_path": self.comparison_path,
            "dominant_matches_contract": self.dominant_matches_contract,
            "deviations": self.deviations,
        }


def _validate_discovery_input(events: pd.DataFrame) -> list[str]:
    """
    Check that discovery has the same four columns as conformance.

    Discovery should use the same event shape as the rest of Fracture:
    pipeline_run_id, activity, timestamp, team.
    """
    missing = [col for col in FRACTURE_COLUMNS if col not in events.columns]
    if missing:
        raise ValueError(
            "Discovery input is missing required columns: "
            + ", ".join(missing)
        )
    return missing


def _contract_path_for_side(contract, log_side: str) -> list[str]:
    """
    Choose the expected path for the selected log side.

    Producer discovery uses the normal required_events path. Middle-pipeline
    consumer discovery can use consumer_required_events when the contract has it.
    """
    if (
        log_side == "consumer"
        and getattr(contract.log_contract, "consumer_required_events", None)
    ):
        return list(contract.log_contract.consumer_required_events)

    return list(contract.log_contract.required_events)


def _comparison_path(contract_path: list[str], optional_activities: Iterable[str], variant: list[str]) -> list[str]:
    """
    Build the contract path that should be compared to this actual variant.

    Optional activities can be legitimately skipped, so if an optional activity
    is absent from the actual variant we remove it from the expected comparison.
    """
    optional = set(optional_activities or [])
    variant_set = set(variant)

    return [
        activity
        for activity in contract_path
        if activity not in optional or activity in variant_set
    ]


def _ordered_variant(trace_df: pd.DataFrame) -> tuple[str, ...]:
    """
    Convert one pipeline_run_id trace into a chronological activity tuple.

    Repeated activities are intentionally preserved here. If the contract sets
    deduplicate_retries=True, normalize_events already removes retry noise first.
    """
    ordered = trace_df.sort_values("timestamp")
    return tuple(ordered["activity"].astype(str).tolist())


def _variant_deviations(
    variant: list[str],
    contract_path: list[str],
    comparison_path: list[str],
    optional_activities: Iterable[str],
) -> list[str]:
    """
    Explain how the dominant actual path differs from the contract path.

    These strings are designed for CLI/dashboard display, not just test asserts.
    """
    deviations = []
    optional = set(optional_activities or [])
    variant_counts = Counter(variant)
    variant_set = set(variant)
    contract_set = set(contract_path)

    missing_required = [
        activity
        for activity in contract_path
        if activity not in optional and activity not in variant_set
    ]
    extra_activities = [
        activity
        for activity in variant
        if activity not in contract_set
    ]
    repeated_activities = [
        activity
        for activity, count in variant_counts.items()
        if count > 1
    ]

    if missing_required:
        deviations.append(
            "missing required activities: " + ", ".join(missing_required)
        )

    if extra_activities:
        deviations.append(
            "extra activities outside contract: " + ", ".join(extra_activities)
        )

    if repeated_activities:
        deviations.append(
            "repeated activities observed: " + ", ".join(repeated_activities)
        )

    # If the same activities appear but in the wrong order, call that out.
    if not missing_required and not extra_activities and variant != comparison_path:
        deviations.append("dominant variant order differs from contract")

    return deviations


def discover_variants(
    events: pd.DataFrame,
    contract,
    log_side: str = "producer",
) -> DiscoverySummary:
    """
    Discover actual process variants from a producer or consumer event log.

    One trace is one pipeline_run_id. A variant is the ordered activity sequence
    observed for that trace.
    """
    if events is None or events.empty:
        return DiscoverySummary(
            pipeline_id=contract.pipeline_id,
            log_side=log_side,
            status="no_input",
            n_traces=0,
            n_variants=0,
            dominant_variant=[],
            dominant_variant_count=0,
            dominant_variant_frequency=0.0,
            contract_path=_contract_path_for_side(contract, log_side),
            comparison_path=[],
            dominant_matches_contract=False,
            deviations=["no input events available for discovery"],
        )

    _validate_discovery_input(events)

    # Use the same normalization rules as conformance so discovery and token
    # replay speak the same activity vocabulary.
    normalized = normalize_events(events, contract)
    normalized = normalized.copy()
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"])

    variants = Counter()
    for _run_id, trace_df in normalized.groupby("pipeline_run_id"):
        variant = _ordered_variant(trace_df)
        if variant:
            variants[variant] += 1

    if not variants:
        return DiscoverySummary(
            pipeline_id=contract.pipeline_id,
            log_side=log_side,
            status="no_traces",
            n_traces=0,
            n_variants=0,
            dominant_variant=[],
            dominant_variant_count=0,
            dominant_variant_frequency=0.0,
            contract_path=_contract_path_for_side(contract, log_side),
            comparison_path=[],
            dominant_matches_contract=False,
            deviations=["no valid traces found after normalization"],
        )

    dominant_variant_tuple, dominant_count = variants.most_common(1)[0]
    dominant_variant = list(dominant_variant_tuple)
    contract_path = _contract_path_for_side(contract, log_side)
    comparison_path = _comparison_path(
        contract_path=contract_path,
        optional_activities=contract.log_contract.optional_activities,
        variant=dominant_variant,
    )
    deviations = _variant_deviations(
        variant=dominant_variant,
        contract_path=contract_path,
        comparison_path=comparison_path,
        optional_activities=contract.log_contract.optional_activities,
    )

    n_traces = sum(variants.values())
    dominant_matches_contract = dominant_variant == comparison_path

    return DiscoverySummary(
        pipeline_id=contract.pipeline_id,
        log_side=log_side,
        status="ok",
        n_traces=n_traces,
        n_variants=len(variants),
        dominant_variant=dominant_variant,
        dominant_variant_count=dominant_count,
        dominant_variant_frequency=dominant_count / n_traces,
        contract_path=contract_path,
        comparison_path=comparison_path,
        dominant_matches_contract=dominant_matches_contract,
        deviations=deviations,
    )


def format_discovery_summary(summary: DiscoverySummary) -> str:
    """
    Format discovery output for the CLI.

    The dashboard can use as_dict(); CLI gets a compact human-readable summary.
    """
    dominant = " -> ".join(summary.dominant_variant) or "n/a"
    contract = " -> ".join(summary.contract_path) or "n/a"
    comparison = " -> ".join(summary.comparison_path) or "n/a"
    match = "yes" if summary.dominant_matches_contract else "no"
    frequency = f"{summary.dominant_variant_frequency:.0%}"

    lines = [
        f"Process Discovery: {summary.pipeline_id}",
        f"Log side: {summary.log_side}",
        f"Status: {summary.status}",
        f"Traces: {summary.n_traces}",
        f"Variants: {summary.n_variants}",
        "",
        "Dominant variant:",
        f"  {dominant}",
        f"  count={summary.dominant_variant_count}, frequency={frequency}",
        "",
        "Contract path:",
        f"  {contract}",
        "",
        "Comparison path:",
        f"  {comparison}",
        "",
        f"Dominant path matches contract: {match}",
    ]

    if summary.deviations:
        lines.extend(["", "Deviation summary:"])
        lines.extend([f"  - {item}" for item in summary.deviations])

    return "\n".join(lines)
