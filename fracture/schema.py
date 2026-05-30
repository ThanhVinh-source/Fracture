"""
fracture/schema.py

The contract. What two teams agreed a pipeline should do.

Design principle: minimum fields Fracture genuinely cannot
function without. Everything else is derived, defaulted,
or belongs somewhere else.

13 fields. A team writes this in five minutes.
A team lead reviews it in thirty seconds.

Validators fire at definition time — not at 3am.
Wrong contract caught before a single event is processed.
"""

from __future__ import annotations

import warnings
from enum import Enum
from pathlib import Path
from typing import Optional, Optional

import yaml
from pydantic import BaseModel, Field, model_validator, field_validator


# ── Enums ─────────────────────────────────────────────────────────────────────

class ContractStatus(str, Enum):
    DRAFT      = "draft"       # bootstrapped — not yet reviewed
    ACTIVE     = "active"      # reviewed — conformance runs
    DEPRECATED = "deprecated"  # no longer monitored


class Criticality(str, Enum):
    HIGH   = "high"    # client SLA or regulatory
    MEDIUM = "medium"  # internal dependency
    LOW    = "low"     # best-effort


class Transport(str, Enum):
    PLAINTEXT = "plaintext"
    JSON_LOG  = "json_log"
    CSV       = "csv"
    SFTP      = "sftp"
    PARQUET   = "parquet"


class NotificationChannel(str, Enum):
    SLACK = "slack"
    EMAIL = "email"
    TEAMS = "teams"
    LOG   = "log"


# ── Sub-models ────────────────────────────────────────────────────────────────

class Notification(BaseModel):
    channel: NotificationChannel
    target:  str


class LogContract(BaseModel):
    """
    How Fracture finds the pipeline's event file.

    Teams produce 4-column parquet/CSV and drop it in
    inputs/{pipeline_id}/. This section tells Fracture
    what format to expect and how to normalise names.
    """
    transport:    Transport = Transport.PLAINTEXT
    source_path:  str

    required_events: list[str] = Field(
        default=["SCHEDULED", "STARTED", "COMPLETED", "DATA_AVAILABLE"],
        description=(
            "Ordered business milestones Fracture expects. "
            "Become transitions in the Petri net. Order matters."
        )
    )


    # Some process steps are legitimate branches, not failures.
    # Example: VALIDATED may run for full loads but be skipped for incremental loads.
    # Petri net construction will add silent bypass arcs for these activities.
    optional_activities: list[str] = Field(
        default_factory=list,
        description=(
            "Activities that may be legitimately skipped. "
            "Fracture adds silent bypass arcs for these activities "
            "in the contract-derived Petri net."
        )
    )

    terminal_event: str = Field(
        default="COMPLETED",
        description=(
            "Event that marks successful completion. "
            "Used for completeness scoring. "
            "Must be in required_events."
        )
    )

    activity_name_map: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Maps your system's activity names to Fracture vocabulary. "
            "Example: {'job_complete': 'COMPLETED'} "
            "Only needed if your names differ from required_events."
        )
    )

    grain: str = Field(
        default='pipeline',
        description=(
            "Unit of conformance measurement. "
            "pipeline: one run of the full pipeline (default). "
            "trade:    one individual transaction or record. "
            "record:   one row-level processing unit. "
            "Every analysis — token replay, variant comparison, "
            "weekday detection — is only valid when analysis grain "
            "matches the declared grain. Mixing grains produces "
            "misleading measurements."
        )
    )

    parent_grain: Optional[str] = Field(
        default=None,
        description=(
            "pipeline_id of the parent-grain contract. "
            "Links trade-level to batch-level contract. "
            "Enables cross-grain: batch healthy, trades broken."
        )
    )

    deduplicate_retries: bool = Field(
        default=False,
        description=(
            "When True, duplicate activities per run are collapsed "
            "to the last occurrence before token replay. "
            "Use for Airflow pipelines with retry=True."
        )
    )

    # ── Multi-process chain support ───────────────────────────────────────
    # For pipelines that act as BOTH consumer (of an upstream process)
    # AND producer (for a downstream process).
    #
    # Example: A → B → C
    #   B receives from A, validates A's data, then produces for C.
    #   B fires consumer events (DATA_RECEIVED, VALIDATION_DONE) AND
    #   producer events (SCHEDULED, STARTED, COMPLETED, DATA_AVAILABLE).
    #
    # upstream_join:   how to measure the A→B gap
    # downstream_join: how to measure the B→C gap
    # consumer_required_events: B's consumer-side process sequence
    # consumer_terminal_event:  what marks B has validated upstream data

    consumer_required_events: list[str] = Field(
        default_factory=list,
        description=(
            "Event sequence for this pipeline's consumer role. "
            "Only needed for middle pipelines (consumer + producer). "
            "Example: ['DATA_RECEIVED', 'VALIDATION_DONE']. "
            "Empty list means this pipeline is producer-only (default)."
        )
    )

    consumer_terminal_event: str = Field(
        default='',
        description=(
            "Terminal event in the consumer sequence. "
            "Marks that upstream data was received and validated. "
            "Example: 'VALIDATION_DONE'. "
            "Empty string = no consumer sequence defined."
        )
    )

    upstream_producer_event: str = Field(
        default='DATA_AVAILABLE',
        description=(
            "Activity name in the UPSTREAM producer log that marks "
            "upstream data is ready. Used to compute the upstream gap. "
            "Default: DATA_AVAILABLE (standard Fracture vocabulary)."
        )
    )

    upstream_consumer_event: str = Field(
        default='DATA_AVAILABLE',
        description=(
            "Activity name in THIS pipeline's consumer log that marks "
            "upstream data was received. "
            "For middle pipelines: set to DATA_RECEIVED. "
            "Default: DATA_AVAILABLE (backward compatible)."
        )
    )

    downstream_producer_event: str = Field(
        default='DATA_AVAILABLE',
        description=(
            "Activity name in THIS pipeline's producer log that marks "
            "output is ready for downstream. "
            "Default: DATA_AVAILABLE."
        )
    )

    downstream_consumer_event: str = Field(
        default='DATA_AVAILABLE',
        description=(
            "Activity name in the DOWNSTREAM consumer log that marks "
            "data was received from this pipeline. "
            "Default: DATA_AVAILABLE (backward compatible)."
        )
    )

    @property
    def is_middle_pipeline(self) -> bool:
        """True if this pipeline acts as both consumer and producer."""
        return bool(self.consumer_required_events)

    @field_validator("terminal_event", mode="before")
    @classmethod
    def terminal_in_required(cls, v, info):
        required = (info.data or {}).get(
            "required_events",
            ["SCHEDULED", "STARTED", "COMPLETED", "DATA_AVAILABLE"]
        )
        if required and v not in required:
            raise ValueError(
                f"terminal_event '{v}' must be in required_events {required}."
            )
        return v
    
    @model_validator(mode="after")
    # Validators are used to block incorrect contracts right from the moment they read the YAML code / create the object.
    def validate_optional_activities(self):
        # Optional activities only make sense if they exist in the process model.
        # If an activity is not in required_events, there is no Petri net transition
        # where Fracture can attach a bypass arc.
        unknown = set(self.optional_activities) - set(self.required_events)
        if unknown:
            raise ValueError(
                f"optional_activities must be in required_events. "
                f"Unknown: {sorted(unknown)}"
            )

        # The terminal event drives completeness scoring.
        # Making it optional can hide incomplete runs, so warn for V1 instead of
        # failing hard to preserve compatibility with existing tests/demo cases.
        if self.terminal_event in self.optional_activities:
            warnings.warn(
                f"terminal_event '{self.terminal_event}' is marked optional. "
                "This can make completeness/conformance interpretation misleading.",
                UserWarning,
            )

        return self


# ── Main contract ─────────────────────────────────────────────────────────────

class PipelineContract(BaseModel):
    """
    The bilateral agreement between producer and consumer.

    13 fields. Write it in five minutes.
    Fracture does the rest.

    Validators enforce correctness at definition time:
      p50 <= p95 <= p99
      p99 + grace fits within the window
      producer_team != consumer_team
      HIGH criticality requires notifications
      grace > 30% of window raises a warning

    DRAFT contracts block conformance until a human
    reviews and sets status: active.
    """

    # Identity
    pipeline_id:   str = Field(..., pattern=r'^[a-z0-9_-]+$')
    owner:         str = Field(..., pattern=r'^[^@]+@[^@]+\.[^@]+$')

    # Teams — bilateral comparison needs two distinct parties
    producer_team: str
    consumer_team: str

    # SLA window
    expected_start: str = Field(..., pattern=r'^\d{2}:\d{2}$')
    expected_end:   str = Field(..., pattern=r'^\d{2}:\d{2}$')
    grace_minutes:  int = Field(..., ge=0, le=120)

    # Percentiles — bootstrapped from logs, never guessed
    p50_minutes:    int = Field(..., ge=1)
    p95_minutes:    int = Field(..., ge=1)
    p99_minutes:    int = Field(..., ge=1)

    # Operational
    criticality:    Criticality = Criticality.MEDIUM
    notifications:  list[Notification] = Field(default_factory=list)
    status:         ContractStatus = ContractStatus.DRAFT

    # How to find the event file
    log_contract:   LogContract

    # ── Validators ─────────────────────────────────────────────

    @model_validator(mode='after')
    def check_percentile_order(self) -> 'PipelineContract':
        if not (self.p50_minutes <= self.p95_minutes <= self.p99_minutes):
            raise ValueError(
                f"Percentiles must be ordered: "
                f"p50({self.p50_minutes}) <= "
                f"p95({self.p95_minutes}) <= "
                f"p99({self.p99_minutes})."
            )
        return self

    @model_validator(mode='after')
    def check_sla_fits_window(self) -> 'PipelineContract':
        window   = self.window_minutes()
        required = self.p99_minutes + self.grace_minutes
        if required > window:
            raise ValueError(
                f"p99({self.p99_minutes}) + grace({self.grace_minutes}) "
                f"= {required} min exceeds the window ({window} min). "
                f"Extend expected_end, reduce p99, or reduce grace."
            )
        return self

    @model_validator(mode='after')
    def check_teams_differ(self) -> 'PipelineContract':
        if self.producer_team == self.consumer_team:
            raise ValueError(
                f"producer_team and consumer_team are both "
                f"'{self.producer_team}'. "
                f"Bilateral comparison requires two distinct teams."
            )
        return self

    @model_validator(mode='after')
    def check_high_criticality_notification(self) -> 'PipelineContract':
        if self.criticality == Criticality.HIGH and not self.notifications:
            raise ValueError(
                f"Pipeline '{self.pipeline_id}' is HIGH criticality "
                f"but has no notification targets. Add at least one."
            )
        return self

    @model_validator(mode='after')
    def warn_grace_padding(self) -> 'PipelineContract':
        window = self.window_minutes()
        if window > 0 and (self.grace_minutes / window) > 0.30:
            warnings.warn(
                f"'{self.pipeline_id}': grace ({self.grace_minutes}m) is "
                f"{self.grace_minutes/window:.0%} of window ({window}m). "
                f"Grace above 30% suggests SLA padding. "
                f"Fix the pipeline instead.",
                UserWarning,
                stacklevel=2,
            )
        return self

    # ── Derived ────────────────────────────────────────────────

    def window_minutes(self) -> int:
        sh, sm = map(int, self.expected_start.split(":"))
        eh, em = map(int, self.expected_end.split(":"))
        start  = sh * 60 + sm
        end    = eh * 60 + em
        if end <= start:
            end += 24 * 60
        return end - start

    def effective_deadline_minutes(self) -> int:
        return self.p99_minutes + self.grace_minutes

    def is_active(self) -> bool:
        return self.status == ContractStatus.ACTIVE

    def is_draft(self) -> bool:
        return self.status == ContractStatus.DRAFT

    def timing_summary(self) -> str:
        return (
            f"p50={self.p50_minutes}m p95={self.p95_minutes}m "
            f"p99={self.p99_minutes}m grace={self.grace_minutes}m "
            f"window={self.window_minutes()}m"
        )


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_contract(path: str) -> PipelineContract:
    """
    Load and validate a contract from YAML or JSON.
    YAML is a superset of JSON — both work.
    Fails loud with a clear fix message.
    """
    raw = Path(path).read_text()
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as e:
        raise ValueError(f"Cannot parse '{path}': {e}") from e

    try:
        return PipelineContract(**data)
    except Exception as e:
        raise ValueError(
            f"Contract validation failed for '{path}':\n{e}"
        ) from e


def validate_contract_file(path: str) -> dict:
    """
    Validate without running conformance.
    Used by: fracture validate contracts/pipeline.yaml
    """
    import warnings as _w
    caught = []
    with _w.catch_warnings(record=True) as w:
        _w.simplefilter("always")
        try:
            c = load_contract(path)
            caught = [str(x.message) for x in w]
            return {
                "valid":       True,
                "pipeline_id": c.pipeline_id,
                "status":      c.status.value,
                "timing":      c.timing_summary(),
                "warnings":    caught,
            }
        except ValueError as e:
            return {"valid": False, "errors": [str(e)]}
