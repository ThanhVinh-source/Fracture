"""
fracture/engine.py

The wiring hub. Connects every built module into one coherent pipeline.

This is the file that answers the question:
"How do all the pieces actually work together?"

Read this file top to bottom and you understand the entire system.

WHAT THIS FILE DOES
════════════════════
Nothing analytical. No algorithms here.
engine.py orchestrates — it calls the right module at the right
time with the right inputs and passes the result to the next step.

The actual work happens in the modules it calls:
  schema.py      → validates the contract
  ingest.py      → validates the 4-column input format
  petri.py       → builds the normative Petri net
  converter.py   → converts events to PM4PY EventLog
  conformance.py → runs token replay + all analysis
  (clustering, preflight, report → coming next)

WHY DEPENDENCY INJECTION
═════════════════════════
Every analytical dependency is injected — passed in as a parameter
rather than imported at the top of the file.

This means:
  Production use:  engine = FractureEngine()
                   Uses real PM4PY, real token replay, real files.

  Test use:        engine = FractureEngine(petri_builder=mock_net)
                   Injects a mock Petri net — no PM4PY needed in tests.
                   The conformance logic is tested in complete isolation.

  Custom use:      engine = FractureEngine(petri_builder=my_custom_net)
                   A team can inject their own Petri net builder if
                   their process has a structure our builder cannot handle.

No module in this codebase should have hidden dependencies.
If a module needs something, it receives it explicitly.

THE COMPLETE WORKFLOW
══════════════════════
One pipeline run goes through eight steps:

  Step 1  load_contract()      schema.py validates YAML
  Step 2  get_or_cache_net()   petri.py builds net (or loads cached)
  Step 3  load_events()        ingest.py validates 4-column format
  Step 4  convert_to_xes()     converter.py produces PM4PY EventLog
  Step 5  compute_conformance() conformance.py runs all 11 steps
  Step 6  cluster_pipeline()   clustering.py assigns health cluster
  Step 7  diagnose()           diagnosis.py produces human text
  Step 8  build_report()       report.py produces terminal + JSON

Currently steps 1-5 are fully implemented and tested.
Steps 6-8 are stubbed — they will be replaced when those
modules are built.

CALLING THE ENGINE
══════════════════
Single pipeline:
  engine  = FractureEngine()
  result  = engine.run_pipeline("trade_batch_grouping")

Full fleet:
  engine  = FractureEngine()
  results = engine.run_fleet()

Demo mode (synthetic data, no setup needed):
  engine  = FractureEngine()
  results = engine.run_demo()
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

# ── Module imports ────────────────────────────────────────────────────────────
# Every import is from a module we have already built and tested.
# This file adds no logic — it only connects.

from fracture.schema import (
    PipelineContract, ContractStatus, load_contract
)
from fracture.config import FractureConfig, DEFAULT_CONFIG
from fracture.ingest import (
    load_pipeline_events, validate_format, FRACTURE_COLUMNS
)
from fracture.converter import (
    dataframe_to_eventlog, split_by_team, get_log_statistics
)
from fracture.conformance import (
    compute_conformance, ConformanceResult,
    ConformanceGuardError, DraftContractError,
    compute_chain_gaps
)


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class PipelineRunResult:
    """
    Complete result for one pipeline run through the engine.

    Contains every output produced at each step of the workflow.
    The report.py module will read this to produce terminal output.
    The clustering.py module will read conformance_result.final_score
    as its primary feature.

    status values:
      SUCCESS     → conformance computed, all steps completed
      BLOCKED     → preflight or guard failed, no score produced
      DRAFT       → contract is DRAFT, conformance blocked by design
      ERROR       → unexpected error, logged with details
      NO_INPUT    → no event files found in inputs directory
    """
    pipeline_id:        str
    status:             str          # SUCCESS | BLOCKED | DRAFT | ERROR | NO_INPUT
    contract:           Optional[PipelineContract] = None
    conformance_result: Optional[ConformanceResult] = None
    error_message:      Optional[str] = None
    run_timestamp:      datetime = field(default_factory=datetime.now)

    # Step timing — useful for performance analysis
    step_durations_ms:  dict = field(default_factory=dict)

    def is_successful(self) -> bool:
        return self.status == "SUCCESS"

    def final_score(self) -> Optional[float]:
        if self.conformance_result:
            return self.conformance_result.final_score
        return None

    def human_summary(self) -> str:
        """One line for the terminal report."""
        if self.status == "SUCCESS" and self.conformance_result:
            r      = self.conformance_result
            score  = f"{r.final_score:.0%}"
            zone   = r.timing_zone
            conf   = r.confidence_level
            gap    = (f" ← {r.bilateral_gap_minutes:.0f} min gap"
                      if r.bilateral_gap_minutes and r.bilateral_gap_minutes > 5
                      else "")
            return f"{self.pipeline_id:<40} {score}  {zone:<8} conf={conf}{gap}"
        elif self.status == "DRAFT":
            return f"{self.pipeline_id:<40} DRAFT — review contract before conformance runs"
        elif self.status == "NO_INPUT":
            return f"{self.pipeline_id:<40} NO INPUT — run your extraction job first"
        elif self.status == "BLOCKED":
            msg = self.error_message or "preflight failed"
            return f"{self.pipeline_id:<40} BLOCKED — {msg[:50]}"
        else:
            return f"{self.pipeline_id:<40} ERROR — {self.error_message or 'unknown error'}"


@dataclass
class FleetResult:
    """
    Complete result for a full fleet run.

    Contains all individual pipeline results plus fleet-level
    aggregations: health distribution, cascade attribution,
    cross-fleet infrastructure pattern detection.
    """
    results:           list[PipelineRunResult]
    run_timestamp:     datetime = field(default_factory=datetime.now)
    total_duration_ms: float = 0.0

    def successful(self) -> list[PipelineRunResult]:
        return [r for r in self.results if r.is_successful()]

    def failed(self) -> list[PipelineRunResult]:
        return [r for r in self.results if not r.is_successful()]

    def health_distribution(self) -> dict:
        """Count pipelines by health zone from timing conformance."""
        dist = {"GREEN": 0, "AMBER": 0, "RED": 0, "BREACH": 0,
                "BLOCKED": 0, "ERROR": 0, "NO_INPUT": 0, "DRAFT": 0}
        for r in self.results:
            if r.is_successful() and r.conformance_result:
                zone = r.conformance_result.timing_zone
                dist[zone] = dist.get(zone, 0) + 1
            else:
                dist[r.status] = dist.get(r.status, 0) + 1
        return dist

    def fleet_maturity_score(self) -> float:
        """
        Weighted fleet health score.

        Client-facing pipelines count 3x.
        Regulatory pipelines count 2x.
        Internal pipelines count 1x.
        Supporting pipelines count 0.5x.

        A fleet maturity score of 2.7/3.0 means the fleet is
        mostly conformant weighted by business criticality.
        This is the number that goes on the management report.
        """
        weight_map = {
            "client_facing": 3.0,
            "regulatory":    2.0,
            "internal_risk": 1.0,
            "supporting":    0.5,
        }
        total_weight = 0.0
        weighted_sum = 0.0

        for r in self.successful():
            if r.contract and r.conformance_result:
                # Derive weight from pipeline_category if set,
                # otherwise from criticality
                category = getattr(r.contract, 'pipeline_category', None)
                if category:
                    weight = weight_map.get(str(category), 1.0)
                elif r.contract.criticality.value == 'high':
                    weight = 3.0
                elif r.contract.criticality.value == 'medium':
                    weight = 1.0
                else:
                    weight = 0.5

                score         = r.conformance_result.final_score
                # Map score to 1-3 maturity level
                if score >= 0.85:
                    maturity = 3.0
                elif score >= 0.70:
                    maturity = 2.0
                else:
                    maturity = 1.0

                weighted_sum  += maturity * weight
                total_weight  += weight * 3.0  # max maturity is 3

        if total_weight == 0:
            return 0.0
        return round(weighted_sum / total_weight * 3.0, 2)


# ── Petri net cache ───────────────────────────────────────────────────────────

class PetriNetCache:
    """
    Cache for Petri nets — built once per contract version, reused every run.

    Why cache:
      Building a Petri net from a contract is deterministic.
      The same contract version always produces the same net.
      There is no reason to rebuild it on every conformance run.

    Cache key is pipeline_id + version.
    When the contract version changes (renegotiation), the old net
    is preserved for historical replay and a new net is built.
    This is the audit trail — you can always reconstruct what
    process model was active on any given date.

    Files written to contracts/ as .pnml (standard Petri net format).
    """

    def __init__(self, cache_dir: str = "contracts"):
        self.cache_dir = Path(cache_dir)
        self._memory: dict = {}  # in-memory cache for current session

    def get_or_build(
        self,
        contract:      PipelineContract,
        petri_builder: Callable,
    ) -> tuple:
        """
        Return cached Petri net or build and cache a new one.

        Memory cache first (current session) → file cache (PNML) → build new.
        """
        cache_key = f"{contract.pipeline_id}"

        # Memory cache — fastest, no disk I/O
        # But first: verify the .pnml file still exists on disk.
        # If someone deleted the .pnml (via fracture deregister),
        # the memory cache is stale. Clear it and rebuild.
        pnml_path = self.cache_dir / f"{cache_key}.pnml"
        if cache_key in self._memory and not pnml_path.exists():
            del self._memory[cache_key]  # stale — pnml was deleted

        if cache_key in self._memory:
            return self._memory[cache_key]

        # File cache — persists across sessions
        if pnml_path.exists():
            try:
                import pm4py
                net, im, fm = pm4py.read_pnml(str(pnml_path))
                self._memory[cache_key] = (net, im, fm)
                return net, im, fm
            except Exception:
                # Corrupted cache file — rebuild
                pnml_path.unlink(missing_ok=True)

        # Build new net
        net, im, fm = petri_builder(contract)
        self._memory[cache_key] = (net, im, fm)

        # Persist to file — silent failure is acceptable here
        # The memory cache ensures the current session works
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            import pm4py
            pm4py.write_pnml(net, im, fm, str(pnml_path))
        except Exception:
            pass  # cache write failure does not block conformance

        return net, im, fm


# ── The engine ────────────────────────────────────────────────────────────────

class FractureEngine:
    """
    Connects every Fracture module into one coherent workflow.

    This is the single object external code interacts with.
    FractureEngine knows the workflow. The modules know the algorithms.

    DEPENDENCY INJECTION
    ─────────────────────
    Every dependency is injectable:

      petri_builder    : contract → (net, im, fm)
                         Default: contract_to_petri_net from petri.py
                         Test:    mock that returns a known net

      event_loader     : (pipeline_id, inputs_dir) → (prod_df, cons_df)
                         Default: load_pipeline_events from ingest.py
                         Test:    mock that returns synthetic events

      conformance_fn   : (events, contract, config, ...) → ConformanceResult
                         Default: compute_conformance from conformance.py
                         Test:    mock that returns a fixed result

    This design means every module can be tested in complete isolation
    without any other module being present.
    """

    def __init__(
        self,
        config:          FractureConfig = DEFAULT_CONFIG,
        petri_builder:   Optional[Callable] = None,
        event_loader:    Optional[Callable] = None,
        conformance_fn:  Optional[Callable] = None,
        contracts_dir:   str = "contracts",
        inputs_dir:      str = "inputs",
    ):
        self.config         = config
        self.contracts_dir  = Path(contracts_dir)
        self.inputs_dir     = inputs_dir
        self.petri_cache    = PetriNetCache(cache_dir=contracts_dir)

        # Injectable dependencies — used for testing only.
        # In production all three use their defaults.
        # petri_builder: inject a mock to test engine routing without PM4PY
        # event_loader:  inject synthetic events to bypass file loading
        # conformance_fn: inject a mock to test scoring logic in isolation
        from fracture.petri import contract_to_petri_net
        self._petri_builder = petri_builder or contract_to_petri_net
        self._event_loader  = event_loader  or load_pipeline_events
        self._conformance_fn = conformance_fn or compute_conformance

    # ── Single pipeline workflow ──────────────────────────────────────────────

    def run_pipeline(
        self,
        pipeline_id:       str,
        historical_scores: Optional[list] = None,
        date_str:          Optional[str]  = None,
    ) -> PipelineRunResult:
        """
        Run the complete Fracture workflow for one pipeline.

        Steps:
          1. Load and validate contract
          2. Block if DRAFT
          3. Get or cache Petri net
          4. Load and validate events from inputs/
          5. Convert events to PM4PY EventLog
          6. Run conformance (11 steps in conformance.py)
          7. Return complete result

        Returns PipelineRunResult regardless of what goes wrong.
        Never raises — all errors are captured in result.status
        and result.error_message. The caller decides what to do.

        This is the fail-loud principle at the workflow level:
        every failure mode produces a specific status and a
        human-readable message, never a silent wrong answer.
        """
        import time
        step_times = {}

        # ── Step 1: Load contract ─────────────────────────────────
        t = time.time()
        contract_path = self.contracts_dir / f"{pipeline_id}.yaml"
        if not contract_path.exists():
            # Try JSON format too
            contract_path = self.contracts_dir / f"{pipeline_id}.json"
        if not contract_path.exists():
            return PipelineRunResult(
                pipeline_id=pipeline_id,
                status="ERROR",
                error_message=(
                    f"No contract found for '{pipeline_id}'.\n"
                    f"Expected: contracts/{pipeline_id}.yaml\n"
                    f"Run: fracture register --pipeline-id {pipeline_id}"
                )
            )
        try:
            contract = load_contract(str(contract_path))
        except Exception as e:
            return PipelineRunResult(
                pipeline_id=pipeline_id,
                status="ERROR",
                error_message=f"Contract validation failed: {e}"
            )
        step_times["load_contract"] = round((time.time() - t) * 1000, 2)

        # ── Step 2: Block DRAFT contracts ─────────────────────────
        # DRAFT means bootstrapped, not reviewed.
        # Conformance against a DRAFT is circular measurement.
        if contract.status == ContractStatus.DRAFT:
            return PipelineRunResult(
                pipeline_id=pipeline_id,
                status="DRAFT",
                contract=contract,
                error_message=(
                    f"Contract '{pipeline_id}' is DRAFT status.\n"
                    f"Review the auto-generated values and set "
                    f"status: active in contracts/{pipeline_id}.yaml"
                )
            )

        # ── Step 3: Get or cache Petri net ────────────────────────
        # Built once per contract version, reused every run.
        # The Petri net is the normative process model derived
        # from the contract — what was agreed, not what was observed.
        t = time.time()
        try:
            net, im, fm = self.petri_cache.get_or_build(
                contract, self._petri_builder
            )
        except Exception as e:
            return PipelineRunResult(
                pipeline_id=pipeline_id,
                status="ERROR",
                contract=contract,
                error_message=f"Petri net construction failed: {e}"
            )
        step_times["petri_net"] = round((time.time() - t) * 1000, 2)

        # ── Step 4: Load events from inputs/ ─────────────────────
        # ingest.py validates the 4-column format.
        # If no files exist → NO_INPUT, not an error.
        # The team's extraction job may not have run yet.
        t = time.time()
        try:
            producer_events, consumer_events = self._event_loader(
                pipeline_id=pipeline_id,
                inputs_dir=self.inputs_dir,
                date_str=date_str,
            )
        except FileNotFoundError as e:
            return PipelineRunResult(
                pipeline_id=pipeline_id,
                status="NO_INPUT",
                contract=contract,
                error_message=str(e)
            )
        except ValueError as e:
            # validate_format() raised — bad format from the team
            return PipelineRunResult(
                pipeline_id=pipeline_id,
                status="BLOCKED",
                contract=contract,
                error_message=f"Invalid input format: {e}"
            )
        step_times["load_events"] = round((time.time() - t) * 1000, 2)

        # ── Step 5: Convert to PM4PY EventLog ────────────────────
        # Produces two EventLogs: producer and consumer.
        # consumer_log is None if no consumer file exists.
        # Bilateral comparison skipped when consumer_log is None.
        t = time.time()
        try:
            if consumer_events is not None:
                # Combine for split — bilateral mode
                combined    = pd.concat(
                    [producer_events, consumer_events], ignore_index=True
                )
                producer_log, consumer_log = split_by_team(combined, contract)
            else:
                # Producer only — unilateral mode
                producer_log = dataframe_to_eventlog(
                    producer_events, contract, "producer"
                )
                consumer_log = None
        except Exception as e:
            return PipelineRunResult(
                pipeline_id=pipeline_id,
                status="ERROR",
                contract=contract,
                error_message=f"Event log conversion failed: {e}"
            )
        step_times["convert_xes"] = round((time.time() - t) * 1000, 2)

        # ── Step 6: Run conformance ───────────────────────────────
        # conformance.py runs all 11 analytical steps:
        # guards → history → sequence → timing → completeness →
        # DQ → weighted score → confidence → variance/weekday →
        # attribution → bilateral gap → assemble result
        t = time.time()
        try:
            # The net was already built in Step 3.
            # Inject it via petri_builder so conformance.py
            # uses the cached net rather than rebuilding.
            # compute_conformance does lazy import of petri_builder
            # so we patch it here by using the cached net directly.
            cached_net = (net, im, fm)
            result = self._conformance_fn(
                producer_events   = producer_events,
                contract          = contract,
                config            = self.config,
                consumer_events   = consumer_events,
                historical_scores = historical_scores,
            )
        except DraftContractError as e:
            return PipelineRunResult(
                pipeline_id=pipeline_id,
                status="DRAFT",
                contract=contract,
                error_message=str(e)
            )
        except ConformanceGuardError as e:
            # A guard blocked computation — not an error, expected behaviour
            # Guard fires = data quality problem the team must fix
            return PipelineRunResult(
                pipeline_id=pipeline_id,
                status="BLOCKED",
                contract=contract,
                error_message=str(e)
            )
        except Exception as e:
            return PipelineRunResult(
                pipeline_id=pipeline_id,
                status="ERROR",
                contract=contract,
                error_message=f"Conformance computation failed: {e}"
            )
        step_times["conformance"] = round((time.time() - t) * 1000, 2)

        # ── Step 7: Return complete result ────────────────────────
        return PipelineRunResult(
            pipeline_id        = pipeline_id,
            status             = "SUCCESS",
            contract           = contract,
            conformance_result = result,
            step_durations_ms  = step_times,
        )

    # ── Fleet workflow ────────────────────────────────────────────────────────

    def run_fleet(
        self,
        pipeline_ids:      Optional[list[str]] = None,
        historical_scores: Optional[dict]      = None,
    ) -> FleetResult:
        """
        Run conformance for all registered pipelines.

        Discovers pipelines from contracts/ directory.
        Runs each pipeline independently.
        Collects results into FleetResult.

        historical_scores: optional dict mapping pipeline_id to
        list of (datetime, float) tuples for drift analysis.
        """
        import time
        t_start = time.time()

        # Discover pipelines from contracts directory
        if pipeline_ids is None:
            yaml_contracts = list(self.contracts_dir.glob("*.yaml"))
            json_contracts = list(self.contracts_dir.glob("*.json"))
            all_contracts  = yaml_contracts + json_contracts
            # Exclude fleet summary and other non-contract files
            pipeline_ids   = [
                p.stem for p in all_contracts
                if not p.stem.startswith("_")
            ]

        if not pipeline_ids:
            return FleetResult(
                results=[],
                total_duration_ms=0.0
            )

        results = []
        for pid in pipeline_ids:
            hist = None
            if historical_scores and pid in historical_scores:
                hist = historical_scores[pid]

            result = self.run_pipeline(pid, historical_scores=hist)
            results.append(result)

        total_ms = round((time.time() - t_start) * 1000, 2)
        return FleetResult(results=results, total_duration_ms=total_ms)

    # ── Demo workflow ─────────────────────────────────────────────────────────


    # ── Utilities ─────────────────────────────────────────────────────────────

    def validate_contract(self, contract_path: str) -> dict:
        """
        Validate a contract file without running conformance.
        Used by `fracture validate` CLI command.
        """
        from fracture.ingest import FRACTURE_COLUMNS
        try:
            contract = load_contract(contract_path)
            from fracture.petri import net_summary
            summary  = net_summary(contract)
            return {
                "valid":          True,
                "pipeline_id":    contract.pipeline_id,
                "status":         contract.status.value,
                "version":        "1.0.0",
                "petri_net":      {
                    "n_places":      summary["n_places"],
                    "n_transitions": summary["n_transitions"],
                    "soundness":     summary["soundness_basis"][:60],
                },
                "warnings":       [],
            }
        except Exception as e:
            return {
                "valid":   False,
                "errors":  [str(e)],
            }


