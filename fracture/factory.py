"""
fracture/factory.py

Generates a realistic population of 100 pipeline contracts
for use in synthetic experiments, demos, and course project evaluation.

Design principle: the synthetic contracts must be realistic enough
that clustering produces meaningful results. Toy contracts with
identical SLA values produce toy clusters. These contracts vary
across domain, criticality, archetype, and SLA ranges in proportions
that reflect a real Dutch financial institution's pipeline portfolio.

The factory produces two outputs:
  1. A list of PipelineContract objects for use in Python code
  2. YAML files written to contracts/ for inspection and editing

Why YAML files matter: a user should be able to look at a generated
contract and recognise it as realistic. If it looks like a textbook
example, the demo loses credibility.
"""

import random
import yaml
from pathlib import Path
from datetime import time, timedelta
from typing import Optional

from fracture.schema import (
    PipelineContract, LogContract,
    Transport, Criticality,
    NotificationChannel, ContractStatus
)
from fracture.config import FractureConfig, DEFAULT_CONFIG


# ── Realistic pipeline name vocabularies ─────────────────────────────────────
# These produce readable pipeline IDs that sound like real bank pipelines.
# Format: {domain_prefix}_{process}_{delivery}

DOMAIN_PREFIXES = {
    "market_risk":  ["mkt", "risk", "var", "pnl"],
    "payments":     ["pay", "settle", "fx", "wire"],
    "compliance":   ["kyc", "aml", "sanction", "ubo"],
    "reporting":    ["rpt", "reg", "basel", "finrep"],
    "finance":      ["fin", "gl", "ledger", "recon"],
    "aml":          ["aml", "fec", "screening", "sar"],
    "operations":   ["ops", "ref", "static", "config"],
}

PROCESSES = [
    "positions", "trades", "exposures", "limits",
    "transactions", "balances", "counterparty",
    "corehours", "batch", "ingest", "transform",
    "aggregation", "reconciliation", "validation",
]

DELIVERIES = [
    "daily", "intraday", "eod", "bod",
    "hourly", "realtime", "batch", "sftp",
]

TEAM_NAMES = {
    "market_risk":  ["market-risk-quant", "risk-technology",
                     "var-platform", "stress-testing"],
    "payments":     ["payments-platform", "settlement-ops",
                     "fx-technology", "wire-transfer"],
    "compliance":   ["kyc-engineering", "compliance-tech",
                     "sanctions-platform", "aml-systems"],
    "reporting":    ["regulatory-reporting", "finance-tech",
                     "basel-platform", "finrep-team"],
    "finance":      ["finance-engineering", "gl-platform",
                     "recon-systems", "ledger-tech"],
    "aml":          ["fec-engineering", "aml-platform",
                     "screening-tech", "fec-risk"],
    "operations":   ["ops-engineering", "ref-data",
                     "static-data", "ops-platform"],
}

CONSUMER_TEAMS = [
    "grid-scheduler", "reporting-platform", "risk-dashboard",
    "data-warehouse", "ml-platform", "analytics-team",
    "regulatory-portal", "trader-workbench", "compliance-portal",
]

SLACK_CHANNELS = {
    "high":   ["#critical-alerts", "#sev1-response", "#platform-oncall"],
    "medium": ["#data-alerts", "#platform-alerts", "#pipeline-health"],
    "low":    ["#data-quality", "#pipeline-info", "#monitoring"],
}


def _pipeline_id(domain: str, seed: int) -> str:
    """
    Generate a readable pipeline ID that looks like a real bank pipeline.
    Uses domain-appropriate vocabulary so the ID is self-documenting.
    """
    random.seed(seed)
    prefix  = random.choice(DOMAIN_PREFIXES[domain])
    process = random.choice(PROCESSES)
    delivery = random.choice(DELIVERIES)
    return f"{prefix}_{process}_{delivery}"


def _expected_times(domain: str, criticality: str) -> tuple[time, time]:
    """
    Generate realistic expected_start and expected_end times.

    Bank pipeline timing follows predictable patterns:
    - BOD (beginning of day) pipelines: 05:00-07:00 start
    - Morning pipelines: 07:00-09:00 start
    - Intraday pipelines: 10:00-14:00 start
    - EOD (end of day) pipelines: 17:00-21:00 start

    High criticality pipelines tend to run earlier because
    downstream consumers (traders, risk managers) need the
    data before their working day starts.
    """
    if criticality == "high":
        # High criticality = early morning, before trading starts
        start_hours = [5, 6, 7]
        weights     = [0.3, 0.5, 0.2]
    elif domain in ("market_risk", "payments"):
        # These domains have strict timing requirements
        start_hours = [6, 7, 8]
        weights     = [0.4, 0.4, 0.2]
    else:
        # Supporting domains can run later
        start_hours = [7, 8, 9, 10]
        weights     = [0.2, 0.3, 0.3, 0.2]

    start_hour   = random.choices(start_hours, weights=weights)[0]
    start_minute = random.choice([0, 15, 30, 45])
    start        = time(start_hour, start_minute)

    # Window duration: 1.5 to 3 hours for most pipelines
    # High criticality pipelines have tighter windows
    if criticality == "high":
        window_minutes = random.randint(60, 150)
    else:
        window_minutes = random.randint(90, 180)

    end_hour   = start_hour + (start_minute + window_minutes) // 60
    end_minute = (start_minute + window_minutes) % 60
    end        = time(min(end_hour, 23), end_minute)

    return start, end


def _sla_values(window_minutes: int, config: FractureConfig) -> tuple[int, int, int, int]:
    """
    Generate realistic p50/p95/p99/grace values that:
    1. Are ordered (p50 < p95 < p99) — enforced by schema validator
    2. Fit within the window — enforced by schema validator
    3. Reflect realistic pipeline performance distributions

    The gap between p50 and p99 reflects execution variance.
    A well-tuned pipeline has low variance (p99 close to p50).
    A poorly-tuned pipeline has high variance (p99 >> p50).
    """
    sla = config.synthetic.sla_ranges

    # p50 is 30-60% of the window
    p50 = int(window_minutes * random.uniform(0.30, 0.55))
    p50 = max(p50, sla["p50"][0])
    p50 = min(p50, sla["p50"][1])

    # p95 is p50 + 20-40% more
    p95 = int(p50 * random.uniform(1.2, 1.6))
    p95 = min(p95, sla["p95"][1])

    # p99 is p95 + 10-25% more
    p99 = int(p95 * random.uniform(1.1, 1.25))
    p99 = min(p99, sla["p99"][1])

    # Grace: 5-20 minutes, but must not push p99+grace past window
    grace = random.randint(
        sla["grace"][0],
        sla["grace"][1]
    )

    # Safety check: ensure p99 + grace fits in window
    # If not, reduce p99 to fit
    # This mirrors the schema validator's check
    if p99 + grace > window_minutes - 5:  # 5 min buffer
        p99 = window_minutes - grace - 5
        p95 = min(p95, int(p99 * 0.90))
        p50 = min(p50, int(p95 * 0.75))

    # Enforce ordering (belt and suspenders alongside schema validator)
    p50 = max(10, p50)
    p95 = max(p50 + 5, p95)
    p99 = max(p95 + 5, p99)

    return p50, p95, p99, grace


def _notifications(criticality: str, domain: str) -> list[dict]:
    """
    Generate realistic notification targets based on criticality.

    High criticality pipelines always have Slack AND email.
    Medium criticality pipelines have Slack only.
    Low criticality pipelines log only (no interrupt).

    This reflects real bank on-call practices — you don't page
    someone at 3am for a low criticality reporting pipeline.
    """
    if criticality == "high":
        channel = random.choice(SLACK_CHANNELS["high"])
        team_domain = domain.replace("_", "-")
        return [
            {"channel": "slack",  "target": channel},
            {"channel": "email",  "target": f"oncall-{team_domain}@bank.com"},
        ]
    elif criticality == "medium":
        channel = random.choice(SLACK_CHANNELS["medium"])
        return [
            {"channel": "slack", "target": channel},
        ]
    else:
        # Low criticality: log only, no interrupt
        return [
            {"channel": "log", "target": "outputs/alerts.log"},
        ]


def _log_contract(transport: str, pipeline_id: str) -> dict:
    """
    Generate a realistic log contract for each transport type.

    For the course project all pipelines use plaintext logs
    because the synthetic generator writes plaintext.
    In production the transport would vary by pipeline.

    The required_events define the process model skeleton —
    these are the activities the Petri net will have transitions for.
    """
    base = {
        "transport":      transport,
        "required_events": [
            "SCHEDULED", "STARTED", "COMPLETED", "DATA_AVAILABLE"
        ],
        "terminal_event":  "COMPLETED",
        "deduplication_key": ["pipeline_run_id", "activity"],
        "max_out_of_order_seconds": 30,
        "known_disruptions": [
            "database restart",
            "deployment in progress",
            "network timeout",
            "grid maintenance",
        ],
    }

    if transport == "plaintext":
        base["source_path"] = f"outputs/logs/{pipeline_id}_producer.parquet"

    elif transport == "json_log":
        base["source_path"] = f"outputs/logs/{pipeline_id}.json"
        base["field_map"] = {
            "case_id":   "pipeline_run_id",
            "activity":  "status",
            "timestamp": "event_time",
            "team":      "emitting_service",
        }

    elif transport == "sftp":
        base["sources"] = {
            "producer_app": {
                "path": f"/logs/{pipeline_id}/transfer.log",
                "format": "plaintext",
            },
            "sftp_server": {
                "path": "/var/log/sftp/access.log",
                "format": "plaintext",
            },
            "consumer_app": {
                "path": f"/logs/{pipeline_id}/consumer.log",
                "format": "plaintext",
            },
        }

    return base



def generate_contract(
    index:       int,
    archetype:   str,
    domain:      str,
    criticality: str,
    config:      FractureConfig,
) -> PipelineContract:
    """
    Generate one realistic PipelineContract.

    Parameters
    ----------
    index       : unique index 0-99, used as random seed for reproducibility
    archetype   : "healthy" | "degrading" | "asymmetry" | "silent"
                  controls how the generator later produces execution events
    domain      : the business domain this pipeline belongs to
    criticality : "high" | "medium" | "low"
    config      : FractureConfig with all tunable parameters

    The archetype is stored in the contract as a tag so the
    clustering validation can use it as ground truth for ARI scoring.
    Later the clustering should be able to recover archetypes from
    conformance features alone — without seeing this tag.
    """
    random.seed(index * 137 + 42)  # deterministic but varied seed

    # Generate pipeline identity
    pipeline_id = _pipeline_id(domain, index)

    # Ensure uniqueness if collision (rare but possible)
    # Simple suffix approach
    pipeline_id = f"{pipeline_id}_{index:03d}"

    # Team assignments
    domain_teams  = TEAM_NAMES.get(domain, ["data-engineering"])
    producer_team = random.choice(domain_teams)

    # Consumer must differ from producer (enforced by schema validator)
    # Pick from cross-functional consumers
    consumer_team = random.choice([
        t for t in CONSUMER_TEAMS
        if t != producer_team
    ])

    owner = f"{producer_team}@bank.com"

    # Timing
    start, end    = _expected_times(domain, criticality)
    window        = (end.hour * 60 + end.minute) - \
                    (start.hour * 60 + start.minute)
    if window <= 0:
        window += 24 * 60  # handle overnight windows

    p50, p95, p99, grace = _sla_values(window, config)

    # Schedule (cron): pipelines run on business days
    # High criticality pipelines run every day including weekends
    if criticality == "high":
        schedule = f"{start.minute} {start.hour} * * *"      # every day
    else:
        schedule = f"{start.minute} {start.hour} * * 1-5"    # weekdays only

    # Notifications
    notifications = _notifications(criticality, domain)

    # Transport — for course project all use plaintext
    # In production this would vary
    transport = "plaintext"

    # Log contract
    log_contract = _log_contract(transport, pipeline_id)

    # Conformance config — varies by criticality and domain

    # Build the raw dict first (easier to see the full contract)
    raw = {
        "pipeline_id":    pipeline_id,
        "owner":          owner,
        "producer_team":  producer_team,
        "consumer_team":  consumer_team,
        "domain":         domain,
        "criticality":    criticality,
        "expected_start": start.strftime("%H:%M"),
        "expected_end":   end.strftime("%H:%M"),
        "grace_minutes":  grace,
        "p50_minutes":    p50,
        "p95_minutes":    p95,
        "p99_minutes":    p99,
        "notifications":  notifications,
        "log_contract":   log_contract,
        "status":         "active",

        # Archetype tag — used for clustering validation (ARI)
        # This field is NOT used by the conformance engine
        # It is the ground truth label for the ML experiment
        # The clustering should recover this grouping from
        # conformance features alone — without seeing this tag
        "_archetype": archetype,
    }

    return PipelineContract(**{
        k: v for k, v in raw.items()
        if not k.startswith("_")
    }), archetype, raw


def generate_fleet(
    config:      FractureConfig = DEFAULT_CONFIG,
    output_dir:  Optional[str]  = None,
) -> tuple[list[PipelineContract], list[str], list[dict]]:
    """
    Generate the full fleet of synthetic pipeline contracts.

    Returns
    -------
    contracts  : list of PipelineContract objects
    archetypes : list of archetype labels (ground truth for ARI)
    raw_dicts  : list of raw dicts (for YAML writing and inspection)

    The archetype list is the ground truth for the ML experiment.
    After computing conformance features and running clustering,
    compare cluster assignments against archetypes using ARI.
    A high ARI means Fracture's features successfully distinguish
    pipeline health categories without being told the archetype.
    """
    n = config.synthetic.n_pipelines

    # Build domain list respecting distribution weights
    domains = list(config.synthetic.domain_weights.keys())
    domain_weights = list(config.synthetic.domain_weights.values())
    assigned_domains = random.choices(domains, weights=domain_weights, k=n)

    # Build criticality list respecting distribution weights
    criticalities = list(config.synthetic.criticality_weights.keys())
    crit_weights  = list(config.synthetic.criticality_weights.values())
    assigned_crits = random.choices(criticalities,
                                    weights=crit_weights, k=n)

    # Build archetype list respecting distribution weights
    archetype_names   = list(config.synthetic.archetype_weights.keys())
    archetype_weights = list(config.synthetic.archetype_weights.values())
    assigned_archetypes = random.choices(archetype_names,
                                         weights=archetype_weights, k=n)

    contracts  = []
    archetypes = []
    raw_dicts  = []
    failures   = []

    for i in range(n):
        try:
            contract, archetype, raw = generate_contract(
                index       = i,
                archetype   = assigned_archetypes[i],
                domain      = assigned_domains[i],
                criticality = assigned_crits[i],
                config      = config,
            )
            contracts.append(contract)
            archetypes.append(archetype)
            raw_dicts.append(raw)

        except Exception as e:
            # Log failures but don't crash — some generated contracts
            # may hit edge cases in the semantic validators
            # We want to know about them, not silently skip them
            failures.append({"index": i, "error": str(e)})

    if failures:
        print(f"[factory] {len(failures)} contracts failed validation:")
        for f in failures[:5]:  # show first 5 only
            print(f"  index={f['index']}: {f['error'][:80]}")

    print(f"[factory] Generated {len(contracts)} / {n} contracts")
    print(f"[factory] Archetype distribution:")
    for a in archetype_names:
        count = archetypes.count(a)
        print(f"  {a:12s}: {count:3d} ({count/len(contracts)*100:.0f}%)")

    # Write YAML files if output_dir is specified
    if output_dir:
        _write_yaml_contracts(raw_dicts, archetypes, output_dir)

    return contracts, archetypes, raw_dicts


def _write_yaml_contracts(
    raw_dicts:  list[dict],
    archetypes: list[str],
    output_dir: str,
) -> None:
    """
    Write contracts to YAML files for inspection.

    Each file is named {archetype}_{pipeline_id}.yaml so you can
    immediately see which archetype a contract belongs to.
    This makes the demo readable — open any file and understand
    what pipeline type it represents.

    The archetype is written as a comment, not a field, because
    the PipelineContract schema doesn't have an archetype field.
    It is metadata for the experiment, not part of the contract.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Write sample contracts — one per archetype (4 files)
    # Writing 100 files would be noise; 4 representative samples
    # show the structure clearly
    written_archetypes = set()

    for raw, archetype in zip(raw_dicts, archetypes):
        if archetype not in written_archetypes:
            filename = f"{archetype}_{raw['pipeline_id']}.yaml"
            filepath = out / filename

            # Add archetype as a comment at the top of the file
            yaml_content = yaml.dump(
                {k: v for k, v in raw.items()
                 if not k.startswith("_")},
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )

            header = (
                f"# Archetype: {archetype}\n"
                f"# This is a synthetic contract generated by factory.py\n"
                f"# It represents a {archetype} pipeline in the demo fleet\n"
                f"# The archetype is the ground truth label for ARI validation\n"
                f"#\n"
            )

            filepath.write_text(header + yaml_content)
            written_archetypes.add(archetype)

    # Also write a full fleet summary as a single YAML list
    # Useful for understanding the distribution
    summary_path = out / "_fleet_summary.yaml"
    summary = [
        {
            "pipeline_id":  r["pipeline_id"],
            "archetype":    a,
            "domain":       r["domain"],
            "criticality":  r["criticality"],
            "p99_minutes":  r["p99_minutes"],
        }
        for r, a in zip(raw_dicts, archetypes)
    ]
    summary_path.write_text(
        "# Fleet summary — 100 synthetic pipelines\n"
        "# archetype is ground truth for ARI validation\n\n"
        + yaml.dump(summary, default_flow_style=False)
    )

    print(f"[factory] Wrote 4 sample contracts + fleet summary to {output_dir}")


def generate_demo_contracts(
    config: FractureConfig = DEFAULT_CONFIG,
) -> tuple[list[PipelineContract], list[str]]:
    """
    Generate the 6-pipeline demo fleet for the before/after story.

    These are hand-crafted contracts that tell a specific narrative:
      1. healthy pipeline     — baseline, what good looks like
      2. assumption asymmetry — the Citi bilateral gap problem
      3. degrading pipeline   — silent performance drift
      4. schema drift         — AI model getting wrong inputs
      5. silent pipeline      — missed regulatory runs
      6. cascade source       — root cause of 3 downstream failures

    Used for the demo run.py output and presentation slides.
    """
    demo_specs = [
        {
            "pipeline_id":   "payment_settlements_daily",
            "owner":         "payments-platform@bank.com",
            "producer_team": "payments-platform",
            "consumer_team": "settlement-ops",
            "criticality":   "high",
            "expected_start": "06:00",
            "expected_end":   "08:00",
            "grace_minutes":  10,
            "p50_minutes":    40,
            "p95_minutes":    65,
            "p99_minutes":    75,
            "notifications":  [
                {"channel": "slack", "target": "#critical-alerts"},
                {"channel": "email", "target": "oncall-payments@bank.com"},
            ],
            "log_contract": {
                "transport": "plaintext",
                "source_path": "outputs/logs/payment_settlements_daily_producer.parquet",
            },
            "_archetype": "healthy",
            "_story":     "Baseline — what good looks like",
        },
        {
            "pipeline_id":   "trade_positions_sftp",
            "owner":         "market-risk-quant@bank.com",
            "producer_team": "market-risk-quant",
            "consumer_team": "grid-scheduler",
            "criticality":   "high",
            "expected_start": "06:00",
            "expected_end":   "08:30",
            "grace_minutes":  15,
            "p50_minutes":    45,
            "p95_minutes":    75,
            "p99_minutes":    90,
            "notifications": [
                {"channel": "slack", "target": "#sev1-response"},
                {"channel": "email", "target": "oncall-market-risk@bank.com"},
            ],
            "log_contract": {
                "transport": "plaintext",
                "source_path": "outputs/logs/trade_positions_sftp_producer.parquet",
            },
            "_archetype": "asymmetry",
            "_story":     "Hidden 34-minute bilateral gap — the Citi problem",
        },
        {
            "pipeline_id":   "grid_corehours_calc",
            "owner":         "risk-technology@bank.com",
            "producer_team": "risk-technology",
            "consumer_team": "reporting-platform",
            "criticality":   "medium",
            "expected_start": "05:30",
            "expected_end":   "07:30",
            "grace_minutes":  20,
            "p50_minutes":    55,
            "p95_minutes":    85,
            "p99_minutes":    100,
            "notifications": [
                {"channel": "slack", "target": "#platform-alerts"},
            ],
            "log_contract": {
                "transport": "plaintext",
                "source_path": "outputs/logs/grid_corehours_calc_producer.parquet",
            },
            "_archetype": "degrading",
            "_story":     "Silent performance drift — 2 min/week for 8 weeks",
        },
        {
            "pipeline_id":   "customer_risk_features",
            "owner":         "ml-platform@bank.com",
            "producer_team": "aml-platform",
            "consumer_team": "ml-platform",
            "criticality":   "high",
            "expected_start": "07:00",
            "expected_end":   "09:00",
            "grace_minutes":  15,
            "p50_minutes":    50,
            "p95_minutes":    80,
            "p99_minutes":    95,
            "notifications": [
                {"channel": "slack", "target": "#critical-alerts"},
                {"channel": "email", "target": "oncall-aml@bank.com"},
            ],
            "log_contract": {
                "transport": "plaintext",
                "source_path": "outputs/logs/customer_risk_features_producer.parquet",
            },
            "_archetype": "degrading",
            "_story":     "AI model getting wrong inputs — Platinum added silently",
        },
        {
            "pipeline_id":   "regulatory_batch_dnb",
            "owner":         "regulatory-reporting@bank.com",
            "producer_team": "regulatory-reporting",
            "consumer_team": "regulatory-portal",
            "criticality":   "high",
            "expected_start": "20:00",
            "expected_end":   "22:30",
            "grace_minutes":  10,
            "p50_minutes":    60,
            "p95_minutes":    100,
            "p99_minutes":    120,
            "notifications": [
                {"channel": "slack", "target": "#sev1-response"},
                {"channel": "email", "target": "duty-officer@bank.com"},
            ],
            "log_contract": {
                "transport": "plaintext",
                "source_path": "outputs/logs/regulatory_batch_dnb_producer.parquet",
            },
            "_archetype": "silent",
            "_story":     "5 missed regulatory runs per month — nobody noticed",
        },
        {
            "pipeline_id":   "positions_sftp_ingest",
            "owner":         "market-risk-quant@bank.com",
            "producer_team": "market-risk-quant",
            "consumer_team": "data-warehouse",
            "criticality":   "high",
            "expected_start": "05:00",
            "expected_end":   "07:00",
            "grace_minutes":  15,
            "p50_minutes":    50,
            "p95_minutes":    80,
            "p99_minutes":    95,
            "notifications": [
                {"channel": "slack", "target": "#sev1-response"},
                {"channel": "email", "target": "oncall-market-risk@bank.com"},
            ],
            "log_contract": {
                "transport": "plaintext",
                "source_path": "outputs/logs/positions_sftp_ingest_producer.parquet",
            },
            "_archetype": "degrading",
            "_story":     "Root cause of 3 downstream failures — cascade attribution demo",
        },
    ]

    contracts  = []
    archetypes = []

    for spec in demo_specs:
        archetype = spec.pop("_archetype")
        spec.pop("_story", None)

        try:
            contract = PipelineContract(**spec)
            contracts.append(contract)
            archetypes.append(archetype)
        except Exception as e:
            print(f"[factory] Demo contract failed: {spec.get('pipeline_id')} — {e}")

    print(f"[factory] Generated {len(contracts)} demo contracts")
    return contracts, archetypes
