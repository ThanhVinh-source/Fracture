"""
fracture/config.py

Single source of truth for all tunable parameters.

Design principle: nothing in the codebase should have magic numbers.
Every threshold, every weight, every limit lives here with an explanation
of why it was set to that value. When a bank wants to customise Fracture
for their environment, this is the first file they read.

The defaults here are calibrated for a Dutch financial services environment
with hard client SLAs. A startup with loose SLAs would set different values.
A real-time payments system would weight timing even higher.
"""

from dataclasses import dataclass, field


@dataclass
class ConformanceWeights:
    """
    How much each conformance dimension contributes to the final score.

    Bank-calibrated defaults:
      - timing is the primary dimension (0.50) because missing a client
        SLA deadline has direct financial consequences — penalty clauses,
        relationship damage, regulatory exposure
      - sequence matters a lot (0.35) because wrong execution order
        means the data may be corrupt regardless of timing
      - completeness is lowest (0.15) because occasional silent runs
        happen legitimately — maintenance windows, bank holidays —
        and should not dominate the score

    These are overridable per contract in the YAML.
    A regulatory pipeline might set sequence=0.60 because audit
    integrity matters more than timing.
    A real-time fraud pipeline might set timing=0.65 because
    a 2-second delay means a fraudulent transaction goes through.
    """
    sequence:     float = 0.35
    timing:       float = 0.50
    completeness: float = 0.15

    def __post_init__(self):
        total = round(self.sequence + self.timing + self.completeness, 6)
        if abs(total - 1.0) > 0.001:
            raise ValueError(
                f"Weights must sum to 1.0 — got {total:.3f}. "
                f"Adjust sequence, timing, or completeness."
            )


@dataclass
class ConformanceThresholds:
    """
    Score thresholds that map to health states and severity levels.

    These define where the cluster boundaries sit and where
    alerts fire. Calibrated for bank environments.

    GREEN  → pipeline is healthy, no action needed
    AMBER  → pipeline is drifting, monitor closely
    RED    → pipeline is critical, immediate investigation
    BREACH → SLA is already violated, penalty clock may have started

    The timing zone thresholds are separate from the overall score
    thresholds because timing gets its own zone analysis.
    """
    # Overall conformance score thresholds
    healthy_above:  float = 0.85  # above this = HEALTHY cluster
    drifting_above: float = 0.70  # above this = DRIFTING cluster
                                  # below 0.70 = CRITICAL cluster

    # Timing zone thresholds (fraction of p99)
    # GREEN:  completed at or before p95
    # AMBER:  completed between p95 and p99
    # RED:    completed between p99 and p99+grace
    # BREACH: completed after p99+grace
    timing_green_at_p95:  bool  = True   # green ends at p95, not p99
    timing_amber_score:   float = 0.75   # score when entering AMBER
    timing_red_score:     float = 0.40   # score when entering RED
    timing_breach_floor:  float = 0.10   # minimum score in BREACH zone
                                         # never goes to 0 — pipeline ran,
                                         # just very late

    # Confidence thresholds
    # Confidence tells you how much to trust the conformance score
    # given the quality of the underlying logs
    confidence_high:   float = 0.90   # HIGH — trust the score fully
    confidence_medium: float = 0.70   # MEDIUM — trust with caveats
    confidence_low:    float = 0.50   # LOW — interpret cautiously
                                      # below 0.50 = UNRELIABLE

    # Drift alarm
    # A drift rate more negative than this triggers an early warning
    # -0.05 means the pipeline is losing 5 conformance points per week
    # At that rate a healthy pipeline (0.90) hits the critical threshold
    # (0.70) in 4 weeks — enough time to intervene
    drift_alarm_per_week: float = -0.05


@dataclass
class OutlierConfig:
    """
    How Fracture handles execution traces that are statistical outliers.

    The core tension: a pipeline that ran 3x its normal duration because
    of a database restart should not permanently drag down the conformance
    score. But a pipeline that runs 3x normal with no explanation is
    real signal that something is wrong.

    Fracture resolves this with weighted outlier handling — explained
    outliers get reduced weight (0.1x), unexplained outliers get full
    weight (1.0x). Confidence is reduced when outliers are present.

    The safety valve (max_excluded_per_window) prevents gaming:
    a team cannot declare every slow run an infrastructure event
    and maintain a falsely healthy conformance score.
    """
    # IQR multiplier for outlier detection
    # 1.5 is the standard statistical threshold
    # 1.5 × IQR above Q3 or below Q1 = outlier
    iqr_multiplier: float = 1.5

    # Weight applied to outlier traces in drift calculation
    # 0.1 means one outlier day has 10% the influence of a normal day
    # Setting this to 0.0 would exclude outliers entirely — too aggressive
    # Setting this to 1.0 would treat outliers as normal — too lenient
    outlier_weight: float = 0.1

    # Maximum traces that can receive reduced weight per 30-day window
    # Even if 10 traces look like outliers, only 3 get reduced weight
    # This prevents teams from having every slow run "explained away"
    max_reduced_per_window: int = 3

    # Rolling window for drift calculation
    # 30 days gives enough data for trend detection without being
    # too slow to respond to sudden changes
    rolling_window_days: int = 30


@dataclass
class ClusteringConfig:
    """
    Parameters for the k-Means and DBSCAN clustering algorithms.

    k=3 for k-Means because three health states are meaningful
    and actionable: healthy (do nothing), drifting (monitor),
    critical (intervene). More clusters fragment the actionability.
    Fewer clusters lose the drifting signal.

    DBSCAN parameters are data-dependent and will need tuning
    once you have real conformance score distributions.
    The defaults are reasonable starting points.

    random_state=42 is the standard for reproducibility —
    running Fracture twice on the same data gives the same clusters.
    """
    # k-Means
    n_clusters:     int   = 3      # healthy / drifting / critical
    kmeans_random:  int   = 42     # reproducibility seed
    kmeans_init:    str   = 'k-means++'  # smarter initialisation
                                          # than random — fewer iterations
                                          # to convergence
    kmeans_n_init:  int   = 10     # run k-means 10 times, take best
                                   # prevents bad random initialisation

    # DBSCAN
    # epsilon: max distance between two points to be considered neighbours
    # 0.5 in normalised feature space is a reasonable starting point
    # If you get too many noise points, increase epsilon
    # If you get too few clusters, decrease epsilon
    dbscan_epsilon:      float = 0.5
    dbscan_min_samples:  int   = 5  # minimum cluster size
                                    # pipelines with fewer than 5
                                    # neighbours become noise points

    # Validation
    # Minimum silhouette score to consider clustering meaningful
    # 0.5 = moderate cluster separation
    # Below 0.3 = clusters are not well-separated — features may not
    # be discriminating enough
    min_silhouette:  float = 0.40


@dataclass
class SyntheticConfig:
    """
    Parameters for the synthetic log generator.

    These control how realistic and diverse the synthetic data is.
    Realistic data is essential for the clustering to produce
    meaningful results — toy data produces toy clusters.

    The archetype distribution is calibrated to what a real bank's
    pipeline fleet looks like:
      - Most pipelines are healthy most of the time
      - A significant minority are degrading — this is the most
        common real-world failure mode
      - Assumption asymmetry is common but often undetected
      - Silent pipelines are the rarest but most dangerous

    The domain and criticality distributions reflect a financial
    institution's pipeline portfolio.
    """
    # Number of synthetic pipelines to generate
    n_pipelines: int = 100

    # Days of history to generate per pipeline
    # 30 days gives enough data for meaningful drift calculation
    # and clustering while keeping the demo fast
    history_days: int = 30

    # Archetype distribution — must sum to 1.0
    # Based on production observation: most pipelines are healthy,
    # degrading is the most common failure mode, assumption asymmetry
    # is underdetected, silent is rare but severe
    archetype_weights: dict = field(default_factory=lambda: {
        "healthy":     0.40,  # 40 pipelines — the baseline
        "degrading":   0.30,  # 30 pipelines — most common failure
        "asymmetry":   0.20,  # 20 pipelines — hidden assumption gap
        "silent":      0.10,  # 10 pipelines — rarest but most severe
    })

    # Domain distribution — reflects a bank's pipeline portfolio
    # Market risk and payments are dominant because they are
    # the most operationally critical domains
    domain_weights: dict = field(default_factory=lambda: {
        "market_risk":  0.25,
        "payments":     0.20,
        "compliance":   0.15,
        "reporting":    0.15,
        "finance":      0.10,
        "aml":          0.10,
        "operations":   0.05,
    })

    # Criticality distribution
    # High criticality pipelines are deliberately minority —
    # if everything is high criticality, nothing is
    criticality_weights: dict = field(default_factory=lambda: {
        "high":   0.25,  # client-facing, regulatory
        "medium": 0.50,  # internal risk, important but not critical
        "low":    0.25,  # supporting, analytics
    })

    # Realistic SLA ranges for a bank (in minutes)
    # These produce p50/p95/p99 values that make sense
    # for different pipeline types
    sla_ranges: dict = field(default_factory=lambda: {
        "p50": (20, 60),    # median 20-60 minutes
        "p95": (50, 90),    # 95th percentile 50-90 minutes
        "p99": (70, 120),   # 99th percentile 70-120 minutes
        "grace": (5, 20),   # grace period 5-20 minutes
    })

    # Drift parameters for degrading archetype
    # 2 minutes per week is realistic — slow enough that nobody
    # notices individually but significant over 8 weeks
    degrading_drift_rate_per_week: float = 2.0  # minutes per week

    # Assumption gap for asymmetry archetype
    # 15-35 minutes is realistic for SFTP + transformation gaps
    asymmetry_gap_range: tuple = (15, 35)  # minutes

    # Silent rate for silent archetype
    # 2 silent days per week = ~28% of runs
    # Enough to be clearly detectable by clustering
    silent_days_per_week: int = 2


@dataclass
class FractureConfig:
    """
    Master configuration object.

    Pass this through every layer of the pipeline.
    Never use magic numbers in any other module —
    always reference a field from this config.

    Usage:
        config = FractureConfig()          # bank defaults
        config = FractureConfig(
            weights=ConformanceWeights(
                timing=0.60,
                sequence=0.25,
                completeness=0.15
            )
        )  # real-time payments override
    """
    weights:     ConformanceWeights  = field(
        default_factory=ConformanceWeights
    )
    thresholds:  ConformanceThresholds = field(
        default_factory=ConformanceThresholds
    )
    outliers:    OutlierConfig       = field(
        default_factory=OutlierConfig
    )
    clustering:  ClusteringConfig    = field(
        default_factory=ClusteringConfig
    )
    synthetic:   SyntheticConfig     = field(
        default_factory=SyntheticConfig
    )

    # Output paths — relative to project root
    output_logs:    str = "outputs/logs"
    output_xes:     str = "outputs/xes"
    output_reports: str = "outputs/reports"

    # Minimum log quality before conformance runs
    # Below this threshold, Fracture refuses to compute
    # a conformance score and marks the result UNRELIABLE
    min_preflight_confidence: float = 0.40


# Module-level default instance
# Import this in other modules:
#   from fracture.config import DEFAULT_CONFIG
DEFAULT_CONFIG = FractureConfig()
