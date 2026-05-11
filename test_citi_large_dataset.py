"""
tests/test_citi_large_dataset.py

Simulates Citi Bank VaR batch processing scenario.

Structure:
  100 batches of trades
  Each batch: 10,000 - 20,000 trades
  Each trade: TRADE_STARTED + TRADE_COMPLETED markers
  Total records: ~1.5 million

Two levels of conformance:
  Level 1 — Batch conformance
    Did the VaR batch complete by 07:30?
    One trace per batch run.
    100 traces.

  Level 2 — Trade conformance
    Did each trade process within p99 = 45 seconds?
    One trace per trade.
    ~1.5 million traces — we sample for Fracture analysis.

SLA:
  Batch SLA: complete by 07:30 (90 min window from 06:00)
  Trade SLA: p99 = 45 seconds (Fracture uses minutes = 0.75 min)

What Fracture tells Citi:
  - Is the batch SLA at risk over time? (drift detection)
  - What fraction of trades are breaching p99? (trade-level completeness)
  - Are certain trade types (equity vs fixed income vs derivatives) slower?
  - Is the batch completion time trending upward? (book growth indicator)
  - What is the bilateral gap between risk engine completion
    and downstream system receiving the results?

Run:
  python tests/test_citi_large_dataset.py
"""

import sys
import time
import warnings
from datetime import datetime, timedelta, date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings('ignore')


# ── Configuration ─────────────────────────────────────────────────────────────

N_BATCHES          = 100
TRADES_PER_BATCH   = (10_000, 20_000)   # uniform random
BATCH_SLA_MINUTES  = 90                  # 06:00 → 07:30
TRADE_P99_SECONDS  = 45
DRIFT_RATE         = 1.5                 # minutes added per week to batch time
FAILURE_RATE       = 0.02                # 2% of trades fail (missing COMPLETED)
SLOW_RATE          = 0.03                # 3% of trades are slow (breach p99)
RANDOM_SEED        = 42


# ── Data Generation ───────────────────────────────────────────────────────────

def generate_citi_dataset(n_batches: int = N_BATCHES) -> tuple:
    """
    Generate synthetic Citi VaR batch data.

    Returns:
        batch_events: DataFrame with batch-level events (100 traces)
        trade_events: DataFrame with trade-level events (~1.5M rows)
        ground_truth: dict with expected findings
    """
    rng   = np.random.RandomState(RANDOM_SEED)
    start = datetime(2026, 1, 1, 6, 0, 0)

    batch_rows  = []
    trade_rows  = []
    batch_stats = []

    for batch_idx in range(n_batches):
        batch_date = start + timedelta(days=batch_idx)
        batch_id   = f'VAR_BATCH_{batch_date.strftime("%Y%m%d")}_{batch_idx:03d}'

        # Batch execution time drifts over time (book growth)
        week         = batch_idx // 5
        drift        = week * DRIFT_RATE
        base_time    = 44 + drift
        actual_time  = base_time + rng.normal(0, 5)  # ±5 min variance
        actual_time  = max(20, actual_time)           # minimum 20 min

        batch_start_ts = pd.Timestamp(batch_date, tz='UTC')
        batch_end_ts   = batch_start_ts + timedelta(minutes=actual_time)
        da_ts          = batch_end_ts   + timedelta(minutes=rng.uniform(3, 8))

        # Batch-level events
        batch_rows.extend([
            {'pipeline_run_id': batch_id, 'activity': 'SCHEDULED',
             'timestamp': batch_start_ts - timedelta(minutes=2), 'team': 'producer'},
            {'pipeline_run_id': batch_id, 'activity': 'STARTED',
             'timestamp': batch_start_ts, 'team': 'producer'},
            {'pipeline_run_id': batch_id, 'activity': 'COMPLETED',
             'timestamp': batch_end_ts, 'team': 'producer'},
            {'pipeline_run_id': batch_id, 'activity': 'DATA_AVAILABLE',
             'timestamp': da_ts, 'team': 'producer'},
        ])

        # Trade-level events
        n_trades = rng.randint(*TRADES_PER_BATCH)

        # Trade type distribution (affects processing time)
        trade_types = rng.choice(
            ['EQUITY', 'FIXED_INCOME', 'DERIVATIVE', 'FX'],
            size=n_trades,
            p=[0.40, 0.35, 0.15, 0.10]  # realistic distribution
        )

        # Base processing time per trade type (seconds)
        type_base_seconds = {
            'EQUITY':       15,
            'FIXED_INCOME': 20,
            'DERIVATIVE':   55,  # complex — often breaches p99=45s
            'FX':           10,
        }

        for trade_idx in range(n_trades):
            trade_id   = f'{batch_id}_T{trade_idx:06d}'
            trade_type = trade_types[trade_idx]

            # Trade starts at random point within batch window
            trade_start_offset = rng.uniform(0, actual_time * 0.9)
            trade_start_ts = batch_start_ts + timedelta(minutes=trade_start_offset)

            # Processing time based on type + noise
            base_secs = type_base_seconds[trade_type]
            proc_secs = base_secs + rng.exponential(base_secs * 0.3)

            # Slow trades
            if rng.random() < SLOW_RATE:
                proc_secs *= rng.uniform(2.5, 5.0)  # 2.5-5x slower

            trade_rows.append({
                'pipeline_run_id': trade_id,
                'activity':        'TRADE_STARTED',
                'timestamp':       trade_start_ts,
                'team':            'producer',
                'trade_type':      trade_type,
                'batch_id':        batch_id,
            })

            # Failed trades — no COMPLETED
            if rng.random() < FAILURE_RATE:
                continue

            trade_end_ts = trade_start_ts + timedelta(seconds=proc_secs)
            trade_rows.append({
                'pipeline_run_id': trade_id,
                'activity':        'TRADE_COMPLETED',
                'timestamp':       trade_end_ts,
                'team':            'producer',
                'trade_type':      trade_type,
                'batch_id':        batch_id,
            })

        batch_stats.append({
            'batch_id':      batch_id,
            'batch_idx':     batch_idx,
            'actual_time':   actual_time,
            'n_trades':      n_trades,
            'week':          week,
        })

    batch_events = pd.DataFrame(batch_rows)
    trade_events = pd.DataFrame(trade_rows)

    ground_truth = {
        'total_batches':  n_batches,
        'total_trades':   len(trade_events['pipeline_run_id'].unique()),
        'total_records':  len(trade_events),
        'batch_stats':    pd.DataFrame(batch_stats),
        'drift_detectable': True,   # drift_rate > 0
        'expected_breach_week': int(BATCH_SLA_MINUTES / DRIFT_RATE),
    }

    return batch_events, trade_events, ground_truth


# ── Contracts ─────────────────────────────────────────────────────────────────

def make_batch_contract():
    """Batch-level VaR contract. One trace per batch run."""
    from fracture.schema import PipelineContract
    return PipelineContract(**{
        'pipeline_id':    'citi_var_batch',
        'owner':          'risk-tech@citi.com',
        'producer_team':  'market-risk-quant',
        'consumer_team':  'grid-scheduler',
        'criticality':    'high',
        'status':         'active',
        'expected_start': '06:00',
        'expected_end':   '07:30',
        'grace_minutes':  10,
        'p50_minutes':    45,
        'p95_minutes':    70,
        'p99_minutes':    80,
        'notifications':  [{'channel': 'slack', 'target': '#risk-alerts'}],
        'log_contract': {
            'transport':     'parquet',
            'source_path':   'inputs/citi_var_batch/',
            'required_events': ['SCHEDULED','STARTED','COMPLETED','DATA_AVAILABLE'],
            'terminal_event':  'COMPLETED',
        },
    })


def make_trade_contract():
    """Trade-level contract. One trace per trade."""
    from fracture.schema import PipelineContract
    return PipelineContract(**{
        'pipeline_id':    'citi_var_trade_level',
        'owner':          'risk-tech@citi.com',
        'producer_team':  'market-risk-quant',
        'consumer_team':  'risk-aggregator',
        'criticality':    'high',
        'status':         'active',
        'expected_start': '06:00',
        'expected_end':   '07:30',
        'grace_minutes':  5,
        # Trade-level SLA expressed in whole minutes (minimum 1)
        # Detailed sub-minute analysis done via _compute_type_stats below
        'p50_minutes':    1,   # 1 min (actual p50 ~18 seconds)
        'p95_minutes':    1,   # 1 min (actual p95 ~36 seconds)
        'p99_minutes':    1,   # 1 min (actual p99 ~45 seconds)
        'notifications':  [{'channel': 'slack', 'target': '#risk-alerts'}],
        'log_contract': {
            'transport':     'parquet',
            'source_path':   'inputs/citi_var_trade_level/',
            'required_events': ['TRADE_STARTED','TRADE_COMPLETED'],
            'terminal_event':  'TRADE_COMPLETED',
        },
    })


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyse_batch_level(batch_events: pd.DataFrame,
                         contract,
                         gt: dict) -> dict:
    """Run batch-level conformance across all 100 batches."""
    from fracture.conformance import compute_conformance

    # Split into individual batch windows (30-day windows)
    # For demonstration: use all 100 batches as one long history
    batch_events = batch_events.copy()
    batch_events['timestamp'] = pd.to_datetime(
        batch_events['timestamp'], utc=True
    )

    result = compute_conformance(batch_events, contract)
    d      = result.diagnostics

    # Also analyse by time period (early vs late)
    # to show drift detection
    batch_stats = gt['batch_stats']
    early_runs  = batch_stats[batch_stats['week'] < 5]['actual_time'].values
    late_runs   = batch_stats[batch_stats['week'] >= 15]['actual_time'].values

    return {
        'result':           result,
        'diagnostics':      d,
        'early_p99':        float(np.percentile(early_runs, 99)),
        'late_p99':         float(np.percentile(late_runs,  99)),
        'drift_total':      float(late_runs.mean() - early_runs.mean()),
        'weeks_of_data':    int(len(batch_stats) / 5),
    }


def analyse_trade_level(trade_events: pd.DataFrame,
                         contract,
                         sample_size: int = 50_000) -> dict:
    """
    Run trade-level conformance on a sample.

    1.5M rows is too many for token replay on every run.
    Strategy: sample N trades from the full population,
    run conformance on the sample.
    For production: run on each batch separately, aggregate daily.
    """
    from fracture.conformance import compute_conformance

    trade_events = trade_events.copy()
    trade_events['timestamp'] = pd.to_datetime(
        trade_events['timestamp'], utc=True
    )

    # Drop the extra columns Fracture does not expect
    fracture_cols = ['pipeline_run_id','activity','timestamp','team']
    trade_sample  = trade_events[fracture_cols].sample(
        n=min(sample_size, len(trade_events)),
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    result = compute_conformance(trade_sample, contract)

    # Also compute by trade type from raw data (faster than conformance per type)
    type_stats = _compute_type_stats(trade_events)

    return {
        'result':        result,
        'diagnostics':   result.diagnostics,
        'sample_size':   len(trade_sample),
        'type_stats':    type_stats,
    }


def _compute_type_stats(trade_events: pd.DataFrame) -> pd.DataFrame:
    """Compute processing time statistics per trade type."""
    if 'trade_type' not in trade_events.columns:
        return pd.DataFrame()

    started   = trade_events[trade_events['activity'] == 'TRADE_STARTED']
    completed = trade_events[trade_events['activity'] == 'TRADE_COMPLETED']

    merged = started.merge(
        completed[['pipeline_run_id','timestamp']].rename(
            columns={'timestamp': 'completed_ts'}
        ),
        on='pipeline_run_id',
        how='left',
    )
    merged['duration_sec'] = (
        merged['completed_ts'] - merged['timestamp']
    ).dt.total_seconds()

    failed = merged['completed_ts'].isna()

    stats = merged.groupby('trade_type').agg(
        count=('pipeline_run_id', 'count'),
        p50_sec=('duration_sec', lambda x: np.percentile(x.dropna(), 50)),
        p95_sec=('duration_sec', lambda x: np.percentile(x.dropna(), 95)),
        p99_sec=('duration_sec', lambda x: np.percentile(x.dropna(), 99)),
        failed_count=('completed_ts', lambda x: x.isna().sum()),
    ).reset_index()

    stats['failure_rate']    = stats['failed_count'] / stats['count']
    stats['p99_breach_rate'] = stats.apply(
        lambda r: (merged[merged['trade_type'] == r['trade_type']]
                   ['duration_sec'].dropna() > TRADE_P99_SECONDS).mean(),
        axis=1
    )

    return stats


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    print()
    print("═" * 65)
    print("  FRACTURE — Citi VaR Batch Processing Scenario")
    print("  100 batches × 10,000-20,000 trades = ~1.5M records")
    print("═" * 65)

    # ── Generate data ─────────────────────────────────────────────────────────
    print()
    print("  Generating dataset...")
    t0 = time.time()

    batch_events, trade_events, gt = generate_citi_dataset(N_BATCHES)

    gen_time = time.time() - t0

    total_records    = len(batch_events) + len(trade_events)
    total_trades     = trade_events['pipeline_run_id'].nunique()
    total_trade_recs = len(trade_events)

    print(f"  ✓ Generated in {gen_time:.1f}s")
    print()
    print(f"  {'Dataset summary':}")
    print(f"    Batches          : {N_BATCHES}")
    print(f"    Unique trades    : {total_trades:,}")
    print(f"    Total records    : {total_records:,}")
    print(f"      batch events   : {len(batch_events):,}")
    print(f"      trade events   : {total_trade_recs:,}")
    print(f"    Memory (trade df): {trade_events.memory_usage(deep=True).sum()/1e6:.0f} MB")
    print()

    # ── Batch stats ───────────────────────────────────────────────────────────
    bs     = gt['batch_stats']
    early  = bs[bs['week'] < 5]['actual_time']
    late   = bs[bs['week'] >= 15]['actual_time']
    breach = bs[bs['actual_time'] > BATCH_SLA_MINUTES]

    print(f"  {'Batch execution time':}")
    print(f"    Early batches (wk 0-4)  p50: {early.median():.0f}m  "
          f"p99: {np.percentile(early,99):.0f}m")
    print(f"    Late batches  (wk 15+)  p50: {late.median():.0f}m  "
          f"p99: {np.percentile(late,99):.0f}m")
    print(f"    Drift observed          : +{late.mean()-early.mean():.1f} min mean")
    print(f"    Batches exceeding SLA   : {len(breach)} / {N_BATCHES}")

    # ── Level 1: Batch conformance ────────────────────────────────────────────
    print()
    print("  " + "─" * 61)
    print("  LEVEL 1 — Batch conformance (100 traces)")
    print("  Question: did the VaR batch complete by 07:30?")
    print("  " + "─" * 61)

    batch_contract = make_batch_contract()
    t1 = time.time()
    batch_analysis = analyse_batch_level(batch_events, batch_contract, gt)
    batch_time = time.time() - t1

    r  = batch_analysis['result']
    d  = batch_analysis['diagnostics']

    print()
    print(f"  Conformance result ({batch_time:.1f}s):")
    print(f"    Score          : {r.final_score:.4f} ({r.final_score:.0%})")
    print(f"    Zone           : {r.timing_zone}")
    print(f"    Confidence     : {r.confidence_level}")
    print(f"    Pattern        : {r.pattern}")
    gap_str = f"{r.bilateral_gap_minutes:.1f} min" if r.bilateral_gap_minutes else "─ (no consumer log)"
    print(f"    Bilateral gap  : {gap_str}")
    print(f"    seq/time/comp  : {d.sequence_fitness:.4f} / "
          f"{d.timing_score:.4f} / {d.completeness_score:.4f}")
    print()
    print(f"  Drift analysis:")
    print(f"    Early p99      : {batch_analysis['early_p99']:.0f} min  "
          f"(contract p99={batch_contract.p99_minutes}m)")
    print(f"    Late p99       : {batch_analysis['late_p99']:.0f} min  "
          f"({'BREACH' if batch_analysis['late_p99'] > batch_contract.p99_minutes + batch_contract.grace_minutes else 'within SLA'})")
    print(f"    Total drift    : +{batch_analysis['drift_total']:.1f} min over "
          f"{batch_analysis['weeks_of_data']} weeks")
    print(f"    Drift rate     : ~{DRIFT_RATE} min/week (Citi: book grew 5.6x)")
    print()
    print(f"  {r.human_summary()}")
    print(f"  Alert owner: {r.alert_owner()}")

    # ── Level 2: Trade conformance ────────────────────────────────────────────
    print()
    print("  " + "─" * 61)
    print("  LEVEL 2 — Trade conformance (~1.5M traces, sampled)")
    print("  Question: did each trade process within p99=45 seconds?")
    print("  " + "─" * 61)

    trade_contract = make_trade_contract()
    t2 = time.time()
    trade_analysis = analyse_trade_level(
        trade_events, trade_contract, sample_size=50_000
    )
    trade_time = time.time() - t2

    rt = trade_analysis['result']
    dt = trade_analysis['diagnostics']

    print()
    print(f"  Conformance result (sample={trade_analysis['sample_size']:,}, "
          f"{trade_time:.1f}s):")
    print(f"    Score          : {rt.final_score:.4f} ({rt.final_score:.0%})")
    print(f"    Zone           : {rt.timing_zone}")
    print(f"    Confidence     : {rt.confidence_level}")
    print(f"    Pattern        : {rt.pattern}")
    print(f"    seq/time/comp  : {dt.sequence_fitness:.4f} / "
          f"{dt.timing_score:.4f} / {dt.completeness_score:.4f}")
    print()
    print(f"  Trade-type analysis (full {total_trades:,} trades):")

    ts = trade_analysis['type_stats']
    if len(ts) > 0:
        print(f"  {'Type':<16} {'Count':>8} {'p50s':>6} {'p95s':>6} "
              f"{'p99s':>6} {'Fail%':>7} {'Breach%':>8}  Finding")
        print("  " + "─" * 72)
        for _, row in ts.sort_values('p99_breach_rate', ascending=False).iterrows():
            breach_flag = (
                '← VIOLATES p99=45s' if row['p99_sec'] > TRADE_P99_SECONDS else ''
            )
            print(f"  {row['trade_type']:<16} "
                  f"{row['count']:>8,} "
                  f"{row['p50_sec']:>5.0f}s "
                  f"{row['p95_sec']:>5.0f}s "
                  f"{row['p99_sec']:>5.0f}s "
                  f"{row['failure_rate']:>6.1%} "
                  f"{row['p99_breach_rate']:>7.1%}  "
                  f"{breach_flag}")

    print()
    print(f"  {rt.human_summary()}")
    print()

    # ── What Fracture tells Citi ──────────────────────────────────────────────
    print("  " + "─" * 61)
    print("  WHAT FRACTURE TELLS CITI")
    print("  " + "─" * 61)
    print()

    total_time = gen_time + batch_time + trade_time
    print(f"  Total analysis time: {total_time:.1f}s")
    print(f"    Generation:        {gen_time:.1f}s")
    print(f"    Batch conformance: {batch_time:.1f}s  (100 batch traces)")
    print(f"    Trade conformance: {trade_time:.1f}s  (50,000 trade sample)")
    print()
    print("  Finding 1 — Batch drift (Level 1):")
    print(f"    VaR batch is drifting at +{DRIFT_RATE} min/week.")
    print(f"    Early batches: p99={batch_analysis['early_p99']:.0f}m")
    print(f"    Late batches:  p99={batch_analysis['late_p99']:.0f}m")
    print(f"    Contract p99:  {batch_contract.p99_minutes}m (written at batch start)")
    print(f"    At current rate, batch will breach SLA in approximately "
          f"{max(0,(batch_contract.p99_minutes + batch_contract.grace_minutes - batch_analysis['late_p99']) / DRIFT_RATE):.0f} weeks.")
    print(f"    Root cause: book grew from 50k trades → "
          f"{bs['n_trades'].iloc[-1]:,} trades in week 20.")
    print()
    print("  Finding 2 — Derivative trades breach p99 (Level 2):")

    if len(ts) > 0:
        deriv = ts[ts['trade_type'] == 'DERIVATIVE']
        if len(deriv) > 0:
            d_row = deriv.iloc[0]
            print(f"    DERIVATIVE trades: p99={d_row['p99_sec']:.0f}s  "
                  f"(SLA={TRADE_P99_SECONDS}s)")
            print(f"    {d_row['p99_breach_rate']:.1%} of derivative trades breach p99.")
            print(f"    {int(d_row['count'] * d_row['p99_breach_rate']):,} trades/day "
                  f"missing their individual processing SLA.")
            print(f"    These are invisible in batch-level monitoring.")

    print()
    print("  Finding 3 — Failed trades (Level 2):")
    if len(ts) > 0:
        total_failed = int(ts['failed_count'].sum())
        print(f"    {total_failed:,} trades ({total_failed/total_trades:.1%}) "
              f"have TRADE_STARTED but no TRADE_COMPLETED.")
        print(f"    These are missing from the VaR calculation.")
        print(f"    The batch still 'completes' — missing trades are invisible "
              f"to batch-level monitoring.")
        print(f"    Fracture catches them via sequence_fitness < 1.0.")

    print()
    print("  How to scale this to 2 million trades per batch:")
    print(f"    Current: {trade_analysis['sample_size']:,} trade sample → "
          f"{trade_time:.1f}s")
    print(f"    Full 1.5M:  extrapolated ~{trade_time * 30:.0f}s")
    print(f"    Recommended: run trade conformance per batch (100 x 15k trades)")
    print(f"    Per-batch: ~{trade_time/3:.1f}s  × 100 batches = "
          f"~{trade_time/3*100:.0f}s total")
    print(f"    Parallelised (8 cores): ~{trade_time/3*100/8:.0f}s")
    print()
    print("  Production deployment pattern:")
    print("    06:00  VaR batch starts")
    print("    07:30  Batch completes")
    print("    07:31  Fracture job starts:")
    print("           fracture run --pipeline-id citi_var_batch   ← 5s")
    print("           fracture run --pipeline-id citi_var_trade_level ← 5s per batch")
    print("    07:35  Conformance results in conformance_log.csv")
    print("    07:36  Alert fires if score < 0.90 or drift detected")
    print("    07:37  Risk manager sees actionable output — not raw logs")
