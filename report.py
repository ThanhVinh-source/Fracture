"""
report.py

Fleet health report for Fracture.

Reads conformance_log.csv and cluster_assignments.csv (from clustering.py)
and produces a human-readable fleet health report.

The goal: every person in the room — risk manager, operations lead,
data engineer, CTO — reads the same output and understands what to do.

Each pipeline gets:
  - A plain-language status
  - The key metric that drives the status
  - A concrete action
  - The team to call

Usage:
  python report.py
  python report.py --team market-risk-quant
  python report.py --csv path/to/conformance_log.csv
  python report.py --format executive    # one-page summary only
  python report.py --format engineer     # full diagnostic per pipeline
"""

import argparse
import csv
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path


# ── Formatting helpers ────────────────────────────────────────────────────────

BAR = "─" * 68

def score_bar(score: float, width: int = 20) -> str:
    """Visual bar for conformance score."""
    pct    = min(1.05, max(0.0, score))
    filled = int(pct / 1.05 * width)
    empty  = width - filled
    return f"[{'█' * filled}{'░' * empty}]"

def zone_icon(zone: str) -> str:
    return {'GREEN': '✓', 'AMBER': '⚠', 'RED': '⚠',
            'BREACH': '✗', 'SUSPICIOUS_EARLY': '?'}.get(zone, '○')

def cluster_badge(cluster: str) -> str:
    return {'HEALTHY': '● HEALTHY', 'DRIFTING': '◐ DRIFTING',
            'CRITICAL': '○ CRITICAL', 'ANOMALY': '! ANOMALY'}.get(cluster, cluster)

def score_pct(v) -> str:
    try:
        return f"{float(v):.0%}"
    except (ValueError, TypeError):
        return '─'

def fmt_gap(v) -> str:
    try:
        return f"{float(v):.1f} min"
    except (ValueError, TypeError):
        return '─'


# ── Plain language generators ─────────────────────────────────────────────────

def plain_status(row: dict) -> str:
    """One sentence a manager reads."""
    zone    = row.get('timing_zone', '')
    pattern = row.get('pattern', '')
    gap     = row.get('bilateral_gap_minutes', '')
    comp    = float(row.get('completeness_score', 1) or 1)

    try:
        gap_min = float(gap)
    except (ValueError, TypeError):
        gap_min = 0

    if zone == 'BREACH':
        return "SLA BREACHED. Immediate action required."
    if zone == 'RED':
        return "Approaching breach. Act before end of day."
    if comp < 0.80:
        return f"Only {comp:.0%} of scheduled runs completed. Investigate missing runs."
    if pattern == 'DRIFTING':
        return "Score declining over time. Review before it breaches SLA."
    if pattern == 'INTERMITTENT':
        return "Fails on specific days. Likely infrastructure problem — route to platform team."
    if gap_min > 20:
        return f"Pipeline healthy but consumer waits {gap_min:.0f} min for data. Review handoff."
    if zone == 'AMBER':
        return "Nearing SLA limit. Monitor closely this week."
    return "Conformant. No action required."


def action(row: dict) -> str:
    """Concrete next step."""
    zone    = row.get('timing_zone', '')
    pattern = row.get('pattern', '')
    cluster = row.get('cluster_name', '')
    comp    = float(row.get('completeness_score', 1) or 1)

    if zone == 'BREACH':
        return f"Escalate to {row.get('alert_owner','pipeline team')} now."
    if comp < 0.80:
        return "Check event files for missing runs. Run: fracture explain --verbose"
    if pattern == 'DRIFTING':
        return "Run: fracture sla-recommend to see renegotiation options."
    if pattern == 'INTERMITTENT':
        return "Escalate to platform-infrastructure. Check scheduler logs."
    if cluster == 'CRITICAL':
        return "Run: fracture explain --pipeline-id X --verbose for full diagnosis."
    if zone in ('AMBER', 'RED'):
        return "Run: fracture explain --pipeline-id X to see timing breakdown."
    return "No action required. Continue monitoring."


# ── Report sections ───────────────────────────────────────────────────────────

def print_executive_summary(rows: list, clusters: dict):
    """One-page summary for managers."""
    total      = len(rows)
    breach     = sum(1 for r in rows if r.get('timing_zone') == 'BREACH')
    red        = sum(1 for r in rows if r.get('timing_zone') == 'RED')
    amber      = sum(1 for r in rows if r.get('timing_zone') == 'AMBER')
    green      = sum(1 for r in rows if r.get('timing_zone') == 'GREEN')
    drifting   = sum(1 for r in rows if r.get('pattern') == 'DRIFTING')
    intermit   = sum(1 for r in rows if r.get('pattern') == 'INTERMITTENT')
    gap_pipelines = [r for r in rows
                     if r.get('bilateral_gap_minutes') and
                     float(r.get('bilateral_gap_minutes') or 0) > 20]

    print()
    print("╔" + "═" * 66 + "╗")
    print(f"║  FLEET HEALTH REPORT — {date.today().strftime('%A %d %B %Y'):<43}║")
    print("╠" + "═" * 66 + "╣")
    print(f"║  Pipelines monitored : {total:<43}║")
    print(f"║  Date                : {date.today().strftime('%Y-%m-%d'):<43}║")
    print("╠" + "═" * 66 + "╣")
    print(f"║  SLA STATUS                                                    ║")
    print(f"║    ✓  GREEN         {green:>4}   Comfortable — no action         ║")
    print(f"║    ⚠  AMBER         {amber:>4}   Watch closely this week         ║")
    print(f"║    ⚠  RED           {red:>4}   Act before end of day            ║")
    print(f"║    ✗  BREACH        {breach:>4}   Immediate action required       ║")
    print("╠" + "═" * 66 + "╣")
    print(f"║  PATTERNS                                                      ║")
    print(f"║    Drifting  {drifting:>4}   Score declining — breach coming         ║")
    print(f"║    Intermit  {intermit:>4}   Specific days fail (infrastructure)    ║")
    print("╠" + "═" * 66 + "╣")
    print(f"║  BILATERAL GAPS > 20 MIN                                       ║")
    if gap_pipelines:
        for r in gap_pipelines[:3]:
            g = float(r.get('bilateral_gap_minutes', 0))
            pid = r['pipeline_id'][:35]
            print(f"║    {pid:<35}  gap={g:.0f}min         ║")
    else:
        print(f"║    None — all handoffs within acceptable range                 ║")
    print("╚" + "═" * 66 + "╝")

    if clusters:
        print()
        print("  CLUSTER DISTRIBUTION (from clustering.py):")
        for name, pids in sorted(clusters.items()):
            print(f"    {cluster_badge(name):<20}: {len(pids)} pipelines")


def print_pipeline_detail(rows: list, clusters: dict, team_filter: str = None):
    """Full diagnostic per pipeline — for engineers."""

    # Group by cluster
    cluster_order = ['CRITICAL', 'DRIFTING', 'HEALTHY', 'ANOMALY', 'UNKNOWN']
    by_cluster = defaultdict(list)
    cluster_lookup = {}
    for name, pids in clusters.items():
        for pid in pids:
            cluster_lookup[pid] = name

    for row in rows:
        if team_filter and row.get('producer_team','') != team_filter:
            continue
        cluster = cluster_lookup.get(row['pipeline_id'], 'UNKNOWN')
        by_cluster[cluster].append(row)

    for cluster_name in cluster_order:
        pipelines = by_cluster.get(cluster_name, [])
        if not pipelines:
            continue

        print()
        print(f"  {cluster_badge(cluster_name)}")
        print(f"  {BAR}")

        for row in pipelines:
            pid   = row['pipeline_id']
            score = float(row.get('final_score', 0) or 0)
            zone  = row.get('timing_zone', '─')
            conf  = row.get('confidence_level', '─')
            pat   = row.get('pattern', '─')
            gap   = row.get('bilateral_gap_minutes', '')
            seq   = float(row.get('sequence_fitness', 1) or 1)
            comp  = float(row.get('completeness_score', 1) or 1)
            timing = float(row.get('timing_score', 1) or 1)
            owner = row.get('alert_owner', '─')

            icon  = zone_icon(zone)
            bar   = score_bar(score)

            print()
            print(f"  {icon}  {pid}")
            print(f"     Score      : {score_pct(score)}  {bar}  {zone}")
            print(f"     Confidence : {conf}")
            print()
            print(f"     Formula breakdown:")
            print(f"       sequence   : {seq:.3f} × 0.35 = {seq * 0.35:.3f}")
            print(f"       timing     : {timing:.3f} × 0.50 = {timing * 0.50:.3f}")
            print(f"       completeness: {comp:.3f} × 0.15 = {comp * 0.15:.3f}")
            print(f"                              total = {score:.3f}")
            print()
            print(f"     Pattern    : {pat}")
            if gap:
                try:
                    print(f"     Bilateral gap: {fmt_gap(gap)}")
                    print(f"       Producer marks DATA_AVAILABLE.")
                    print(f"       Consumer receives it {fmt_gap(gap)} later.")
                    print(f"       This gap is invisible to Airflow and Datadog.")
                except (ValueError, TypeError):
                    pass
            print()
            print(f"     Status  : {plain_status(row)}")
            print(f"     Action  : {action(row)}")
            print(f"     Contact : {owner}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(csv_path: str = 'conformance_log.csv',
         cluster_path: str = 'cluster_assignments.csv',
         team_filter: str = None,
         fmt: str = 'full'):

    # Load conformance log
    log_path = Path(csv_path)
    if not log_path.exists():
        print(f"✗  {csv_path} not found.")
        print(f"   Run: python scripts/05_generate_clustering_data.py --days 7")
        print(f"   Then: python clustering.py")
        sys.exit(1)

    with open(log_path) as f:
        all_rows = list(csv.DictReader(f))

    # Latest row per pipeline
    rows_by_pipeline = {}
    for row in all_rows:
        pid = row['pipeline_id']
        if pid not in rows_by_pipeline or \
           row['run_date'] > rows_by_pipeline[pid]['run_date']:
            rows_by_pipeline[pid] = row
    rows = list(rows_by_pipeline.values())

    # Load cluster assignments
    clusters = defaultdict(list)
    cl_path  = Path(cluster_path)
    if cl_path.exists():
        with open(cl_path) as f:
            for cr in csv.DictReader(f):
                clusters[cr.get('cluster_name', 'UNKNOWN')].append(cr['pipeline_id'])
    else:
        print(f"  ⚠  {cluster_path} not found.")
        print(f"     Run: python clustering.py first for cluster labels.")
        print(f"     Continuing without cluster information.")
        print()

    if fmt in ('executive', 'exec'):
        print_executive_summary(rows, clusters)
    else:
        print_executive_summary(rows, clusters)
        print()
        print()
        print("  PIPELINE DETAIL")
        print_pipeline_detail(rows, clusters, team_filter)

    print()
    print(f"  {BAR}")
    print(f"  Full history: conformance_log.csv  ({len(all_rows)} rows)")
    print(f"  Run: fracture explain --pipeline-id X --verbose  for deep diagnosis")
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='report.py')
    parser.add_argument('--csv',     default='conformance_log.csv')
    parser.add_argument('--clusters',default='cluster_assignments.csv')
    parser.add_argument('--team',    default=None)
    parser.add_argument('--format',  default='full',
                        choices=['full', 'executive', 'exec'])
    args = parser.parse_args()
    main(args.csv, args.clusters, args.team, args.format)
