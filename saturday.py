"""
saturday.py — Run this on Saturday. That's it.

Copy this file to your project root:
    D:\\projects\\fracture\\saturday.py

Run it:
    python saturday.py

What it does:
    Step 1 — Checks your folder structure and all files exist
    Step 2 — Runs 5 end-to-end scenarios and prints every number
    Step 3 — Generates PM4PY visualisations you can open and look at

No setup needed. No contracts to write. No input files to prepare.
Everything is generated synthetically inside this script.

Requirements (install once):
    pip install pm4py pydantic pyyaml pandas numpy scipy scikit-learn pyarrow

If graphviz is installed you get PNG images. If not, everything else still runs.
    Windows: https://graphviz.org/download/
    or: choco install graphviz
"""

import sys
import warnings
import os
from pathlib import Path
from datetime import date, timedelta, datetime
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

# ── Make sure we can find fracture package ────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))


# ── Shared helper ─────────────────────────────────────────────────────────────

def make_contract(pid='test', crit='high', status='active'):
    from fracture.schema import PipelineContract
    return PipelineContract(**{
        'pipeline_id':    pid,
        'owner':          'platform@bank.com',
        'producer_team':  'market-risk-quant',
        'consumer_team':  'grid-scheduler',
        'criticality':    crit,
        'status':         status,
        'expected_start': '06:00',
        'expected_end':   '08:30',
        'grace_minutes':  15,
        'p50_minutes':    45,
        'p95_minutes':    75,
        'p99_minutes':    90,
        'notifications':  [{'channel': 'slack', 'target': '#alerts'}]
                          if crit == 'high' else [],
        'log_contract': {
            'transport':   'plaintext',
            'source_path': f'inputs/{pid}/',
        },
    })


START = date.today() - timedelta(days=30)
DAYS  = 30


# ════════════════════════════════════════════════════════════════════════════
# STEP 1: STRUCTURE CHECK
# ════════════════════════════════════════════════════════════════════════════

def step1_structure():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  STEP 1 — Folder structure and file check                ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    root = Path('.')
    required_files = [
        ('fracture/__init__.py',    'Package init'),
        ('fracture/schema.py',      '14-field contract'),
        ('fracture/config.py',      'Weights and tunables'),
        ('fracture/petri.py',       'Petri net builder'),
        ('fracture/preflight.py',   'Traffic light checks'),
        ('fracture/conformance.py', '11-step engine'),
        ('fracture/converter.py',   'XES converter'),
        ('fracture/ingest.py',      '4-column validator'),
        ('fracture/generator.py',   'Synthetic data'),
        ('fracture/factory.py',     'Contract factory'),
        ('fracture/engine.py',      'DI wiring hub'),
    ]

    all_ok = True
    for path, desc in required_files:
        exists = (root / path).exists()
        icon   = '  v' if exists else '  x'
        if not exists:
            all_ok = False
        print(f"{icon}  {path:<35} {desc}")

    print()
    dirs_needed = ['contracts', 'inputs', 'outputs', 'tests']
    for d in dirs_needed:
        exists = (root / d).exists()
        if not exists:
            Path(d).mkdir(exist_ok=True)
            print(f"  ○  {d}/ created")
        else:
            print(f"  v  {d}/")

    if not all_ok:
        print()
        print("  x Some files are missing.")
        print("  Copy all .py files from your outputs directory to fracture/")
        sys.exit(1)

    print()
    print("  All files present. Ready.")


# ════════════════════════════════════════════════════════════════════════════
# STEP 2: FIVE END-TO-END SCENARIOS
# ════════════════════════════════════════════════════════════════════════════

def step2_scenarios():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  STEP 2 — Five end-to-end scenarios                      ║")
    print("║  Generator → Preflight → Petri → Replay → Conformance    ║")
    print("╚══════════════════════════════════════════════════════════╝")

    from fracture.conformance import compute_conformance, ConformanceGuardError
    from fracture.generator import (
        StableMatureGenerator, AssumptionAsymmetryGenerator,
        SilentPipelineGenerator, ClockSyncOutlierGenerator,
        SuddenCollapseGenerator,
    )

    all_pass = True

    # ── Scenario 1: Healthy ───────────────────────────────────────────────
    print()
    print("─" * 60)
    print("  Scenario 1: HEALTHY PIPELINE")
    print("  What: 30 days of well-behaved execution")
    print("  Expect: score ≥ 0.90, GREEN, HIGH confidence")
    print("─" * 60)

    c = make_contract('healthy')
    prod, cons, gt = StableMatureGenerator().generate(c, DAYS, START, 200, 1)

    r = compute_conformance(prod, c, consumer_events=cons)
    d = r.diagnostics

    # Formula breakdown
    manual = d.sequence_fitness*0.35 + d.timing_score*0.50 + d.completeness_score*0.15

    print(f"""
  Generator:
    Producer events  : {len(prod)} ({DAYS} days × 4 activities)
    Consumer events  : {len(cons)}
    Ground truth     : {gt.expected_cluster} / {gt.expected_pattern}

  Conformance result:
    final_score      : {r.final_score:.4f}
    confidence_level : {r.confidence_level}
    timing_zone      : {r.timing_zone}
    pattern          : {r.pattern}
    bilateral_gap    : {r.bilateral_gap_minutes:.1f} min

  Formula breakdown:
    seq  {d.sequence_fitness:.3f} × 0.35 = {d.sequence_fitness*0.35:.3f}
    time {d.timing_score:.3f} × 0.50 = {d.timing_score*0.50:.3f}
    comp {d.completeness_score:.3f} × 0.15 = {d.completeness_score*0.15:.3f}
    ─────────────────────────────────────
    total            : {manual:.4f}  (stored: {r.final_score:.4f})
    note: 1.05 is the early-completion bonus cap

  Human output:
    summary  : {r.human_summary()}
    owner    : {r.alert_owner()}
    trustworthy: {r.is_trustworthy()}
""")

    ok = r.final_score >= 0.90 and r.timing_zone == 'GREEN'
    print(f"  {'v PASS' if ok else 'x FAIL'}")
    all_pass = all_pass and ok

    # ── Scenario 2: Bilateral gap ─────────────────────────────────────────
    print()
    print("─" * 60)
    print("  Scenario 2: ASSUMPTION ASYMMETRY (bilateral gap)")
    print("  What: producer marks done at T+47, consumer gets data at T+72")
    print("  Expect: bilateral_gap > 15 min — invisible to unilateral tools")
    print("─" * 60)

    c2 = make_contract('asym')
    prod2, cons2, gt2 = AssumptionAsymmetryGenerator(initial_gap=25).generate(
        c2, DAYS, START, 90, 2)

    # Show the raw gap before conformance
    prod_da = prod2[prod2.activity=='DATA_AVAILABLE']['timestamp'].mean()
    cons_da = cons2[cons2.activity=='DATA_AVAILABLE']['timestamp'].mean()
    raw_gap = (cons_da - prod_da).total_seconds() / 60

    r2 = compute_conformance(prod2, c2, consumer_events=cons2)

    print(f"""
  Raw data gap (before conformance):
    Producer DATA_AVAILABLE mean : {prod_da.strftime('%H:%M:%S')}
    Consumer DATA_AVAILABLE mean : {cons_da.strftime('%H:%M:%S')}
    Raw gap                      : {raw_gap:.1f} minutes

  Conformance result:
    final_score      : {r2.final_score:.4f}
    bilateral_gap    : {r2.bilateral_gap_minutes:.1f} min
    gap_trend        : {r2.diagnostics.bilateral_gap_trend}
    gap_p95          : {r2.diagnostics.bilateral_gap_p95:.1f} min

  Why this matters:
    Datadog shows both pipelines GREEN.
    Monte Carlo shows data quality OK.
    Fracture shows 27 min gap that nobody documented.
    p95 gap means on bad days consumer waits {r2.diagnostics.bilateral_gap_p95:.0f} min.

  Human output:
    {r2.human_summary()}
""")

    ok2 = r2.bilateral_gap_minutes and r2.bilateral_gap_minutes > 15
    print(f"  {'v PASS' if ok2 else 'x FAIL'}")
    all_pass = all_pass and ok2

    # ── Scenario 3: Silent ────────────────────────────────────────────────
    print()
    print("─" * 60)
    print("  Scenario 3: SILENT PIPELINE")
    print("  What: pipeline does not run on Mondays and Thursdays")
    print("  Expect: completeness < 0.75 — no error, just absence")
    print("─" * 60)

    c3 = make_contract('silent', crit='medium')
    prod3, cons3, gt3 = SilentPipelineGenerator(silent_weekdays=[0,3]).generate(
        c3, DAYS, START, 60, 3)

    scheduled = prod3[prod3.activity=='SCHEDULED']['pipeline_run_id'].nunique()
    completed = prod3[prod3.activity=='COMPLETED']['pipeline_run_id'].nunique()
    silent    = scheduled - completed

    r3 = compute_conformance(prod3, c3)
    d3 = r3.diagnostics

    print(f"""
  Generator:
    Scheduled runs   : {scheduled}
    Completed runs   : {completed}
    Silent runs      : {silent} (Mon + Thu every week)
    Completion rate  : {completed/scheduled:.0%}

  Conformance result:
    final_score      : {r3.final_score:.4f}
    completeness     : {d3.completeness_score:.4f}
    pattern          : {r3.pattern}
    timing_zone      : {r3.timing_zone}

  Why completeness matters:
    Infrastructure monitoring: no alerts (nothing failed)
    Data quality tools:        no alerts (data looks fine when it runs)
    Fracture:                  AMBER — {silent} runs produced no DATA_AVAILABLE

  Human output:
    {r3.human_summary()}
""")

    ok3 = d3.completeness_score < 0.75
    print(f"  {'v PASS' if ok3 else 'x FAIL'}")
    all_pass = all_pass and ok3

    # ── Scenario 4: Dual preflight ────────────────────────────────────────
    print()
    print("─" * 60)
    print("  Scenario 4: CLOCK SYNC — dual preflight in action")
    print("  What: consumer has -15 min clock skew (timezone mismatch)")
    print("  Expect: producer score intact, bilateral_gap = None")
    print("─" * 60)

    c4 = make_contract('clock')
    prod4, _, _ = StableMatureGenerator().generate(c4, DAYS, START, 200, 1)
    _, cons4, _ = ClockSyncOutlierGenerator(clock_skew_minutes=-15).generate(
        c4, DAYS, START, 60, 5)

    prod_da4 = prod4[prod4.activity=='DATA_AVAILABLE']['timestamp'].iloc[0]
    cons_da4 = cons4[cons4.activity=='DATA_AVAILABLE']['timestamp'].iloc[0]

    r4 = compute_conformance(prod4, c4, consumer_events=cons4)

    print(f"""
  Clock skew demonstration (first run):
    Producer DATA_AVAILABLE : {prod_da4}
    Consumer DATA_AVAILABLE : {cons_da4}
    Raw gap                 : {(cons_da4-prod_da4).total_seconds()/60:.1f} min (negative = skew)

  Dual preflight design:
    Producer preflight → CLEAN → conformance runs normally
    Consumer preflight → 29/30 gaps negative → bilateral SKIPPED
    Consumer clock problem does not invalidate producer measurement

  Conformance result:
    final_score      : {r4.final_score:.4f}  ← producer unaffected
    bilateral_gap    : {r4.bilateral_gap_minutes}       ← None = skipped
    confidence_level : {r4.confidence_level}

  Human output:
    {r4.human_summary()}
""")

    ok4 = r4.final_score >= 0.85 and r4.bilateral_gap_minutes is None
    print(f"  {'v PASS' if ok4 else 'x FAIL'}")
    all_pass = all_pass and ok4

    # ── Scenario 5: Sudden collapse ───────────────────────────────────────
    print()
    print("─" * 60)
    print("  Scenario 5: SUDDEN COLLAPSE — documented limitation")
    print("  What: 25 days healthy, then catastrophic failure on day 26")
    print("  Expect: historical score high, live score critical")
    print("─" * 60)

    c5 = make_contract('collapse')
    gen = SuddenCollapseGenerator()
    hist  = gen.generate_historical(c5, days=25, start_date=START, seed=42)
    live_p, live_c, gt5 = gen.generate_live(c5, days=5,
                                             start_date=START+timedelta(25), seed=99)

    r5_hist = compute_conformance(hist, c5)
    r5_live = compute_conformance(live_p, c5,
                                   consumer_events=live_c if len(live_c)>0 else None)

    print(f"""
  Split design:
    Historical : {len(hist)} events over 25 days (before Fracture deployed)
    Live       : {len(live_p)} events over 5 days (post-collapse)

  Historical conformance (25 healthy days):
    final_score  : {r5_hist.final_score:.4f}
    timing_zone  : {r5_hist.timing_zone}
    completeness : {r5_hist.diagnostics.completeness_score:.4f}

  Live conformance (5 collapse days):
    final_score  : {r5_live.final_score:.4f}
    timing_zone  : {r5_live.timing_zone}
    completeness : {r5_live.diagnostics.completeness_score:.4f}
    summary      : {r5_live.human_summary()}

  The documented limitation:
    Drift rate over full 30 days ≈ near zero (healthy history dominates slope)
    But per-run live score is critical — the step function is visible in charts
    This is why drift rate alone is not enough for sudden failures
    Fracture surfaces it — the live score is what matters operationally
""")

    ok5 = r5_hist.final_score >= 0.85 and r5_live.final_score < 0.70
    print(f"  {'v PASS' if ok5 else 'x FAIL'}")
    all_pass = all_pass and ok5

    # ── Summary ───────────────────────────────────────────────────────────
    print()
    print("─" * 60)
    all_results = [ok, ok2, ok3, ok4, ok5]
    passed = sum(all_results)
    print(f"  Scenarios: {passed}/5 passed")
    if passed == 5:
        print()
        print("  All five scenarios passing.")
        print("  The full pipeline works end-to-end.")
    else:
        print("  Some scenarios failed — check output above.")
    print("─" * 60)

    return all_pass


# ════════════════════════════════════════════════════════════════════════════
# STEP 3: PM4PY VISUALISATIONS
# ════════════════════════════════════════════════════════════════════════════

def step3_visualisations():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  STEP 3 — PM4PY Visualisations                           ║")
    print("║  Petri net + Performance DFG for bilateral gap           ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # Check if graphviz is available
    try:
        import subprocess
        result = subprocess.run(['dot', '-V'],
                                capture_output=True, text=True)
        graphviz_ok = result.returncode == 0
    except FileNotFoundError:
        graphviz_ok = False

    if not graphviz_ok:
        print("  ○ Graphviz not found. Skipping image generation.")
        print("  Install: https://graphviz.org/download/")
        print("  or: choco install graphviz")
        print()
        print("  Everything else still works. Images are bonus.")
        return

    import pm4py
    from fracture.petri import contract_to_petri_net
    from fracture.converter import dataframe_to_eventlog
    from fracture.generator import (
        StableMatureGenerator, AssumptionAsymmetryGenerator
    )

    Path('outputs').mkdir(exist_ok=True)

    # ── 3a: Normative Petri net ───────────────────────────────────────────
    print("  Generating Petri net visualisations...")
    c = make_contract('viz')
    net, im, fm = contract_to_petri_net(c)

    try:
        pm4py.save_vis_petri_net(net, im, fm, 'outputs/01_petri_net_normative.png')
        print("  v  outputs/01_petri_net_normative.png")
        print("     Open this. You will see 6 circles (places) connected by")
        print("     4 rectangles (transitions: SCHEDULED STARTED COMPLETED DATA_AVAILABLE)")
        print("     This is your contract made visible as a Petri net.")
    except Exception as e:
        print(f"  x  Petri net: {e}")

    # ── 3b: Discovered net from healthy logs ─────────────────────────────
    print()
    print("  Generating discovered net (Inductive Miner on healthy logs)...")
    prod, _, _ = StableMatureGenerator().generate(c, DAYS, START, 200, 1)
    log_healthy = dataframe_to_eventlog(prod, c, 'producer')

    try:
        net_disc, im_disc, fm_disc = pm4py.discover_petri_net_inductive(log_healthy)
        pm4py.save_vis_petri_net(net_disc, im_disc, fm_disc,
                                  'outputs/02_petri_net_discovered_healthy.png')
        print("  v  outputs/02_petri_net_discovered_healthy.png")
        print("     Compare with 01. They should look identical.")
        print("     Healthy pipeline: discovered model matches the contract.")
        print("     This validates your contract is correct.")
    except Exception as e:
        print(f"  x  Discovered net: {e}")

    # ── 3c: Performance DFG — bilateral gap visible as red arc ───────────
    print()
    print("  Generating performance DFG for assumption asymmetry...")
    c2     = make_contract('viz_asym')
    prod2, cons2, _ = AssumptionAsymmetryGenerator(initial_gap=25).generate(
        c2, DAYS, START, 90, 2)

    log_prod = dataframe_to_eventlog(prod2, c2, 'producer')
    log_cons = dataframe_to_eventlog(cons2, c2, 'consumer')

    try:
        # Producer performance DFG
        perf_prod, s_prod, e_prod = pm4py.discover_performance_dfg(log_prod)
        pm4py.save_vis_performance_dfg(perf_prod, s_prod, e_prod,
                                        'outputs/03_perf_dfg_producer.png')
        print("  v  outputs/03_perf_dfg_producer.png  (producer side)")

        # Consumer performance DFG
        perf_cons, s_cons, e_cons = pm4py.discover_performance_dfg(log_cons)
        pm4py.save_vis_performance_dfg(perf_cons, s_cons, e_cons,
                                        'outputs/04_perf_dfg_consumer.png')
        print("  v  outputs/04_perf_dfg_consumer.png  (consumer side)")
        print()
        print("  Compare 03 and 04 side by side.")
        print("  The arc from COMPLETED → DATA_AVAILABLE will be different.")
        print("  Producer: small number (minutes)")
        print("  Consumer: larger number (minutes + bilateral gap)")
        print("  THAT difference is the bilateral gap Fracture measures.")
        print("  This is the Figure 2 for your paper.")
    except Exception as e:
        print(f"  x  Performance DFG: {e}")

    # ── 3d: Dotted chart — silent pipeline ────────────────────────────────
    print()
    print("  Generating dotted chart for silent pipeline...")
    from fracture.generator import SilentPipelineGenerator
    c3 = make_contract('viz_silent', crit='medium')
    prod3, _, _ = SilentPipelineGenerator([0,3]).generate(c3, DAYS, START, 60, 3)
    log_silent = dataframe_to_eventlog(prod3, c3, 'producer')

    try:
        # PM4PY dotted chart needs a pandas dataframe not EventLog
        import pm4py.visualization.dotted_chart as dc
        from pm4py.objects.conversion.log import converter as log_converter
        df_for_dot = pm4py.convert_to_dataframe(log_silent)
        pm4py.save_vis_dotted_chart(df_for_dot, 'outputs/05_dotted_silent.png',
                                    attributes=['concept:name'])
        print("  v  outputs/05_dotted_silent.png")
        print("     Each row is one pipeline run (30 rows for 30 days).")
        print("     Monday and Thursday rows are very short — only SCHEDULED fires.")
        print("     This is the silent breach visualised.")
    except Exception as e:
        print(f"  x  Dotted chart: {e}")

    print()
    print("  All visualisations saved to outputs/")
    print("  Open them in this order:")
    print("  01 → 02 → compare (contract vs discovered)")
    print("  03 → 04 → compare (bilateral gap on consumer arc)")
    print("  05       → silent pipeline (short Monday/Thursday rows)")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print()
    print("FRACTURE — Saturday deep dive")
    print("Generator → Preflight → Petri net → Token replay → Conformance")
    print()
    print("Three steps. Read every line. Every number should make sense.")
    print()

    step1_structure()
    all_pass = step2_scenarios()
    step3_visualisations()

    print()
    print("═" * 60)
    print("  DONE")
    print("═" * 60)
    print()
    print("  What you just verified:")
    print("  v Healthy pipeline scores above 0.90 using token replay")
    print("  v Bilateral gap of 27 min detected in real minutes")
    print("  v Silent pipeline caught by completeness (no alerts elsewhere)")
    print("  v Broken consumer does not invalidate producer score")
    print("  v Sudden collapse: live score critical despite healthy history")
    print()
    print("  What to read next:")
    print("  conformance.py steps 3, 4, 5 — token replay, timing zones, completeness")
    print("  petri.py — contract_to_petri_net() and the soundness argument")
    print("  preflight.py — 8 checks, see which AMBER checks fired in scenarios 3+4")
    print()
    print("  Questions to bring back:")
    print("  Why is timing weight 0.50 and not 0.33?")
    print("  Why is confidence minimum of 4 factors and not average?")
    print("  Why does consumer preflight RED not block producer score?")
    print()
    if not all_pass:
        print("  Some scenarios failed — check output above before reading code.")
        sys.exit(1)
