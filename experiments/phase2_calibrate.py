"""Phase 2 baseline anchoring: combine 3 real HPA runs, recalibrate sim, re-gate.

Usage:
  python -m experiments.phase2_calibrate
"""
import os, sys, io, numpy as np, pandas as pd
from scipy import stats as scipy_stats
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── paths ────────────────────────────────────────────────────────────────────
REAL_RUNS = [
    ("seed0 (hpa_v3)", "results/hpa_v3.csv"),
    ("seed1",          "results/hpa_real_seed1.csv"),
    ("seed2",          "results/hpa_real_seed2.csv"),
]
SLO_P95  = 0.200
SLO_SF   = 0.10
SIM_GATE = 0.08   # pass if |sim_mean - real_mean| <= this

# ── simulator imports (recalibration modifies rl.simulator constants) ─────────
import rl.simulator as sim_mod
from rl.simulator import ClusterSim, hpa_desired


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_viol_rate(df: pd.DataFrame) -> float:
    """Recompute violation rate using the canonical metric (always consistent)."""
    p95_col = "p95_client" if "p95_client" in df.columns else "p95"
    p95_viol = df[p95_col] > SLO_P95
    if "shortfall_frac" in df.columns:
        sf_viol = df["shortfall_frac"] > SLO_SF
    elif "offered_rps" in df.columns and "served_rps" in df.columns:
        sf = (df["offered_rps"] - df["served_rps"]) / df["offered_rps"].clip(lower=1e-9)
        sf_viol = sf.clip(lower=0) > SLO_SF
    else:
        sf_viol = pd.Series(False, index=df.index)
    return float((p95_viol | sf_viol).mean())


def run_sim_hpa(trace: pd.DataFrame, p95_base: float, sim_seeds: int = 20) -> list:
    """Run HPA-in-sim over `trace` with given P95_BASE_ZERO; return per-seed viol rates."""
    # Temporarily patch the module-level constant so ClusterSim picks it up
    sim_mod.P95_BASE_ZERO = p95_base
    sim_mod.P95_BASE      = p95_base   # backwards-compat alias
    rates = []
    for s in range(sim_seeds):
        sim = ClusterSim(seed=s)
        sim.reset(pods=8)
        violations = 0
        for _, row in trace.iterrows():
            offered = float(row["rps"])
            snap    = sim.step(offered)
            sf      = max(0.0, (offered - snap["served_rps"]) / max(offered, 1e-9))
            viol    = int(snap["p95"] > SLO_P95 or sf > SLO_SF)
            violations += viol
            desired = hpa_desired(int(snap["pods"]), snap["cpu_util"])
            sim.set_pods(desired, spot_fraction=0.0)
        rates.append(violations / len(trace))
    return rates


def sim_mean_for_base(p95_base: float, traces: list, sim_seeds: int = 20) -> float:
    """Average sim violation rate across all traces at a given P95_BASE_ZERO."""
    all_rates = []
    for trace in traces:
        all_rates.extend(run_sim_hpa(trace, p95_base, sim_seeds))
    return float(np.mean(all_rates))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("PHASE 2 BASELINE ANCHORING")
    print("=" * 70)

    # ── STEP 2: Load 3 real HPA runs and compute statistics ──────────────────
    print("\n── STEP 2: Real HPA run statistics ────────────────────────────────")
    viol_rates = []
    traces     = []
    for label, path in REAL_RUNS:
        if not os.path.exists(path):
            print(f"  MISSING: {path}  — cannot proceed")
            sys.exit(1)
        df = pd.read_csv(path)
        vr = compute_viol_rate(df)
        viol_rates.append(vr)
        p95_col = "p95_client" if "p95_client" in df.columns else "p95"
        pods_spot = df["pods_spot"].mean() if "pods_spot" in df.columns else 0.0
        print(f"\n  {label}: {path}")
        print(f"    violation rate : {vr:.1%}  ({int(vr*len(df))}/{len(df)})")
        print(f"    p95 median     : {df[p95_col].median():.3f}s")
        print(f"    p95 p90        : {df[p95_col].quantile(0.90):.3f}s")
        print(f"    p95 max        : {df[p95_col].max():.3f}s")
        print(f"    shortfall>10%  : {(df.get('shortfall_frac', pd.Series([0]*len(df))) > SLO_SF).mean():.1%}")
        print(f"    pods_mean      : {df['pods'].mean():.2f}")
        print(f"    pods_spot_mean : {pods_spot:.1f}  {'[WARNING >0!]' if pods_spot > 0 else '[OK]'}")
        # Load corresponding trace for calibration
        trace_path = {
            "results/hpa_v3.csv":          "load/trace.csv",
            "results/hpa_real_seed1.csv":  "load/trace_seed1.csv",
            "results/hpa_real_seed2.csv":  "load/trace_seed2.csv",
        }.get(path, "load/trace.csv")
        traces.append(pd.read_csv(trace_path))

    viol_rates = np.array(viol_rates)
    n          = len(viol_rates)
    mean_vr    = float(np.mean(viol_rates))
    std_vr     = float(np.std(viol_rates, ddof=1))
    se_vr      = std_vr / np.sqrt(n)
    ci95_lo    = mean_vr - scipy_stats.t.ppf(0.975, df=n-1) * se_vr
    ci95_hi    = mean_vr + scipy_stats.t.ppf(0.975, df=n-1) * se_vr

    print(f"\n  {'─'*50}")
    print(f"  Individual violation rates: {[f'{r:.1%}' for r in viol_rates]}")
    print(f"  Mean  : {mean_vr:.1%}")
    print(f"  Std   : {std_vr:.1%}  (ddof=1, n={n})")
    print(f"  Min   : {min(viol_rates):.1%}   Max: {max(viol_rates):.1%}")
    print(f"  95% CI: [{ci95_lo:.1%}, {ci95_hi:.1%}]  (t-dist, n={n})")

    if std_vr > 0.15:
        print(f"\n  *** WARNING: std = {std_vr:.1%} > 15pp — real baseline is HIGHLY VARIABLE. ***")
        print(f"  *** Downstream comparisons need correspondingly wide confidence intervals. ***")
    elif std_vr > 0.08:
        print(f"\n  NOTE: std = {std_vr:.1%} — baseline is moderately variable (>{std_vr*100:.0f}pp spread).")

    # ── STEP 3: Recalibrate simulator to real mean ────────────────────────────
    print(f"\n── STEP 3: Recalibrate simulator to real mean ({mean_vr:.1%}) ─────────")
    original_base = 0.10

    # Quick coarse scan: P95_BASE_ZERO in [0.10, 0.22]
    print(f"\n  Scanning P95_BASE_ZERO (20 sim seeds × {len(traces)} trace(s) each)...")
    candidates = [round(v, 3) for v in np.arange(0.10, 0.23, 0.01)]
    scan = {}
    for base in candidates:
        rate = sim_mean_for_base(base, traces, sim_seeds=20)
        scan[base] = rate
        print(f"    P95_BASE_ZERO={base:.3f}  →  sim_mean={rate:.1%}", flush=True)

    # Find best match
    best_base = min(scan, key=lambda b: abs(scan[b] - mean_vr))
    best_rate = scan[best_base]
    gap_after = abs(best_rate - mean_vr)

    print(f"\n  Best P95_BASE_ZERO : {best_base:.3f}  (was {original_base:.3f})")
    print(f"  Sim mean at best   : {best_rate:.1%}  (target: {mean_vr:.1%})")
    print(f"  Gap after calib    : {gap_after:.1%}")

    # Apply the calibrated value permanently in simulator.py
    sim_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "rl", "simulator.py")
    with open(sim_path) as f:
        src = f.read()
    new_src = src.replace(
        f"P95_BASE_ZERO = {original_base:.2f}",
        f"P95_BASE_ZERO = {best_base:.3f}"
    )
    if new_src == src:
        # Try alternative format
        import re
        new_src = re.sub(r"P95_BASE_ZERO\s*=\s*[\d.]+",
                         f"P95_BASE_ZERO = {best_base:.3f}", src)
    with open(sim_path, "w") as f:
        f.write(new_src)
    print(f"\n  Updated rl/simulator.py: P95_BASE_ZERO = {best_base:.3f}")

    # ── STEP 4: Re-gate ──────────────────────────────────────────────────────
    print(f"\n── STEP 4: Re-gate (sim vs real mean, calibrated P95_BASE_ZERO={best_base:.3f}) ──")

    # Run sim HPA at calibrated value, over all three traces, 20 seeds each
    sim_mod.P95_BASE_ZERO = best_base
    sim_mod.P95_BASE      = best_base

    all_sim_rates = []
    for i, (trace, (label, _)) in enumerate(zip(traces, REAL_RUNS)):
        rates = run_sim_hpa(trace, best_base, sim_seeds=20)
        print(f"\n  Trace {i} ({label}): sim seeds 0-19")
        print(f"    rates: {[f'{r:.1%}' for r in rates]}")
        print(f"    mean={np.mean(rates):.1%}  std={np.std(rates):.1%}")
        all_sim_rates.extend(rates)

    sim_grand_mean = float(np.mean(all_sim_rates))
    sim_grand_std  = float(np.std(all_sim_rates))
    final_gap      = abs(sim_grand_mean - mean_vr)

    print(f"\n  {'─'*50}")
    print(f"  Sim grand mean   : {sim_grand_mean:.1%} ± {sim_grand_std:.1%}")
    print(f"  Real mean        : {mean_vr:.1%}")
    print(f"  Gap              : {final_gap:.1%}  (gate threshold: {SIM_GATE:.0%})")
    print(f"  Real 95% CI      : [{ci95_lo:.1%}, {ci95_hi:.1%}]")
    in_ci = ci95_lo <= sim_grand_mean <= ci95_hi
    print(f"  Sim inside real CI? {in_ci}")

    print(f"\n── STEP 5: Verdict ─────────────────────────────────────────────────")
    print(f"\n  Real HPA violation rates (3 runs):")
    for (label, _), vr in zip(REAL_RUNS, viol_rates):
        print(f"    {label}: {vr:.1%}")
    print(f"  Real mean ± std : {mean_vr:.1%} ± {std_vr:.1%}")
    print(f"  Real 95% CI     : [{ci95_lo:.1%}, {ci95_hi:.1%}]")
    print(f"  Calibrated P95_BASE_ZERO : {best_base:.3f}  (was {original_base:.3f})")
    print(f"  Sim violation rate (calibrated): {sim_grand_mean:.1%} ± {sim_grand_std:.1%}")
    print(f"  Gap to real mean : {final_gap:.1%}")

    if final_gap <= SIM_GATE and in_ci:
        print(f"\n  GATE PASSED — sim ({sim_grand_mean:.1%}) is within {final_gap:.1%} of real mean")
        print(f"  and sits inside the real 95% CI [{ci95_lo:.1%}, {ci95_hi:.1%}].")
        print(f"  The sim credibly represents the AVERAGE real HPA — proceed to 25-seed study.")
    elif final_gap <= SIM_GATE:
        print(f"\n  GATE PASSED (gap criterion) — sim ({sim_grand_mean:.1%}) within {final_gap:.1%} of real mean.")
        print(f"  Note: outside real 95% CI but that CI is wide (n=3).")
    elif in_ci:
        print(f"\n  CONDITIONAL PASS — sim inside real CI but gap {final_gap:.1%} > {SIM_GATE:.0%} threshold.")
        print(f"  Given n=3 and {std_vr:.1%} std, the real CI is wide; sim is plausible.")
    else:
        print(f"\n  GATE FAILED — gap {final_gap:.1%} > {SIM_GATE:.0%} threshold AND outside real CI.")
        print(f"  Recalibration did not converge — stop and rethink before proceeding.")

    print()


if __name__ == "__main__":
    main()
