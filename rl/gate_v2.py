"""Step 5 gate: validate simulator under the NEW metric before retraining.

New SLO violation definition (consistent everywhere):
    p95 > 0.200 s  OR  served_rps < 0.90 * offered_rps

Usage:
    python -m rl.gate_v2                          # sim only
    python -m rl.gate_v2 --real results/hpa_v2.csv  # sim vs real comparison
    python -m rl.gate_v2 --seeds 20
"""
import argparse, sys
import numpy as np
import pandas as pd
from rl.simulator import ClusterSim, hpa_desired

SLO_P95      = 0.200
SLO_SHORTFALL = 0.10
TRACE        = "load/trace.csv"
GATE_THRESH  = 0.05   # 5 percentage-point tolerance


def run_sim_hpa(trace: pd.DataFrame, seed: int = 0, warmup_pods: int = 8) -> pd.DataFrame:
    """Run HPA control policy inside ClusterSim with new violation metric."""
    sim = ClusterSim(seed=seed)
    sim.reset(pods=warmup_pods)
    rows = []
    for _, row in trace.iterrows():
        offered_rps = float(row["rps"])
        snap        = sim.step(offered_rps)
        shortfall   = max(0.0, (offered_rps - snap["served_rps"]) / max(offered_rps, 1e-9))
        slo_viol    = int(snap["p95"] > SLO_P95 or shortfall > SLO_SHORTFALL)
        rows.append({
            **snap,
            "t_s":            row["t_s"],
            "seed":           seed,
            "offered_rps":    offered_rps,
            "shortfall_frac": shortfall,
            "slo_violation":  slo_viol,
            "p95_client":     snap["p95"],   # sim p95 = modelled client-side
        })
        desired = hpa_desired(int(snap["pods"]), snap["cpu_util"])
        sim.set_pods(desired, spot_fraction=0.0)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, label: str) -> dict:
    n        = len(df)
    viols    = int(df["slo_violation"].sum())
    sf_steps = int((df["shortfall_frac"] > SLO_SHORTFALL).sum()) if "shortfall_frac" in df else 0
    p95_col  = "p95_client" if "p95_client" in df.columns else "p95"
    print(f"\n{label}")
    print(f"  steps:           {n}")
    print(f"  violations:      {viols}/{n}  ({viols/n:.1%})")
    print(f"  shortfall steps: {sf_steps}/{n}  ({sf_steps/n:.1%})")
    print(f"  p95_mean:        {df[p95_col].mean():.3f}s")
    print(f"  p95_median:      {df[p95_col].median():.3f}s")
    print(f"  p95_max:         {df[p95_col].max():.3f}s")
    print(f"  cost_mean:       ${df['cost_per_hr'].mean():.2f}/hr")
    if "pods" in df.columns:
        print(f"  pods_mean:       {df['pods'].mean():.2f}")
    if "pods_spot" in df.columns:
        print(f"  pods_spot_mean:  {df['pods_spot'].mean():.2f}")
    return dict(viol_rate=viols/n, n=n, viols=viols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--real",  default=None,
                    help="Path to real HPA episode CSV (hpa_v2.csv) for gate comparison")
    a = ap.parse_args()

    trace = pd.read_csv(TRACE)

    print("=" * 70)
    print("GATE v2 — Simulator validation under new SLO metric")
    print(f"  SLO:        p95 > {SLO_P95}s  OR  shortfall > {SLO_SHORTFALL:.0%}")
    print(f"  obs[7]:     EWMA(served_rps, a=0.3)/200  [oracle removed]")
    print("=" * 70)

    # --- Sim HPA ---
    print("\nRunning sim-HPA over trace...")
    sim_rates = []
    for s in range(a.seeds):
        df   = run_sim_hpa(trace, seed=s)
        rate = float(df["slo_violation"].mean())
        sim_rates.append(rate)
        print(f"  seed={s:2d}  viol={df['slo_violation'].sum():3d}/60  "
              f"rate={rate:.1%}  p95_mean={df['p95_client'].mean():.3f}s  "
              f"cost=${df['cost_per_hr'].mean():.2f}/hr")

    sim_mean = float(np.mean(sim_rates))
    sim_std  = float(np.std(sim_rates))
    df0      = run_sim_hpa(trace, seed=0)
    sim_stats = summarise(df0, "Sim-HPA (seed=0, representative)")
    print(f"\n  Sim-HPA mean over {a.seeds} seeds: {sim_mean:.1%} ± {sim_std:.1%}")

    # --- Real HPA v2 ---
    if a.real:
        try:
            real = pd.read_csv(a.real)
            # Ensure slo_violation column uses new rule
            if "slo_violation" not in real.columns:
                p95_col = "p95_client" if "p95_client" in real.columns else "p95"
                sf = ((real.get("offered_rps", real.get("rps", 0)) - real.get("served_rps", real.get("rps", 0)))
                      / real.get("offered_rps", real.get("rps", 1)).clip(lower=1e-9))
                real["shortfall_frac"] = sf.clip(lower=0)
                real["slo_violation"] = ((real[p95_col] > SLO_P95) | (real["shortfall_frac"] > SLO_SHORTFALL)).astype(int)
            real_stats = summarise(real, f"Real HPA v2  ({a.real})")
            real_rate  = real_stats["viol_rate"]

            print("\n" + "=" * 70)
            print("GATE COMPARISON")
            print(f"  Sim-HPA  violation rate : {sim_mean:.1%} ± {sim_std:.1%}")
            print(f"  Real-HPA violation rate : {real_rate:.1%}")
            gap = abs(sim_mean - real_rate)
            print(f"  Gap                     : {gap:.1%}  (threshold = {GATE_THRESH:.0%})")
            print()
            if gap <= GATE_THRESH:
                print("GATE PASSED  — proceed to Step 6 (retrain).")
                rc = 0
            else:
                print("GATE FAILED  — recalibrate simulator before retraining.")
                print("  Hints: tune MU, SAT_THRESHOLD, or P95_SIGMA in rl/simulator.py")
                rc = 1
        except FileNotFoundError:
            print(f"\nReal CSV not found: {a.real}")
            print("Run:  python -m experiments.run_episode --policy hpa --out results/hpa_v2.csv")
            rc = 2
    else:
        print(f"\nSim-only run.  Add --real results/hpa_v2.csv for full gate comparison.")
        rc = 0

    sys.exit(rc)


if __name__ == "__main__":
    main()
