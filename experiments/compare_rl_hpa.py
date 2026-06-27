"""Compare RL vs HPA results. Run after rl_homogeneous episode completes.

Note on p95 comparison: HPA baseline (hpa.csv) used port-forward to one pod (client-side
p95 ~135ms). RL episode (rl_homogeneous.csv) used Docker k6 to NodePort (client-side p95
~190-280ms). Direct p95 comparison is invalid due to infrastructure difference.
The valid comparison is: pods_mean (scaling efficiency) and cost_per_hr.
"""
import pandas as pd, numpy as np, sys

def load(path):
    df = pd.read_csv(path)
    df["violation"] = (df["p95"] > 0.200).astype(int)
    return df

def report(df, name):
    print(f"\n=== {name} ===")
    print(f"  steps:         {len(df)}")
    print(f"  violations:    {df['violation'].sum()}/{len(df)}  ({df['violation'].mean()*100:.1f}%)")
    print(f"  pods_mean:     {df['pods'].mean():.2f}")
    print(f"  cost_mean:     ${df['cost_per_hr'].mean():.2f}/hr")
    print(f"  p95_median:    {df['p95'].median():.3f}s")
    print(f"  p95_max:       {df['p95'].max():.3f}s")
    if "served_rps" in df.columns:
        mean_rps = pd.to_numeric(df["served_rps"], errors="coerce").mean()
        print(f"  served_rps:    {mean_rps:.1f}")
    if "slo_violation" in df.columns:
        sv = df["slo_violation"].mean()
        print(f"  slo_viol(srv): {sv:.3f}  (server-side Prometheus p95)")
    print(f"  p95 per step (first 10): {list(df['p95'].round(3).head(10))}")

hpa = load("results/hpa.csv")
report(hpa, "HPA baseline (port-forward, 1-pod)")

try:
    rl = load("results/rl_homogeneous.csv")
    report(rl, "RL homogeneous (Docker+NodePort, 8-pod distributed)")

    # Cost and pods comparison
    print("\n=== COMPARISON (valid metrics only) ===")
    print(f"  pods_mean:  HPA={hpa['pods'].mean():.2f}  RL={rl['pods'].mean():.2f}"
          f"  delta={rl['pods'].mean()-hpa['pods'].mean():+.2f}")
    print(f"  cost/hr:    HPA=${hpa['cost_per_hr'].mean():.2f}  RL=${rl['cost_per_hr'].mean():.2f}"
          f"  delta=${rl['cost_per_hr'].mean()-hpa['cost_per_hr'].mean():+.2f}")

    # Burst behavior (steps 34-36 in trace, index 33-35)
    if len(rl) >= 36:
        burst_rl = rl.iloc[33:36]
        burst_hpa = hpa.iloc[33:36]
        print(f"\n  Burst steps 34-36 (210 RPS):")
        print(f"    HPA  pods={burst_hpa['pods'].mean():.1f} p95={burst_hpa['p95'].mean():.3f}s")
        print(f"    RL   pods={burst_rl['pods'].mean():.1f} p95={burst_rl['p95'].mean():.3f}s")
except FileNotFoundError:
    print("\nrl_homogeneous.csv not found — episode not complete yet")
