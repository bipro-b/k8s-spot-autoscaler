"""Turn the per-run CSVs in results/ into the paper's headline table + figures.

Expects files named like results/<policy>_seed<k>.csv  (or any csv with a
'policy' and 'seed' column). Produces:
  - results/summary.csv      : per-policy mean +/- std of cost, p95, SLA-violation
  - results/significance.txt : Welch t-tests vs the spot-aware agent (p<0.01)
  - results/figures/*.png    : cost-vs-SLO scatter, cost bars, latency CDF

  python experiments/analyze.py results/
"""
import sys, glob, os, numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SLO = 0.2  # p95 threshold (s). TODO: match your app/SLO.

def load(folder):
    frames = []
    for f in glob.glob(os.path.join(folder, "*.csv")):
        if os.path.basename(f) in ("summary.csv",): continue
        df = pd.read_csv(f)
        if "policy" not in df.columns: continue
        frames.append(df)
    if not frames:
        sys.exit("No result CSVs with a 'policy' column found.")
    return pd.concat(frames, ignore_index=True)

def per_run(df):
    """Collapse each (policy, seed) episode into one row of summary metrics."""
    g = df.groupby(["policy", "seed"])
    out = g.agg(
        cost=("cost_per_hr", "mean"),
        p95=("p95", "mean"),
        sla_violation_rate=("p95", lambda s: float((s > SLO).mean())),
    ).reset_index()
    return out

def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "results/"
    os.makedirs(os.path.join(folder, "figures"), exist_ok=True)
    runs = per_run(load(folder))

    summary = runs.groupby("policy").agg(["mean", "std"])
    summary.to_csv(os.path.join(folder, "summary.csv"))
    print("\n=== Summary (mean +/- std over runs) ===")
    print(summary.round(3))

    # Significance: every baseline vs the spot-aware agent
    target = "rl_spot"
    lines = []
    if target in runs.policy.unique():
        base = runs[runs.policy == target]
        for pol in sorted(runs.policy.unique()):
            if pol == target: continue
            other = runs[runs.policy == pol]
            for metric in ["cost", "sla_violation_rate"]:
                t, p = stats.ttest_ind(base[metric], other[metric], equal_var=False)
                sig = "**SIGNIFICANT (p<0.01)**" if p < 0.01 else "n.s."
                lines.append(f"{target} vs {pol:16s} | {metric:18s} "
                             f"t={t:+.2f} p={p:.4g} {sig}")
    txt = "\n".join(lines) if lines else "rl_spot runs not found; add them to compare."
    open(os.path.join(folder, "significance.txt"), "w").write(txt)
    print("\n=== Significance vs rl_spot ===\n" + txt)

    # Figure: the money plot — cost vs SLA violations (lower-left = best)
    plt.figure(figsize=(6, 4.5))
    for pol, grp in runs.groupby("policy"):
        plt.scatter(grp.sla_violation_rate, grp.cost, label=pol, alpha=0.7)
    plt.xlabel("SLA violation rate"); plt.ylabel("Cost ($/hr)")
    plt.title("Cost vs Reliability (lower-left is better)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(folder, "figures", "cost_vs_slo.png"), dpi=160)
    print(f"\nFigures -> {os.path.join(folder, 'figures')}")

if __name__ == "__main__":
    main()
