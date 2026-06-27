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

SLO = 0.2        # p95 threshold (s)
SLO_SF = 0.10   # shortfall threshold (fraction of offered load)


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


def _violation_rate(df):
    """Use pre-computed slo_violation if present; else fall back to p95_client or p95."""
    if "slo_violation" in df.columns:
        return float(df["slo_violation"].mean())
    # Legacy fallback
    p95_col = "p95_client" if "p95_client" in df.columns else "p95"
    p95_viol = df[p95_col] > SLO
    if "shortfall_frac" in df.columns:
        sf_viol = df["shortfall_frac"] > SLO_SF
        return float((p95_viol | sf_viol).mean())
    return float(p95_viol.mean())


def _shortfall_rate(df):
    if "shortfall_frac" in df.columns:
        return float((df["shortfall_frac"] > SLO_SF).mean())
    if "offered_rps" in df.columns and "served_rps" in df.columns:
        sf = (df["offered_rps"] - df["served_rps"]) / df["offered_rps"].clip(lower=1e-9)
        return float((sf > SLO_SF).mean())
    return float("nan")


def per_run(df):
    """Collapse each (policy, seed) episode into one row of summary metrics."""
    def agg_fn(grp):
        p95_col = "p95_client" if "p95_client" in grp.columns else "p95"
        return pd.Series({
            "cost":              grp["cost_per_hr"].mean(),
            "p95_mean":          grp[p95_col].mean(),
            "p95_max":           grp[p95_col].max(),
            "pods_mean":         grp["pods"].mean() if "pods" in grp.columns else float("nan"),
            "sla_violation_rate": _violation_rate(grp),
            "shortfall_rate":    _shortfall_rate(grp),
        })
    return df.groupby(["policy", "seed"]).apply(agg_fn).reset_index()

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
