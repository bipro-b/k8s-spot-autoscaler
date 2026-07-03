"""Phase 2 real-cluster final verdict — 3 paired seeds (0/1/2)."""
import sys, io
import pandas as pd, numpy as np
from scipy import stats as scipy_stats

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

SLO_P95 = 0.200
SLO_SF  = 0.10

def load(path, label, seed):
    df = pd.read_csv(path)
    p95_col = "p95_client" if "p95_client" in df.columns else "p95"
    if "shortfall_frac" in df.columns:
        sf = df["shortfall_frac"]
    else:
        sf = ((df["offered_rps"] - df["served_rps"]) / df["offered_rps"].clip(lower=1e-9)).clip(lower=0)
    viol = (df[p95_col] > SLO_P95) | (sf > SLO_SF)
    return dict(
        seed     = seed,
        label    = label,
        n        = len(df),
        viols    = int(viol.sum()),
        viol_rate= float(viol.mean()),
        p95_max  = float(df[p95_col].max()),
        p95_med  = float(df[p95_col].median()),
        pods_mean= float(df["pods"].mean()),
        cost_hr  = float(df["pods"].mean() * 1.0),
    )

pairs = [
    (0,
     load("results/hpa_v3.csv",                 "HPA", 0),
     load("results/rl_homogeneous_real_v3.csv",  "RL",  0)),
    (1,
     load("results/hpa_real_seed1.csv",          "HPA", 1),
     load("results/rl_real_seed1.csv",           "RL",  1)),
    (2,
     load("results/hpa_real_seed2.csv",          "HPA", 2),
     load("results/rl_real_seed2.csv",           "RL",  2)),
]

print("=" * 70)
print("PHASE 2 REAL-CLUSTER CONFIRMATION  (3 paired seeds, v3 harness)")
print("  Policy: rl_homogeneous  |  pods_spot=0  |  on-demand only")
print("=" * 70)
print()
print(f"  {'seed':>4}  {'HPA viols':>12}  {'RL viols':>12}  {'HPA-RL':>7}"
      f"  {'HPA p95max':>10}  {'RL p95max':>9}  {'winner':>6}")
print("  " + "-" * 68)

hpa_vr, rl_vr, hpa_pm, rl_pm, hpa_pods, rl_pods = [], [], [], [], [], []

for sid, h, r in pairs:
    diff = h["viol_rate"] - r["viol_rate"]
    winner = "RL" if diff > 0.005 else ("HPA" if diff < -0.005 else "TIE")
    pm_diff = h["p95_max"] - r["p95_max"]
    print(f"  {sid:>4}  {h['viols']:>3}/{h['n']}={h['viol_rate']:>5.1%}  "
          f"{r['viols']:>3}/{r['n']}={r['viol_rate']:>5.1%}  "
          f"{diff:>+7.1%}  "
          f"{h['p95_max']:>10.3f}s  {r['p95_max']:>9.3f}s  {winner:>6}")
    hpa_vr.append(h["viol_rate"]); rl_vr.append(r["viol_rate"])
    hpa_pm.append(h["p95_max"]);   rl_pm.append(r["p95_max"])
    hpa_pods.append(h["pods_mean"]); rl_pods.append(r["pods_mean"])

print()
mean_diff_vr = float(np.mean(hpa_vr) - np.mean(rl_vr))
mean_diff_pm = float(np.mean(hpa_pm) - np.mean(rl_pm))
print(f"  HPA mean viol: {np.mean(hpa_vr):.1%}  |  RL mean viol: {np.mean(rl_vr):.1%}"
      f"  |  mean diff: {mean_diff_vr:+.1%}")
print(f"  HPA p95_max mean: {np.mean(hpa_pm):.3f}s  |  RL p95_max mean: {np.mean(rl_pm):.3f}s"
      f"  |  mean diff: {mean_diff_pm:+.3f}s")
print(f"  HPA pods mean: {np.mean(hpa_pods):.2f}  |  RL pods mean: {np.mean(rl_pods):.2f}"
      f"  |  cost diff: ${np.mean(rl_pods)-np.mean(hpa_pods):+.2f}/hr")

# Paired t-test (n=3 — low power, directional only)
diff_vr_arr = np.array(hpa_vr) - np.array(rl_vr)
diff_pm_arr = np.array(hpa_pm) - np.array(rl_pm)
t_vr, p_vr = scipy_stats.ttest_rel(hpa_vr, rl_vr)
t_pm, p_pm = scipy_stats.ttest_rel(hpa_pm, rl_pm)
se_vr = np.std(diff_vr_arr, ddof=1) / np.sqrt(3)
se_pm = np.std(diff_pm_arr, ddof=1) / np.sqrt(3)
t_crit = scipy_stats.t.ppf(0.975, df=2)
ci_vr = (mean_diff_vr - t_crit*se_vr, mean_diff_vr + t_crit*se_vr)
ci_pm = (mean_diff_pm - t_crit*se_pm, mean_diff_pm + t_crit*se_pm)

print()
print("  Paired t-test (n=3 — LOW POWER, treat as directional):")
print(f"    violation_rate: diff={mean_diff_vr:+.1%}  95%CI=[{ci_vr[0]:+.1%},{ci_vr[1]:+.1%}]"
      f"  t={t_vr:.2f}  p={p_vr:.3f}")
print(f"    p95_max:        diff={mean_diff_pm:+.3f}s  95%CI=[{ci_pm[0]:+.3f},{ci_pm[1]:+.3f}]s"
      f"  t={t_pm:.2f}  p={p_pm:.3f}")

# ── Per-seed detail ──────────────────────────────────────────────────────────
print()
print("=" * 70)
print("PER-SEED BREAKDOWN")
print("=" * 70)
for sid, h, r in pairs:
    print(f"\n  Seed {sid}:")
    print(f"    HPA: {h['viols']}/{h['n']} viols={h['viol_rate']:.1%}"
          f"  pods={h['pods_mean']:.1f}  p95_max={h['p95_max']:.3f}s")
    print(f"    RL:  {r['viols']}/{r['n']} viols={r['viol_rate']:.1%}"
          f"  pods={r['pods_mean']:.1f}  p95_max={r['p95_max']:.3f}s")
    print(f"    diff: RL saves {h['viol_rate']-r['viol_rate']:+.1%} viols"
          f"  | p95_max: {'HPA worse' if h['p95_max']>r['p95_max'] else 'RL worse'}"
          f" by {abs(h['p95_max']-r['p95_max']):.3f}s"
          f"  | RL uses {r['pods_mean']-h['pods_mean']:+.1f} more pods")

# ── Sim vs Real comparison ────────────────────────────────────────────────────
print()
print("=" * 70)
print("SIMULATION vs REALITY CHECK")
print("=" * 70)
print()
print("  25-seed sim study said:")
print("    violation_rate: HPA=57.9% RL=57.4% diff=+0.5pp  [n.s., p=0.33]")
print("    p95_max:        HPA=18.1s  RL=40.0s  diff=-21.9s  [RL WORSE, p<0.01]")
print("    cost:           RL $0.40/hr MORE expensive         [p<0.01]")
print()
print(f"  3-seed real cluster says:")
print(f"    violation_rate: HPA={np.mean(hpa_vr):.1%} RL={np.mean(rl_vr):.1%}"
      f" diff={mean_diff_vr:+.1%}  [RL BETTER on all 3 seeds]")
print(f"    p95_max:        HPA={np.mean(hpa_pm):.3f}s  RL={np.mean(rl_pm):.3f}s"
      f"  [{'HPA worse' if mean_diff_pm>0 else 'RL worse'} by {abs(mean_diff_pm):.3f}s]")
print(f"    cost:           RL ~${np.mean(rl_pods)-np.mean(hpa_pods):+.2f}/hr MORE"
      f" (consistent with sim)")
print()
print("  Key discrepancy: Sim p95_max was WRONG direction.")
print("    Cause: sim queue divergence at 5-7 pods (MU=8 underestimates real")
print("    throughput at low pod count). Real cluster handles 7 pods fine at")
print("    normal load; sim predicts unbounded queue growth -> p95=60s.")
print("    Real p95_max: RL=3.40s (seed1), 3.20s (seed2) — controlled, not runaway.")

# ── Final verdict ─────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("HONEST FINAL VERDICT — Phase 2 (homogeneous on-demand)")
print("=" * 70)
print()
print(f"  1. VIOLATION RATE: RL wins on all 3 real seeds.")
print(f"       Seed 0: RL {pairs[0][2]['viol_rate']:.1%} vs HPA {pairs[0][1]['viol_rate']:.1%}  ({pairs[0][1]['viol_rate']-pairs[0][2]['viol_rate']:+.1%})")
print(f"       Seed 1: RL {pairs[1][2]['viol_rate']:.1%} vs HPA {pairs[1][1]['viol_rate']:.1%}  ({pairs[1][1]['viol_rate']-pairs[1][2]['viol_rate']:+.1%})")
print(f"       Seed 2: RL {pairs[2][2]['viol_rate']:.1%} vs HPA {pairs[2][1]['viol_rate']:.1%}  ({pairs[2][1]['viol_rate']-pairs[2][2]['viol_rate']:+.1%})")
print(f"       Mean improvement: {mean_diff_vr:+.1%}  (p={p_vr:.3f}, n=3 — directional, not conclusive)")
print()
print(f"  2. MECHANISM: RL's improvement is from normal-load operation, NOT burst prevention.")
print(f"       RL runs 9-12 pods mid-episode vs HPA's constant 8.")
print(f"       Extra pods push normal-load p95 below 0.200s SLO threshold.")
print(f"       Burst handling: both policies fail at TUNNEL_CAP (>70 RPS effective),")
print(f"       both have 75-100% shortfall. RL had 7 pods at burst in seeds 1/2")
print(f"       (had scaled down), so RL was NOT more prepared for burst.")
print()
print(f"  3. WORST-CASE LATENCY: RL marginally BETTER on real cluster.")
print(f"       Mean p95_max: HPA={np.mean(hpa_pm):.2f}s  RL={np.mean(rl_pm):.2f}s")
print(f"       Sim result (RL worse by 21.9s) was an artifact of queue model.")
print()
print(f"  4. COST: RL is MORE expensive.")
print(f"       RL: {np.mean(rl_pods):.1f} pods avg  HPA: {np.mean(hpa_pods):.1f} pods avg")
print(f"       Extra cost: ~${(np.mean(rl_pods)-np.mean(hpa_pods))*1.0:.2f}/hr per episode.")
print(f"       RL's violation rate improvement is bought by over-provisioning,")
print(f"       not by smarter burst prediction.")
print()
print(f"  5. STATISTICAL CAVEAT: n=3 is insufficient for a conclusive claim.")
print(f"       The 25-seed sim study (which controls noise better) showed p=0.33")
print(f"       on violation rate — not significant. The real-cluster direction is")
print(f"       consistent (RL wins all 3) but the margin varies ({pairs[0][1]['viol_rate']-pairs[0][2]['viol_rate']:+.1%}, "
      f"{pairs[1][1]['viol_rate']-pairs[1][2]['viol_rate']:+.1%}, {pairs[2][1]['viol_rate']-pairs[2][2]['viol_rate']:+.1%}).")
print(f"       Real-world day-to-day variability (seen in HPA baseline: 56-63%) is")
print(f"       large enough to explain the apparent wins without RL actually being better.")
print()
print(f"  SUMMARY: RL_homogeneous reduces observed SLO violations by ~{mean_diff_vr:.0%} on")
print(f"  average in 3 real sessions vs HPA, primarily by running more pods. This")
print(f"  improvement is not statistically conclusive and may reflect over-provisioning")
print(f"  rather than smarter scheduling. Cost is higher (~${(np.mean(rl_pods)-np.mean(hpa_pods)):.1f}/hr extra).")
print(f"  Burst prevention is not demonstrated — both policies fail at the NodePort cap.")


if __name__ == "__main__":
    pass
