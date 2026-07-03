"""Phase 2 twenty-five seed simulator study — HPA vs rl_homogeneous.

For seeds 0..24:
  - Generate a fresh 60-step trace (1 hour, diurnal + burst, seed N)
  - Run HPA rule in ClusterSim (sim_seed=N, same noise realization as RL -> tightest pairing)
  - Run rl_homogeneous policy in SimEnv (sim_seed=N)
  - Aggregate per-episode metrics

Statistical tests: paired t-test + Wilcoxon on violation_rate, cost_hr, p95_max.
Saves: results/phase2_sim_seeds.csv  (50 rows: 25 seeds x 2 policies)

Usage:
  python -m experiments.phase2_sim_study
"""
import sys, io, os
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from load.trace import make_trace
from rl.simulator import ClusterSim, hpa_desired
from rl.sim_env import SimEnv

SLO_P95 = 0.200
SLO_SF  = 0.10
N_SEEDS = 25
OUT_CSV = "results/phase2_sim_seeds.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Episode runners
# ─────────────────────────────────────────────────────────────────────────────

def run_hpa_episode(trace: pd.DataFrame, sim_seed: int) -> list:
    sim = ClusterSim(seed=sim_seed)
    sim.reset(pods=8)
    rows = []
    for _, row in trace.iterrows():
        offered = float(row["rps"])
        snap    = sim.step(offered)
        sf      = max(0.0, (offered - snap["served_rps"]) / max(offered, 1e-9))
        viol    = int(snap["p95"] > SLO_P95 or sf > SLO_SF)
        rows.append({
            "offered_rps":    offered,
            "served_rps":     snap["served_rps"],
            "p95_client":     snap["p95"],
            "pods":           snap["pods"],
            "pods_spot":      snap["pods_spot"],
            "cost_per_hr":    snap["cost_per_hr"],
            "slo_violation":  viol,
            "shortfall_frac": sf,
        })
        desired = hpa_desired(int(snap["pods"]), snap["cpu_util"])
        sim.set_pods(desired, spot_fraction=0.0)
    return rows


def run_rl_episode(trace: pd.DataFrame, model, vn, sim_seed: int) -> list:
    env = SimEnv(trace, homogeneous=True)
    obs, _ = env.reset(seed=sim_seed)
    rows = []
    for _ in range(len(trace)):
        obs_in = vn.normalize_obs(obs[np.newaxis, :])[0] if vn is not None else obs
        obs_in = np.nan_to_num(obs_in, nan=0.0, posinf=10.0, neginf=-10.0)
        act, _ = model.predict(obs_in, deterministic=True)
        obs, _, done, _, info = env.step(act)
        rows.append({
            "offered_rps":    info["offered_rps"],
            "served_rps":     info["served_rps"],
            "p95_client":     info["p95_client"],
            "pods":           info["pods"],
            "pods_spot":      info["pods_spot"],
            "cost_per_hr":    info["cost_per_hr"],
            "slo_violation":  info["slo_violation"],
            "shortfall_frac": info["shortfall_frac"],
        })
    return rows


def summarise_episode(rows: list, policy: str, seed: int) -> dict:
    df = pd.DataFrame(rows)
    return {
        "policy":         policy,
        "seed":           seed,
        "violation_rate": float(df["slo_violation"].mean()),
        "cost_hr":        float(df["cost_per_hr"].mean()),
        "pods_mean":      float(df["pods"].mean()),
        "pods_spot_mean": float(df["pods_spot"].mean()),
        "p95_median":     float(df["p95_client"].median()),
        "p95_p90":        float(df["p95_client"].quantile(0.90)),
        "p95_max":        float(df["p95_client"].max()),
        "shortfall_rate": float((df["shortfall_frac"] > SLO_SF).mean()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Statistics helpers
# ─────────────────────────────────────────────────────────────────────────────

def paired_test(hpa_vals, rl_vals, metric_name: str) -> dict:
    """Paired t-test + Wilcoxon on (HPA - RL). Positive diff = HPA worse."""
    diff      = np.array(hpa_vals) - np.array(rl_vals)
    n         = len(diff)
    mean_diff = float(np.mean(diff))
    std_diff  = float(np.std(diff, ddof=1))
    se_diff   = std_diff / np.sqrt(n)
    t_crit    = scipy_stats.t.ppf(0.975, df=n - 1)
    ci_lo     = mean_diff - t_crit * se_diff
    ci_hi     = mean_diff + t_crit * se_diff

    t_stat, t_pval = scipy_stats.ttest_rel(hpa_vals, rl_vals)
    w_stat, w_pval = scipy_stats.wilcoxon(diff, alternative="two-sided")

    pooled_std = np.sqrt(
        (np.std(hpa_vals, ddof=1) ** 2 + np.std(rl_vals, ddof=1) ** 2) / 2
    )
    cohen_d = mean_diff / pooled_std if pooled_std > 1e-9 else float("nan")

    def sig(p):
        return "p<0.01**" if p < 0.01 else ("p<0.05*" if p < 0.05 else "n.s.")

    print(f"\n  [{metric_name}]  (HPA - RL), positive = HPA worse")
    print(f"    HPA mean +/- std : {np.mean(hpa_vals):.4f} +/- {np.std(hpa_vals, ddof=1):.4f}")
    print(f"    RL  mean +/- std : {np.mean(rl_vals):.4f}  +/- {np.std(rl_vals, ddof=1):.4f}")
    print(f"    Mean diff        : {mean_diff:+.4f}  (std {std_diff:.4f})")
    print(f"    95% CI on diff   : [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"    Paired t-test    : t={t_stat:+.3f}  p={t_pval:.4g}  [{sig(t_pval)}]")
    print(f"    Wilcoxon rank    : W={w_stat:.0f}    p={w_pval:.4g}  [{sig(w_pval)}]")
    print(f"    Cohen's d        : {cohen_d:.3f}")

    return dict(metric=metric_name,
                mean_hpa=float(np.mean(hpa_vals)), std_hpa=float(np.std(hpa_vals, ddof=1)),
                mean_rl=float(np.mean(rl_vals)),   std_rl=float(np.std(rl_vals, ddof=1)),
                mean_diff=mean_diff, ci_lo=ci_lo, ci_hi=ci_hi,
                t_stat=float(t_stat), t_pval=float(t_pval),
                w_stat=float(w_stat), w_pval=float(w_pval),
                cohen_d=cohen_d,
                sig_t=sig(t_pval), sig_w=sig(w_pval))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print(f"PHASE 2 SIM STUDY — {N_SEEDS} seeds, HPA vs rl_homogeneous")
    print("  Sim: P95_BASE_ZERO=0.140 (calibrated v3 to real mean 59.4%)")
    print("  Pairing: same trace seed AND same ClusterSim noise seed per pair")
    print("=" * 70)

    # Load RL model + VecNormalize
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    model = PPO.load("results/rl_homogeneous.zip")
    vn    = None
    vn_path = "results/rl_homogeneous_vecnorm.pkl"
    if os.path.exists(vn_path):
        _dummy_trace = make_trace(1, 60, seed=0)
        _dummy = DummyVecEnv([lambda: SimEnv(_dummy_trace, homogeneous=True)])
        vn = VecNormalize.load(vn_path, _dummy)
        vn.training = False
        vn.norm_reward = False
        print(f"  VecNormalize loaded from {vn_path}")
    print(f"  RL model loaded from results/rl_homogeneous.zip")
    print()

    header = f"  {'seed':>4}  {'HPA%':>6}  {'RL%':>6}  {'diff':>6}  {'HPA p95max':>10}  {'RL p95max':>9}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    summary_rows = []

    for seed in range(N_SEEDS):
        trace    = make_trace(1, 60, seed=seed)
        hpa_rows = run_hpa_episode(trace, sim_seed=seed)
        rl_rows  = run_rl_episode(trace, model, vn, sim_seed=seed)
        hpa_s    = summarise_episode(hpa_rows, "hpa",            seed)
        rl_s     = summarise_episode(rl_rows,  "rl_homogeneous", seed)

        diff_viol = hpa_s["violation_rate"] - rl_s["violation_rate"]
        print(f"  {seed:>4}  {hpa_s['violation_rate']:>6.1%}  {rl_s['violation_rate']:>6.1%}"
              f"  {diff_viol:>+6.1%}  {hpa_s['p95_max']:>10.3f}s  {rl_s['p95_max']:>9.3f}s",
              flush=True)

        summary_rows.extend([hpa_s, rl_s])

    df = pd.DataFrame(summary_rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n  Wrote {len(df)} rows -> {OUT_CSV}")

    # ── Aggregate per-policy ─────────────────────────────────────────────────
    hpa = df[df.policy == "hpa"]
    rl  = df[df.policy == "rl_homogeneous"]

    print("\n" + "=" * 70)
    print("AGGREGATE  (mean +/- std over 25 seeds)")
    print("=" * 70)
    metrics = [
        ("violation_rate", "Violation rate"),
        ("cost_hr",        "Cost/hr ($)   "),
        ("pods_mean",      "Pods mean     "),
        ("p95_median",     "p95 median (s)"),
        ("p95_p90",        "p95 p90 (s)   "),
        ("p95_max",        "p95 max (s)   "),
        ("shortfall_rate", "Shortfall rate"),
    ]
    print(f"  {'Metric':<20}  {'HPA mean +/- std':>22}  {'RL mean +/- std':>22}")
    print("  " + "-" * 68)
    for col, label in metrics:
        h, r = hpa[col], rl[col]
        print(f"  {label:<20}  {h.mean():>10.4f} +/- {h.std():<8.4f}  "
              f"{r.mean():>10.4f} +/- {r.std():<8.4f}")

    # ── Significance tests ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SIGNIFICANCE TESTS  (paired, n=25, same trace+noise per pair)")
    print("=" * 70)

    results = {}
    for col in ("violation_rate", "cost_hr", "p95_max"):
        results[col] = paired_test(hpa[col].values, rl[col].values, col)

    # ── Honest verdict ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("HONEST VERDICT")
    print("=" * 70)

    vr = results["violation_rate"]
    co = results["cost_hr"]
    pm = results["p95_max"]

    viol_sig = vr["t_pval"] < 0.01
    cost_sig = co["t_pval"] < 0.01
    p95m_sig = pm["t_pval"] < 0.01

    print(f"\n  violation_rate : HPA={vr['mean_hpa']:.1%}  RL={vr['mean_rl']:.1%}"
          f"  diff={vr['mean_diff']:+.1%}  CI=[{vr['ci_lo']:+.1%},{vr['ci_hi']:+.1%}]"
          f"  [{vr['sig_t']}]")
    print(f"  cost_hr        : HPA=${co['mean_hpa']:.2f}  RL=${co['mean_rl']:.2f}"
          f"  diff=${co['mean_diff']:+.2f}  [{co['sig_t']}]")
    print(f"  p95_max        : HPA={pm['mean_hpa']:.3f}s  RL={pm['mean_rl']:.3f}s"
          f"  diff={pm['mean_diff']:+.3f}s  [{pm['sig_t']}]")

    print()
    if viol_sig and p95m_sig:
        print("  RL beats HPA on BOTH violation rate AND worst-case p95 (both p<0.01).")
    elif p95m_sig and not viol_sig:
        print("  RL's robust win is worst-case latency (p95_max, p<0.01).")
        print("  Overall violation rate difference is NOT statistically significant (p>=0.01).")
        print("  The ~" + f"{vr['mean_diff']:+.1%}" + " violation-rate gap could be noise.")
    elif viol_sig and not p95m_sig:
        print("  RL beats HPA on violation rate (p<0.01) but p95_max is not significant.")
    else:
        print("  NEITHER violation rate NOR p95_max reaches p<0.01 in simulation.")
        print("  Cannot claim RL outperforms HPA at this significance threshold.")

    if cost_sig:
        print(f"\n  WARNING: Cost difference is significant — RL uses more pods on average.")
        print(f"    HPA pods_mean={hpa['pods_mean'].mean():.2f}  RL pods_mean={rl['pods_mean'].mean():.2f}")
    else:
        print("\n  Cost: no significant difference (both policies at ~8 pods on-demand-only).")

    print()
    print("  Sim is calibrated to real mean (59.4%); real-cluster confirmation")
    print("  (3 seeds already collected: hpa_v3/seed1/seed2) should verify direction.")


if __name__ == "__main__":
    main()
