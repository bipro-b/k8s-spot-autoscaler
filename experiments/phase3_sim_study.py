"""Phase 3 Step 5: 4-way paired simulator study.
Policies: hpa_ondemand, naive_spot, rl_homogeneous, rl_spot.
All evaluated under the SAME (trace_seed, eviction_seed) per simulation seed.
Saves results/phase3_sim_seeds.csv.

Run: .venv\Scripts\python.exe experiments\phase3_sim_study.py
"""
import io, sys, os, numpy as np, pandas as pd
from scipy import stats as scipy_stats

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from load.trace import make_trace
from rl.simulator import ClusterSim, hpa_desired
from rl.spot_simulator import SpotSimulator

N_SEEDS     = 25
STEP_S      = 60
SLO_P95     = 0.200
SLO_SF      = 0.10
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── HPA on-demand policy ────────────────────────────────────────────────────

def run_hpa_ondemand(trace, cluster_seed, spot_seed):
    """HPA scaling, all pods on on-demand (spot_fraction=0)."""
    sim  = ClusterSim(seed=cluster_seed)
    spot = SpotSimulator(step_s=STEP_S, seed=spot_seed, sim_mode=True)  # evictions injected but pods=ondemand
    sim.reset(pods=8)

    rows = []
    for _, row in trace.iterrows():
        rps      = float(row["rps"])
        snap     = sim.step(rps, step_s=STEP_S)
        pods_now = int(snap["pods"])
        # HPA: cpu-based desired replicas (capped 1-8)
        desired  = hpa_desired(pods_now, snap["cpu_util"], min_replicas=1, max_replicas=8)
        sim.set_pods(desired, spot_fraction=0.0)
        # SpotSimulator ticks but can't evict ondemand pods (no spot pods)
        spot.step()
        sf   = max(0.0, (rps - snap["served_rps"]) / max(rps, 1e-9))
        viol = int(snap["p95"] > SLO_P95 or sf > SLO_SF)
        rows.append({**snap, "offered_rps": rps, "shortfall_frac": sf,
                     "slo_violation": viol, "evicted": 0, "policy": "hpa_ondemand"})
    return pd.DataFrame(rows)


# ── Naive-spot policy ────────────────────────────────────────────────────────

def run_naive_spot(trace, cluster_seed, spot_seed, spot_frac=0.5):
    """HPA scaling but half the pods placed on spot — no risk awareness."""
    sim  = ClusterSim(seed=cluster_seed)
    spot = SpotSimulator(step_s=STEP_S, seed=spot_seed, sim_mode=True)
    sim.reset(pods=8)
    sim.set_pods(8, spot_fraction=spot_frac)

    rows = []
    for _, row in trace.iterrows():
        rps     = float(row["rps"])
        snap    = sim.step(rps, step_s=STEP_S)
        pods_now = int(snap["pods"])
        desired  = hpa_desired(pods_now, snap["cpu_util"], min_replicas=1, max_replicas=8)
        sim.set_pods(desired, spot_fraction=spot_frac)
        # Spot eviction: may remove 1 pod regardless of risk
        evicted = spot.step()
        if evicted:
            sim.ready_pods = max(1, sim.ready_pods - 1)
        sf   = max(0.0, (rps - snap["served_rps"]) / max(rps, 1e-9))
        viol = int(snap["p95"] > SLO_P95 or sf > SLO_SF)
        rows.append({**snap, "offered_rps": rps, "shortfall_frac": sf,
                     "slo_violation": viol, "evicted": int(evicted),
                     "spot_risk": spot.risk_signal(), "policy": "naive_spot"})
    return pd.DataFrame(rows)


# ── RL homogeneous policy (Phase 2 model) ────────────────────────────────────

def _run_rl_episode(trace, cluster_seed, spot_seed, model, vn_path, homogeneous, policy_name):
    """Shared RL episode runner for both rl_homogeneous and rl_spot."""
    from rl.sim_env import SimEnv
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    raw_env = SimEnv(trace, homogeneous=homogeneous)
    vec_env = DummyVecEnv([lambda: raw_env])

    if vn_path is not None and os.path.exists(str(vn_path)):
        env = VecNormalize.load(str(vn_path), vec_env)
        env.training    = False
        env.norm_reward = False
    else:
        env = vec_env

    # env.reset() initialises raw_env._sim and raw_env._spot — override immediately after
    obs = env.reset()
    raw_env._sim  = ClusterSim(seed=cluster_seed)
    raw_env._sim.reset(pods=8)
    raw_env._spot = SpotSimulator(step_s=STEP_S, seed=spot_seed, sim_mode=True)
    raw_env._ewma_rps = float(trace.iloc[0]["rps"])
    raw_env.i         = 0
    # Re-derive initial obs from the new sim state; re-normalise if needed
    obs_raw = raw_env._obs(raw_env._null_snap())
    if hasattr(env, "normalize_obs"):
        obs = env.normalize_obs(obs_raw[np.newaxis, :])
    else:
        obs = obs_raw[np.newaxis, :]

    rows = []
    done = False
    while not done:
        act, _ = model.predict(obs, deterministic=True)
        result = env.step(act)
        obs, _, terminated_arr, info_arr = result[0], result[1], result[2], result[3]
        info = info_arr[0] if isinstance(info_arr, (list, tuple)) else info_arr
        if not isinstance(info, dict):
            info = dict(info)
        sf   = info.get("shortfall_frac", 0.0)
        p95  = info.get("p95_client", info.get("p95", 0))
        viol = int(p95 > SLO_P95 or sf > SLO_SF)
        rows.append({**info, "slo_violation": viol, "policy": policy_name})
        done = bool(terminated_arr[0] if hasattr(terminated_arr, "__len__") else terminated_arr)
    return pd.DataFrame(rows)


def run_rl_homogeneous(trace, cluster_seed, spot_seed, model, vn):
    return _run_rl_episode(trace, cluster_seed, spot_seed, model, vn,
                            homogeneous=True, policy_name="rl_homogeneous")


def run_rl_spot(trace, cluster_seed, spot_seed, model, vn):
    return _run_rl_episode(trace, cluster_seed, spot_seed, model, vn,
                            homogeneous=False, policy_name="rl_spot")


# ── Summarise one episode ─────────────────────────────────────────────────────

def summarise(df, seed, policy):
    p95_col  = "p95_client" if "p95_client" in df.columns else "p95"
    sf_col   = "shortfall_frac"
    viol     = df["slo_violation"].mean() if "slo_violation" in df.columns else (df[p95_col] > SLO_P95).mean()
    p95_max  = float(df[p95_col].max())
    cost     = float(df["cost_per_hr"].mean()) if "cost_per_hr" in df.columns else float("nan")
    pods     = float(df["pods"].mean()) if "pods" in df.columns else float("nan")
    sf_mean  = float(df[sf_col].mean()) if sf_col in df.columns else 0.0
    spot_f   = float(df["pods_spot"].mean() / df["pods"].mean()) if "pods_spot" in df.columns else 0.0

    # Post-eviction metric: for each eviction event, compute peak p95 and shortfall
    # in the 3 steps following the eviction
    post_evict_p95  = float("nan")
    post_evict_sf   = float("nan")
    evict_count     = 0
    if "evicted" in df.columns:
        evict_idx = df.index[df["evicted"] > 0].tolist()
        evict_count = len(evict_idx)
        if evict_idx:
            post_peaks = []
            post_sfs   = []
            for idx in evict_idx:
                window = df.loc[idx : idx + 3]
                post_peaks.append(float(window[p95_col].max()))
                if sf_col in df.columns:
                    post_sfs.append(float(window[sf_col].max()))
            post_evict_p95 = float(np.mean(post_peaks))
            post_evict_sf  = float(np.mean(post_sfs)) if post_sfs else float("nan")

    return dict(seed=seed, policy=policy, viol_rate=float(viol),
                p95_max=p95_max, cost_hr=cost, pods_mean=pods, spot_frac=spot_f,
                sf_rate=sf_mean, evict_count=evict_count,
                post_evict_p95=post_evict_p95, post_evict_sf=post_evict_sf)


# ── Statistical tests ─────────────────────────────────────────────────────────

def paired_test(a_arr, b_arr, name):
    arr_a = np.array(a_arr)
    arr_b = np.array(b_arr)
    diff  = arr_a - arr_b
    t, p  = scipy_stats.ttest_rel(arr_a, arr_b)
    se    = np.std(diff, ddof=1) / np.sqrt(len(diff))
    t_crit = scipy_stats.t.ppf(0.975, df=len(diff) - 1)
    ci    = (float(np.mean(diff) - t_crit * se), float(np.mean(diff) + t_crit * se))
    d     = float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-9))
    print(f"  {name}: mean_diff={np.mean(diff):+.4f}  "
          f"95%CI=[{ci[0]:+.4f},{ci[1]:+.4f}]  t={t:.2f}  p={p:.4f}  d={d:.2f}",
          flush=True)
    return dict(name=name, mean_diff=float(np.mean(diff)), ci_lo=ci[0], ci_hi=ci[1],
                t=float(t), p=float(p), cohens_d=d)


def main():
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from rl.sim_env import SimEnv

    # Load models
    homo_model = PPO.load("results/rl_homogeneous.zip")
    homo_vn    = "results/rl_homogeneous_vecnorm.pkl"

    if not os.path.exists("results/rl_spot.zip"):
        print("ERROR: results/rl_spot.zip not found. Run training first.", flush=True)
        sys.exit(1)
    spot_model = PPO.load("results/rl_spot.zip")
    spot_vn    = "results/rl_spot_vecnorm.pkl" if os.path.exists("results/rl_spot_vecnorm.pkl") else None

    base_trace = pd.read_csv("load/trace.csv")

    print("="*72, flush=True)
    print("PHASE 3 — 4-WAY SIM STUDY  (25 seeds, paired per seed)", flush=True)
    print("="*72, flush=True)

    all_rows = []

    for seed in range(N_SEEDS):
        trace = make_trace(hours=1, step_s=STEP_S, seed=seed)

        # NOTE: all 4 policies use the SAME spot_seed per simulation seed
        # so eviction events are identical — only the policy response differs
        spot_seed = seed + 1000

        print(f"\n--- Seed {seed:02d} ---", flush=True)

        hpa_df  = run_hpa_ondemand(trace, cluster_seed=seed, spot_seed=spot_seed)
        ns_df   = run_naive_spot(trace,   cluster_seed=seed, spot_seed=spot_seed)
        rl_h_df = run_rl_homogeneous(trace, cluster_seed=seed, spot_seed=spot_seed,
                                     model=homo_model, vn=homo_vn)
        rl_s_df = run_rl_spot(trace, cluster_seed=seed, spot_seed=spot_seed,
                               model=spot_model, vn=spot_vn)

        for df, policy in [(hpa_df,"hpa_ondemand"),(ns_df,"naive_spot"),
                            (rl_h_df,"rl_homogeneous"),(rl_s_df,"rl_spot")]:
            s = summarise(df, seed, policy)
            all_rows.append(s)
            evct = int(s["evict_count"])
            pe   = f"{s['post_evict_p95']:.3f}s" if not np.isnan(s["post_evict_p95"]) else "N/A"
            print(f"  {policy:18s} viol={s['viol_rate']:.1%} cost=${s['cost_hr']:.2f}/hr "
                  f"spot={s['spot_frac']:.0%} evict={evct} post_evict_p95={pe}", flush=True)

    out = pd.DataFrame(all_rows)
    out.to_csv(os.path.join(RESULTS_DIR, "phase3_sim_seeds.csv"), index=False)
    print(f"\nSaved -> results/phase3_sim_seeds.csv", flush=True)

    # ── Headline stats ──────────────────────────────────────────────────────
    print("\n" + "="*72, flush=True)
    print("HEADLINE METRICS (mean ± std over 25 seeds)", flush=True)
    print("="*72, flush=True)
    for pol in ["hpa_ondemand", "naive_spot", "rl_homogeneous", "rl_spot"]:
        g = out[out.policy == pol]
        print(f"  {pol:18s}  viol={g.viol_rate.mean():.1%}±{g.viol_rate.std():.1%}  "
              f"cost=${g.cost_hr.mean():.2f}±{g.cost_hr.std():.2f}/hr  "
              f"spot_frac={g.spot_frac.mean():.0%}  "
              f"post_evict_p95={g.post_evict_p95.mean():.3f}s", flush=True)

    # ── Paired significance tests ────────────────────────────────────────────
    def arr(pol, col):
        return out[out.policy == pol][col].values

    print("\n" + "="*72, flush=True)
    print("PAIRED SIGNIFICANCE TESTS", flush=True)
    print("="*72, flush=True)

    print("\n  rl_spot vs hpa_ondemand (COST — thesis claim: rl_spot cheaper):", flush=True)
    paired_test(arr("hpa_ondemand","cost_hr"), arr("rl_spot","cost_hr"), "hpa_cost - rl_spot_cost")

    print("\n  rl_spot vs naive_spot (POST-EVICTION SLO — thesis claim: rl_spot absorbs better):", flush=True)
    naive_pe  = arr("naive_spot",  "post_evict_p95")
    rl_s_pe   = arr("rl_spot",     "post_evict_p95")
    valid     = ~(np.isnan(naive_pe) | np.isnan(rl_s_pe))
    if valid.sum() >= 3:
        paired_test(naive_pe[valid], rl_s_pe[valid], "naive_post_evict_p95 - rl_post_evict_p95")
    else:
        print(f"  Insufficient eviction events for test (n={valid.sum()})", flush=True)

    print("\n  rl_spot vs naive_spot (OVERALL SLO):", flush=True)
    paired_test(arr("naive_spot","viol_rate"), arr("rl_spot","viol_rate"), "naive_viol - rl_spot_viol")

    print("\n  rl_spot vs hpa_ondemand (SLO — verify rl_spot doesn't degrade):", flush=True)
    paired_test(arr("hpa_ondemand","viol_rate"), arr("rl_spot","viol_rate"),
                "hpa_viol - rl_spot_viol (positive means rl_spot better)")


if __name__ == "__main__":
    main()
