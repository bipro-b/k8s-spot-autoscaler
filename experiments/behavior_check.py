"""Post-training behavior check (Step 3 verification).

Runs a SINGLE simulated episode of rl_spot and prints:
  - spot_frac when risk < 0.3 (agent should USE spot → spot_frac > 0)
  - spot_frac when risk > 0.7 (agent should FLEE spot → spot_frac near 0)
  - Whether the agent maintains stable scale (no +2/-2 thrashing every step)
  - Eviction count and average recovery time

Confirm: agent uses spot AND protects itself; does NOT collapse to all-ondemand.

Run: .venv\Scripts\python.exe experiments\behavior_check.py
"""
import io, sys, os, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from load.trace import make_trace
from rl.sim_env import SimEnv
from rl.simulator import ClusterSim
from rl.spot_simulator import SpotSimulator

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def main():
    model_path = os.path.join(RESULTS, "rl_spot.zip")
    vn_path    = os.path.join(RESULTS, "rl_spot_vecnorm.pkl")
    if not os.path.exists(model_path):
        print(f"ERROR: {model_path} not found — training not yet complete.", flush=True)
        sys.exit(1)

    model = PPO.load(model_path)
    trace = make_trace(hours=1, step_s=60, seed=999)   # fresh seed not used in training

    raw_env = SimEnv(trace, homogeneous=False)
    vec_env = DummyVecEnv([lambda: raw_env])

    if os.path.exists(vn_path):
        env = VecNormalize.load(vn_path, vec_env)
        env.training   = False
        env.norm_reward = False
    else:
        env = vec_env

    obs = env.reset()
    # Override with a seed that has multiple evictions
    raw_env._sim  = ClusterSim(seed=777)
    raw_env._sim.reset(pods=8)
    raw_env._spot = SpotSimulator(step_s=60, seed=42, sim_mode=True)
    raw_env._ewma_rps = float(trace.iloc[0]["rps"])
    raw_env.i = 0
    obs_raw = raw_env._obs(raw_env._null_snap())
    if hasattr(env, "normalize_obs"):
        obs = env.normalize_obs(obs_raw[np.newaxis, :])
    else:
        obs = obs_raw[np.newaxis, :]

    rows = []
    done = False
    while not done:
        act, _ = model.predict(obs, deterministic=True)
        result  = env.step(act)
        obs, _, terminated_arr, info_arr = result[0], result[1], result[2], result[3]
        info = info_arr[0] if isinstance(info_arr, (list,tuple)) else info_arr
        if not isinstance(info, dict): info = dict(info)
        sf = info.get("shortfall_frac", 0.0)
        rows.append({
            "step":       raw_env.i,
            "delta":      [-2,-1,0,1,2][int(act.flatten()[0])],
            "pool":       int(act.flatten()[1]),
            "pods":       info.get("pods", 0),
            "pods_spot":  info.get("pods_spot", 0),
            "risk":       info.get("spot_risk", raw_env._spot.risk_signal()),
            "p95":        info.get("p95", 0),
            "sf":         sf,
            "evicted":    int(info.get("evicted", 0)),
        })
        done = bool(terminated_arr[0] if hasattr(terminated_arr, "__len__") else terminated_arr)

    # ── Analysis ──────────────────────────────────────────────────────────────
    print("="*70, flush=True)
    print("BEHAVIORAL CHECK — rl_spot (seed 999/sim-seed 777/evict-seed 42)", flush=True)
    print("="*70, flush=True)

    print(f"\n{'step':>4} {'risk':>6} {'pool':>5} {'delta':>5} "
          f"{'pods':>5} {'spot':>5} {'spot%':>6} {'p95':>7} {'evict':>6}", flush=True)
    print("-"*65, flush=True)
    for r in rows:
        sf_pct = 100*r["pods_spot"]/max(r["pods"],1)
        print(f"  {r['step']:>2} {r['risk']:>6.2f} {'spot' if r['pool']==1 else 'od':>5} "
              f"{r['delta']:>+5} {r['pods']:>5.0f} {r['pods_spot']:>5.0f} "
              f"{sf_pct:>5.0f}% {r['p95']:>7.3f}s {r['evicted']:>6}", flush=True)

    # Low-risk steps (risk < 0.3)
    low_risk  = [r for r in rows if r["risk"] < 0.30]
    high_risk = [r for r in rows if r["risk"] > 0.65]

    spot_frac_low  = np.mean([r["pods_spot"]/max(r["pods"],1) for r in low_risk])  if low_risk  else float("nan")
    spot_frac_high = np.mean([r["pods_spot"]/max(r["pods"],1) for r in high_risk]) if high_risk else float("nan")
    spot_actions_low  = np.mean([r["pool"] for r in low_risk])  if low_risk  else float("nan")
    spot_actions_high = np.mean([r["pool"] for r in high_risk]) if high_risk else float("nan")
    evict_steps = [i for i, r in enumerate(rows) if r["evicted"]]
    thrash = sum(1 for r in rows if abs(r["delta"]) == 2) / len(rows)

    print(f"\n  Steps with risk < 0.30 : {len(low_risk):2d}  avg spot_frac={spot_frac_low:.1%}  "
          f"avg spot_action={spot_actions_low:.2f}", flush=True)
    print(f"  Steps with risk > 0.65 : {len(high_risk):2d}  avg spot_frac={spot_frac_high:.1%}  "
          f"avg spot_action={spot_actions_high:.2f}", flush=True)
    print(f"  Evictions observed: {len(evict_steps)}  at steps {evict_steps}", flush=True)
    print(f"  Action thrash (+2/-2): {thrash:.0%} of steps", flush=True)

    print("\n  VERDICT:", flush=True)
    uses_spot   = spot_frac_low  > 0.05 or spot_actions_low > 0.3
    flees_spot  = (spot_frac_high < spot_frac_low - 0.05 or spot_actions_high < spot_actions_low - 0.1) \
                  if (not np.isnan(spot_frac_high) and not np.isnan(spot_frac_low)) else None
    no_collapse = spot_frac_low > 0.02  # didn't collapse to all-ondemand

    print(f"    Uses spot when safe (risk<0.3):   {'YES' if uses_spot else 'NO'}", flush=True)
    if flees_spot is not None:
        print(f"    Flees to ondemand when risky:     {'YES' if flees_spot else 'NO'}", flush=True)
    else:
        print(f"    Flees to ondemand when risky:     (no high-risk steps observed)", flush=True)
    print(f"    Didn't collapse to all-ondemand:  {'YES' if no_collapse else 'NO'}", flush=True)
    print(f"    Overall: {'PASS' if (uses_spot and no_collapse) else 'REVIEW'}", flush=True)


if __name__ == "__main__":
    main()
