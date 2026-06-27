"""Step 7: Evaluate RL vs HPA in the simulator over multiple seeds.

Usage:
    python -m rl.sim_eval                        # compare rl_homogeneous vs sim-HPA
    python -m rl.sim_eval --seeds 20 --plot
"""
import argparse, os, sys
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl.simulator import ClusterSim, hpa_desired
from rl.sim_env    import SimEnv
from rl.gate_validate import run_sim_hpa

SLO      = 0.2
TRACE    = "load/trace.csv"


def run_rl_episode(model, trace: pd.DataFrame, seed: int = 0,
                   homogeneous: bool = True,
                   vecnorm_path: str = None) -> pd.DataFrame:
    raw_env = SimEnv(trace, homogeneous=homogeneous)
    vec_env = DummyVecEnv([lambda: raw_env])

    if vecnorm_path and os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, vec_env)
        env.training   = False
        env.norm_reward = False
    else:
        env = vec_env

    obs = env.reset()
    rows = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        result = env.step(action)
        obs, _, terminated_arr, info_arr = result[0], result[1], result[2], result[3]
        info = info_arr[0]
        info["seed"] = seed
        rows.append(info)
        done = bool(terminated_arr[0])
    return pd.DataFrame(rows)


def stats(df: pd.DataFrame, label: str) -> dict:
    n       = len(df)
    viols   = int((df["p95"] > SLO).sum())
    rate    = viols / n
    cost    = df["cost_per_hr"].mean()
    p95_med = df["p95"].median()
    print(f"{label:35s}  viol={viols:3d}/{n} ({rate:.1%})  "
          f"mean_cost=${cost:.2f}/hr  median_p95={p95_med:.3f}s")
    return dict(viols=viols, n=n, viol_rate=rate, mean_cost=cost, median_p95=p95_med)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",   default="results/rl_homogeneous.zip")
    ap.add_argument("--vecnorm", default="results/rl_homogeneous_vecnorm.pkl")
    ap.add_argument("--seeds",   type=int, default=10)
    ap.add_argument("--plot",    action="store_true")
    a = ap.parse_args()

    trace = pd.read_csv(TRACE)

    print("Loading model:", a.model)
    if a.vecnorm and os.path.exists(a.vecnorm):
        print("Loading VecNormalize stats:", a.vecnorm)
    model = PPO.load(a.model)

    print("=" * 70)
    print("Step 7 — RL vs HPA comparison in simulator")
    print("-" * 70)

    hpa_rates, hpa_costs, hpa_peak_p95 = [], [], []
    rl_rates,  rl_costs,  rl_peak_p95  = [], [], []

    BURST_RPS = 100.0  # steps with offered_rps > this are burst steps

    for s in range(a.seeds):
        hpa_df = run_sim_hpa(trace, seed=s)
        rl_df  = run_rl_episode(model, trace, seed=s, vecnorm_path=a.vecnorm)

        h = stats(hpa_df, f"Sim-HPA  seed={s}")
        r = stats(rl_df,  f"RL-homo  seed={s}")

        hpa_rates.append(h["viol_rate"]); hpa_costs.append(h["mean_cost"])
        rl_rates.append(r["viol_rate"]);  rl_costs.append(r["mean_cost"])

        # Peak p95 during burst window
        hpa_burst = hpa_df[hpa_df["offered_rps"] > BURST_RPS]["p95"] if "offered_rps" in hpa_df.columns else hpa_df.tail(3)["p95"]
        rl_burst  = rl_df[rl_df["offered_rps"]  > BURST_RPS]["p95"] if "offered_rps"  in rl_df.columns  else rl_df.tail(3)["p95"]
        hpa_peak_p95.append(float(hpa_burst.max()) if len(hpa_burst) else float("nan"))
        rl_peak_p95.append( float(rl_burst.max())  if len(rl_burst)  else float("nan"))

    print()
    print(f"{'':35s}  {'HPA':>20s}  {'RL-homogeneous':>20s}")
    print(f"{'Violation rate (mean+-std)':35s}  "
          f"{np.mean(hpa_rates):.1%}+-{np.std(hpa_rates):.1%}   "
          f"{np.mean(rl_rates):.1%}+-{np.std(rl_rates):.1%}")
    print(f"{'Mean cost (mean+-std)':35s}  "
          f"${np.mean(hpa_costs):.2f}+-{np.std(hpa_costs):.2f}   "
          f"${np.mean(rl_costs):.2f}+-{np.std(rl_costs):.2f}")
    print(f"{'Peak p95 burst (mean)':35s}  "
          f"{np.nanmean(hpa_peak_p95):>18.2f}s   "
          f"{np.nanmean(rl_peak_p95):>18.2f}s")

    cost_saving = (np.mean(hpa_costs) - np.mean(rl_costs)) / np.mean(hpa_costs)
    viol_delta  = np.mean(rl_rates) - np.mean(hpa_rates)
    peak_improvement = np.nanmean(hpa_peak_p95) - np.nanmean(rl_peak_p95)

    print()
    print(f"RL cost saving       : {cost_saving:.1%}")
    print(f"RL violation delta   : {viol_delta:+.1%} vs HPA")
    print(f"RL burst peak p95 cut: {peak_improvement:.2f}s")

    if cost_saving > 0 and viol_delta <= 0.05:
        print("RESULT: PHASE-2 PASS — RL lower cost, comparable SLO.")
    elif viol_delta < -0.05 and cost_saving > -0.30:
        print("RESULT: PHASE-2 PASS — RL dramatically fewer violations "
              f"({-viol_delta:.1%} improvement), cost delta {-cost_saving:.1%}.")
    elif cost_saving <= 0 and viol_delta >= 0:
        print("RESULT: FAIL — RL neither cheaper nor fewer violations.")
    elif cost_saving <= 0:
        print("RESULT: RL did NOT reduce cost -- retrain with higher w_cost.")
    else:
        print("RESULT: RL reduced cost but SLO degraded -- increase w_slo.")

    if a.plot:
        try:
            import matplotlib.pyplot as plt
            hpa0 = run_sim_hpa(trace, seed=0)
            rl0  = run_rl_episode(model, trace, seed=0)
            t    = trace["t_s"]

            fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

            axes[0].plot(hpa0["t_s"], hpa0["p95"], label="HPA", lw=1.2)
            axes[0].plot(rl0.get("t_s", range(len(rl0))), rl0["p95"], label="RL", lw=1.2)
            axes[0].axhline(SLO, color="red", ls="--", lw=0.8, label="SLO")
            axes[0].set_ylabel("p95 (s)"); axes[0].set_yscale("symlog", linthresh=0.5)
            axes[0].legend(loc="upper left")

            axes[1].step(hpa0["t_s"], hpa0["pods"],  label="HPA pods",  lw=1.2)
            axes[1].step(rl0.get("t_s", range(len(rl0))), rl0["pods"], label="RL pods", lw=1.2)
            axes[1].set_ylabel("pods"); axes[1].legend(loc="upper left")

            axes[2].step(hpa0["t_s"], hpa0["cost_per_hr"],  label="HPA cost",  lw=1.2)
            axes[2].step(rl0.get("t_s", range(len(rl0))), rl0["cost_per_hr"], label="RL cost", lw=1.2)
            axes[2].set_ylabel("cost $/hr"); axes[2].set_xlabel("time (s)")
            axes[2].legend(loc="upper left")

            plt.tight_layout()
            plt.savefig("results/sim_eval.png", dpi=150)
            print("Saved results/sim_eval.png")
            plt.show()
        except ImportError:
            print("(matplotlib not available)")


if __name__ == "__main__":
    main()
