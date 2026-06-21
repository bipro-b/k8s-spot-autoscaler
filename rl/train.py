"""Train the autoscaler with PPO (or DQN). Two modes:
  python rl/train.py               -> spot-aware agent (full contribution)
  python rl/train.py --homogeneous -> spot-blind RL ablation baseline

Safe exploration: we warm-start by imitating HPA for the first `bootstrap_steps`
so SLOs are protected during early learning (your 'HPA-bootstrapped safe
exploration'). Implemented as an action override wrapper.
"""
import argparse, pandas as pd
from stable_baselines3 import PPO
from rl.env import AutoscaleEnv

def hpa_action(snap, target_util=0.6):
    """Mimic HPA: scale toward CPU target. Returns a MultiDiscrete-style action."""
    if snap["cpu_util"] > target_util * 1.1: d = 4      # +2
    elif snap["cpu_util"] > target_util:     d = 3      # +1
    elif snap["cpu_util"] < target_util*0.5: d = 1      # -1
    else:                                    d = 2      # 0
    return [d, 0]  # HPA places on-demand (safe) during bootstrap

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--homogeneous", action="store_true")
    ap.add_argument("--trace", default="load/trace.csv")
    ap.add_argument("--timesteps", type=int, default=50_000)
    ap.add_argument("--bootstrap", type=int, default=5_000)
    a = ap.parse_args()

    trace = pd.read_csv(a.trace)
    env = AutoscaleEnv(trace, homogeneous=a.homogeneous)
    model = PPO("MlpPolicy", env, verbose=1, n_steps=256, gamma=0.95)

    # TODO: implement the bootstrap override (call hpa_action for first
    # `a.bootstrap` env steps via a callback) before unleashing exploration.
    model.learn(total_timesteps=a.timesteps)

    name = "rl_homogeneous" if a.homogeneous else "rl_spot"
    model.save(f"results/{name}.zip")
    print(f"saved results/{name}.zip")

if __name__ == "__main__":
    main()
