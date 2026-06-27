"""Train the autoscaler with PPO (or DQN). Two modes:
  python rl/train.py --sim               -> fast sim training (Phase 2)
  python rl/train.py --sim --homogeneous -> spot-blind RL ablation baseline
  python rl/train.py                     -> live-cluster training (Phase 3+)

Safe exploration: the first `bootstrap_steps` env steps use HPA-imitation actions
so SLOs are protected during early exploration. Implemented via ActionOverrideCallback.
"""
import argparse, numpy as np, pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


# ------------------------------------------------------------------ #
# HPA imitation action                                                #
# ------------------------------------------------------------------ #

def hpa_action(obs: np.ndarray,
               cpu_request: float = 0.1, target_util: float = 0.6) -> np.ndarray:
    """Mimic HPA from the observation vector [rps,cpu_util,p95,pods,sf,risk,hod].

    cpu_util (obs[1]) = avg per container including pause → multiply by 2 for workload.
    HPA target = target_util × cpu_request = 0.06 cores per workload pod.
    """
    pods     = max(1, int(round(float(obs[3]))))
    cpu_avg  = float(obs[1])
    cpu_pod  = cpu_avg * 2.0                          # undo pause-container averaging
    desired  = int(np.ceil(pods * cpu_pod / (cpu_request * target_util)))
    desired  = int(np.clip(desired, 1, 8))
    delta    = int(np.clip(desired - pods, -2, 2))
    d        = delta + 2                              # map {-2,-1,0,+1,+2} → {0,1,2,3,4}
    return np.array([d, 0], dtype=np.int64)           # placement = ondemand (safe)


# ------------------------------------------------------------------ #
# Bootstrap callback                                                  #
# ------------------------------------------------------------------ #

class BootstrapCallback(BaseCallback):
    """Override agent actions with HPA imitation for the first N env steps."""

    def __init__(self, bootstrap_steps: int, verbose: int = 0):
        super().__init__(verbose)
        self.bootstrap_steps = bootstrap_steps

    def _on_step(self) -> bool:
        if self.num_timesteps > self.bootstrap_steps:
            return True
        # Replace the action the model just chose with the HPA imitation action
        obs = self.locals["obs_tensor"].cpu().numpy()   # shape (n_envs, obs_dim)
        for i in range(len(self.locals["actions"])):
            self.locals["actions"][i] = hpa_action(obs[i])
        return True


# ------------------------------------------------------------------ #
# Main                                                                #
# ------------------------------------------------------------------ #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim",         action="store_true",
                    help="Train in fast simulator (no live cluster required)")
    ap.add_argument("--homogeneous", action="store_true",
                    help="Spot-blind ablation: disable spot pool and risk term")
    ap.add_argument("--trace",       default="load/trace.csv")
    ap.add_argument("--timesteps",   type=int, default=200_000,
                    help="Total PPO training steps (sim is fast → use large budget)")
    ap.add_argument("--bootstrap",   type=int, default=10_000,
                    help="Steps to imitate HPA before unleashing exploration")
    ap.add_argument("--w_cost",   type=float, default=1.0)
    ap.add_argument("--w_slo",   type=float, default=10.0)
    ap.add_argument("--w_risk",  type=float, default=2.0)
    ap.add_argument("--w_smooth", type=float, default=0.3)
    a = ap.parse_args()

    trace = pd.read_csv(a.trace)

    if a.sim:
        from rl.sim_env import SimEnv
        raw_env = SimEnv(trace, homogeneous=a.homogeneous,
                         w_cost=a.w_cost, w_slo=a.w_slo, w_risk=a.w_risk, w_smooth=a.w_smooth)
        # VecNormalize: normalises obs (mean≈0 std≈1) and rewards to [-1,1].
        # Critical here because obs features span very different scales:
        # rps(0-70), p95(0-60), pods(1-20), next_rps(0-1.05), hod(0-24).
        vec_env = DummyVecEnv([lambda: raw_env])
        env     = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        print(f"Training in SIMULATOR  homogeneous={a.homogeneous}  "
              f"timesteps={a.timesteps:,}  VecNormalize=ON")
    else:
        from rl.env import AutoscaleEnv
        raw_env = AutoscaleEnv(trace, homogeneous=a.homogeneous,
                               w_cost=a.w_cost, w_slo=a.w_slo, w_risk=a.w_risk)
        env     = DummyVecEnv([lambda: raw_env])
        print(f"Training on LIVE CLUSTER  homogeneous={a.homogeneous}  "
              f"timesteps={a.timesteps:,}  bootstrap={a.bootstrap:,}")

    model = PPO(
        "MlpPolicy", env,
        verbose=1,
        n_steps=1024,
        batch_size=128,
        gamma=0.95,
        learning_rate=3e-4,
        ent_coef=0.05,    # raised from 0.01 — forces exploration of delta=±1,±2
        clip_range=0.2,
    )

    if a.sim:
        # No bootstrap in sim: env resets to pods=8, and w_slo makes 1-pod clearly
        # worse than 8-pod, so random exploration quickly learns to stay near 8 pods.
        model.learn(total_timesteps=a.timesteps)
        name = "rl_homogeneous" if a.homogeneous else "rl_spot"
        env.save(f"results/{name}_vecnorm.pkl")
    else:
        callback = BootstrapCallback(bootstrap_steps=a.bootstrap)
        model.learn(total_timesteps=a.timesteps, callback=callback)

    name = "rl_homogeneous" if a.homogeneous else "rl_spot"
    model.save(f"results/{name}.zip")
    print(f"Saved results/{name}.zip")


if __name__ == "__main__":
    main()
