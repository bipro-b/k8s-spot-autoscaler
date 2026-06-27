"""Quick episode tracer — run once to understand per-step rewards."""
import os, numpy as np, pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from rl.sim_env import SimEnv

trace   = pd.read_csv("load/trace.csv")
model   = PPO.load("results/rl_homogeneous.zip")
raw_env = SimEnv(trace, homogeneous=True, w_cost=2.0, w_slo=50.0)
vec_env = DummyVecEnv([lambda: raw_env])

vecnorm_path = "results/rl_homogeneous_vecnorm.pkl"
if os.path.exists(vecnorm_path):
    env = VecNormalize.load(vecnorm_path, vec_env)
    env.training = False   # freeze stats at eval time
    env.norm_reward = False
    print("Loaded VecNormalize stats.")
else:
    env = vec_env
    print("No VecNormalize stats found — using raw obs.")

obs = env.reset()

total = 0
action_names = {0:"-2", 1:"-1", 2:"0", 3:"+1", 4:"+2"}
print(f"{'step':>4}  {'offered':>7}  {'next_rps_raw':>12}  {'act':>3}  {'p95':>7}  {'pods':>4}  {'reward':>9}")
for i in range(60):
    next_rps_raw = float(obs[0][7]) if hasattr(obs, '__len__') and len(obs.shape) > 1 else float(obs[7])
    action, _ = model.predict(obs, deterministic=True)
    result = env.step(action)
    obs, reward_arr = result[0], result[1]
    # unwrap VecEnv info
    info = result[4][0] if len(result) == 5 else result[3][0]
    reward = float(reward_arr[0]) if hasattr(reward_arr, '__len__') else float(reward_arr)
    total += reward
    mark = " <BURST" if info.get("offered_rps", 0) > 100 else ""
    if i >= 29 and i <= 40:
        print(f"{i:4d}  {info.get('offered_rps',0):7.1f}  {next_rps_raw:12.4f}  "
              f"{action_names[int(action[0][0]) if hasattr(action[0],'__len__') else int(action[0])]:>3}  "
              f"{info.get('p95',0):7.3f}  {int(info.get('pods',0)):4d}  {reward:9.1f}{mark}")

print(f"\nTotal episode reward: {total:.1f}")
print(f"(next_rps_raw > 0.5 means burst signal present in obs)")
