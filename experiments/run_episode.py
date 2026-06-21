"""Run ONE evaluation episode of a policy over the trace, logging every step.
Works for any policy so all baselines share identical measurement code (fairness).

  python experiments/run_episode.py --policy hpa     --out results/hpa.csv
  python experiments/run_episode.py --policy rl_spot --out results/rl_spot.csv --seed 3

For HPA/VPA/KEDA: apply their manifest first; this runner just drives load and logs.
For rl_*: this runner loads the trained model and lets it act.

Run >=20 times per policy with different --seed to get the variance the paper needs.
"""
import argparse, subprocess, pandas as pd
from metrics.collect import Metrics

def drive_load(rps, url, step_s):
    """Fire k6 for one trace step at the target arrival rate (blocking)."""
    subprocess.run(
        ["k6", "run", "-e", f"TARGET_RPS={int(rps)}", "-e", f"TARGET_URL={url}",
         "-e", f"STEP_DURATION={step_s}s", "load/k6-replay.js"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)          # hpa|vpa|keda|rl_homogeneous|rl_spot
    ap.add_argument("--trace", default="load/trace.csv")
    ap.add_argument("--url", default="http://localhost:8080/work")
    ap.add_argument("--step", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    trace = pd.read_csv(a.trace)
    m = Metrics()
    model = None
    if a.policy.startswith("rl_"):
        from stable_baselines3 import PPO
        from rl.env import AutoscaleEnv
        model = PPO.load(f"results/{a.policy}.zip")
        env = AutoscaleEnv(trace, homogeneous=(a.policy == "rl_homogeneous"))
        obs, _ = env.reset(seed=a.seed)

    rows = []
    for i, row in trace.iterrows():
        drive_load(row["rps"], a.url, a.step)            # same load for every policy
        if model is not None:
            act, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, info = env.step(act)
            snap = info
        else:
            snap = m.snapshot()                          # HPA/VPA/KEDA scale themselves
        snap["t_s"] = row["t_s"]; snap["policy"] = a.policy; snap["seed"] = a.seed
        rows.append(snap)

    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"wrote {len(rows)} steps -> {a.out}")

if __name__ == "__main__":
    main()
