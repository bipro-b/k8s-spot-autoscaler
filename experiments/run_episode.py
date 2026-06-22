"""Run ONE evaluation episode of a policy over the trace, logging every step.
Works for any policy so all baselines share identical measurement code (fairness).

  python experiments/run_episode.py --policy hpa     --out results/hpa.csv
  python experiments/run_episode.py --policy rl_spot --out results/rl_spot.csv --seed 3

For HPA/VPA/KEDA: apply their manifest first; this runner just drives load and logs.
For rl_*: this runner loads the trained model and lets it act.

Run >=20 times per policy with different --seed to get the variance the paper needs.
"""
import argparse, json, subprocess, tempfile, os, time, pandas as pd
from metrics.collect import Metrics

def warmup(url, target_pods, step_s):
    """Send load until HPA reaches target_pods, then drain the queue."""
    m = Metrics()
    # Drive continuous load so HPA scales up
    proc = subprocess.Popen(
        ["k6", "run", "-e", "TARGET_RPS=40", "-e", f"TARGET_URL={url}",
         "-e", "STEP_DURATION=300s", "load/k6-replay.js"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 300
    while time.time() < deadline:
        snap = m.snapshot()
        pods = int(snap.get("pods", 0))
        print(f"  warmup: {pods}/{target_pods} pods", flush=True)
        if pods >= target_pods:
            break
        time.sleep(15)
    proc.kill(); proc.wait()
    # One extra step_s drain so in-flight requests clear before episode starts
    drive_load(40, url, step_s)
    print("  warmup done", flush=True)

def drive_load(rps, url, step_s):
    """Fire k6 for one trace step; return client-side p95 latency in seconds."""
    summary = os.path.join(tempfile.gettempdir(), "k6_summary.json")
    subprocess.run(
        ["k6", "run", "-e", f"TARGET_RPS={int(rps)}", "-e", f"TARGET_URL={url}",
         "-e", f"STEP_DURATION={step_s}s",
         f"--summary-export={summary}", "load/k6-replay.js"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        with open(summary) as f:
            data = json.load(f)
        return data["metrics"]["http_req_duration"]["p(95)"] / 1000.0
    except Exception:
        return None

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
    # Pre-warm: scale HPA to max replicas so the episode starts at steady state
    print("Pre-warming cluster...", flush=True)
    warmup(a.url, 8, a.step)

    model = None
    if a.policy.startswith("rl_"):
        from stable_baselines3 import PPO
        from rl.env import AutoscaleEnv
        model = PPO.load(f"results/{a.policy}.zip")
        env = AutoscaleEnv(trace, homogeneous=(a.policy == "rl_homogeneous"))
        obs, _ = env.reset(seed=a.seed)

    rows = []
    for i, row in trace.iterrows():
        k6_p95 = drive_load(row["rps"], a.url, a.step)  # same load for every policy
        if model is not None:
            act, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, info = env.step(act)
            snap = info
        else:
            snap = m.snapshot()                          # HPA/VPA/KEDA scale themselves
        if k6_p95 is not None:
            snap["p95"] = k6_p95   # override with client-side latency (captures queuing)
        snap["offered_rps"] = row["rps"]
        snap["served_rps"] = snap.get("rps")
        snap["t_s"] = row["t_s"]; snap["policy"] = a.policy; snap["seed"] = a.seed
        rows.append(snap)

    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"wrote {len(rows)} steps -> {a.out}")

if __name__ == "__main__":
    main()
