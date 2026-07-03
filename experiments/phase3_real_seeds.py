"""Phase 3 Step 5: Run one real-cluster episode for a Phase 3 policy.

Usage:
  .venv\Scripts\python.exe experiments\phase3_real_seeds.py --policy hpa_ondemand --seed 0 --out results/p3_hpa_od_s0.csv
  .venv\Scripts\python.exe experiments\phase3_real_seeds.py --policy naive_spot    --seed 0 --out results/p3_naive_spot_s0.csv
  .venv\Scripts\python.exe experiments\phase3_real_seeds.py --policy rl_spot       --seed 0 --out results/p3_rl_spot_s0.csv
  .venv\Scripts\python.exe experiments\phase3_real_seeds.py --policy rl_homogeneous --seed 0 --out results/p3_rl_homo_s0.csv

Eviction schedule (for spot policies): same SpotSimulator seed per episode seed, so all
spot-eligible policies face the same eviction events per seed.

Phase 3 SLO: same as Phase 2 (client p95 > 0.200s OR shortfall_frac > 0.10).
Harness: v3 order (scale -> settle -> load -> observe).
"""
import argparse, json, subprocess, os, time, sys, io
import numpy as np, pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from metrics.collect import Metrics
from load.trace import make_trace
from rl.spot_simulator import SpotSimulator

NODEPORT_URL  = "http://192.168.49.2:30080/work"
PROJ_DIR      = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SLO_P95       = 0.200
SLO_SHORTFALL = 0.10
EWMA_ALPHA    = 0.3
SPOT_LAMBDA   = 5.0     # evictions/hr (same as SpotSimulator default)
STEP_S        = 60


def settle_for_replicas(target, timeout_s=90):
    deadline = time.time() + timeout_s
    prev = -1
    while time.time() < deadline:
        r = subprocess.run(
            ["kubectl","get","deployment","workload","-n","bench",
             "-o","jsonpath={.status.readyReplicas}"],
            capture_output=True, text=True)
        ready = int(r.stdout.strip() or "0")
        if ready != prev:
            print(f"  settle: {ready} -> {target}", flush=True)
            prev = ready
        if ready == target:
            return ready
        time.sleep(5)
    r2 = subprocess.run(
        ["kubectl","get","deployment","workload","-n","bench",
         "-o","jsonpath={.status.readyReplicas}"],
        capture_output=True, text=True)
    actual = int(r2.stdout.strip() or "0")
    print(f"  WARNING settle timeout: target={target} actual={actual}", flush=True)
    return actual


def drive_load(rps, step_s=STEP_S):
    host_summary = os.path.join(PROJ_DIR, "k6_summary_tmp.json")
    try:
        subprocess.run(
            ["docker","run","--rm","--network","minikube",
             "-v", f"{PROJ_DIR}:/app",
             "grafana/k6","run",
             "-e", f"TARGET_RPS={int(rps)}",
             "-e", f"TARGET_URL={NODEPORT_URL}",
             "-e", f"STEP_DURATION={step_s}s",
             "--summary-export=/app/k6_summary_tmp.json",
             "/app/load/k6-replay.js"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=step_s + 90)
        with open(host_summary) as f:
            data = json.load(f)
        return data["metrics"]["http_req_duration"]["p(95)"] / 1000.0
    except Exception as e:
        print(f"  drive_load error: {e}", flush=True)
        return None


def apply_ondemand_affinity():
    subprocess.run(
        ["kubectl","apply","-f",
         os.path.join(PROJ_DIR,"deploy","workload-deployment.yaml")],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("  applied hard on-demand affinity", flush=True)


def apply_spot_affinity(lifecycle="ondemand"):
    """Use preferred affinity for spot-eligible policies. Bias toward lifecycle."""
    patch = {"spec":{"template":{"spec":{"affinity":{"nodeAffinity":{
        "preferredDuringSchedulingIgnoredDuringExecution":[{
            "weight": 100,
            "preference": {"matchExpressions":[{
                "key":"node-lifecycle","operator":"In","values":[lifecycle]}]}}],
        "requiredDuringSchedulingIgnoredDuringExecution": None
    }}}}}}
    subprocess.run(
        ["kubectl","patch","deployment","workload","-n","bench",
         "--type","merge","-p", json.dumps(patch)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"  applied preferred affinity -> {lifecycle}", flush=True)


def remove_required_affinity():
    """Remove the hard requiredDuring constraint so pods can land on spot nodes."""
    subprocess.run(
        ["kubectl","apply","-f",
         os.path.join(PROJ_DIR,"deploy","workload-deployment-spot.yaml")],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("  applied spot-eligible affinity (no hard constraint)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True,
                    choices=["hpa_ondemand","naive_spot","rl_homogeneous","rl_spot"])
    ap.add_argument("--seed",    type=int, default=0)
    ap.add_argument("--step",    type=int, default=STEP_S)
    ap.add_argument("--out",     required=True)
    a = ap.parse_args()

    trace = make_trace(hours=1, step_s=a.step, seed=a.seed)
    m     = Metrics()

    # Shared eviction schedule: spot_seed = seed + 1000 (same as sim study)
    spot_seed = a.seed + 1000
    use_spot  = a.policy in ("naive_spot", "rl_spot")
    spot_sim  = SpotSimulator(step_s=a.step, lambda_per_hr=SPOT_LAMBDA,
                               seed=spot_seed, sim_mode=False) if use_spot else None

    print(f"\n{'='*70}", flush=True)
    print(f"Phase 3 real eval: policy={a.policy}  seed={a.seed}", flush=True)
    print(f"{'='*70}", flush=True)

    # ── Warmup ───────────────────────────────────────────────────────────────
    subprocess.run(["kubectl","delete","hpa","workload-hpa","-n","bench",
                    "--ignore-not-found"], capture_output=True)
    if a.policy == "hpa_ondemand":
        apply_ondemand_affinity()
        subprocess.run(["kubectl","apply","-f",
                        os.path.join(PROJ_DIR,"deploy","hpa.yaml")],
                       capture_output=True)
        print("  HPA manifest applied", flush=True)
    elif a.policy in ("naive_spot", "rl_spot"):
        remove_required_affinity()
    else:  # rl_homogeneous
        apply_ondemand_affinity()

    subprocess.run(["kubectl","scale","deployment/workload","-n","bench","--replicas=8"],
                   capture_output=True)
    settle_for_replicas(8)
    drive_load(40, step_s=a.step)
    print("  warmup done", flush=True)

    # ── RL model loading ──────────────────────────────────────────────────────
    model = None; vn = None; env = None; obs = None

    if a.policy.startswith("rl_"):
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from rl.env import AutoscaleEnv

        model = PPO.load(f"results/{a.policy}.zip")
        vn_path = f"results/{a.policy}_vecnorm.pkl"
        homogeneous = (a.policy == "rl_homogeneous")

        env = AutoscaleEnv(trace, step_s=a.step, homogeneous=homogeneous)
        env.i = 0
        env._ewma_rps = float(trace.iloc[0]["rps"])

        if os.path.exists(vn_path):
            _dummy = DummyVecEnv([lambda: AutoscaleEnv(trace, step_s=a.step, homogeneous=homogeneous)])
            vn = VecNormalize.load(vn_path, _dummy)
            vn.training = False; vn.norm_reward = False
            print(f"  Loaded VecNormalize {vn_path}", flush=True)

        init_snap = m.snapshot()
        env._ewma_rps = EWMA_ALPHA * init_snap.get("rps",0) + (1-EWMA_ALPHA) * env._ewma_rps
        obs = env._obs(init_snap)
        print(f"  init obs: rps={obs[0]:.1f} p95={obs[2]:.3f} pods={obs[3]:.0f} "
              f"risk={obs[5]:.2f}", flush=True)

    # ── Episode loop ──────────────────────────────────────────────────────────
    rows = []

    for step_num, (i, row) in enumerate(trace.iterrows()):
        offered_rps = float(row["rps"])

        if a.policy == "hpa_ondemand":
            k6_p95 = drive_load(offered_rps, a.step)
            snap   = m.snapshot()
            evicted = 0
            current_pods = int(snap.get("pods", 0))
            pods_spot_cnt = int(snap.get("pods_spot", 0))

        elif a.policy == "naive_spot":
            # Drive load, observe, HPA rule applies. Spot eviction may happen.
            k6_p95 = drive_load(offered_rps, a.step)
            snap   = m.snapshot()
            evicted_bool = spot_sim.step()
            evicted = int(evicted_bool)
            if evicted:
                print(f"  EVICTION at step {step_num+1}!", flush=True)
            current_pods = int(snap.get("pods", 0))
            pods_spot_cnt = int(snap.get("pods_spot", 0))

        else:  # rl_homogeneous or rl_spot
            # v3 order: scale -> settle -> load -> observe
            obs_in = vn.normalize_obs(obs[np.newaxis,:])[0] if vn is not None else obs
            obs_in = np.nan_to_num(obs_in, nan=0.0, posinf=10.0, neginf=-10.0)
            act, _ = model.predict(obs_in, deterministic=True)

            delta = [-2,-1,0,1,2][int(act[0])]
            current = env.k.get_replicas()
            target  = int(np.clip(current + delta, 1, 20))

            # Placement (rl_spot only)
            if a.policy == "rl_spot" and not env.homogeneous:
                lifecycle = "spot" if int(act[1]) == 1 else "ondemand"
                apply_spot_affinity(lifecycle)

            env.k.set_replicas(target)
            actual_pods = settle_for_replicas(target)

            # Spot eviction (rl_spot only): fires BEFORE load step so agent sees impact
            evicted = 0
            if spot_sim is not None:
                evicted_bool = spot_sim.step()
                evicted = int(evicted_bool)
                if evicted:
                    print(f"  EVICTION at step {step_num+1}! risk was {obs[5]:.2f}", flush=True)
                # Always sync risk signal so agent sees warning 2 steps before eviction
                env.spot._risk = spot_sim.risk_signal()

            k6_p95 = drive_load(offered_rps, a.step)
            snap   = m.snapshot()

            # Update env state
            env._ewma_rps = EWMA_ALPHA * snap.get("rps",0) + (1-EWMA_ALPHA) * env._ewma_rps
            env.i += 1
            obs = env._obs(snap)

            current_pods = actual_pods
            pods_spot_cnt = int(snap.get("pods_spot", 0))

        # ── Metrics ──────────────────────────────────────────────────────────
        snap["offered_rps"] = offered_rps
        snap["served_rps"]  = float(snap.get("rps") or 0.0)
        snap["p95_client"]  = k6_p95
        if k6_p95 is not None:
            snap["p95"] = k6_p95

        p95c = k6_p95 if k6_p95 is not None else 999.0
        sf   = max(0.0, (offered_rps - snap["served_rps"]) / max(offered_rps, 1e-9))
        viol = int(p95c > SLO_P95 or sf > SLO_SHORTFALL)

        snap["slo_violation"]  = viol
        snap["shortfall_frac"] = sf
        snap["t_s"]    = row["t_s"]
        snap["policy"] = a.policy
        snap["seed"]   = a.seed
        snap["evicted"] = evicted
        snap["spot_risk"] = spot_sim.risk_signal() if spot_sim else 0.0
        rows.append(snap)

        risk_str = f"risk={spot_sim.risk_signal():.2f}" if spot_sim else ""
        p95s = f"{k6_p95:.3f}s" if k6_p95 else "None"
        print(f"  step {step_num+1:02d}/60 rps={offered_rps:.0f} "
              f"pods={snap.get('pods',0):.0f}(spot={pods_spot_cnt}) "
              f"p95_c={p95s} sf={sf:.1%} viol={viol} evict={evicted} {risk_str}",
              flush=True)

    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"\nwrote {len(rows)} steps -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
