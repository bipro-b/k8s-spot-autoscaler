"""Run ONE evaluation episode of a policy over the trace, logging every step.

  python -m experiments.run_episode --policy hpa            --out results/hpa_v3.csv
  python -m experiments.run_episode --policy rl_homogeneous --out results/rl_homogeneous_real_v3.csv

  # Validate obs-timing fix first (5 steps only, no CSV written):
  python -m experiments.run_episode --policy rl_homogeneous --out results/rl_check.csv --validate_only

Phase 2/3 design choices (both policies):
  - Load:       Docker k6 on the minikube bridge network -> NodePort 30080.
  - Prometheus: kubectl port-forward on :9090 via stdin=PIPE (stays alive without TTY).
  - Affinity:   workload-deployment.yaml applied before every episode. pods_spot must be 0.
  - Metric:     slo_violation = (p95_client > 0.200 s) OR (shortfall_frac > 0.10).
  - RL obs timing (v3 fix): scale -> settle -> load -> observe.
    Matches sim_env.py step order: policy always acts on under-load observations.
    env.step() is NOT called; the control loop is driven directly from run_episode.py.
"""
import argparse, json, subprocess, os, time
import numpy as np, pandas as pd
from metrics.collect import Metrics

NODEPORT_URL  = "http://192.168.49.2:30080/work"
PROJ_DIR      = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SLO_P95       = 0.200
SLO_SHORTFALL = 0.10
EWMA_ALPHA    = 0.3

_prom_pf_proc = None


# ------------------------------------------------------------------ #
# Infrastructure                                                       #
# ------------------------------------------------------------------ #

def start_prom_pf(local_port=9090, ns="monitoring"):
    """Start Prometheus port-forward. stdin=PIPE keeps kubectl alive (no TTY needed)."""
    global _prom_pf_proc
    if _prom_pf_proc is not None:
        try: _prom_pf_proc.kill(); _prom_pf_proc.wait()
        except Exception: pass
    subprocess.run(
        ["powershell", "-Command",
         f"Get-NetTCPConnection -LocalPort {local_port} -State Listen "
         f"-ErrorAction SilentlyContinue | ForEach-Object "
         f"{{ Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1)
    _prom_pf_proc = subprocess.Popen(
        ["kubectl", "port-forward", "-n", ns,
         "svc/prom-kube-prometheus-stack-prometheus", f"{local_port}:9090"],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(6)
    print(f"  Prometheus port-forward :{local_port}", flush=True)


def drive_load(rps, step_s):
    """Fire Docker k6 at NodePort for one trace step; return client-side p95 in seconds."""
    host_summary = os.path.join(PROJ_DIR, "k6_summary_tmp.json")
    try:
        subprocess.run(
            ["docker", "run", "--rm", "--network", "minikube",
             "-v", f"{PROJ_DIR}:/app",
             "grafana/k6", "run",
             "-e", f"TARGET_RPS={int(rps)}",
             "-e", f"TARGET_URL={NODEPORT_URL}",
             "-e", f"STEP_DURATION={step_s}s",
             "--summary-export=/app/k6_summary_tmp.json",
             "/app/load/k6-replay.js"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=step_s + 60)
    except subprocess.TimeoutExpired:
        print(f"  WARNING: k6 Docker timed out after {step_s+60}s — killing container",
              flush=True)
        subprocess.run(["docker", "rm", "-f", "k6-step"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return None
    except Exception as e:
        print(f"  WARNING: drive_load error: {e}", flush=True)
        return None
    try:
        with open(host_summary) as f:
            data = json.load(f)
        return data["metrics"]["http_req_duration"]["p(95)"] / 1000.0
    except Exception:
        return None


def ensure_ondemand_affinity():
    """Apply workload-deployment.yaml to restore hard on-demand nodeAffinity."""
    subprocess.run(
        ["kubectl", "apply", "-f",
         os.path.join(PROJ_DIR, "deploy", "workload-deployment.yaml")],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("  on-demand affinity applied (hard requiredDuringScheduling)", flush=True)


def wait_for_pods(target=8, timeout_s=180):
    """Wait until readyReplicas >= target (used for scale-up in warmup)."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        r = subprocess.run(
            ["kubectl", "get", "deployment", "workload", "-n", "bench",
             "-o", "jsonpath={.status.readyReplicas}"],
            capture_output=True, text=True)
        ready = int(r.stdout.strip() or "0")
        print(f"  pods ready: {ready}/{target}", flush=True)
        if ready >= target:
            return True
        time.sleep(10)
    return False


def settle_for_replicas(target, timeout_s=90):
    """Wait until readyReplicas == target (handles scale-up and scale-down).

    Polls every 5 s. Returns actual ready count; warns on timeout.
    """
    deadline = time.time() + timeout_s
    prev = -1
    while time.time() < deadline:
        r = subprocess.run(
            ["kubectl", "get", "deployment", "workload", "-n", "bench",
             "-o", "jsonpath={.status.readyReplicas}"],
            capture_output=True, text=True)
        ready = int(r.stdout.strip() or "0")
        if ready != prev:
            print(f"  settle: {ready} -> {target} ready", flush=True)
            prev = ready
        if ready == target:
            return ready
        time.sleep(5)
    r = subprocess.run(
        ["kubectl", "get", "deployment", "workload", "-n", "bench",
         "-o", "jsonpath={.status.readyReplicas}"],
        capture_output=True, text=True)
    actual = int(r.stdout.strip() or "0")
    print(f"  WARNING settle timeout: target={target} actual={actual}", flush=True)
    return actual


def warmup_rl(step_s):
    """RL warmup: enforce on-demand pin, delete HPA, scale to 8, run warmup k6."""
    ensure_ondemand_affinity()
    subprocess.run(
        ["kubectl", "delete", "hpa", "workload-hpa", "-n", "bench", "--ignore-not-found"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(
        ["kubectl", "scale", "deployment/workload", "-n", "bench", "--replicas=8"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    wait_for_pods(target=8)
    drive_load(40, step_s)
    print("  warmup done", flush=True)


def warmup_hpa(step_s):
    """HPA warmup: enforce on-demand pin, apply HPA, scale to 8 pods, then load."""
    ensure_ondemand_affinity()
    subprocess.run(
        ["kubectl", "rollout", "status", "deployment/workload", "-n", "bench",
         "--timeout=120s"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(
        ["kubectl", "apply", "-f", os.path.join(PROJ_DIR, "deploy", "hpa.yaml")],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("  HPA manifest applied", flush=True)
    subprocess.run(
        ["kubectl", "scale", "deployment/workload", "-n", "bench", "--replicas=8"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    wait_for_pods(target=8)
    drive_load(40, step_s)
    print("  warmup done", flush=True)


# ------------------------------------------------------------------ #
# Metric helpers                                                       #
# ------------------------------------------------------------------ #

def compute_slo(p95_client, offered_rps, served_rps):
    """Return (slo_violation: int, shortfall_frac: float)."""
    p95_c = p95_client if p95_client is not None else 999.0
    shortfall = max(0.0, (offered_rps - served_rps) / max(offered_rps, 1e-9))
    viol = int(p95_c > SLO_P95 or shortfall > SLO_SHORTFALL)
    return viol, shortfall


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy",        required=True)
    ap.add_argument("--trace",         default="load/trace.csv")
    ap.add_argument("--step",          type=int, default=60)
    ap.add_argument("--seed",          type=int, default=0)
    ap.add_argument("--out",           required=True)
    ap.add_argument("--validate_only", action="store_true",
                    help="Run only 5 steps to validate obs timing (no CSV written).")
    a = ap.parse_args()

    trace  = pd.read_csv(a.trace)
    m      = Metrics()
    is_rl  = a.policy.startswith("rl_")

    start_prom_pf()

    if is_rl:
        print("Pre-warming cluster (RL)...", flush=True)
        warmup_rl(a.step)
    else:
        print("Pre-warming cluster (HPA)...", flush=True)
        warmup_hpa(a.step)

    model = None; vn = None; env = None; obs = None

    if is_rl:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from rl.env import AutoscaleEnv

        model = PPO.load(f"results/{a.policy}.zip")
        vecnorm_path = f"results/{a.policy}_vecnorm.pkl"
        vn = None
        if os.path.exists(vecnorm_path):
            _dummy = DummyVecEnv([lambda: AutoscaleEnv(trace,
                                  homogeneous=(a.policy == "rl_homogeneous"))])
            vn = VecNormalize.load(vecnorm_path, _dummy)
            vn.training = False; vn.norm_reward = False
            print(f"  Loaded VecNormalize from {vecnorm_path}", flush=True)

        env = AutoscaleEnv(trace, homogeneous=(a.policy == "rl_homogeneous"))

        # Manual init — bypass env.reset() to avoid an idle Prometheus snapshot.
        # warmup_rl() just finished drive_load(40, step_s), so the cluster is under load.
        # Taking init_snap NOW captures a loaded obs (rps~40, p95~0.15) as step-1's prior.
        env.i = 0
        env._ewma_rps = float(trace.iloc[0]["rps"])   # same prior as sim_env.py reset()
        init_snap = m.snapshot()                       # under-load snapshot from warmup
        env._ewma_rps = (EWMA_ALPHA * init_snap.get("rps", 0.0)
                         + (1 - EWMA_ALPHA) * env._ewma_rps)
        obs = env._obs(init_snap)
        print(f"  init obs: rps={obs[0]:.1f} cpu={obs[1]:.3f} p95={obs[2]:.3f} "
              f"pods={obs[3]:.0f} ewma_feat={obs[7]:.3f}", flush=True)
        if obs[0] < 5.0:
            print("  WARNING: init rps looks idle — warmup may not have populated Prometheus",
                  flush=True)

    VALIDATE_N = 5
    rows = []
    for step_num, (i, row) in enumerate(trace.iterrows()):
        offered_rps = float(row["rps"])

        if model is not None:
            # ============================================================
            # RL step order (v3): scale -> settle -> load -> observe
            # Matches sim_env.py: obs always reflects cluster UNDER load.
            # env.step() is deliberately NOT called (it sleeps 15 s post-load
            # and snapshots an idle cluster, causing the timing mismatch).
            # ============================================================

            # 1. Predict from under-load obs (previous step's snapshot / warmup)
            obs_in = vn.normalize_obs(obs[np.newaxis, :])[0] if vn is not None else obs
            obs_in = np.nan_to_num(obs_in, nan=0.0, posinf=10.0, neginf=-10.0)
            act, _ = model.predict(obs_in, deterministic=True)
            delta = [-2, -1, 0, 1, 2][int(act[0])]

            # Obs-timing validation: print obs the policy acts on for first VALIDATE_N steps
            if step_num < VALIDATE_N:
                print(f"  [obs-check {step_num+1}/{VALIDATE_N}] "
                      f"rps={obs[0]:.1f} cpu={obs[1]:.3f} p95={obs[2]:.3f} "
                      f"pods={obs[3]:.0f} ewma_feat={obs[7]:.3f} -> delta={delta:+d}",
                      flush=True)

            # 2. Apply scale action (Phase 2 = homogeneous: ignore act[1])
            current = env.k.get_replicas()
            target  = int(np.clip(current + delta, 1, 20))
            env.k.set_replicas(target)

            # 3. Settle: poll until readyReplicas == target (no blind sleep)
            actual_pods = settle_for_replicas(target, timeout_s=90)

            # 4. Drive load on the stable, settled cluster
            k6_p95 = drive_load(offered_rps, a.step)

            # 5. Observe immediately after load (under-load Prometheus snapshot)
            snap = m.snapshot()
            env._ewma_rps = (EWMA_ALPHA * snap.get("rps", 0.0)
                             + (1 - EWMA_ALPHA) * env._ewma_rps)
            env.i += 1
            obs = env._obs(snap)   # feeds into next step's prediction

        else:
            # HPA path: unchanged — drive load then snapshot (HPA acts autonomously)
            k6_p95 = drive_load(offered_rps, a.step)
            snap   = m.snapshot()

        # --- Metric computation (identical for HPA and RL) ---
        snap["offered_rps"] = offered_rps
        snap["served_rps"]  = float(snap.get("rps") or 0.0)
        snap["p95_client"]  = k6_p95
        if k6_p95 is not None:
            snap["p95"] = k6_p95

        viol, shortfall = compute_slo(k6_p95, offered_rps, snap["served_rps"])
        snap["slo_violation"]  = viol
        snap["shortfall_frac"] = shortfall
        snap["t_s"]    = row["t_s"]
        snap["policy"] = a.policy
        snap["seed"]   = a.seed
        rows.append(snap)

        p95_str   = f"{k6_p95:.3f}s" if k6_p95 is not None else "None"
        pods_spot = int(snap.get("pods_spot", 0))
        print(f"  step {step_num+1:02d}/60 rps={offered_rps:.0f} "
              f"pods={snap.get('pods', 0):.0f}(spot={pods_spot}) "
              f"p95_c={p95_str} sf={shortfall:.1%} viol={viol}", flush=True)

        if pods_spot > 0:
            print(f"  WARNING: pods_spot={pods_spot} — on-demand affinity not enforced!",
                  flush=True)

        if a.validate_only and step_num + 1 >= VALIDATE_N:
            print(f"\n=== VALIDATION COMPLETE ({VALIDATE_N} steps) ===", flush=True)
            print("Check the [obs-check] lines above:", flush=True)
            print("  GOOD: rps > 5 and p95 > 0.05 -> obs reflects loaded cluster", flush=True)
            print("  BAD : rps ~ 0 and p95 ~ 0    -> obs still idle; fix is broken", flush=True)
            print("Rerun without --validate_only to run the full 60-step episode.", flush=True)
            return

    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"wrote {len(rows)} steps -> {a.out}")


if __name__ == "__main__":
    main()
