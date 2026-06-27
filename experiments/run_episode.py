"""Run ONE evaluation episode of a policy over the trace, logging every step.

  python -m experiments.run_episode --policy hpa            --out results/hpa_v2.csv
  python -m experiments.run_episode --policy rl_homogeneous --out results/rl_homogeneous_real_v2.csv

Phase 2 design choices (both policies):
  - Load:       Docker k6 on the minikube bridge network → NodePort 30080.
                Bypasses kubectl port-forward entirely; eliminates the WSL2/API-server
                serialisation bottleneck that inflated p95 at >10 RPS in earlier runs.
  - Prometheus: kubectl port-forward on :9090 via stdin=PIPE (stays alive without TTY).
  - Affinity:   workload-deployment.yaml is applied before every episode to restore the
                hard on-demand nodeAffinity.  pods_spot MUST be 0 for Phase 2.
  - Metric:     slo_violation = (p95_client > 0.200 s) OR (shortfall_frac > 0.10).
                p95_client is the client-side p95 from k6 (user-experienced latency).
                shortfall_frac = max(0, (offered - served) / offered).
"""
import argparse, json, subprocess, os, time
import numpy as np, pandas as pd
from metrics.collect import Metrics

NODEPORT_URL = "http://192.168.49.2:30080/work"
PROJ_DIR     = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SLO_P95      = 0.200   # seconds
SLO_SHORTFALL = 0.10   # fraction of offered load

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
            timeout=step_s + 60)   # hard timeout: step + 60s grace
    except subprocess.TimeoutExpired:
        print(f"  WARNING: k6 Docker timed out after {step_s+60}s — killing container",
              flush=True)
        subprocess.run(["docker", "rm", "-f", "k6-step"], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
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
    """Apply workload-deployment.yaml to restore hard on-demand nodeAffinity.

    This is the Phase 2 invariant: pods_spot == 0 for both HPA and RL.
    The RL k8s_actions.set_node_pref() patches to preferredDuringScheduling (soft),
    which breaks the Phase 2 constraint.  Re-applying the manifest restores the
    requiredDuringScheduling (hard) affinity before each episode.
    """
    subprocess.run(
        ["kubectl", "apply", "-f",
         os.path.join(PROJ_DIR, "deploy", "workload-deployment.yaml")],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("  on-demand affinity applied (hard requiredDuringScheduling)", flush=True)


def wait_for_pods(target=8, timeout_s=180):
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


def warmup_rl(step_s):
    """RL warmup: enforce on-demand pin, delete HPA, scale to 8, run warmup k6."""
    ensure_ondemand_affinity()
    # Delete HPA so RL can control replicas directly
    subprocess.run(
        ["kubectl", "delete", "hpa", "workload-hpa", "-n", "bench",
         "--ignore-not-found"],
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
    # Wait for any rolling update triggered by the affinity patch to finish
    subprocess.run(
        ["kubectl", "rollout", "status", "deployment/workload", "-n", "bench",
         "--timeout=120s"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(
        ["kubectl", "apply", "-f",
         os.path.join(PROJ_DIR, "deploy", "hpa.yaml")],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("  HPA manifest applied", flush=True)
    # Scale to 8 directly so HPA starts from a warm state (HPA will maintain 8 at 40 RPS)
    subprocess.run(
        ["kubectl", "scale", "deployment/workload", "-n", "bench", "--replicas=8"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    wait_for_pods(target=8)
    # One warmup step at 40 RPS to populate Prometheus metrics
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
    ap.add_argument("--policy", required=True)
    ap.add_argument("--trace",  default="load/trace.csv")
    ap.add_argument("--step",   type=int, default=60)
    ap.add_argument("--seed",   type=int, default=0)
    ap.add_argument("--out",    required=True)
    a = ap.parse_args()

    trace  = pd.read_csv(a.trace)
    m      = Metrics()
    is_rl  = a.policy.startswith("rl_")

    # Prometheus port-forward needed for Prometheus metrics (both HPA and RL)
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
        if os.path.exists(vecnorm_path):
            _dummy = DummyVecEnv([lambda: AutoscaleEnv(trace,
                                  homogeneous=(a.policy == "rl_homogeneous"))])
            vn = VecNormalize.load(vecnorm_path, _dummy)
            vn.training = False; vn.norm_reward = False
            print(f"  Loaded VecNormalize from {vecnorm_path}", flush=True)
        env = AutoscaleEnv(trace, homogeneous=(a.policy == "rl_homogeneous"))
        obs, _ = env.reset(seed=a.seed)
        time.sleep(10)
        fresh_snap = env.m.snapshot()
        # Sync EWMA with current observed load after warmup
        env._ewma_rps = (0.3 * float(fresh_snap.get("rps", 0.0))
                         + 0.7 * env._ewma_rps)
        obs = env._obs(fresh_snap)
        print(f"  start obs: pods={fresh_snap.get('pods', 0):.0f} "
              f"rps={fresh_snap.get('rps', 0):.1f} "
              f"ewma_rps={env._ewma_rps:.1f}", flush=True)

    rows = []
    for i, row in trace.iterrows():
        offered_rps = float(row["rps"])
        k6_p95      = drive_load(offered_rps, a.step)

        if model is not None:
            # Snapshot immediately after k6 finishes — before env.step()'s 15s settle
            # sleep — so served_rps reflects the actual traffic rate during this step.
            # env.step() sleeps then re-queries Prometheus for the NEXT step's obs.
            eval_snap = m.snapshot()
            obs_in = vn.normalize_obs(obs[np.newaxis, :])[0] if vn is not None else obs
            obs_in = np.nan_to_num(obs_in, nan=0.0, posinf=10.0, neginf=-10.0)
            act, _ = model.predict(obs_in, deterministic=True)
            obs, _, done, _, info = env.step(act)
            snap = dict(info)
            # Override served_rps with the timely measurement (env.step lags by 15s)
            snap["rps"] = eval_snap.get("rps", snap.get("rps", 0.0))
        else:
            snap = m.snapshot()

        # --- Fair metric computation (identical for HPA and RL) ---
        snap["offered_rps"] = offered_rps
        snap["served_rps"]  = float(snap.get("rps") or 0.0)

        # p95_client: true user-experienced latency from k6 (Docker → NodePort)
        snap["p95_client"]  = k6_p95

        # Override p95 with client-side for logging and analysis consistency
        if k6_p95 is not None:
            snap["p95"] = k6_p95

        viol, shortfall = compute_slo(k6_p95, offered_rps, snap["served_rps"])
        snap["slo_violation"]  = viol
        snap["shortfall_frac"] = shortfall

        snap["t_s"]    = row["t_s"]
        snap["policy"] = a.policy
        snap["seed"]   = a.seed
        rows.append(snap)

        p95_str = f"{k6_p95:.3f}s" if k6_p95 is not None else "None"
        pods_spot = int(snap.get("pods_spot", 0))
        print(f"  step {i+1:02d}/60 rps={offered_rps:.0f} "
              f"pods={snap.get('pods', 0):.0f}(spot={pods_spot}) "
              f"p95_c={p95_str} sf={shortfall:.1%} viol={viol}", flush=True)

        # Phase 2 invariant check: alert if pods land on spot
        if pods_spot > 0:
            print(f"  WARNING: pods_spot={pods_spot} — on-demand affinity not enforced!",
                  flush=True)

    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"wrote {len(rows)} steps -> {a.out}")


if __name__ == "__main__":
    main()
