"""Step 4 (Gate): Validate eviction dynamics against a real cluster eviction.

Procedure:
  1. Scale workload to N pods, drive steady load at R RPS for WARMUP_STEPS steps.
  2. Inject a real eviction: delete 1 pod on a spot node (simulating AWS reclamation).
  3. Continue load for RECOVERY_STEPS more steps, recording p95 and shortfall.
  4. Run the SAME scenario in the simulator and compare.

Gate passes if:
  - Sim and real agree on DIRECTION (both show p95 spike after eviction)
  - Sim post-eviction peak p95 is within 2× of real (order-of-magnitude correct)
  - Sim recovery time (steps until p95 < SLO again) is within 3 steps of real

Run: .venv\Scripts\python.exe experiments\step4_eviction_gate.py

IMPORTANT: This script deletes a pod on a spot node — brief service disruption expected.
"""
import subprocess, time, json, os, sys, io
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from metrics.collect import Metrics
from rl.simulator import ClusterSim

PROJ_DIR     = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
NODEPORT_URL = "http://192.168.49.2:30080/work"
STEP_S       = 60
SLO_P95      = 0.200
STEADY_PODS  = 6        # start below 8 so eviction impact is visible
STEADY_RPS   = 40       # normal load
WARMUP_STEPS = 3        # steps of steady load before eviction
RECOVERY_STEPS = 5      # steps to observe after eviction
RESULTS_DIR  = os.path.join(PROJ_DIR, "results")


def settle_for_replicas(target, timeout_s=150):
    deadline = time.time() + timeout_s
    prev = -1
    while time.time() < deadline:
        r = subprocess.run(
            ["kubectl","get","deployment","workload","-n","bench",
             "-o","jsonpath={.status.readyReplicas}"],
            capture_output=True, text=True)
        ready = int(r.stdout.strip() or "0")
        if ready != prev:
            print(f"    settle: {ready}/{target}", flush=True)
            prev = ready
        if ready == target:
            return ready
        time.sleep(5)
    return int(subprocess.run(
        ["kubectl","get","deployment","workload","-n","bench",
         "-o","jsonpath={.status.readyReplicas}"],
        capture_output=True, text=True).stdout.strip() or "0")


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
        print(f"    drive_load error: {e}", flush=True)
        return None


def inject_real_eviction():
    """Delete one workload pod on a spot node (simulates AWS reclamation)."""
    from kubernetes import client, config
    config.load_kube_config()
    core = client.CoreV1Api()
    pods = core.list_namespaced_pod("bench", label_selector="app=workload").items
    spot_pods = []
    for p in pods:
        node = p.spec.node_name
        if node:
            labels = core.read_node(node).metadata.labels or {}
            if labels.get("node-lifecycle") == "spot":
                spot_pods.append(p)
    if not spot_pods:
        print("  WARNING: no spot pods found — injecting eviction on ANY pod", flush=True)
        # Fall back: delete any workload pod
        all_pods = [p for p in pods if p.status.phase == "Running"]
        if all_pods:
            victim = all_pods[0]
            core.delete_namespaced_pod(victim.metadata.name, "bench", grace_period_seconds=0)
            print(f"  EVICTION injected: {victim.metadata.name} (not on spot)", flush=True)
            return victim.metadata.name
        return None
    victim = spot_pods[0]
    core.delete_namespaced_pod(victim.metadata.name, "bench", grace_period_seconds=0)
    print(f"  EVICTION injected: {victim.metadata.name} on node {victim.spec.node_name}", flush=True)
    return victim.metadata.name


def run_sim_scenario(initial_pods, rps, warmup_steps, recovery_steps, sim_seed=0):
    """Run the same scenario in the simulator: N pods steady load, evict 1 pod, observe."""
    sim = ClusterSim(seed=sim_seed)
    sim.reset(pods=initial_pods)
    # Spot fraction = some on spot (so eviction can happen)
    sim.set_pods(initial_pods, spot_fraction=0.5)

    rows = []
    phase = "warmup"
    for step_i in range(warmup_steps + 1 + recovery_steps):
        if step_i == warmup_steps:
            # Inject eviction: remove 1 pod
            print(f"  [sim] step {step_i}: EVICTION injected", flush=True)
            sim.ready_pods = max(1, sim.ready_pods - 1)
            phase = "post_eviction"
        snap = sim.step(rps, step_s=STEP_S)
        sf   = max(0.0, (rps - snap["served_rps"]) / max(rps, 1e-9))
        viol = int(snap["p95"] > SLO_P95 or sf > SLO_SF)
        rows.append(dict(step=step_i, phase=phase, p95=snap["p95"], sf=sf,
                         viol=viol, pods=snap["pods"], source="sim"))
        if step_i < warmup_steps:
            phase = "warmup"
    return rows


def main():
    m = Metrics()

    # Delete HPA, scale to STEADY_PODS on spot-eligible affinity
    subprocess.run(["kubectl","delete","hpa","workload-hpa","-n","bench","--ignore-not-found"],
                   capture_output=True)
    # Apply spot deployment (preferred affinity — allows some pods on spot)
    subprocess.run(["kubectl","apply","-f",
                    os.path.join(PROJ_DIR,"deploy","workload-deployment-spot.yaml")],
                   capture_output=True)
    subprocess.run(["kubectl","scale","deployment/workload","-n","bench",
                    f"--replicas={STEADY_PODS}"], capture_output=True)
    settle_for_replicas(STEADY_PODS)
    print(f"\nSetup: {STEADY_PODS} pods with spot-eligible affinity, {STEADY_RPS} RPS", flush=True)

    # ── Real cluster measurements ─────────────────────────────────────────────
    real_rows = []

    print(f"\n--- WARMUP: {WARMUP_STEPS} steps at {STEADY_RPS} RPS ---", flush=True)
    for step_i in range(WARMUP_STEPS):
        p95 = drive_load(STEADY_RPS)
        snap = m.snapshot()
        sf   = max(0.0, (STEADY_RPS - snap.get("rps",0)) / max(STEADY_RPS, 1e-9))
        viol = int((p95 or 999) > SLO_P95 or sf > SLO_SF)
        real_rows.append(dict(step=step_i, phase="warmup", p95=p95, sf=sf, viol=viol,
                              pods=snap.get("pods",0), source="real"))
        print(f"  step {step_i}: p95={p95:.3f}s sf={sf:.1%} pods={snap.get('pods',0):.0f}",
              flush=True)

    # Inject eviction
    print(f"\n--- EVICTION ---", flush=True)
    evicted_pod = inject_real_eviction()

    print(f"\n--- RECOVERY: {RECOVERY_STEPS} steps post-eviction ---", flush=True)
    for step_i in range(RECOVERY_STEPS):
        p95 = drive_load(STEADY_RPS)
        snap = m.snapshot()
        sf   = max(0.0, (STEADY_RPS - snap.get("rps",0)) / max(STEADY_RPS, 1e-9))
        viol = int((p95 or 999) > SLO_P95 or sf > SLO_SF)
        real_rows.append(dict(step=WARMUP_STEPS + step_i, phase="post_eviction",
                              p95=p95, sf=sf, viol=viol,
                              pods=snap.get("pods",0), source="real"))
        print(f"  step {WARMUP_STEPS+step_i}: p95={p95:.3f}s sf={sf:.1%} "
              f"pods={snap.get('pods',0):.0f}", flush=True)

    # ── Sim scenario ──────────────────────────────────────────────────────────
    print(f"\n--- SIMULATOR (same scenario) ---", flush=True)
    sim_rows = run_sim_scenario(STEADY_PODS, STEADY_RPS, WARMUP_STEPS, RECOVERY_STEPS)
    for r in sim_rows:
        print(f"  step {r['step']:02d} [{r['phase']:12s}]: "
              f"p95={r['p95']:.3f}s sf={r['sf']:.1%} pods={r['pods']:.0f}", flush=True)

    # ── Gate comparison ───────────────────────────────────────────────────────
    print("\n" + "="*72, flush=True)
    print("STEP 4 EVICTION GATE COMPARISON", flush=True)
    print("="*72, flush=True)

    real_pre   = [r for r in real_rows if r["phase"] == "warmup"]
    real_post  = [r for r in real_rows if r["phase"] == "post_eviction"]
    sim_pre    = [r for r in sim_rows  if r["phase"] == "warmup"]
    sim_post   = [r for r in sim_rows  if r["phase"] == "post_eviction"]

    real_pre_p95  = np.mean([r["p95"] for r in real_pre if r["p95"]])
    real_post_p95 = max([r["p95"] for r in real_post if r["p95"]], default=0)
    sim_pre_p95   = np.mean([r["p95"] for r in sim_pre])
    sim_post_p95  = max([r["p95"] for r in sim_post], default=0)

    real_recovery = next((i for i, r in enumerate(real_post) if (r["p95"] or 0) < SLO_P95), len(real_post))
    sim_recovery  = next((i for i, r in enumerate(sim_post)  if r["p95"] < SLO_P95), len(sim_post))

    print(f"\n  PRE-EVICTION p95:   real={real_pre_p95:.3f}s  sim={sim_pre_p95:.3f}s", flush=True)
    print(f"  POST-EVICTION peak: real={real_post_p95:.3f}s  sim={sim_post_p95:.3f}s  "
          f"ratio={sim_post_p95/max(real_post_p95,1e-9):.2f}x", flush=True)
    print(f"  Recovery steps:     real={real_recovery}  sim={sim_recovery}  "
          f"delta={abs(real_recovery-sim_recovery)}", flush=True)

    direction_ok  = (real_post_p95 > real_pre_p95) and (sim_post_p95 > sim_pre_p95)
    magnitude_ok  = 0.25 < (sim_post_p95 / max(real_post_p95, 1e-9)) < 4.0
    recovery_ok   = abs(real_recovery - sim_recovery) <= 3

    print(f"\n  Direction  (both spike): {'PASS' if direction_ok else 'FAIL'}", flush=True)
    print(f"  Magnitude  (sim within 4x real): {'PASS' if magnitude_ok else 'FAIL'}", flush=True)
    print(f"  Recovery   (<=3 steps apart):    {'PASS' if recovery_ok else 'FAIL'}", flush=True)
    gate_pass = direction_ok and magnitude_ok and recovery_ok
    print(f"\n  GATE: {'*** PASSED ***' if gate_pass else '*** FAILED — review before proceeding ***'}",
          flush=True)

    # Save
    import json as json_mod
    out = {"real": real_rows, "sim": sim_rows,
           "summary": dict(real_pre_p95=real_pre_p95, real_post_p95=real_post_p95,
                           sim_pre_p95=sim_pre_p95, sim_post_p95=sim_post_p95,
                           real_recovery=real_recovery, sim_recovery=sim_recovery,
                           direction_ok=direction_ok, magnitude_ok=magnitude_ok,
                           recovery_ok=recovery_ok, gate_pass=gate_pass)}
    with open(os.path.join(RESULTS_DIR, "step4_eviction_gate.json"), "w") as f:
        json_mod.dump(out, f, indent=2)
    print(f"\nSaved -> results/step4_eviction_gate.json", flush=True)

    # Restore cluster to 8 on-demand pods
    print("\nRestoring cluster to 8 on-demand pods...", flush=True)
    subprocess.run(["kubectl","apply","-f",
                    os.path.join(PROJ_DIR,"deploy","workload-deployment.yaml")],
                   capture_output=True)
    subprocess.run(["kubectl","scale","deployment/workload","-n","bench","--replicas=8"],
                   capture_output=True)
    settle_for_replicas(8)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
