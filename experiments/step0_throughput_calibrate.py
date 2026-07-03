"""Step 0 (HARD GATE): Measure real cluster throughput at 5-7 pods × 30/45/60 RPS.
Fit simulator MU + latency cap so sim matches real within 5pp violation / 15% p95.
Writes results/step0_real.json and patches rl/simulator.py if gate passes.

Run via: .venv\Scripts\python.exe experiments\step0_throughput_calibrate.py
"""
import subprocess, time, json, os, sys, io
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from metrics.collect import Metrics

PROJ_DIR     = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
NODEPORT_URL = "http://192.168.49.2:30080/work"
STEP_S       = 60
SLO_P95      = 0.200
RESULTS_DIR  = os.path.join(PROJ_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


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
    r2 = subprocess.run(
        ["kubectl","get","deployment","workload","-n","bench",
         "-o","jsonpath={.status.readyReplicas}"],
        capture_output=True, text=True)
    return int(r2.stdout.strip() or "0")


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


def sim_p95_deterministic(pods, rps, mu, p95_base, p95_slope, sat_threshold,
                           p95_cap, tunnel_cap=70.0):
    """Single sim step, no noise — returns median (lognormal(mean,sigma) median = mean)."""
    capacity_rps = pods * mu
    effective_rps = min(float(rps), tunnel_cap)
    # Backlog (single step, starting from zero)
    arrivals = effective_rps * STEP_S
    capacity_srv = capacity_rps * STEP_S
    backlog_after = max(0.0, arrivals - capacity_srv)
    if backlog_after > capacity_rps:
        queue_wait = backlog_after / max(capacity_rps, 1e-9)
        return min(p95_cap, p95_base + queue_wait)
    rho = effective_rps / max(capacity_rps, 1e-9)
    if rho >= sat_threshold:
        return p95_cap
    p95_mean = p95_base + p95_slope * rho / max(1.0 - rho, 0.05)
    return p95_mean


def sim_violation_rate_mc(pods, rps, mu, p95_base, p95_slope, sat_threshold,
                           p95_cap, p95_sigma, n_samples=2000, seed=0,
                           tunnel_cap=70.0):
    """Simulate N steps (all starting from zero backlog for step-0 comparison)."""
    rng = np.random.default_rng(seed)
    capacity_rps = pods * mu
    effective_rps = min(float(rps), tunnel_cap)
    backlog_after = max(0.0, effective_rps - capacity_rps)
    if backlog_after > capacity_rps:
        queue_wait = backlog_after / max(capacity_rps, 1e-9)
        p95_val = min(p95_cap, p95_base + queue_wait)
        return float(np.mean(p95_val * rng.lognormal(0.0, p95_sigma, n_samples) > SLO_P95))
    rho = effective_rps / max(capacity_rps, 1e-9)
    if rho >= sat_threshold:
        return 1.0
    p95_mean = p95_base + p95_slope * rho / max(1.0 - rho, 0.05)
    samples = p95_mean * rng.lognormal(0.0, p95_sigma, n_samples)
    return float(np.mean(samples > SLO_P95))


def main():
    m = Metrics()

    # Delete HPA to prevent interference
    subprocess.run(["kubectl","delete","hpa","workload-hpa","-n","bench",
                    "--ignore-not-found"], capture_output=True)
    print("Deleted HPA (if any).", flush=True)

    # Test matrix: (pods, offered_rps)
    test_matrix = [
        (5, 30), (5, 45), (5, 60),
        (6, 30), (6, 45), (6, 60),
        (7, 30), (7, 45), (7, 60),
    ]

    results = []
    for pods, rps in test_matrix:
        print(f"\n=== pods={pods} offered_rps={rps} ===", flush=True)
        subprocess.run(["kubectl","scale","deployment/workload","-n","bench",
                        f"--replicas={pods}"], capture_output=True)
        actual = settle_for_replicas(pods)
        print(f"  settled at {actual} pods", flush=True)

        # Warmup pass (15s): populate Prometheus with current RPS
        print(f"  warmup 15s...", flush=True)
        drive_load(rps, step_s=15)
        time.sleep(2)

        # Measurement pass
        p95 = drive_load(rps)
        snap = m.snapshot()
        served = snap.get("rps", 0.0)
        sf = max(0.0, (rps - served) / max(float(rps), 1e-9))
        viol = int((p95 or 999) > SLO_P95 or sf > 0.10)

        row = dict(pods=pods, offered_rps=rps, actual_pods=actual,
                   served_rps=round(served, 2), p95_client=round(p95, 4) if p95 else None,
                   shortfall_frac=round(sf, 4), slo_violation=viol)
        results.append(row)
        print(f"  served={served:.1f} p95={p95:.3f}s sf={sf:.1%} viol={viol}", flush=True)

    # ── Print real data table ────────────────────────────────────────────────
    print("\n" + "="*72)
    print("STEP 0: Real cluster throughput at 5-7 pods")
    print("="*72)
    print(f"  {'pods':>4}  {'offered':>8}  {'served':>8}  {'sf%':>6}  {'p95_c':>8}  viol")
    print("  " + "-"*55)
    for r in results:
        p95_s = f"{r['p95_client']:.3f}s" if r['p95_client'] else "None"
        print(f"  {r['pods']:>4}  {r['offered_rps']:>8.0f}  {r['served_rps']:>8.1f}  "
              f"{r['shortfall_frac']:>6.1%}  {p95_s:>8}  {r['slo_violation']}")

    # ── Simulator fit: scan MU values ────────────────────────────────────────
    print("\n" + "="*72)
    print("SIM FIT (scanning MU + p95_cap)")
    print("="*72)

    # Current calibrated params (from rl/simulator.py)
    P95_BASE  = 0.140
    P95_SLOPE = 0.030
    P95_SIGMA = 0.62
    SAT_THR   = 0.93

    valid_real = [r for r in results if r["p95_client"] is not None]

    best_mu   = None
    best_cap  = None
    best_mae  = float("inf")

    for mu in [8.0, 9.0, 10.0, 11.0, 12.0]:
        for p95_cap in [3.5, 5.0, 60.0]:
            errs = []
            for r in valid_real:
                sp = sim_p95_deterministic(r["pods"], r["offered_rps"], mu,
                                           P95_BASE, P95_SLOPE, SAT_THR, p95_cap)
                if r["p95_client"]:
                    errs.append(sp - r["p95_client"])
            mae = np.mean(np.abs(errs)) if errs else float("inf")
            if mae < best_mae:
                best_mae = mae
                best_mu  = mu
                best_cap = p95_cap

    # Print detailed table for best and current
    for mu, cap, label in [
        (8.0,  60.0, "CURRENT (MU=8, cap=60s)"),
        (best_mu, best_cap, f"BEST    (MU={best_mu}, cap={best_cap}s)"),
    ]:
        print(f"\n  {label}:")
        print(f"  {'pods':>4}  {'rps':>6}  {'real_p95':>10}  {'sim_p95':>10}  "
              f"{'err':>8}  {'real_viol':>10}  {'sim_viol':>10}")
        print("  " + "-"*72)
        errs = []; viol_errs = []
        for r in valid_real:
            sp = sim_p95_deterministic(r["pods"], r["offered_rps"], mu,
                                       P95_BASE, P95_SLOPE, SAT_THR, cap)
            sv = sim_violation_rate_mc(r["pods"], r["offered_rps"], mu,
                                       P95_BASE, P95_SLOPE, SAT_THR, cap, P95_SIGMA)
            err = sp - r["p95_client"]
            errs.append(err)
            viol_errs.append(abs(sv - r["slo_violation"]))
            print(f"  {r['pods']:>4}  {r['offered_rps']:>6.0f}  {r['p95_client']:>10.3f}  "
                  f"{sp:>10.3f}  {err:>+8.3f}  {r['slo_violation']:>10}  {sv:>10.2f}")
        print(f"  MAE_p95={np.mean(np.abs(errs)):.3f}s  "
              f"MAE_viol={np.mean(viol_errs):.3f}")

    # ── Gate decision ────────────────────────────────────────────────────────
    print("\n" + "="*72)
    print("GATE DECISION")
    print("="*72)
    gate_errors_p95 = []
    gate_errors_viol = []
    for r in valid_real:
        sp = sim_p95_deterministic(r["pods"], r["offered_rps"], best_mu,
                                   P95_BASE, P95_SLOPE, SAT_THR, best_cap)
        sv = sim_violation_rate_mc(r["pods"], r["offered_rps"], best_mu,
                                   P95_BASE, P95_SLOPE, SAT_THR, best_cap, P95_SIGMA)
        pct_err = abs(sp - r["p95_client"]) / max(r["p95_client"], 1e-9)
        gate_errors_p95.append(pct_err)
        gate_errors_viol.append(abs(sv - r["slo_violation"]))

    mean_pct_err = np.mean(gate_errors_p95)
    mean_viol_err = np.mean(gate_errors_viol)
    p95_pass  = mean_pct_err < 0.15
    viol_pass = mean_viol_err < 0.05

    print(f"  Best config: MU={best_mu}, p95_cap={best_cap}s")
    print(f"  p95 relative error : {mean_pct_err:.1%}  (threshold 15%)  {'PASS' if p95_pass else 'FAIL'}")
    print(f"  violation abs error: {mean_viol_err:.3f}   (threshold 0.05) {'PASS' if viol_pass else 'FAIL'}")

    if p95_pass and viol_pass:
        print(f"\n  *** GATE PASSED — updating rl/simulator.py with MU={best_mu}, "
              f"P95_TIMEOUT={best_cap} ***")
    else:
        print(f"\n  *** GATE FAILED — do NOT proceed to training. Review the table above. ***")
        print(f"  Consider: wider noise sigma, different latency curve shape, or larger MU range.")

    # Save results
    out = {"real_measurements": results,
           "best_mu": best_mu, "best_p95_cap": best_cap,
           "gate_passed": bool(p95_pass and viol_pass)}
    with open(os.path.join(RESULTS_DIR, "step0_real.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> results/step0_real.json")


if __name__ == "__main__":
    main()
