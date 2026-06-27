"""Step 3+4: Calibrate and gate-validate the simulator.

Runs the HPA rule inside the simulator over load/trace.csv and compares the
resulting violation rate to results/hpa.csv (ground truth: 14/60 = 23.3%).

GATE: simulator must reproduce the real violation rate within ±5 percentage
points. If it fails, STOP — do not train until the model is fixed.

Usage:
    python -m rl.gate_validate
    python -m rl.gate_validate --seeds 20          # average over N seeds
    python -m rl.gate_validate --plot              # show p95 comparison chart
"""
import argparse, sys
import numpy as np
import pandas as pd
from rl.simulator import ClusterSim, hpa_desired, MU, TUNNEL_CAP, SAT_THRESHOLD, P95_BASE, P95_SLOPE

SLO      = 0.2   # seconds
REAL_CSV = "results/hpa.csv"
TRACE    = "load/trace.csv"


def run_sim_hpa(trace: pd.DataFrame, seed: int = 0, warmup_pods: int = 8) -> pd.DataFrame:
    """Run HPA policy inside the simulator. Pre-warms to `warmup_pods`."""
    sim = ClusterSim(seed=seed)
    sim.reset(pods=warmup_pods)          # start warm, like the real episode
    rows = []
    for _, row in trace.iterrows():
        snap = sim.step(row["rps"])
        snap["t_s"] = row["t_s"]
        snap["seed"] = seed
        rows.append(snap)
        # HPA decision for NEXT step
        desired = hpa_desired(
            current_pods=int(snap["pods"]),
            cpu_util_avg_containers=snap["cpu_util"],
        )
        sim.set_pods(desired, spot_fraction=0.0)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, label: str) -> dict:
    n_total   = len(df)
    n_viols   = int((df["p95"] > SLO).sum())
    viol_rate = n_viols / n_total
    mean_p95  = df["p95"].mean()
    mean_cost = df["cost_per_hr"].mean()
    print(f"{label:30s}  violations={n_viols:3d}/{n_total}  "
          f"rate={viol_rate:.1%}  mean_p95={mean_p95:.3f}s  "
          f"mean_cost=${mean_cost:.2f}/hr")
    return dict(n_viols=n_viols, n_total=n_total, viol_rate=viol_rate,
                mean_p95=mean_p95, mean_cost=mean_cost)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds",  type=int, default=10)
    ap.add_argument("--plot",   action="store_true")
    a = ap.parse_args()

    trace = pd.read_csv(TRACE)
    real  = pd.read_csv(REAL_CSV)

    print("=" * 70)
    print("STEP 3 — Calibration parameters")
    print(f"  MU={MU} RPS/pod  TUNNEL_CAP={TUNNEL_CAP}  "
          f"SAT_THRESHOLD={SAT_THRESHOLD}  P95_BASE_ZERO={P95_BASE}  P95_SLOPE={P95_SLOPE}")
    print()

    print("STEP 4 — Gate validation")
    print("-" * 70)

    real_stats = summarise(real, "Real HPA (ground truth)")

    sim_viol_rates = []
    for s in range(a.seeds):
        df = run_sim_hpa(trace, seed=s)
        stats = summarise(df, f"Sim HPA  seed={s}")
        sim_viol_rates.append(stats["viol_rate"])

    mean_sim_rate = float(np.mean(sim_viol_rates))
    std_sim_rate  = float(np.std(sim_viol_rates))
    gap           = abs(mean_sim_rate - real_stats["viol_rate"])

    print()
    print(f"  Real violation rate : {real_stats['viol_rate']:.1%}")
    print(f"  Sim  violation rate : {mean_sim_rate:.1%} ± {std_sim_rate:.1%}  (mean ± std over {a.seeds} seeds)")
    print(f"  Gap                 : {gap:.1%}  (gate threshold = 5.0%)")
    print()

    if gap <= 0.05:
        print("GATE PASSED  Simulator matches real HPA within 5 pp.")
        print("  -> Proceed to Step 5 (sim_env.py) and Step 6 (training).")
        rc = 0
    else:
        print("GATE FAILED  Gap exceeds 5 pp -- fix simulator before training.")
        print("  -> Tune MU, SAT_THRESHOLD, or P95_SIGMA in rl/simulator.py.")
        rc = 1

    if a.plot:
        try:
            import matplotlib.pyplot as plt
            df0 = run_sim_hpa(trace, seed=0)
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(real["t_s"], real["p95"],  label="Real HPA p95",  lw=1.5)
            ax.plot(df0["t_s"],  df0["p95"],   label="Sim HPA p95 (seed=0)", lw=1.5, alpha=0.8)
            ax.axhline(SLO, color="red", ls="--", label=f"SLO={SLO}s")
            ax.set_xlabel("time (s)"); ax.set_ylabel("p95 latency (s)")
            ax.set_title("Simulator calibration — HPA baseline")
            ax.legend(); ax.set_yscale("symlog", linthresh=1)
            plt.tight_layout()
            plt.savefig("results/gate_validate.png", dpi=150)
            print("  Saved results/gate_validate.png")
            plt.show()
        except ImportError:
            print("  (matplotlib not available — skipping plot)")

    sys.exit(rc)


if __name__ == "__main__":
    main()
