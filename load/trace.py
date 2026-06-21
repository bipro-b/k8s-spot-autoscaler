"""Generate a reproducible workload trace (requests/sec over time) with a
diurnal cycle plus random bursts. Every policy is evaluated on the SAME trace
so comparisons are fair. This is your 'realistic diurnal and bursty workloads'.

Usage: python load/trace.py --hours 24 --step 60 --out load/trace.csv
"""
import argparse, numpy as np, pandas as pd

def make_trace(hours, step_s, base=30, peak=180, seed=0):
    rng = np.random.default_rng(seed)
    n = int(hours * 3600 / step_s)
    t = np.arange(n)
    tod = (t * step_s / 3600.0) % 24                      # hour of day
    diurnal = base + (peak - base) * (0.5 - 0.5*np.cos(2*np.pi*(tod-3)/24))
    noise = rng.normal(0, base*0.08, n)
    rps = diurnal + noise
    # inject sharp bursts (the part that breaks reactive autoscalers)
    for _ in range(int(hours)):                            # ~1 burst/hour
        s = rng.integers(0, n)
        w = rng.integers(2, 6)
        rps[s:s+w] += rng.uniform(peak*0.8, peak*1.4)
    rps = np.clip(rps, 1, None)
    return pd.DataFrame({"t_s": t*step_s, "rps": rps.round(1)})

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=24)
    ap.add_argument("--step", type=int, default=60)
    ap.add_argument("--out", default="load/trace.csv")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    df = make_trace(a.hours, a.step, seed=a.seed)
    df.to_csv(a.out, index=False)
    print(f"wrote {len(df)} steps -> {a.out}  (mean {df.rps.mean():.0f} rps, max {df.rps.max():.0f})")
