# Phase 2 Summary — Homogeneous RL vs HPA

## Objective

Determine whether `rl_homogeneous` (all on-demand pods, no spot) beats the Kubernetes HPA
baseline on SLO violation rate, using the real minikube cluster and a paired statistical design.

---

## Setup

| Item | Value |
|---|---|
| Policy A | HPA (`averageUtilization: 60`, cpu request 100m, min=1, max=8) |
| Policy B | `rl_homogeneous` (PPO, 5-action delta ∈ {−2,−1,0,+1,+2}, homogeneous mode) |
| Pods spot | 0 (on-demand affinity hard-enforced via `requiredDuringSchedulingIgnoredDuringExecution`) |
| Load | Docker k6 → minikube NodePort 30080, 60 steps × 60 s |
| Traces | `make_trace(hours=1, step_s=60, seed=N)` — diurnal ~50 RPS + one random burst (144–252 RPS add, width 2–5 steps) |
| SLO | `p95_client > 0.200 s` OR `shortfall_frac > 0.10` |
| Harness | v3 (scale → settle → load → observe), bypasses `env.step()` to match sim semantics |

---

## Key Fix: Observation Timing (v3 Harness)

The original eval harness drove load, slept 15 s, then snapshotted an **idle** cluster — mismatching
the simulator's `scale → drive_load → observe` order. The fix (`experiments/run_episode.py`):

1. **Scale** (`env.k.set_replicas(target)`)
2. **Settle** (`settle_for_replicas(target, timeout_s=90)`) — polls `readyReplicas == target`
3. **Drive load** (`drive_load(rps, step_s)`)
4. **Observe** (`m.snapshot()`) — under-load, matches sim semantics

---

## Simulator Calibration (v3)

Gate v1 failed (17.2 pp gap) because the sim was calibrated to a stale single HPA run.

**Option B — 3-run anchoring:**

| Run | Violation rate |
|---|---|
| HPA seed 0 (`hpa_v3.csv`) | 63.3% |
| HPA seed 1 (`hpa_real_seed1.csv`) | 56.7% |
| HPA seed 2 (`hpa_real_seed2.csv`) | 58.3% |
| **Mean ± std** | **59.4% ± 3.5%** |

Calibration scan → best `P95_BASE_ZERO = 0.140` (was 0.100).
Gate v3: sim 58.2% ± 6.6% vs real mean 59.4% → **gap = 1.3 pp < 8 pp threshold → PASSED**.

---

## 25-Seed Simulator Study

Paired design: same trace seed and same ClusterSim noise seed per (HPA, RL) pair.

| Metric | HPA mean | RL mean | Diff (HPA−RL) | p-value | Result |
|---|---|---|---|---|---|
| Violation rate | 57.9% | 57.4% | +0.5 pp | 0.33 | **n.s.** |
| Cost/hr | $8.00 | $8.40 | −$0.40 | <0.01 | **RL costs more** |
| p95_max | 18.1 s | 40.0 s | −21.9 s | <0.01 | **RL appears worse** |

The `p95_max` result is an **artifact** of the simulator's queueing model: when RL scales down to
5–7 pods post-burst, the sim computes ρ > 0.93 and queue diverges to p95 = 60 s. The real cluster
at 5–7 pods with 45 RPS shows p95 = 0.4–0.8 s (no queue collapse). `MU = 8 RPS/pod` underestimates
real throughput at low pod counts. The sim correctly captures violation rate in aggregate but cannot
reliably model RL's post-burst latency regime.

---

## Real-Cluster Confirmation (3 Paired Seeds)

| Seed | HPA | RL | RL saves | HPA p95\_max | RL p95\_max |
|---|---|---|---|---|---|
| 0 | 38/60 = 63.3% | 31/60 = 51.7% | **+11.7 pp** | 54.558 s | 2.091 s |
| 1 | 34/60 = 56.7% | 18/60 = 30.0% | **+26.7 pp** | 4.051 s | 3.399 s |
| 2 | 35/60 = 58.3% | 22/60 = 36.7% | **+21.7 pp** | 24.923 s | 3.202 s |
| **Mean** | **59.4%** | **39.4%** | **+20.0 pp** | 27.8 s | 2.9 s |

Paired t-test: t = 4.54, **p = 0.045**, 95% CI = [+1.0%, +39.0%] (n = 3).

---

## Honest Verdict

### 1. RL reduces SLO violations by ~20 pp on the real cluster — consistently across all 3 seeds.

Direction is unambiguous (RL wins every seed). The margin is large (12–27 pp). Paired t = 4.54,
p = 0.045. The CI is wide ([+1%, +39%]) at n = 3, and the 25-seed sim study showed p = 0.33
(no significance) — so the real-cluster result should be treated as **strongly directional but
not statistically conclusive** without more runs.

### 2. The mechanism is over-provisioning, not burst prediction.

RL runs **9–12 pods** during normal load (vs HPA's constant 8). Extra capacity pushes normal-load
p95 below 0.200 s where HPA marginally exceeds it. When bursts arrived (seeds 1 and 2), RL had
**7 pods** — *fewer* than HPA's 8 — and suffered similar shortfall (75–100%). RL does not
reliably pre-scale for bursts because the EWMA signal cannot anticipate a step-change arrival.

### 3. Worst-case latency: RL is better on the real cluster.

HPA p95\_max = 27.8 s mean (dominated by seed 0's 54.6 s burst). RL p95\_max = 2.9 s mean.
The sim predicted the opposite (RL worse by 21.9 s) — that was a queue-model artifact.
HPA's catastrophic burst latency comes from hard saturation at the NodePort cap (>70 RPS
effective, 8 pods × 8 RPS = 64 capacity). RL's post-burst scaling was faster in seed 0
(11 → 9 → 7 pods, recovered quickly), keeping p95 controlled even post-burst.

### 4. Cost: RL is more expensive by ~$1.00/hr.

RL averages 9.0 pods vs HPA's 8.0. On-demand at $1/hr/pod → +$1.00/hr = +12.5% cost.
The improvement in violation rate is purchased by this over-provisioning, not by smarter
allocation decisions in the sense of spot/on-demand mixing.

### 5. Burst prevention is not demonstrated.

Both policies fail when offered RPS > TUNNEL\_CAP (70 RPS effective). The NodePort physical
ceiling (not pod count) is the bottleneck at burst time. A policy that could *pre-route*
or *shed* load before the tunnel saturates would be needed to address this.

---

## Sim-to-Real Gap Analysis

| Metric | Sim prediction | Real result | Gap direction |
|---|---|---|---|
| Violation rate diff | +0.5 pp (n.s.) | +20.0 pp (RL wins) | Sim underestimates RL |
| p95\_max diff | −21.9 s (RL worse) | +24.9 s (RL better) | Sim inverts direction |
| Cost diff | −$0.40/hr (RL more) | −$1.00/hr (RL more) | Sim underestimates magnitude |

Root cause: `MU = 8 RPS/pod` (calibrated to match HPA violation rate on the full trace)
underestimates single-pod throughput at low utilisation. When RL scales to 5–7 pods, sim
capacity = 40–56 RPS < normal load (45–52 RPS), causing persistent queue growth → p95 = 60 s.
Real throughput at 5–7 pods is sufficient for normal load. The sim was well-calibrated for
HPA-at-8-pods but not for RL's variable pod regime.

---

## Files

| File | Description |
|---|---|
| `experiments/run_episode.py` | v3 eval harness (scale → settle → load → observe) |
| `experiments/phase2_calibrate.py` | 3-run HPA anchoring + sim recalibration scan |
| `experiments/phase2_sim_study.py` | 25-seed paired simulator study + significance tests |
| `experiments/phase2_real_verdict.py` | Final real-cluster analysis (3 paired seeds) |
| `rl/simulator.py` | Recalibrated: `P95_BASE_ZERO = 0.140` |
| `load/trace_seed1.csv` | 60-step trace, seed=1, burst at steps 56–57 |
| `load/trace_seed2.csv` | 60-step trace, seed=2, burst at steps 59–60 |
| `results/hpa_v3.csv` | Real HPA episode, seed 0 (63.3% viols) |
| `results/hpa_real_seed1.csv` | Real HPA episode, seed 1 (56.7% viols) |
| `results/hpa_real_seed2.csv` | Real HPA episode, seed 2 (58.3% viols) |
| `results/rl_homogeneous_real_v3.csv` | Real RL episode, seed 0 (51.7% viols) |
| `results/rl_real_seed1.csv` | Real RL episode, seed 1 (30.0% viols) |
| `results/rl_real_seed2.csv` | Real RL episode, seed 2 (36.7% viols) |
| `results/phase2_sim_seeds.csv` | 25-seed sim study output (50 rows) |

---

## Next Steps (Phase 3)

Phase 2 used on-demand pods only. The original research question is whether RL can
**mix spot and on-demand** to cut costs while preserving SLO. Phase 3 would:

1. Enable `pods_spot > 0` (remove on-demand hard affinity, allow spot node scheduling)
2. Train `rl_heterogeneous` with the two-action head (delta\_pods, spot\_fraction)
3. Compare against HPA + fixed spot mix (e.g. 50% spot) as baseline
4. Measure cost reduction vs SLO violation rate tradeoff
