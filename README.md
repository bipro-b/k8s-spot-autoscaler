# Spot-Instance-Aware Cost & Reliability Autoscaling for Kubernetes

An empirical study of reinforcement-learning autoscalers that decide **when to scale** *and*
**which node type (spot vs. on-demand)** to scale onto under reclamation risk — evaluated
against HPA on a real Kubernetes cluster and in a calibrated simulator with paired
multi-seed statistical testing.

> **Status:** active research. Phases 0–3 complete (simulation + real-cluster runs).
> Real-eviction validation (Step 4 gate) and reward redesign in progress.
> Results below are reported as measured, including the negative ones.

---

## TL;DR — what this study found

**1. Spot-blind RL does not beat HPA.**
A homogeneous (spot-unaware) RL agent tied HPA in a paired 25-seed simulation study
(*p* = 0.33). An apparent advantage in an earlier single run disappeared once the
comparison was made fair: the agent's lower violation rate was bought by running ~9–12
pods against HPA's 8, i.e. **over-provisioning, not smarter scaling**. HPA could buy the
same improvement by raising `maxReplicas`.

**2. A naively-rewarded spot-aware agent reward-hacks into bufferless spot usage.**
Given a linear cost/SLO trade-off, the spot-aware agent learned to run ~85% of pods on spot
with essentially **no on-demand safety buffer** (~0.9 pods). It achieved a large cost
reduction — but became *less* resilient to reclamation than a naive spot baseline that
happened to keep a ~3.8-pod on-demand cushion.

| Policy (25 sim seeds) | Cost/hr | Violation rate | Post-eviction p95 | Spot share |
|---|---|---|---|---|
| HPA (on-demand only) | $8.00 | 43.7% | — | 0% |
| Naive spot | $5.26 | 44.7% | **362 ms** | 51% |
| RL homogeneous | $8.42 | 43.7% | — | 0% |
| **RL spot-aware** | **$2.72** | 54.7% | **1,297 ms** | 85% |

Paired *t*-tests (n = 25): cost reduction vs. HPA is significant (*p* < 0.01, *d* = 31.85);
post-eviction recovery vs. naive spot is significantly **worse** (*p* < 0.01, *d* = −1.37).

**3. The implication.** Cost and reliability cannot be traded off linearly here. Per-step
cost savings dominate the intermittent penalty of a rare eviction, so any soft SLO weight
is eventually out-competed by thin spot usage. Achieving the intended
*spot cost with on-demand reliability* appears to require treating reliability as a
**hard constraint** (or rewarding a risk-proportional standing buffer) rather than as a
penalty term.

**Caveat (important):** the post-eviction figures above come from the simulator's eviction
model, which has **not yet been validated against real reclamation events** on the cluster.
That validation is the current work item; treat those numbers as provisional.

---

## Research question

> Can a learning-based, spot-aware autoscaler reduce infrastructure cost while maintaining
> SLO compliance more effectively than default Kubernetes autoscalers and homogeneous RL
> baselines?

Kubernetes' default autoscalers are reactive and treat every node as interchangeable. Spot
instances cost far less than on-demand but can be reclaimed at short notice, and HPA/VPA/KEDA
cannot reason about that trade-off. This project asks whether an RL controller that also
chooses *node type* can occupy a better point on the cost–reliability frontier.

---

## Method

**Policies compared (identical measurement, identical traces, paired per seed):**

| Policy | Scaling | Placement | Role |
|---|---|---|---|
| `hpa` | CPU-target HPA | on-demand only | reliable, expensive ceiling |
| `naive_spot` | HPA-style | spot, risk-unaware | cheap, reclamation-exposed |
| `rl_homogeneous` | RL (PPO) | on-demand only | ablation: isolates scaling policy |
| `rl_spot` | RL (PPO) | spot ↔ on-demand | the proposed contribution |

**Environment.** 3-node Kubernetes cluster with `node-lifecycle` labels partitioning
spot and on-demand pools. A tunable-CPU HTTP workload exposes a Prometheus latency
histogram; k6 replays diurnal traces with injected bursts.

**Metric.** A step counts as an SLO violation if `client p95 > 200 ms` **or**
`served_rps < 0.90 × offered_rps`. The second clause matters: a saturated service sheds
load that never enters the server-side histogram, so latency-only metrics report a healthy
system while it is failing. Client-side measurement plus a shortfall term makes capacity
loss — the exact failure mode reclamation causes — visible.

**Agent.** PPO (Stable-Baselines3) over a Gymnasium environment.
State: request rate, CPU utilisation, p95, pod count, spot fraction, interruption-risk
signal, time of day. Action: `MultiDiscrete([5, 2])` — replica delta ∈ {−2…+2} × target
pool ∈ {on-demand, spot}. Reward: −(cost + SLO penalty + reclamation-risk exposure +
smoothness penalty).

**Simulator.** Training against the live cluster is infeasible (~60 s per control step ×
10⁵ steps). A discrete-event simulator with an M/M/1-style latency model
(`p95 = a + b·ρ/(1−ρ)`) and explicit pod start-up lag is **calibrated to real cluster
measurements and gated against the real HPA baseline** before any agent is trained in it.
Trained policies are then re-evaluated on the real cluster to quantify the sim-to-real gap.

---

## Methodological notes (things that changed the results)

Several confounds materially altered conclusions and are documented here because they
generalise to similar studies:

- **Placement asymmetry.** An early "−34% cost win" reversed to **+27% more expensive**
  once both policies were pinned to the same node pool. The RL deployment had lost its
  on-demand affinity and was silently drawing a spot discount the baseline could not.
- **Observation-timing mismatch.** The real-cluster harness observed the cluster *after*
  load had finished (idle), while the simulator observed *under* load. The policy therefore
  acted on states it had never trained on. Fixed by reordering to
  scale → settle → load → observe.
- **Invisible load shedding.** Latency-only SLO metrics reported 0% violations while the
  service was dropping most of the offered burst. Fixed by adding the offered-vs-served
  shortfall term.
- **Single-run instability.** The HPA baseline's own violation rate varied 45% → 63%
  across runs — wider than the effect being claimed. All conclusions therefore rest on
  paired multi-seed tests, not single episodes.

---

## Reproducing

### Prerequisites
Docker, `minikube`, `kubectl`, `helm`, `k6`, Python 3.12+.

### Phase 0 — cluster
```bash
minikube start --driver=docker --cpus=4 --memory=8192 --nodes=3
minikube addons enable metrics-server

kubectl label node minikube     node-lifecycle=ondemand
kubectl label node minikube-m02 node-lifecycle=spot
kubectl label node minikube-m03 node-lifecycle=spot

helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prom prometheus-community/kube-prometheus-stack -n monitoring --create-namespace

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> **Gotcha:** kube-prometheus-stack only discovers `ServiceMonitor`s labelled
> `release: prom`, and kube-state-metrics does not expose custom node labels by default —
> without both, latency scraping and the spot/on-demand cost split silently return zeros.
> See `deploy/workload-servicemonitor.yaml`.

### Phase 1 — workload + baseline
```bash
docker build -t spot-workload:v1 workload/ && minikube image load spot-workload:v1
kubectl apply -f deploy/namespace.yaml -f deploy/workload-deployment.yaml \
              -f deploy/workload-service.yaml -f deploy/workload-servicemonitor.yaml
python load/trace.py --hours 1 --out load/trace.csv
kubectl apply -f deploy/hpa.yaml

# keep these alive in separate terminals
kubectl -n monitoring port-forward svc/prom-kube-prometheus-stack-prometheus 9090:9090
kubectl -n bench port-forward svc/workload 8080:80

python experiments/run_episode.py --policy hpa --trace load/trace.csv \
       --url http://localhost:8080/work --out results/hpa.csv
```

### Phases 2–4 — train and evaluate
```bash
python rl/train.py --homogeneous          # spot-blind ablation
python rl/train.py                        # spot-aware agent
python experiments/run_episode.py --policy rl_spot --out results/rl_spot.csv
python experiments/analyze.py results/    # paired tests + cost-vs-reliability figure
```

---

## Repository layout

```
workload/      tunable-CPU HTTP service + Prometheus latency histogram
deploy/        k8s manifests, ServiceMonitor, HPA / KEDA baseline configs
load/          diurnal + bursty trace generator, k6 replay driver
metrics/       Prometheus snapshot collector (pods, CPU, p95, spot/on-demand cost)
rl/            Gymnasium envs (real + simulated), simulator, spot eviction model,
               k8s action layer, PPO training
experiments/   episode runner (policy-agnostic), multi-seed study, analysis + plots
results/       per-run CSVs, multi-seed summaries, figures
```

---

## Limitations

- Single-node on-demand pool (Minikube) caps on-demand capacity and constrains the
  placement search space.
- Spot reclamation is **simulated**, calibrated to published interruption frequencies;
  real-cluster eviction validation is pending.
- Cost model uses representative on-demand/spot prices rather than live market prices.
- No EKS validation yet; all results are Minikube + simulator.

---

## Roadmap

- [ ] **Eviction-dynamics gate** — validate simulated reclamation impact and recovery
      against real cordon/drain events; check whether pod start-up fits inside the
      ~2-minute spot warning window (if not, pre-drain designs are infeasible as such).
- [ ] **Constrained reward redesign** — hard post-eviction SLO floor, or explicit reward
      for a risk-proportional on-demand buffer.
- [ ] Re-run the 4-way paired study under the revised formulation.
- [ ] EKS validation with a managed spot node group and real pricing.
- [ ] Write-up.

---

## Citing / contact

Independent research project. Manuscript in preparation:
*"Cost–Reliability Trade-offs in Spot-Instance-Aware Kubernetes Autoscaling: An Empirical
Study of Reward Design."*

Questions, corrections, and collaboration welcome — [name] · [email] · [profile link]

## License

[MIT / Apache-2.0 — pick one and add the LICENSE file]
