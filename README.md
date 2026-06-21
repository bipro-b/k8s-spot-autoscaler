# Spot-Instance-Aware Cost & Reliability Autoscaling for Kubernetes

Reference implementation + experiment harness for the research outline.
Goal: train an RL autoscaler that decides **when to scale** and **which node type
(spot vs on-demand)** to scale onto, and beat HPA / VPA / KEDA on the
cost-vs-reliability tradeoff with statistical significance.

---

## The critical path (do phases in order — do NOT skip ahead)

| Phase | What you produce | Est. time | "Done" means |
|-------|------------------|-----------|--------------|
| 0 | Working Minikube + Prometheus + Python env | 2-3 days | `kubectl get pods` clean; metrics scraping |
| 1 | Workload deployed + load trace + HPA baseline logging cost/latency | ~1 week | One CSV of HPA results over a full trace |
| 2 | Gym env + homogeneous RL agent that beats HPA | 1-2 weeks | RL CSV beats HPA CSV on cost at equal SLO |
| 3 | Spot awareness: simulated evictions, risk signal, node-type action | ~2 weeks | Spot-aware agent handles evictions gracefully |
| 4 | Full runs: 4 policies x >=20 reps + stats + plots | ~1 week | `analyze.py` prints p<0.01 results + figures |
| 5 | (Optional) EKS validation w/ real spot node group | ~1 week | Same pipeline, real cluster, real prices |
| 6 | Write the paper using your own figures | ~2 weeks | Submittable workshop/conference draft |

The single most common way this project dies: jumping to Phase 3 (spot RL) before
Phase 1 (clean metrics) works. If your cost/latency numbers aren't trustworthy,
nothing downstream matters. **Earn each phase.**

---

## Run order (today)

```bash
# Phase 0
minikube start --cpus=4 --memory=8192 --nodes=3      # 3 nodes so you can label spot vs on-demand
kubectl label node minikube-m02 node-lifecycle=spot   # pretend nodes
kubectl label node minikube-m03 node-lifecycle=spot
kubectl label node minikube     node-lifecycle=ondemand
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prom prometheus-community/kube-prometheus-stack -n monitoring --create-namespace

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Phase 1
docker build -t spot-workload:v1 workload/ && minikube image load spot-workload:v1
kubectl apply -f deploy/namespace.yaml -f deploy/workload-deployment.yaml -f deploy/workload-service.yaml
python load/trace.py --hours 24 --out load/trace.csv          # generate diurnal+bursty trace
kubectl apply -f deploy/hpa.yaml                               # baseline 1
python experiments/run_episode.py --policy hpa --trace load/trace.csv --out results/hpa.csv

# Phase 2+  (after Phase 1 is solid)
python rl/train.py --homogeneous   # trains the no-spot RL baseline
python rl/train.py                 # trains the spot-aware agent
python experiments/run_episode.py --policy rl_spot --out results/rl_spot.csv

# Phase 4
python experiments/analyze.py results/   # t-tests + plots into results/figures/
```

## What you must tune (marked `# TODO` in code)
- SLO latency threshold for your app (start p95 < 200ms).
- Reward weights: cost vs SLO-penalty vs reclamation-risk. This is the heart of the
  contribution — sweep these and report sensitivity.
- Spot interruption rate (Phase 3). Calibrate to AWS published spot interruption
  frequencies for the instance family you cite.

## Layout
```
workload/      a tunable-CPU Node.js API (your scalable service) + Prometheus latency histogram
deploy/        k8s manifests + HPA / KEDA baseline configs
load/          trace generator + k6 replay script (diurnal + bursts)
metrics/       pulls pods, CPU, latency, cost from Prometheus into a snapshot dict
rl/            Gymnasium env, k8s action layer, spot simulator, SB3 training
experiments/   episode runner (any policy) + analysis (t-test, p<0.01, figures)
```
