"""Fast cluster simulator for RL training — no I/O, no sleep, sub-ms per step.

Calibration history:
  v3 (Phase 2): MU=8, P95_BASE=0.140, P95_SIGMA=0.62 → 58.2% HPA viol rate (real=59.4%, gap 1.3pp PASSED)
  v4 (Phase 3 Step 0): MU=11, P95_TIMEOUT=3.5s
    — Step 0 real data at 5-7 pods shows CFS throttling, NOT queueing, governs latency
    — MU=8 caused catastrophic queue divergence (p95=60s) at 5 pods; MU=11 fixes this
    — MAE_p95 drops from 12.758s to 0.303s at 5-7 pods (direction: correct)
    — TRADE-OFF: HPA violation rate at 8 pods drops from 59.4% to ~40% in sim
    — PARTIAL GATE: Step 4 real-eviction gate is the hard validator for Phase 3
    — MAX_BACKLOG cap prevents unbounded queue growth after evictions
"""
import numpy as np

MU              = 11.0   # recal v4: CFS capacity ~6 RPS/pod but effective MU=11 prevents queue collapse
TUNNEL_CAP      = 70.0
P95_BASE_ZERO   = 0.140
P95_SLOPE       = 0.030
P95_TIMEOUT     = 3.5    # recal v4: real cluster max p95 at saturation ~1.5s (capped at 3.5s with margin)
P95_SIGMA       = 0.62
SAT_THRESHOLD   = 0.93
STARTUP_STEPS   = 3
MAX_BACKLOG_STEPS = 2    # backlog cap: at most 2 steps of excess demand, prevents unbounded growth
PRICE           = {"ondemand": 1.00, "spot": 0.30}

# Kept for backwards compat with gate_validate prints
P95_BASE = P95_BASE_ZERO


class ClusterSim:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self._ready_pods    = 1
        self._pods_ondemand = 1
        self._pods_spot     = 0
        self._pending       = []   # [[steps_remaining, count], ...]
        self._backlog       = 0.0  # unserved requests carried over from overload

    # ------------------------------------------------------------------ #
    # Control interface                                                    #
    # ------------------------------------------------------------------ #

    def reset(self, pods: int = 1):
        self._ready_pods    = int(pods)
        self._pods_ondemand = int(pods)
        self._pods_spot     = 0
        self._pending       = []
        self._backlog       = 0.0
        return self

    @property
    def ready_pods(self):
        return self._ready_pods

    @ready_pods.setter
    def ready_pods(self, n):
        self._ready_pods = max(1, int(n))

    def set_pods(self, n: int, spot_fraction: float = 0.0):
        """Request n total pods; delta>0 pods enter startup queue.

        Delta is computed against TOTAL planned (ready + pending) to avoid
        double-scheduling when the agent repeats a scale-up request.
        """
        n = int(np.clip(n, 1, 20))
        total_planned = self._ready_pods + sum(item[1] for item in self._pending)
        delta = n - total_planned
        if delta > 0:
            self._pending.append([STARTUP_STEPS, delta])
        elif delta < 0:
            # scale-down: cut ready pods immediately; cancel pending if over-planned
            shortfall = -delta
            # first cancel youngest pending batches
            new_pending = []
            for item in reversed(self._pending):
                if shortfall <= 0:
                    new_pending.insert(0, item)
                elif shortfall >= item[1]:
                    shortfall -= item[1]   # cancel entire batch
                else:
                    new_pending.insert(0, [item[0], item[1] - shortfall])
                    shortfall = 0
            self._pending = new_pending
            if shortfall > 0:
                self._ready_pods = max(1, self._ready_pods - shortfall)
        total = self._ready_pods + sum(item[1] for item in self._pending)
        self._pods_spot     = int(round(total * float(spot_fraction)))
        self._pods_ondemand = total - self._pods_spot

    # ------------------------------------------------------------------ #
    # Simulation step                                                      #
    # ------------------------------------------------------------------ #

    def step(self, offered_rps: float, step_s: int = 60) -> dict:
        """Advance one step. Returns a metrics dict compatible with env.py info."""

        # 1. Materialise pods that finished starting up
        still_pending = []
        for item in self._pending:
            item[0] -= 1
            if item[0] <= 0:
                self._ready_pods    += item[1]
                self._pods_ondemand += item[1]   # new pods land on ondemand
            else:
                still_pending.append(item)
        self._pending = still_pending

        # 2. Queueing: arrivals vs capacity
        effective_rps = min(float(offered_rps), TUNNEL_CAP)
        capacity_rps  = self._ready_pods * MU
        arrivals      = effective_rps * step_s
        capacity_srv  = capacity_rps  * step_s

        total_demand   = self._backlog + arrivals
        served         = min(total_demand, capacity_srv)
        # Cap backlog to prevent unbounded growth after evictions (real servers reject, not queue)
        max_backlog    = capacity_srv * MAX_BACKLOG_STEPS
        self._backlog  = min(max(0.0, total_demand - served), max_backlog)
        served_rps     = served / step_s

        # 3. p95 latency — utilisation-aware M/D/k approximation
        rho = effective_rps / max(capacity_rps, 1e-9)
        if self._backlog > capacity_rps:
            # Backlog bigger than 1s of capacity: queue draining, model as wait time
            queue_wait = self._backlog / max(capacity_rps, 1e-9)
            p95 = min(P95_TIMEOUT, P95_BASE_ZERO + queue_wait)
        elif rho >= SAT_THRESHOLD:
            p95 = P95_TIMEOUT
        else:
            # Latency mean grows with utilisation: p95_mean = base + slope * rho/(1-rho)
            # At rho=0.70 (normal ops, 45 RPS / 8 pods): mean ≈ 0.141s → ~19% viol rate
            # At rho=0.77 (7 pods, 45 RPS): mean ≈ 0.177s → ~32% viol rate (trade-off visible)
            # At rho=0.58 (8 pods, 37 RPS): mean ≈ 0.111s → ~7% viol rate (scale-down safe)
            p95_mean = P95_BASE_ZERO + P95_SLOPE * rho / max(1.0 - rho, 0.05)
            p95 = p95_mean * float(self.rng.lognormal(0.0, P95_SIGMA))

        # 4. cpu_util: avg over ALL containers incl. pause (hence /2)
        cpu_per_pod = (served_rps / max(self._ready_pods, 1)) * 0.05
        cpu_util    = cpu_per_pod / 2.0

        # 5. Cost
        cost_per_hr = (self._pods_ondemand * PRICE["ondemand"]
                       + self._pods_spot    * PRICE["spot"])

        return {
            "pods":          float(self._ready_pods),
            "pods_ondemand": float(self._pods_ondemand),
            "pods_spot":     float(self._pods_spot),
            "cpu_util":      cpu_util,
            "rps":           served_rps,
            "served_rps":    served_rps,
            "offered_rps":   float(offered_rps),
            "p95":           p95,
            "p99":           p95 * 1.1,
            "cost_per_hr":   cost_per_hr,
        }


# ------------------------------------------------------------------ #
# HPA rule (for gate validation in gate_validate.py)                 #
# ------------------------------------------------------------------ #

def hpa_desired(current_pods: int, cpu_util_avg_containers: float,
                cpu_request: float = 0.1, cpu_target: float = 0.6,
                min_replicas: int = 1, max_replicas: int = 8) -> int:
    """Replicate k8s HPA averageUtilization formula.

    cpu_util_avg_containers: avg(rate(container_cpu_usage_seconds_total)) in cores,
    averaged across ALL containers including pause (hence *2 for workload estimate).
    """
    if current_pods == 0 or cpu_util_avg_containers <= 0:
        return min_replicas
    cpu_per_workload_pod = cpu_util_avg_containers * 2.0   # undo pause-container /2
    desired = int(np.ceil(
        current_pods * cpu_per_workload_pod / (cpu_request * cpu_target)
    ))
    return int(np.clip(desired, min_replicas, max_replicas))
