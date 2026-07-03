"""On Minikube there is no real spot market, so we SIMULATE reclamation:
evict pods from 'spot' nodes on a Poisson schedule, and expose a forward-looking
interruption-risk signal that the agent observes.

WARNING LEAD TIME (matches AWS ~2 min notice):
  At step T:   Poisson fires -> _pending_eviction = WARNING_STEPS, risk rises to ~0.4
  At step T+1: countdown -> risk ~0.75 (agent still has 1 step to pre-drain)
  At step T+2: eviction fires, risk = 1.0
An agent that observes risk > 0.4 and pre-scales to on-demand avoids the impact.

lambda_per_hr=5.0 → P(eviction in any 60-s step) ≈ 8.0% per step, ~5 evictions/hr expected.
Calibrated to realistic AWS spot interruption frequency (single-digit % per hour).

sim_mode=True: skip real k8s API calls (for PPO training — pod removal handled by ClusterSim).
"""
import numpy as np

WARNING_STEPS  = 2      # control steps of lead time before eviction fires (2×60s = 2 min)
LAMBDA_PER_HR  = 5.0    # expected evictions per hour on the spot pool


class SpotSimulator:
    def __init__(self, ns="bench", lambda_per_hr=LAMBDA_PER_HR, step_s=60,
                 seed=0, sim_mode=False):
        self.ns       = ns
        self.step_s   = step_s
        self.sim_mode = sim_mode
        # P(new eviction scheduled this step) — Poisson inter-arrival
        self.p_step   = 1 - np.exp(-lambda_per_hr * step_s / 3600.0)
        self.rng      = np.random.default_rng(seed)
        self._risk    = 0.0
        self._pending = 0   # countdown steps until eviction fires (0 = none pending)

        if not sim_mode:
            from kubernetes import client, config
            try: config.load_incluster_config()
            except Exception: config.load_kube_config()
            self._core = client.CoreV1Api()
        else:
            self._core = None

    def risk_signal(self) -> float:
        """0..1 interruption-imminent signal observed by the agent.
        Rises BEFORE the eviction fires so a smart agent can pre-drain to on-demand."""
        return float(self._risk)

    def step(self) -> int:
        """Advance one control step.
        Returns number of pods evicted this step (0 or 1).
        In sim_mode the caller (SimEnv) decrements sim.ready_pods directly."""
        if self._pending > 0:
            self._pending -= 1
            if self._pending == 0:
                # Eviction fires NOW
                self._risk = 1.0
                if not self.sim_mode:
                    self._evict_one_spot_pod()
                return 1
            else:
                # Still in warning window — risk proportional to urgency
                urgency = 1.0 - self._pending / WARNING_STEPS
                self._risk = 0.40 + 0.55 * urgency
                return 0
        else:
            # No eviction pending; maybe schedule one
            if self.rng.random() < self.p_step:
                self._pending = WARNING_STEPS
                self._risk    = 0.35 + self.rng.random() * 0.10   # first warning
            else:
                # Background noise: small random fluctuation, decays if nothing pending
                self._risk = max(0.0, self._risk * 0.6 + self.rng.random() * 0.05)
            return 0

    # ------------------------------------------------------------------ #
    # Real-cluster eviction (skipped in sim_mode)                         #
    # ------------------------------------------------------------------ #

    def _evict_one_spot_pod(self):
        pods = self._core.list_namespaced_pod(
            self.ns, label_selector="app=workload").items
        spot_pods = [p for p in pods if self._on_spot(p)]
        if spot_pods:
            victim = self.rng.choice(spot_pods)
            self._core.delete_namespaced_pod(
                victim.metadata.name, self.ns, grace_period_seconds=0)

    def _on_spot(self, pod):
        node = pod.spec.node_name
        if not node:
            return False
        labels = self._core.read_node(node).metadata.labels or {}
        return labels.get("node-lifecycle") == "spot"
