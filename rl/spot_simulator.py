"""On Minikube there is no real spot market, so we SIMULATE reclamation:
evict pods from 'spot' nodes on a Poisson schedule, and expose a forward-looking
interruption-risk signal that the agent observes. On real EKS (Phase 5) you
replace this with the AWS Node Termination Handler rebalance/interruption signal.

Calibrate `lambda_per_hr` to the published spot interruption frequency for the
instance family you cite (often single-digit % over an hour).
"""
import numpy as np
from kubernetes import client, config

class SpotSimulator:
    def __init__(self, ns="bench", lambda_per_hr=6.0, step_s=60, seed=0):
        try: config.load_incluster_config()
        except Exception: config.load_kube_config()
        self.core = client.CoreV1Api()
        self.ns = ns
        self.p_step = 1 - np.exp(-lambda_per_hr * step_s / 3600.0)  # P(interruption this step)
        self.rng = np.random.default_rng(seed)
        self._risk = 0.0

    def risk_signal(self):
        """A noisy 0..1 'interruption imminent' signal (state feature).
        Real EKS gives a 2-minute warning; here we emit elevated risk shortly
        before an injected eviction so a smart agent can pre-drain."""
        return float(self._risk)

    def step(self):
        """Call once per control step. Maybe evicts a spot pod; updates risk."""
        fire = self.rng.random() < self.p_step
        self._risk = 0.9 if fire else max(0.0, self._risk * 0.5 + self.rng.random() * 0.1)
        if fire:
            self._evict_one_spot_pod()
        return fire

    def _evict_one_spot_pod(self):
        pods = self.core.list_namespaced_pod(self.ns, label_selector="app=workload").items
        spot_pods = [p for p in pods if self._on_spot(p)]
        if spot_pods:
            victim = self.rng.choice(spot_pods)
            self.core.delete_namespaced_pod(victim.metadata.name, self.ns,
                                            grace_period_seconds=0)

    def _on_spot(self, pod):
        node = pod.spec.node_name
        if not node: return False
        labels = self.core.read_node(node).metadata.labels or {}
        return labels.get("node-lifecycle") == "spot"
