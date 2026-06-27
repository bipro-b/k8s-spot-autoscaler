"""The Gym environment that turns autoscaling into an RL problem.

STATE   : [rps, cpu_util, p95, pods, spot_fraction, spot_risk, hour_of_day, ewma_rps]  (8-dim)
          obs[7] = EWMA(Prometheus served_rps, α=0.3) / 200  — causal load forecast.
          CHANGED from v1: was oracle trace[i+3]/200, now persistence EWMA.

ACTION  : MultiDiscrete([5, 2])
            dim 0 = replica delta in {-2,-1,0,+1,+2}
            dim 1 = target pool in {ondemand, spot}   <-- spot-aware action

REWARD  : -(cost) - slo_penalty - risk_penalty   (see sim_env.py for the exact formula)
          In Phase 2 real-cluster eval, reward is logged but NOT used for training.

Set `homogeneous=True` for the ablation: scaling only, no placement action, no risk term.
"""
import time, numpy as np, gymnasium as gym
from gymnasium import spaces

from metrics.collect import Metrics
from rl.k8s_actions import K8sActions
from rl.spot_simulator import SpotSimulator

EWMA_ALPHA = 0.3


class AutoscaleEnv(gym.Env):
    def __init__(self, trace, step_s=60, slo_p95=0.2, homogeneous=False,
                 w_cost=1.0, w_slo=10.0, w_risk=2.0, w_smooth=0.3, settle_s=15):
        super().__init__()
        self.trace = trace
        self.step_s, self.slo = step_s, slo_p95
        self.homogeneous = homogeneous
        self.w_cost, self.w_slo, self.w_risk, self.w_smooth = w_cost, w_slo, w_risk, w_smooth
        self.settle_s = settle_s
        self.m = Metrics()
        self.k = K8sActions()
        self.spot = SpotSimulator(step_s=step_s)
        self._ewma_rps = 0.0

        self.action_space = spaces.MultiDiscrete([5, 2])
        # obs[7] = EWMA of observed served_rps / 200 (causal forecast, matches sim_env.py)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(8,), dtype=np.float32)
        self.i = 0

    def _obs(self, snap):
        spot_frac = snap.get("pods_spot", 0.0) / max(1.0, snap.get("pods", 1.0))
        hod = (self.i * self.step_s / 3600.0) % 24
        ewma_feat = self._ewma_rps / 200.0
        return np.array([snap.get("rps", 0.0), snap.get("cpu_util", 0.0),
                         snap.get("p95", 0.14), snap.get("pods", 8.0),
                         spot_frac, self.spot.risk_signal(), hod, ewma_feat],
                        dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.i = 0
        # Initialise EWMA to first trace value — same prior as sim_env.py
        self._ewma_rps = float(self.trace.iloc[0]["rps"])
        self.k.set_replicas(8)
        return self._obs(self.m.snapshot()), {}

    def step(self, action):
        delta = [-2, -1, 0, 1, 2][int(action[0])]
        self.k.set_replicas(self.k.get_replicas() + delta)
        if not self.homogeneous:
            self.k.set_node_pref("spot" if int(action[1]) == 1 else "ondemand")
        time.sleep(self.settle_s)
        evicted = self.spot.step() if not self.homogeneous else False

        snap = self.m.snapshot()

        # Update EWMA with observed served_rps (causal — only past data)
        self._ewma_rps = EWMA_ALPHA * snap.get("rps", 0.0) + (1.0 - EWMA_ALPHA) * self._ewma_rps

        current_i = self.i
        offered_rps = float(self.trace.iloc[current_i]["rps"])
        served_rps  = snap.get("rps", 0.0)

        shortfall_frac = max(0.0, (offered_rps - served_rps) / max(offered_rps, 1e-9))
        # Server-side p95 from Prometheus (used for obs); client-side set by run_episode.py
        slo_violation  = 1.0 if (snap.get("p95", 0.0) > self.slo or shortfall_frac > 0.10) else 0.0

        risk_exposure = 0.0 if self.homogeneous else \
            self.spot.risk_signal() * (snap.get("pods_spot", 0.0) / max(1.0, snap.get("pods", 1.0)))

        p95_excess        = max(0.0, snap.get("p95", 0.0) - self.slo)
        shortfall_penalty = max(0.0, shortfall_frac - 0.10)
        reward = -(self.w_cost * snap.get("cost_per_hr", 0.0)
                   + self.w_slo * (p95_excess + 2.0 * shortfall_penalty)
                   + self.w_risk * risk_exposure
                   + self.w_smooth * abs(delta))

        self.i += 1
        terminated = self.i >= len(self.trace)
        info = {**snap,
                "evicted": evicted,
                "slo_violation": slo_violation,
                "shortfall_frac": shortfall_frac,
                "reward": reward,
                "offered_rps": offered_rps,
                "served_rps": served_rps}
        return self._obs(snap), float(reward), terminated, False, info
