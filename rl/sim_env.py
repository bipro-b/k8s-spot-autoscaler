"""Simulator-backed Gymnasium env — identical obs / action / reward to env.py.

Replaces live Kubernetes calls with ClusterSim, so PPO can take thousands of
steps per second instead of one step per 60 s real-world wall-clock time.

STATE   : [rps, cpu_util, p95, pods, spot_fraction, spot_risk, hour_of_day, ewma_rps]  (8-dim)
          obs[7] = EWMA(observed served_rps, α=0.3) / 200  — causal load forecast.
          CHANGED from v1: was oracle trace[i+3]/200, now persistence EWMA.

ACTION  : MultiDiscrete([5, 2])
            dim 0 = replica delta  {-2,-1, 0,+1,+2}
            dim 1 = target pool    {ondemand=0, spot=1}   (ignored when homogeneous=True)

REWARD  : -(w_cost*cost_per_hr
           + w_slo*(max(0,p95-0.2) + 2.0*max(0,shortfall_frac-0.10))
           + w_smooth*|delta_pods|)
          Continuous signal: shortfall term penalises load-shedding so the agent
          cannot hide under-provisioning behind a quiet server-side histogram.

SLO VIOLATION (binary, for logging):
    p95 > 0.200 s  OR  served_rps < 0.90 * offered_rps
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from rl.simulator import ClusterSim
from rl.spot_simulator import SpotSimulator

EWMA_ALPHA = 0.3


class SimEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, trace, step_s: int = 60, slo_p95: float = 0.2,
                 homogeneous: bool = False,
                 w_cost: float = 1.0, w_slo: float = 10.0,
                 w_risk: float = 2.0, w_smooth: float = 0.3):
        super().__init__()
        self.trace       = trace
        self.step_s      = step_s
        self.slo         = slo_p95
        self.homogeneous = homogeneous
        self.w_cost      = w_cost
        self.w_slo       = w_slo
        self.w_risk      = w_risk
        self.w_smooth    = w_smooth

        # Identical spaces to env.py — required for sim-to-real transfer
        # obs[7] = EWMA of observed served_rps / 200  (causal forecast)
        self.action_space      = spaces.MultiDiscrete([5, 2])
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(8,), dtype=np.float32)

        self._sim      = None
        self._spot     = None
        self._ewma_rps = 0.0
        self.i         = 0

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _obs(self, snap: dict) -> np.ndarray:
        spot_frac = snap["pods_spot"] / max(1.0, snap["pods"])
        hod       = (self.i * self.step_s / 3600.0) % 24
        # obs[7]: causal EWMA forecast — updated each step from observed rps
        ewma_feat = self._ewma_rps / 200.0
        return np.array(
            [snap["rps"], snap["cpu_util"], snap["p95"], snap["pods"],
             spot_frac, self._spot.risk_signal(), hod, ewma_feat],
            dtype=np.float32,
        )

    def _null_snap(self) -> dict:
        return {
            "rps": 0.0, "cpu_util": 0.0, "p95": 0.14, "pods": 8.0,
            "pods_ondemand": 8.0, "pods_spot": 0.0, "cost_per_hr": 8.0,
            "served_rps": 0.0, "offered_rps": 0.0,
        }

    # ------------------------------------------------------------------ #
    # Gymnasium interface                                                  #
    # ------------------------------------------------------------------ #

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._sim  = ClusterSim(seed=seed)
        self._sim.reset(pods=8)
        self._spot = SpotSimulator(step_s=self.step_s)
        # Initialise EWMA to first trace value — sensible prior, converges quickly
        self._ewma_rps = float(self.trace.iloc[0]["rps"])
        self.i         = 0
        return self._obs(self._null_snap()), {}

    def step(self, action):
        delta    = [-2, -1, 0, 1, 2][int(action[0])]
        use_spot = (not self.homogeneous) and (int(action[1]) == 1)
        spot_frac = 1.0 if use_spot else 0.0

        # Apply scaling decision (homogeneous → always on-demand, spot_frac=0)
        target_pods = int(np.clip(self._sim.ready_pods + delta, 1, 20))
        self._sim.set_pods(target_pods, spot_fraction=spot_frac)

        # Spot interruption (disabled in Phase 2: homogeneous=True → evicted=0)
        evicted = 0
        if not self.homogeneous:
            evicted = self._spot.step()
            if evicted > 0:
                self._sim.ready_pods = max(1, self._sim.ready_pods - evicted)

        # Drive one trace step through the simulator
        offered_rps = float(self.trace.iloc[self.i]["rps"])
        snap        = self._sim.step(offered_rps, step_s=self.step_s)

        # Update EWMA with this step's observed served_rps (causal — past only)
        self._ewma_rps = EWMA_ALPHA * snap["rps"] + (1.0 - EWMA_ALPHA) * self._ewma_rps

        # ---- Reliability metric (consistent definition everywhere) ----
        # shortfall_frac: fraction of offered load that was not served
        shortfall_frac = max(0.0,
            (offered_rps - snap["served_rps"]) / max(offered_rps, 1e-9))
        # Binary SLO violation: latency OR load-shedding
        slo_violation = 1.0 if (snap["p95"] > self.slo or shortfall_frac > 0.10) else 0.0

        # ---- Continuous reward with shortfall and smoothness terms ----
        p95_excess        = max(0.0, snap["p95"] - self.slo)
        shortfall_penalty = max(0.0, shortfall_frac - 0.10)
        risk_exposure     = 0.0 if self.homogeneous else (
            self._spot.risk_signal() * (snap["pods_spot"] / max(1.0, snap["pods"]))
        )
        reward = -(
            self.w_cost   * snap["cost_per_hr"]
            + self.w_slo  * (p95_excess + 2.0 * shortfall_penalty)
            + self.w_risk * risk_exposure
            + self.w_smooth * abs(delta)
        )

        self.i    += 1
        terminated = self.i >= len(self.trace)

        info = {
            **snap,
            "evicted":        evicted,
            "slo_violation":  slo_violation,
            "shortfall_frac": shortfall_frac,
            "p95_client":     snap["p95"],   # in sim, M/D/k p95 IS the client-side model
            "reward":         reward,
            "offered_rps":    offered_rps,
            "served_rps":     snap["rps"],
        }
        return self._obs(snap), float(reward), terminated, False, info
