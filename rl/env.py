"""The Gym environment that turns autoscaling into an RL problem.

STATE   : [rps, cpu_util, p95, pods, spot_fraction, spot_risk, hour_of_day]
ACTION  : MultiDiscrete([5, 2])
            dim 0 = replica delta in {-2,-1,0,+1,+2}
            dim 1 = target pool in {ondemand, spot}   <-- the spot-aware action
REWARD  : -(cost)  -  slo_penalty  -  risk_penalty
            This reward is THE contribution. Sweep the weights and report
            sensitivity in the paper.

Set `homogeneous=True` to get the ablation baseline: the agent can still scale
but its placement action and the risk term are disabled, so it is spot-blind.
"""
import time, numpy as np, gymnasium as gym
from gymnasium import spaces

from metrics.collect import Metrics
from rl.k8s_actions import K8sActions
from rl.spot_simulator import SpotSimulator

class AutoscaleEnv(gym.Env):
    def __init__(self, trace, step_s=60, slo_p95=0.2, homogeneous=False,
                 w_cost=1.0, w_slo=5.0, w_risk=2.0, settle_s=15):
        super().__init__()
        self.trace = trace                 # DataFrame with column 'rps'
        self.step_s, self.slo = step_s, slo_p95
        self.homogeneous = homogeneous
        self.w_cost, self.w_slo, self.w_risk = w_cost, w_slo, w_risk
        self.settle_s = settle_s
        self.m = Metrics(); self.k = K8sActions(); self.spot = SpotSimulator(step_s=step_s)

        self.action_space = spaces.MultiDiscrete([5, 2])
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(7,), dtype=np.float32)
        self.i = 0

    def _obs(self, snap):
        spot_frac = snap["pods_spot"] / max(1.0, snap["pods"])
        hod = (self.i * self.step_s / 3600.0) % 24
        return np.array([snap["rps"], snap["cpu_util"], snap["p95"], snap["pods"],
                         spot_frac, self.spot.risk_signal(), hod], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.i = 0
        self.k.set_replicas(1)
        return self._obs(self.m.snapshot()), {}

    def step(self, action):
        delta = [-2, -1, 0, 1, 2][int(action[0])]
        # apply scaling
        self.k.set_replicas(self.k.get_replicas() + delta)
        # apply placement (spot-aware agent only)
        if not self.homogeneous:
            self.k.set_node_pref("spot" if int(action[1]) == 1 else "ondemand")
        # advance the world: drive this step's load externally, then maybe evict
        # NOTE: the experiment runner fires k6 at trace.rps[self.i] in parallel.
        time.sleep(self.settle_s)
        evicted = self.spot.step()

        snap = self.m.snapshot()
        slo_violation = max(0.0, snap["p95"] - self.slo)
        risk_exposure = 0.0 if self.homogeneous else \
            self.spot.risk_signal() * (snap["pods_spot"] / max(1.0, snap["pods"]))

        reward = -(self.w_cost * snap["cost_per_hr"]
                   + self.w_slo * slo_violation
                   + self.w_risk * risk_exposure)

        self.i += 1
        terminated = self.i >= len(self.trace)
        info = {**snap, "evicted": evicted, "slo_violation": slo_violation, "reward": reward}
        return self._obs(snap), float(reward), terminated, False, info
