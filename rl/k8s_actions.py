"""Thin layer the RL agent uses to act on the cluster:
  - set replica count
  - steer placement toward spot or on-demand via nodeAffinity
Baselines never touch placement; only the spot-aware agent uses set_node_pref().
"""
from kubernetes import client, config

class K8sActions:
    def __init__(self, ns="bench", deploy="workload"):
        try:
            config.load_incluster_config()
        except Exception:
            config.load_kube_config()
        self.apps = client.AppsV1Api()
        self.ns, self.deploy = ns, deploy

    def get_replicas(self):
        return self.apps.read_namespaced_deployment(self.deploy, self.ns).spec.replicas

    def set_replicas(self, n):
        n = max(1, min(20, int(n)))
        self.apps.patch_namespaced_deployment_scale(
            self.deploy, self.ns, {"spec": {"replicas": n}})
        return n

    def set_node_pref(self, lifecycle):
        """lifecycle in {'spot','ondemand'} — bias scheduling toward that pool."""
        patch = {"spec": {"template": {"spec": {"affinity": {"nodeAffinity": {
            "preferredDuringSchedulingIgnoredDuringExecution": [{
                "weight": 100,
                "preference": {"matchExpressions": [{
                    "key": "node-lifecycle", "operator": "In", "values": [lifecycle]}]}}]}}}}}}
        self.apps.patch_namespaced_deployment(self.deploy, self.ns, patch)
