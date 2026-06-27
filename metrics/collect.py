"""Pulls a single observation snapshot from Prometheus: pod count, CPU/mem
utilization, p95/p99 latency, request rate, and the spot/on-demand split.
Also computes instantaneous infrastructure cost. This is the ground truth that
feeds both the RL state and the evaluation metrics.

TODO: set PROM_URL via `kubectl -n monitoring port-forward svc/prom-...-prometheus 9090`.
"""
import math, os
from prometheus_api_client import PrometheusConnect

PROM_URL = os.environ.get("PROM_URL", "http://localhost:9090")

# Per-hour price per node by lifecycle. TODO: use the real prices you cite in the paper
# (e.g. an AWS instance family). Spot ~ 70% cheaper is the headline of your study.
PRICE = {"ondemand": 1.00, "spot": 0.30}

class Metrics:
    def __init__(self, ns="bench", url=PROM_URL):
        self.p = PrometheusConnect(url=url, disable_ssl=True)
        self.ns = ns

    def _q(self, query, default=0.0):
        try:
            r = self.p.custom_query(query)
            if not r:
                return default
            val = float(r[0]["value"][1])
            return default if math.isnan(val) or math.isinf(val) else val
        except Exception:
            return default

    def snapshot(self):
        ns = self.ns
        pods_on = self._q(
            f'count(kube_pod_info{{namespace="{ns}"}} * on(node) '
            f'group_left(label_node_lifecycle) kube_node_labels{{label_node_lifecycle="ondemand"}})')
        pods_spot = self._q(
            f'count(kube_pod_info{{namespace="{ns}"}} * on(node) '
            f'group_left(label_node_lifecycle) kube_node_labels{{label_node_lifecycle="spot"}})')
        return {
            "pods": self._q(f'count(kube_pod_info{{namespace="{ns}"}})'),
            "pods_ondemand": pods_on,
            "pods_spot": pods_spot,
            "cpu_util": self._q(
                f'avg(rate(container_cpu_usage_seconds_total{{namespace="{ns}"}}[1m]))'),
            "rps": self._q(
                f'sum(rate(http_request_duration_seconds_count{{namespace="{ns}"}}[1m]))'),
            "p95": self._q(
                f'histogram_quantile(0.95, sum(rate('
                f'http_request_duration_seconds_bucket{{namespace="{ns}"}}[1m])) by (le))'),
            "p99": self._q(
                f'histogram_quantile(0.99, sum(rate('
                f'http_request_duration_seconds_bucket{{namespace="{ns}"}}[1m])) by (le))'),
            "cost_per_hr": pods_on * PRICE["ondemand"] + pods_spot * PRICE["spot"],
        }
