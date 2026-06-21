// Stateless API whose per-request CPU cost is tunable, so latency responds
// predictably to load and to replica count. Exposes a Prometheus histogram so
// you can measure p95/p99 latency (your SLO metric). This is your scalable service.
// (Built on Node/Express to match your MERN background — swap in your own app if you
//  have one; the only contract is: expose /metrics with a request-latency histogram.)

const express = require("express");
const client = require("prom-client");

const app = express();
const register = client.register;
client.collectDefaultMetrics({ register });

const latency = new client.Histogram({
  name: "http_request_duration_seconds",
  help: "Request latency",
  buckets: [0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2], // tuned around a 200ms SLO
});

// WORK_MS = synthetic CPU cost per request. Raise it to make the service
// saturate sooner (so scaling decisions matter). TODO: calibrate for your SLO.
const WORK_MS = parseInt(process.env.WORK_MS || "40", 10);

function burnCpu(ms) {
  const end = Date.now() + ms;
  let x = 0;
  while (Date.now() < end) { x += Math.sqrt(Math.random() * 1e6); }
  return x;
}

app.get("/work", (req, res) => {
  const stop = latency.startTimer();
  burnCpu(WORK_MS);
  stop();
  res.json({ ok: true, pod: process.env.HOSTNAME });
});

app.get("/healthz", (_, res) => res.send("ok"));
app.get("/metrics", async (_, res) => {
  res.set("Content-Type", register.contentType);
  res.end(await register.metrics());
});

app.listen(8080, () => console.log(`workload up, WORK_MS=${WORK_MS}`));
