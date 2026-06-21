// k6 load replayer: reads one trace step's target RPS from env and drives it.
// The experiment runner calls k6 once per trace step with TARGET_RPS set, so the
// arrival pattern matches load/trace.csv exactly.  Install: https://k6.io
import http from "k6/http";
import { sleep } from "k6";

const RPS = parseInt(__ENV.TARGET_RPS || "30");
const URL = __ENV.TARGET_URL || "http://localhost:8080/work";

export const options = {
  scenarios: {
    step: {
      executor: "constant-arrival-rate",
      rate: RPS, timeUnit: "1s",
      duration: __ENV.STEP_DURATION || "60s",
      preAllocatedVUs: Math.max(20, RPS), maxVUs: Math.max(50, RPS * 3),
    },
  },
};

export default function () { http.get(URL); }
