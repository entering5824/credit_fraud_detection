/**
 * k6 Load Test — Fraud Detection AI Agent
 *
 * Tests two endpoints with configurable concurrency:
 *   1. POST /agent/analyze   — fast path (scoring only), p95 SLO <= 200 ms
 *   2. POST /agent/investigate — full investigation, p95 SLO <= 1000 ms
 *
 * Run:
 *   k6 run tests/load/agent_load_test.js
 *   k6 run --vus 20 --duration 60s tests/load/agent_load_test.js
 *   k6 run -e BASE_URL=http://localhost:8000 tests/load/agent_load_test.js
 *
 * Environment variables:
 *   BASE_URL      – defaults to http://localhost:8000
 *   VUS_ANALYZE   – virtual users for /agent/analyze (default: 10)
 *   VUS_INVEST    – virtual users for /agent/investigate (default: 5)
 *   DURATION      – test duration (default: 60s)
 */

import http from 'k6/http';
import { check, sleep, group } from 'k6';
import { Trend, Counter, Rate } from 'k6/metrics';

// ── Custom metrics ──────────────────────────────────────────────────────────
const analyzeLatency   = new Trend('analyze_latency_ms', true);
const investLatency    = new Trend('invest_latency_ms',  true);
const errorRate        = new Rate('error_rate');
const partialReports   = new Counter('partial_reports');

// ── Config ──────────────────────────────────────────────────────────────────
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';

export const options = {
  scenarios: {
    analyze_fast: {
      executor: 'constant-vus',
      vus: parseInt(__ENV.VUS_ANALYZE || '10'),
      duration: __ENV.DURATION || '60s',
      exec: 'analyzeScenario',
    },
    investigate_full: {
      executor: 'constant-vus',
      vus: parseInt(__ENV.VUS_INVEST || '5'),
      duration: __ENV.DURATION || '60s',
      exec: 'investigateScenario',
    },
  },
  thresholds: {
    // p95 SLOs
    'analyze_latency_ms{p(95)}':  ['p(95)<200'],
    'invest_latency_ms{p(95)}':   ['p(95)<1000'],
    // Error rate must stay below 1%
    'error_rate':                  ['rate<0.01'],
    // Standard http_req_failed as fallback
    'http_req_failed':             ['rate<0.01'],
  },
};

// ── Payload generators ──────────────────────────────────────────────────────
function baseFeatures(amount = 120.5) {
  const feat = { Amount: amount, Time: 3600.0 };
  for (let i = 1; i <= 28; i++) {
    feat[`V${i}`] = (Math.random() - 0.5) * 4;
  }
  return feat;
}

function highRiskFeatures() {
  return baseFeatures(5000.0);
}

// ── Scenarios ───────────────────────────────────────────────────────────────

export function analyzeScenario() {
  group('/agent/analyze', () => {
    const payload = JSON.stringify({
      transaction_id: `tx_load_${Date.now()}_${Math.random().toString(36).slice(2)}`,
      features: baseFeatures(),
    });

    const headers = { 'Content-Type': 'application/json' };
    const res = http.post(`${BASE_URL}/agent/analyze`, payload, { headers, timeout: '5s' });

    analyzeLatency.add(res.timings.duration);

    const ok = check(res, {
      'analyze: status 200':              (r) => r.status === 200,
      'analyze: has fraud_probability':   (r) => {
        try { return JSON.parse(r.body).fraud_probability !== undefined; }
        catch { return false; }
      },
      'analyze: has risk_level':          (r) => {
        try { return JSON.parse(r.body).risk_level !== undefined; }
        catch { return false; }
      },
    });

    if (!ok || res.status !== 200) {
      errorRate.add(1);
    } else {
      errorRate.add(0);
      try {
        const body = JSON.parse(res.body);
        if (body.partial_report === true) partialReports.add(1);
      } catch {}
    }
  });

  sleep(0.1);
}

export function investigateScenario() {
  group('/agent/investigate', () => {
    const payload = JSON.stringify({
      transaction_id: `tx_inv_${Date.now()}_${Math.random().toString(36).slice(2)}`,
      features: highRiskFeatures(),
      include_explanation: true,
      include_behavior: true,
    });

    const headers = { 'Content-Type': 'application/json' };
    const res = http.post(`${BASE_URL}/agent/investigate`, payload, { headers, timeout: '10s' });

    investLatency.add(res.timings.duration);

    const ok = check(res, {
      'investigate: status 200':           (r) => r.status === 200,
      'investigate: has recommended_action': (r) => {
        try { return !!JSON.parse(r.body).recommended_action; }
        catch { return false; }
      },
      'investigate: has session_id':       (r) => {
        try { return !!JSON.parse(r.body).session_id; }
        catch { return false; }
      },
      'investigate: has _meta':            (r) => {
        try { return JSON.parse(r.body)._meta !== undefined; }
        catch { return false; }
      },
    });

    if (!ok || res.status !== 200) {
      errorRate.add(1);
    } else {
      errorRate.add(0);
      try {
        const body = JSON.parse(res.body);
        if (body.partial_report === true) partialReports.add(1);
      } catch {}
    }
  });

  sleep(0.5);
}

// ── Smoke test entrypoint ───────────────────────────────────────────────────
export default function () {
  // Runs when no specific scenario is selected (for quick sanity checks)
  analyzeScenario();
  investigateScenario();
}
