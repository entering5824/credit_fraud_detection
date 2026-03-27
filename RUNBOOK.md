# Fraud Detection AI Agent — Operational Runbook

## SLOs

| Endpoint | p95 Latency Target | Error Rate |
|---|---|---|
| `POST /agent/analyze` | ≤ 200 ms | < 1 % |
| `POST /agent/investigate` | ≤ 1 000 ms | < 1 % |
| `POST /agent/explain` | ≤ 800 ms | < 1 % |
| `GET /agent/session/{id}` | ≤ 50 ms | < 0.1 % |

**Fraud alert threshold for manual review queue**: probability > 0.85 (business-configurable via `config/thresholds.yaml`).

---

## Prometheus Alert Rules

Paste these into an `alertmanager_rules.yml` or your existing alert config:

```yaml
groups:
  - name: fraud_agent_slos
    rules:

      - alert: AnalyzeP95LatencyHigh
        expr: histogram_quantile(0.95, rate(inference_latency_ms_bucket{request_type="analyze"}[5m])) > 200
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "/agent/analyze p95 latency > 200 ms"
          description: "p95={{ $value | printf \"%.0f\" }}ms over last 5 minutes"

      - alert: InvestigateP95LatencyHigh
        expr: histogram_quantile(0.95, rate(inference_latency_ms_bucket{request_type="investigate"}[5m])) > 1000
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "/agent/investigate p95 latency > 1 000 ms"

      - alert: ToolErrorRateHigh
        expr: rate(tool_errors_total[10m]) / rate(tool_calls_total[10m]) > 0.01
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Tool error rate > 1% for {{ $labels.tool_name }}"

      - alert: FeatureDriftCritical
        # Requires drift_monitoring_tool to expose a custom metric per feature.
        # Until then, alert when > 5% of investigations return partial_report.
        expr: rate(partial_reports_total[15m]) / rate(agent_requests_total[15m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "More than 5% of investigations are returning partial (degraded) reports"

      - alert: FraudAlertRateSpike
        expr: rate(model_prediction_total{prediction="fraud"}[10m]) / rate(model_prediction_total[10m]) > 0.3
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Fraud prediction rate > 30% over 10 minutes — possible data quality or attack issue"
```

---

## Incident Response Playbooks

### Alert: AnalyzeP95LatencyHigh

1. Check `inference_latency_ms` histogram in Grafana: is this all endpoints or one?
2. Check `tool_calls_total` and `tool_errors_total` — if error rate is also spiking, it's a downstream tool issue, not latency.
3. If only `analyze`, check CPU/memory on the API container: `docker stats`.
4. Scale API replicas if needed: `docker compose up --scale api=4`.
5. Verify `fraud_scoring` tool is using the cached model (check logs for `load_artifacts_from_registry` calls — it should only appear once per process).
6. If not resolved in 15 min, page on-call ML engineer.

### Alert: InvestigateP95LatencyHigh

1. Check if `/agent/investigate` is being called with `include_explanation=true` and `include_behavior=true` — SHAP + behavior can push p95 high.
2. Consider routing high-volume callers to `/agent/analyze` (fast path).
3. Check SHAP explainer warm-start: look for `TreeExplainer created for key=` in logs. If absent, model file may have changed and cache was cleared — this causes a one-time spike.
4. If latency is > 3× SLO and sustained, temporarily disable `include_explanation` in the planner config and route explain requests to an async job queue.

### Alert: ToolErrorRateHigh

1. Identify which `tool_name` label has the highest error rate: `rate(tool_errors_total[5m]) by (tool_name)`.
2. `feature_explanation`: check SHAP import; if `shap` package is missing or incompatible, reinstall.
3. `drift_monitoring`: check that baseline data file exists at `results/monitoring/baseline.parquet`.
4. `fraud_scoring`: model file may be missing or corrupt; verify `models/xgboost.pkl` size and checksum.
5. `behavior_analysis`: check `add_synthetic_behavior_features` function for schema changes.
6. All failures are captured in `partial_report: true` responses — investigations continue in degraded mode. Confidence scores are penalised automatically.

### Alert: FeatureDriftCritical

Per-feature PSI thresholds:
- PSI 0.10–0.20: investigate (warn)
- PSI > 0.20: high drift (alert)
- PSI > 0.30: retrain required (critical)

Steps:
1. Run full drift report: `POST /agent/investigate` with `include_drift=true` on a recent batch.
2. Check `drift_analysis.top_drifted_features` in the response.
3. If drift is concentrated in PCA components (V1–V28): likely upstream data pipeline change — check ETL logs.
4. If drift is in `Amount`: possible new fraud campaign or seasonal pattern.
5. If > 3 features have PSI > 0.20: schedule model retrain. Use `model_version` and `feature_vector_hash_full` from `_meta` for audit trail.
6. Swap model artifact in `models/` and clear SHAP cache: call `clear_explainer_cache()` in a maintenance endpoint.

### Alert: FraudAlertRateSpike

Possible causes:
- **Attack campaign**: alert fraud ops immediately.
- **Model degradation**: compare current `fraud_probability` distribution with training-time baseline.
- **Data pipeline issue**: upstream features may be corrupted (all V features = 0, etc).

Steps:
1. Sample 20 recent high-probability transactions from `/agent/session/{id}` and inspect `feature_vector_hash_full`, `behavioral_anomalies`, `fraud_pattern`.
2. If `fraud_pattern = testing_attack` or `velocity_fraud` is dominant → escalate to fraud ops.
3. If `fraud_pattern = unknown` for most → likely a model/data issue, not a real attack.
4. Temporarily raise the alert threshold from 0.85 to 0.95 to reduce false positives while investigating.

---

## Model Rollback Procedure

1. Identify the last known-good `model_version` from `_meta.model_version` in stored sessions.
2. Copy the old artifact to `models/xgboost.pkl` (from S3 or artifact store).
3. Restart API containers: `docker compose restart api`.
4. The SHAP explainer cache auto-clears on restart — warm-up takes ~1 request per worker.
5. Monitor `fraud_probability_distribution` histogram: confirm distribution returns to baseline within 10 minutes.

---

## Session Access and Privacy

- `GET /agent/session/{id}` returns full investigation including raw features. Ensure this endpoint is behind JWT or mTLS in production.
- Session LRU cap is 500. Eviction is FIFO. If PII data-retention policy requires earlier eviction, reduce `_MAX_SESSIONS` in `src/memory/investigation_memory.py`.
- To redact PII before saving sessions, override `store_investigation()` and mask fields like `user_id`, `card_last_4`, etc.
