# AI Fraud Investigation Agent — Architecture

## Overview

This project has evolved from a standalone ML fraud detection system into a
**multi-capability AI Fraud Investigation Agent Platform**.

The agent embodies three core properties of autonomous AI agents:

| Property | Implementation |
|---|---|
| **Reason** | `RiskReasoner` — fraud pattern classification, confidence scoring, signal ranking |
| **Use Tools** | `Tool` registry — fraud scoring, SHAP, behavior analysis, graph, drift |
| **Maintain State/Memory** | `InvestigationSession` memory + `FraudCase` case lifecycle |

---

## Full System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         EXTERNAL INTERFACES                         │
│                                                                     │
│   Analyst Browser      REST API Clients      Kafka Topic            │
│         │                    │                   │                  │
│         ▼                    ▼                   ▼                  │
│   ┌───────────────┐   ┌────────────┐    ┌─────────────────┐        │
│   │ Streamlit App │   │  FastAPI   │    │  Kafka Consumer │        │
│   │  (3 pages)    │   │  /agent/*  │    │  (stream proc.) │        │
│   │               │   │  /cases/*  │    └────────┬────────┘        │
│   └───────┬───────┘   └─────┬──────┘             │                 │
│           │                 │                     │                 │
└───────────┼─────────────────┼─────────────────────┼─────────────────┘
            │                 │                     │
            ▼                 ▼                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        AGENT RUNTIME LAYER                          │
│                                                                     │
│   AgentLoop (background workers — Observe/Plan/Act/Evaluate)        │
│   ├── AgentTaskQueue  (priority queue — CRITICAL→BACKGROUND)        │
│   └── AgentExecutor   (stateless task runner)                       │
│                                                                     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        AGENT CORE                                   │
│                                                                     │
│   AgentOrchestrator                                                 │
│   ├── 1. OBSERVE  — receive features + context                      │
│   ├── 2. PLAN     — BasePlanner → ToolPlan                          │
│   │       ├── RuleBasedPlanner  (fast, deterministic)               │
│   │       └── LLMPlanner        (GPT-4o / Ollama, fallback to rule) │
│   ├── 3. ACT      — execute ToolPlan (graceful degradation)         │
│   │       Tool Registry (self-registering):                         │
│   │         • fraud_scoring       (ML model → probability)          │
│   │         • feature_explanation (SHAP TreeExplainer, cached)      │
│   │         • behavior_analysis   (spending spike, velocity, merch) │
│   │         • drift_monitoring    (PSI per feature)                 │
│   │         • transaction_history (TransactionStore abstraction)    │
│   │         • graph_analysis      (TransactionGraph patterns)       │
│   ├── 4. REASON   — RiskReasoner                                    │
│   │         • fraud pattern classification (8 patterns)             │
│   │         • confidence scoring                                    │
│   │         • signal ranking                                        │
│   │         • analyst_summary + recommended_action                  │
│   └── 5. REPORT   — TransactionRiskReport (normalised JSON)         │
│                                                                     │
└──────────────┬───────────────────────────────┬──────────────────────┘
               │                               │
               ▼                               ▼
┌──────────────────────────┐   ┌───────────────────────────────────────┐
│     MEMORY LAYER         │   │         CASE MANAGEMENT               │
│                          │   │                                       │
│  InvestigationSession    │   │  CaseManager                          │
│  (LRU-capped, 500 max)   │   │  ├── auto-open on prob ≥ 0.50         │
│  • tool_calls audit      │   │  ├── lifecycle: open→review→confirmed │
│  • agent_decisions       │   │  ├── analyst notes + assignment       │
│  • final_reports         │   │  └── CaseStore (pluggable backend)    │
│                          │   │                                       │
└──────────────────────────┘   └───────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                     SUPPORTING LAYERS                               │
│                                                                     │
│  Graph Layer                                                        │
│  ├── TransactionGraph    (bipartite user-merchant graph)            │
│  └── GraphFraudDetector  (velocity, card-testing, dormant, cluster) │
│                                                                     │
│  Knowledge Base                                                     │
│  └── FraudKnowledgeBase  (8 patterns + triage rules + guidelines)   │
│                                                                     │
│  Simulation                                                         │
│  └── FraudSimulator      (normal, fraud, attack scenarios)          │
│                                                                     │
│  Observability                                                      │
│  ├── Prometheus metrics  (GET /metrics)                             │
│  ├── OpenTelemetry spans (Jaeger)                                   │
│  └── AgentLogger         (JSONL audit trail)                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Agent Capabilities

### 1. Transaction Risk Analysis

**Entry point**: `POST /agent/analyze` (fast) or `POST /agent/investigate` (full)

The agent scores any transaction and returns:
- `fraud_probability` — model output
- `risk_score` — 0–100
- `risk_level` — low / medium / high / critical
- `prediction` — 0 or 1 at configured threshold
- `recommended_action` — approve / monitor / flag for manual review

### 2. SHAP Model Explanation

**Entry point**: `POST /agent/explain` or `include_explanation=true`

- `TreeExplainer` cached per model version (warm-started at process launch)
- Returns `top_k` features with SHAP value, feature value, and direction
- Generates human-readable narrative (`_narrative_from_top_features`)

### 3. Behaviour Analysis

**Tool**: `behavior_analysis`

Computes from raw transaction features:
- `spending_spike_ratio` — amount vs. account average
- `merchant_frequency_score` — novelty of merchant
- `transaction_velocity_1h` — short-window count
- `is_new_merchant_for_user` — first-seen flag
- `flags` — human-readable signals

### 4. Fraud Pattern Detection

**Layer**: `RiskReasoner` + `GraphFraudDetector`

Single-transaction patterns (rule-based):
- `velocity_fraud`, `account_takeover`, `testing_attack`, `large_anomalous_purchase`, `unknown`

Multi-transaction patterns (graph-based):
- `velocity_burst`, `card_testing_attack`, `dormant_account`, `high_degree_anomaly`, `merchant_cluster`

### 5. Investigation Report Generation

Every investigation produces a `TransactionRiskReport` with:

```json
{
  "transaction_id": "...",
  "fraud_probability": 0.92,
  "risk_score": 92.0,
  "risk_level": "critical",
  "fraud_pattern": "account_takeover",
  "confidence_score": 0.87,
  "key_risk_signals": [...],
  "ranked_signals": [...],
  "behavioral_anomalies": [...],
  "model_explanation": {"top_features_contributing": [...], "narrative": "..."},
  "recommended_action": "flag for manual review",
  "analyst_summary": "...",
  "partial_report": false,
  "session_id": "...",
  "case_id": "...",
  "_meta": {
    "tool_calls": [...],
    "failed_tools": [],
    "feature_vector_hash_full": "...",
    "execution_time_ms": 124.3
  }
}
```

### 6. Case Management

High-risk investigations automatically create a `FraudCase`:

```
open → under_review → confirmed_fraud
                    → false_positive
                    → dismissed
```

API: `GET /cases`, `GET /cases/{id}`, `POST /cases/{id}/status`, `/note`, `/assign`

### 7. Agent Memory (Session)

Each investigation is stored in an `InvestigationSession`:
- LRU-capped at 500 sessions
- Full audit trail: tool_calls, agent_decisions, final_reports
- Retrievable via `GET /agent/session/{session_id}`

### 8. Autonomous Runtime Loop

`AgentLoop` runs in background thread(s):

```
submit_task(features)
    │
    ▼  OBSERVE: extract features + context
    │  PLAN:    planner selects tools
    │  ACT:     executor runs tools (with degradation policy)
    │  EVALUATE: escalation check, callback, result storage
    ▼
get_result(task_id)
```

Priority queue ensures critical-risk transactions are processed first.

---

## Data Flow

```
Transaction event
    │
    ├── (Kafka) FraudEventProcessor → AgentLoop.submit_task()
    │
    └── (API) POST /agent/investigate → run_investigation()
                                              │
                                    AgentOrchestrator.run()
                                              │
                             ┌────────────────┼────────────────┐
                             │                │                │
                     fraud_scoring    feature_explanation  behavior_analysis
                     [REQUIRED]       [if requested]       [if requested]
                             │                │                │
                             └────────────────┼────────────────┘
                                              │
                                       RiskReasoner
                                       .reason(...)
                                              │
                                    TransactionRiskReport
                                              │
                          ┌───────────────────┼───────────────────┐
                          │                   │                   │
                 InvestigationSession    CaseManager         AgentLogger
                 (memory)                (auto-open)         (JSONL + OTel)
```

---

## Module Reference

| Module | Location | Purpose |
|---|---|---|
| Agent Loop | `src/agent_runtime/agent_loop.py` | Autonomous observe/plan/act/evaluate runtime |
| Task Queue | `src/agent_runtime/task_queue.py` | Priority queue for investigation tasks |
| Executor | `src/agent_runtime/agent_executor.py` | Stateless task runner |
| Orchestrator | `src/agents/agent_orchestrator.py` | Stateful investigation coordinator |
| Rule Planner | `src/agents/planners/rule_based_planner.py` | Deterministic tool selector |
| LLM Planner | `src/agents/planners/llm_planner.py` | OpenAI-compatible dynamic planner |
| Risk Reasoner | `src/agents/reasoning/risk_reasoner.py` | Pattern + confidence + signals |
| Tool Registry | `src/tools/registry.py` | Self-registering tool catalogue |
| Tools | `src/tools/*.py` | Scoring, SHAP, behaviour, drift, history, graph |
| Case Manager | `src/cases/case_manager.py` | Fraud case lifecycle |
| Case Store | `src/cases/case_store.py` | Pluggable case persistence |
| Memory | `src/memory/investigation_memory.py` | Session LRU store |
| Graph | `src/graph/transaction_graph.py` | User-merchant bipartite graph |
| Graph Patterns | `src/graph/fraud_patterns.py` | Multi-tx fraud detection |
| Knowledge Base | `src/knowledge/fraud_knowledge_base.py` | Pattern catalogue + LLM context |
| Simulator | `src/simulation/fraud_simulator.py` | Synthetic transaction generation |
| Kafka Consumer | `pipeline/stream/kafka_consumer.py` | Stream ingestion |
| Event Processor | `pipeline/stream/fraud_event_processor.py` | Transport-agnostic event handler |
| Metrics | `src/monitoring/metrics.py` | Prometheus counters/histograms |
| Tracing | `src/monitoring/tracing.py` | OpenTelemetry spans |
| Agent Logger | `src/monitoring/agent_logger.py` | JSONL audit trail |
| Analyst UI | `app/pages/*.py` | Streamlit analyst console |

---

## Extension Points

| What to extend | How |
|---|---|
| Add a new tool | Create `src/tools/my_tool.py`, call `register_tool(...)` |
| Use Redis for cases | Implement `CaseStore` ABC, call `set_case_store(RedisCaseStore())` |
| Use Redis for history | Implement `TransactionStore` ABC, call `set_transaction_store(...)` |
| Switch to LLM planner | `AgentOrchestrator(planner=LLMPlanner())` |
| Add graph storage | Swap `TransactionGraph` with NetworkX or Neo4j backend |
| Add new fraud pattern | Extend `_PATTERNS` in `risk_reasoner.py` and `fraud_knowledge_base.py` |
| Add a Kafka sink | Implement alert callback and pass to `FraudKafkaConsumer` |
