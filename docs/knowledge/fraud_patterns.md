# Fraud Pattern Knowledge Base

Reference document for the Fraud Investigation Agent.
Updated by the fraud operations team. The agent's `FraudKnowledgeBase`
module loads this file to enrich investigation reports.

---

## Pattern Catalog

### 1. Velocity Fraud (`velocity_fraud`)

**Description**  
The cardholder (or attacker) performs many transactions in a short time
window, typically within one hour.

**Signals**
- `transactions_last_1h >= 5`
- `transaction_velocity_1h` spike

**Risk level**: High → Critical  
**Recommended action**: Block card, escalate to fraud ops  
**False positive risk**: Low (surge purchases at concerts, travel)

**Investigation steps**
1. Review the past 24 h transaction history.
2. Check if transactions span multiple merchants or one repeated merchant.
3. Confirm with cardholder via OTP.

---

### 2. Account Takeover (`account_takeover`)

**Description**  
An attacker gains access to a legitimate account and transacts at new
merchants with atypical amounts.

**Signals**
- `is_new_merchant_for_user == 1`
- `spending_spike_ratio >= 3`
- IP/device change (if available)
- Password change in last 24 h (external signal)

**Risk level**: High → Critical  
**Recommended action**: Lock account, escalate to fraud ops  
**False positive risk**: Medium (travel, gift purchases)

---

### 3. Card Testing Attack (`testing_attack`)

**Description**  
Attacker probes card validity with very small amounts across multiple
merchants before proceeding to large purchases.

**Signals**
- Amount ≤ $5 on 3+ distinct merchants within 1 h
- `transaction_velocity_1h >= 3`
- `merchant_frequency_score` very low for all merchants

**Risk level**: Critical  
**Recommended action**: Block card immediately  
**False positive risk**: Very low

---

### 4. Large Anomalous Purchase (`large_anomalous_purchase`)

**Description**  
A single transaction with an amount far above the account's historical
average with no behavioral context (no velocity, no new merchant pattern).

**Signals**
- `spending_spike_ratio >= 4`
- `Amount >= 500`
- Low `transaction_velocity_1h`

**Risk level**: High  
**Recommended action**: Flag for manual review, optional 2FA  
**False positive risk**: High (luxury, travel, medical)

---

### 5. Velocity Burst — Graph Level (`velocity_burst`)

**Description**  
Detected at the graph level from `TransactionGraph`.  
User fires ≥ 5 transactions in 1 h.  Differs from `velocity_fraud` in
that this is detected from historical graph data, not just the current
transaction's behavioral features.

**Risk level**: High → Critical  
**Recommended action**: Temporary card hold

---

### 6. Merchant Cluster (`merchant_cluster`)

**Description**  
A merchant has a historical fraud rate ≥ 15%.  Any transaction at this
merchant is elevated to at least HIGH risk, regardless of other signals.

**Signals**
- `merchant_fraud_rate >= 0.15`

**Risk level**: High  
**Recommended action**: Flag for manual review

---

### 7. Dormant Account (`dormant_account`)

**Description**  
The account has been inactive for ≥ 30 days and suddenly executes a
transaction, especially at a new merchant.  Common in account-takeover
scenarios where the attacker waits for fraud controls to relax.

**Risk level**: Medium  
**Recommended action**: Send OTP, monitor next 24 h

---

### 8. High Degree Anomaly (`high_degree_anomaly`)

**Description**  
The user transacted at ≥ 5 distinct merchants in 24 h, which is unusual
for most consumer profiles.

**Risk level**: High  
**Recommended action**: Monitor, optional 2FA

---

## Investigation Guidelines

### Triage Priority

| Risk Level | Probability | Action           | SLA        |
|------------|-------------|------------------|------------|
| Critical   | ≥ 0.85      | Immediate review | 15 minutes |
| High       | 0.50–0.84   | Same-day review  | 4 hours    |
| Medium     | 0.25–0.49   | Weekly review    | 3 days     |
| Low        | < 0.25      | Auto-approve     | —          |

### Evidence Collection Checklist

- [ ] Run SHAP explanation (`include_explanation: true`)
- [ ] Run behavior analysis (`include_behavior: true`)
- [ ] Check transaction history for last 30 days
- [ ] Check graph patterns (velocity, degree, merchant cluster)
- [ ] Review `feature_vector_hash_full` and `model_version` for reproducibility
- [ ] Note if `partial_report: true` and which tools failed

### Common False Positive Scenarios

| Scenario              | Distinguishing Signal              |
|-----------------------|------------------------------------|
| Holiday shopping      | Multiple merchants, normal amounts |
| International travel  | New merchants, high Amount         |
| Online subscription   | Recurring at known merchant        |
| Large medical bill    | Single transaction, no velocity    |

### Escalation Contacts

| Situation              | Team              |
|------------------------|-------------------|
| Active fraud campaign  | Fraud Operations  |
| Model drift suspected  | ML Engineering    |
| System tool failures   | Platform SRE      |
| > 5% alert rate spike  | Risk Management   |
