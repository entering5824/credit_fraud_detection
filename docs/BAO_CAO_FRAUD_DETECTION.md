# Báo cáo Hệ thống Phát hiện Gian lận (Fraud Detection)

Báo cáo này trả lời các câu hỏi về bối cảnh nghiệp vụ, dữ liệu, feature engineering, mô hình ML, hệ thống real-time và an ninh của dự án **Credit Card Fraud Detection** trong repository này.

---

## 1. Bối cảnh nghiệp vụ (Business & Banking Context)

### Hệ thống fraud detection áp dụng cho credit card, debit card hay cả hai?

- **Credit card** là đối tượng chính (dataset Kaggle creditcard, creditcard_2023, và các bộ tổng hợp có schema tương tự).
- Một số dataset trong repo mô phỏng **giao dịch trực tuyến / PaySim** (onlinefraud, PS_log, transactions) — có thể coi là **card-not-present / online payment**; pipeline xử lý chung qua schema chuẩn hóa nên **cả hai bối cảnh (credit card và giao dịch online) đều được hỗ trợ** ở mức dữ liệu và mô hình.

### Hệ thống phục vụ ngân hàng thật, dữ liệu mô phỏng hay dataset public?

- **Dataset public và mô phỏng**: Kaggle creditcard (PCA ẩn danh), PaySim-style (onlinefraud, PS_log, Synthetic_Financial_datasets_log), credit_card_fraud_10k, transactions, bank_transactions_data_2.
- **Không phải dữ liệu ngân hàng thật**; hệ thống thiết kế cho **nghiên cứu, portfolio và mô phỏng pipeline production-style**, có thể mở rộng sang dữ liệu thật khi có.

### Fraud được định nghĩa cụ thể thế nào trong dataset?

- Trong schema chuẩn của pipeline: **Fraud = nhãn `Class = 1`**, Normal = `Class = 0`.
- Các file gốc dùng tên cột khác nhau (`Class`, `is_fraud`, `isFraud`) và được map về cột chuẩn `Class` (0/1) khi load.
- **Không có định nghĩa nghiệp vụ chi tiết** (ví dụ chargeback, dispute) — fraud được hiểu theo nhãn có sẵn trong từng dataset.

### Fraud detection được thực hiện real-time hay batch?

- **Cả hai**:
  - **Real-time (hoặc near real-time)**: API `POST /predict` (FastAPI) nhận từng giao dịch, trả probability, risk score, prediction và giải thích SHAP.
  - **Batch**: CLI `python -m src.models.batch_scoring` score một file CSV và ghi kết quả ra file; pipeline training low-memory cũng xử lý theo batch (extract → sample → train).

### Thời gian tối đa để phát hiện fraud (latency requirement)?

- **Không được định nghĩa cứng trong code** (không có SLA ms/giây trong config).
- API ghi **latency từng request** (ms) vào `results/monitoring/api_perf.jsonl`; có script tổng hợp `latency_ms_p50`, `latency_ms_p95`, `latency_ms_p99`. Hệ thống thiết kế để inference nhanh (single transaction, model đã load sẵn); latency thực tế phụ thuộc phần cứng và có thể đạt **vài chục đến vài trăm ms** cho một request.

### Hệ thống chỉ cảnh báo (alert) hay chặn giao dịch (block)?

- **Chỉ cảnh báo**: API trả `prediction` (0/1) và `risk_score`; **không có logic block transaction** trong repo. Quyết định chặn hay chỉ flag cho analyst nằm ngoài phạm vi hệ thống ML (business rule / downstream system).

### Có human review / fraud analyst trong workflow không?

- **Có hỗ trợ gián tiếp**: Dashboard Streamlit dùng để **điều tra giao dịch đáng ngờ** (lọc theo threshold, xem danh sách risk, SHAP explanation từng giao dịch). Workflow “alert → analyst review” có thể xây trên API + dashboard; **không có module case management hay ticketing** trong repo.

### Một giao dịch bị nghi ngờ được xử lý thế nào trong business flow?

- Trong repo: giao dịch được **score → trả probability/risk + explanation**. Việc “xử lý” (hold, block, gửi sang team review) **không được triển khai**; đây là bước nghiệp vụ cần tích hợp với hệ thống bên ngoài (core banking, case management).

### KPI quan trọng nhất: giảm fraud loss hay giảm false positive?

- **Cả hai** được cân nhắc:
  - **Cost-sensitive threshold**: cấu hình `cost_fn` (chi phí bỏ sót fraud) và `cost_fp` (chi phí false positive); tìm ngưỡng tối ưu theo expected cost.
  - **Recall tại FPR cố định** (ví dụ recall@FPR=1%, 0.1%) để đảm bảo bắt đủ fraud khi giới hạn tỉ lệ báo động sai.
  - **PR-AUC** là metric chính cho class imbalance; ROC-AUC cũng được báo cáo.
- Trong code, **không ưu tiên rõ ràng** “chỉ giảm loss” hay “chỉ giảm FP” — tùy cách set `cost_fn`/`cost_fp` và threshold.

### Hệ thống có cần tuân thủ quy định ngân hàng / PCI DSS / AML không?

- **Không**: Đây là dự án ML/demo, **không có** module tuân thủ PCI DSS, AML hay quy định ngân hàng cụ thể. Dữ liệu dùng là public/synthetic; nếu triển khai thật cần bổ sung encryption, access control, audit log theo chuẩn ngành.

---

## 2. Dataset & Data Engineering

### Dataset có bao nhiêu transactions?

- **Phụ thuộc file** và giới hạn khi load:
  - `creditcard.csv` (Kaggle): ~284k dòng (có thể giới hạn bởi `max_rows` hoặc `max_rows_per_dataset`).
  - `transactions.csv`: ~299k dòng (loader mặc định có thể giới hạn 200k/dataset).
  - Các file khác: credit_card_fraud_10k (~10k), onlinefraud, PS_log, Synthetic_Financial_datasets_log, bank_transactions_data_2 (~2.5k), v.v.
- **Tổng khi train trên tất cả dataset** (với giới hạn 200k/dataset): có thể lên đến **hàng trăm nghìn đến hơn một triệu dòng** tùy cấu hình; pipeline low-memory hỗ trợ **sample có giới hạn** (ví dụ 80k) để chạy trên máy 8GB RAM.

### Tỷ lệ fraud vs normal transaction?

- **Rất không cân bằng**: Ví dụ Kaggle creditcard thường **&lt; 1% fraud**; các dataset PaySim/synthetic có thể vài phần trăm. Tỷ lệ chính xác tùy từng file và sample.

### Dataset có bị class imbalance nghiêm trọng không?

- **Có**; đây là vấn đề trung tâm. README và code nhấn mạnh **accuracy không phù hợp**; dùng **PR-AUC, recall@FPR, cost-sensitive threshold** và **class_weight / scale_pos_weight** trong mô hình để xử lý imbalance.

### Dữ liệu bao gồm những loại feature nào?

- **Chuẩn hóa sau load**: `Time` (thời gian, đơn vị giây hoặc bước), `Amount` (số tiền), `Class` (nhãn).
- **Creditcard-style**: **V1–V28** (PCA ẩn danh từ Kaggle), `Time`, `Amount`.
- **Dataset generic** (không có V1–V28): giữ **numeric** (clip -1e6..1e6) và **categorical** (one-hot); cột gốc tùy từng file (transaction amount, step, type, v.v.).
- **Không có** trong schema chuẩn: location, merchant category code, device ID, IP — trừ khi nằm trong cột raw của từng CSV và được đưa vào nhánh generic (numeric/categorical).

### Có thông tin user behavior history không?

- **Có, nhưng synthetic**: Trên dữ liệu kiểu creditcard (có V1–V28), pipeline tạo **synthetic_user_id** và **synthetic_merchant_id** từ Amount + V1–V5 (hash/quantize) để xây **behavioral features** (user_avg_amount, spending_spike_ratio, is_new_merchant_for_user, transactions_last_1h, …). **Không phải** user/merchant thật.

### Dữ liệu có timestamp để phân tích sequence không?

- **Có**: Cột `Time` (số giây hoặc bước) được chuẩn hóa từ các cột gốc (`Time`, `step`, `transaction_time`, `transaction_hour`, `TransactionDate`). Dùng để **temporal features** (time_of_day_sin/cos, time_since_last_transaction, transactions_last_1h/24h, velocity).

### Có geolocation hoặc IP data không?

- **Không**: Các dataset được hỗ trợ trong loader **không** có cột địa lý hay IP; không có feature distance anomaly (hai giao dịch hai quốc gia gần nhau) trong code.

### Dataset có missing values không? Xử lý thế nào?

- **Có**: Validation schema chỉ kiểm tra dataset không &gt; 90% NaN tổng thể.
- **Xử lý**: Trong feature engineering: numeric **fillna(0)** trước quantize; generic numeric **clip** rồi **fillna(0.0)**; categorical **fillna("NA")** rồi one-hot; cuối **fillna(0.0)** cho toàn bộ feature matrix. Inf/-inf được thay bằng NaN rồi fill 0.

### Dataset có anonymized features (V1..V28) hay raw features?

- **Cả hai**: (1) **Creditcard (Kaggle)**: V1–V28 là **PCA anonymized**; (2) **Dataset generic** (PaySim, transactions, …): giữ **raw** (numeric + categorical) rồi clip/one-hot. Pipeline tự nhận diện theo tỉ lệ NaN trên V1–V28 để chọn nhánh xử lý.

### Data được lưu ở đâu?

- **Raw**: file **CSV** trong thư mục `data/`.
- **Sau khi xử lý**: **Parquet** trong `data/processed/` (pipeline low-memory); **model artifacts** trong `models/` (pkl, registry.json).
- **Không** dùng database hay data warehouse trong repo; có thể mở rộng (đọc từ DB, ghi feature store) bên ngoài.

---

## 3. Feature Engineering

### Có sử dụng behavioral features không?

- **Có**: Trên dữ liệu creditcard-like: **user_avg_amount**, **user_med_amount**, **spending_spike_ratio**, **user_merchant_count**, **is_new_merchant_for_user**, **user_distinct_merchants_so_far**, **merchant_frequency_score**. Dựa trên synthetic user/merchant ID và **Time** (expanding/rolling trong group).

### Có feature velocity detection không?

- **Có**: **transactions_last_1h**, **transactions_last_24h**, **transaction_velocity_1h**, **transaction_velocity_24h** (số giao dịch trong 1h/24h trước đó, theo synthetic user, dựa trên `Time`).

### Có feature spending pattern deviation không?

- **Có**: **spending_spike_ratio** = Amount / user_med_amount (rolling median trước đó); phản ánh độ lệch so với mức chi tiêu thường lệ của “user” synthetic.

### Có sử dụng merchant category risk score không?

- **Không**: Không có cột merchant category hay risk score theo danh mục; chỉ có **merchant_frequency_score** (1/(1+user_merchant_count)) và **is_new_merchant_for_user**.

### Có feature distance anomaly (2 giao dịch 2 quốc gia gần nhau) không?

- **Không**: Không có geolocation/IP nên không có feature khoảng cách địa lý.

### Feature engineering được làm offline hay real-time?

- **Offline** cho training: toàn bộ feature được build trong pipeline (batch) khi train và khi extract (low-memory pipeline). **Real-time**: API nhận dict features (V1..V28, Amount, hoặc full unified); inference dùng **cùng schema đã train** (feature list từ registry). Engineered features (behavioral/temporal) **có thể** tính real-time nếu caller gửi đủ trường; pipeline hiện tại chủ yếu thiết kế cho **offline build**, inference dùng feature đã có hoặc base features.

### Có sử dụng sliding window features không?

- **Có**: **transactions_last_1h**, **transactions_last_24h** là sliding window (rolling count) theo `Time` trong group synthetic_user_id; **user_avg_amount** / **user_med_amount** dùng expanding (history đến thời điểm hiện tại).

### Feature pipeline được xây bằng công cụ gì?

- **Pandas** (trong `src/features/`): unified_features, behavior_features, temporal_features, feature_engineering. **Không** dùng Spark hay feature store chuyên dụng; có shim `src/feature_store/` (legacy) và optional persistence, nhưng pipeline chính là pandas.

---

## 4. Machine Learning / AI Model

### Đang dùng model nào?

- **Logistic Regression**, **Random Forest**, **XGBoost**; **LightGBM** (optional, bỏ qua nếu không cài). Benchmark chạy stratified CV trên cả ba (hoặc bốn) model; model mặc định cho API/dashboard thường là **XGBoost** (đăng ký trong registry).

### Vì sao chọn model đó?

- **LR**: baseline, dễ giải thích, nhanh. **RF / XGBoost**: hiệu năng tốt trên tabular, imbalance (class_weight/scale_pos_weight), hỗ trợ SHAP TreeExplainer. **LightGBM**: tốc độ và bộ nhớ phù hợp dataset lớn. Lựa chọn phù hợp bài toán **imbalanced classification** và **explainability**.

### Có so sánh nhiều model không?

- **Có**: Stratified cross-validation so sánh LogReg, RF, XGBoost (và LightGBM nếu có); kết quả ghi vào **results/model_benchmark.csv** (ROC-AUC, PR-AUC, recall@FPR, precision/recall, cost threshold, …).

### Training offline hay online learning?

- **Offline**: Training chạy batch (full hoặc sampled dataset); **không có** cập nhật mô hình online (incremental/streaming learning).

### Xử lý class imbalance bằng cách nào?

- **Class weight**: Logistic Regression và Random Forest dùng **class_weight="balanced"** (hoặc balanced_subsample); XGBoost dùng **scale_pos_weight** (tỉ lệ neg/pos). **Cost-sensitive threshold**: tìm ngưỡng tối ưu theo cost_fn/cost_fp. **Không** dùng SMOTE/undersampling trong pipeline benchmark chính (train.py); SMOTE chỉ xuất hiện trong legacy preprocessing (data_preprocessing, preprocessing).

### Có dùng SMOTE / undersampling / cost-sensitive learning không?

- **SMOTE/undersampling**: Có trong code legacy (imblearn); **không** dùng trong pipeline train chính. **Cost-sensitive learning**: **Có** — cost_sensitive_threshold, find_optimal_threshold, cấu hình cost_fn/cost_fp; threshold có thể load từ config cho API.

### Metric chính để đánh giá model?

- **PR-AUC (Average Precision)**, **ROC-AUC**, **Recall tại FPR cố định** (1%, 0.1%), **precision/recall** tại threshold, **cost-sensitive expected cost** và threshold tối ưu. F1 và confusion matrix cũng có trong thư viện metrics.

### Recall của fraud class / False Positive Rate?

- **Phụ thuộc run**: Ví dụ benchmark (README) với 30k rows, 3-fold: Recall ~0.94–0.97, Recall@FPR=1% ~0.97–0.98, Recall@FPR=0.1% ~0.91–0.97. FPR không báo trực tiếp trong một con số; được kiểm soát qua **recall@FPR** và **cost-sensitive threshold**.

### Có dùng model explainability không?

- **Có**: **SHAP** (TreeExplainer) trong `src/explainability/shap_explainer.py` — trả **top_features_contributing** và **narrative** (mô tả ngắn). API `POST /predict` trả kèm explanation; dashboard Streamlit hiển thị SHAP theo từng giao dịch.

---

## 5. Real-time Detection System

### Hệ thống fraud detection được deploy như thế nào?

- **API**: FastAPI (`uvicorn src.api.main:app`); **Dashboard**: Streamlit (`streamlit run app/streamlit_app.py`). Có thể chạy local hoặc deploy lên server/cloud; **không** mô tả container/K8s trong repo. Có tham chiếu **Streamlit Cloud** trong README.

### Transaction data đi qua pipeline gì?

- **Real-time**: Client gửi **POST /predict** với payload features → validate schema → **score** (model + scaler từ registry) → **explain_transaction** (SHAP) → trả probability, risk_score, prediction, explanation; **ghi latency** vào api_perf.jsonl.
- **Batch**: CSV → batch_scoring (load model, đọc từng phần hoặc toàn bộ file, score, ghi ra CSV). **Không** có Kafka/queue trong luồng chính; pipeline có thư mục `pipeline/` (Kafka demo) nhưng không tích hợp bắt buộc.

### Latency của model inference?

- **Không cố định trong code**; đo và ghi **latency_ms** từng request. Tổng hợp qua script performance_monitoring (p50, p95, p99). Thực tế thường **vài chục đến vài trăm ms** cho một request (model + SHAP), tùy phần cứng.

### Hệ thống có scalable architecture không?

- **Ở mức ứng dụng**: API stateless, có thể chạy nhiều instance phía sau load balancer. **Không** có thiết kế distributed training hay distributed inference (Spark, Ray) trong repo; scaling chủ yếu horizontal (nhiều replica API).

### Có fallback rule-based nếu model fail không?

- **Không**: Không có fallback rule-based khi model lỗi; API trả exception cho client. Có **legacy endpoints** (/score, /score/features) tương thích cũ, không phải rule-based fallback.

### Có hybrid system (rule + ML) không?

- **Không**: Quyết định hoàn toàn từ **ML model** (probability + threshold). Threshold có thể cấu hình (cost-sensitive hoặc config); không có rule nghiệp vụ (ví dụ “chặn nếu Amount &gt; X và country khác”) trong code.

---

## 6. Security & Fraud Prevention

### Hệ thống có phát hiện account takeover fraud không?

- **Không chuyên biệt**: Không có module ATO; có thể **phản ánh gián tiếp** qua behavioral/temporal features (velocity, new merchant, spending spike) nếu hành vi bất thường tương ứng với nhãn fraud trong dữ liệu. Không có định danh account thật (chỉ synthetic ID).

### Có phát hiện card-not-present fraud (online payment) không?

- **Có thể**: Các dataset PaySim/onlinefraud mô phỏng giao dịch online; pipeline train và score trên chung schema. **Không** có nhãn hay logic riêng cho “CNP”; fraud được hiểu chung là Class=1.

### Có cơ chế device fingerprinting không?

- **Không**: Không có device ID hay fingerprint trong schema hay feature.

### Data có được encrypt hoặc tokenization không?

- **Không**: Repo không triển khai encryption at rest/transit hay tokenization cho dữ liệu; phù hợp môi trường demo/research. Production cần bổ sung theo PCI DSS nếu xử lý dữ liệu thẻ thật.

### Có hệ thống audit log / monitoring cho fraud detection không?

- **Monitoring có**: **api_perf.jsonl** (latency, probability, prediction mỗi request); script tổng hợp latency và alert rate. **Drift**: `drift_detection.py` — PSI, KS, fraud rate; ghi **drift_report.json**. **Experiment tracking**: run metadata, params, metrics, artifacts. **Không** có audit log nghiệp vụ (ai xem gì, ai duyệt case) trong repo.

### Hệ thống có cơ chế model drift detection khi behavior thay đổi không?

- **Có**: `src/monitoring/drift_detection.py` — so sánh **baseline** vs **current** (CSV hoặc DataFrame): **PSI** theo feature, **KS test** (p-value) nếu có SciPy, **fraud rate**; output JSON. Có thể chạy định kỳ (cron/script) để phát hiện drift; **không** có trigger tự động retrain hay alert trong code.

---

*Báo cáo dựa trên code và tài liệu trong repository tại thời điểm viết; không thêm tính năng hay dữ liệu không tồn tại trong repo.*
