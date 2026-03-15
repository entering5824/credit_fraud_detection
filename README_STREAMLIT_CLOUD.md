# Deploy lên Streamlit Cloud

1. Đẩy repo (hoặc folder `credit_fraud_detection`) lên GitHub.
2. Vào [share.streamlit.io](https://share.streamlit.io) → New app.
3. **Repository**: `your-username/your-repo`
4. **Branch**: `main`
5. **Main file path**: `app/streamlit_app.py`
6. **Root directory**: (để trống nếu repo root là credit_fraud_detection; nếu repo chứa nhiều project thì điền `credit_fraud_detection`)
7. Deploy. Lần đầu cần train model và commit thư mục `models/` + `data/creditcard.csv` (hoặc dùng sample data nhỏ).

Chạy local: `streamlit run app/streamlit_app.py` (từ thư mục credit_fraud_detection).
