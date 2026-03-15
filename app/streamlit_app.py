"""
Credit Card Fraud Detection - Streamlit app for Streamlit Cloud.
Dự đoán gian lận + Explainability (SHAP). Chạy: streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path
import pickle
import pandas as pd
import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
from src.data_preprocessing import scale_features
from src.models import load_model
from src.cost_sensitive import load_cost_config
from src.explain import global_importance, local_shap_values, FEATURE_COLS

st.set_page_config(page_title="Fraud Detection", page_icon="💳", layout="wide")
st.title("💳 Phát hiện gian lận thẻ tín dụng")

tab1, tab2, tab3 = st.tabs(["🔍 Dự đoán", "📊 Giải thích (SHAP)", "ℹ️ Hướng dẫn"])

with tab1:
    @st.cache_resource
    def load_models():
        models_dir = project_root / "models"
        scaler_path = models_dir / "scaler.pkl"
        if not scaler_path.exists():
            return None, None, None
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        models = {}
        for name in ["Random Forest", "AdaBoost", "XGBoost"]:
            try:
                models[name] = load_model(name, model_dir=models_dir)
            except FileNotFoundError:
                pass
        cfg = load_cost_config()
        threshold = cfg.get("optimal_threshold", 0.5)
        return scaler, models, threshold

    scaler, models, threshold = load_models()
    if scaler is None or not models:
        st.warning("Chưa có model. Chạy notebook 01 & 02 để train và lưu scaler/models vào thư mục `models/`.")
    else:
        st.caption(f"Ngưỡng cost-sensitive: {threshold:.4f}")
        with st.form("transaction_form"):
            amount = st.number_input("Số tiền (Amount)", min_value=0.0, value=100.0, step=1.0)
            st.markdown("**V1–V28** (có thể để mặc định 0):")
            cols = st.columns(4)
            v_vals = {}
            for i in range(1, 29):
                v_vals[f"V{i}"] = cols[(i - 1) % 4].number_input(f"V{i}", value=0.0, key=f"v{i}")
            submitted = st.form_submit_button("Dự đoán")
            if submitted:
                transaction = {**v_vals, "Amount": amount}
                df = pd.DataFrame([transaction])
                X_scaled, _ = scale_features(df, feature_cols=FEATURE_COLS, scaler=scaler, fit=False)
                preds, probs = {}, {}
                for name, model in models.items():
                    proba = model.predict_proba(X_scaled)[0][1]
                    probs[name] = proba
                    preds[name] = 1 if proba >= threshold else 0
                st.subheader("Kết quả")
                st.metric("Số tiền", f"${amount:,.2f}")
                avg = np.mean(list(probs.values()))
                fraud_count = sum(preds.values())
                st.metric("Xác suất gian lận (TB)", f"{avg*100:.1f}%")
                st.metric("Kết luận", "🚨 GIAN LẬN" if fraud_count > len(models) / 2 else "✅ Bình thường")
                for name in preds:
                    st.write(f"**{name}**: {probs[name]*100:.1f}% → {'Gian lận' if preds[name] else 'Bình thường'}")

with tab2:
    try:
        import pandas as pd
        from src.data_preprocessing import split_data
        df = pd.read_csv(project_root / "data" / "creditcard.csv")
        X = df[FEATURE_COLS]
        y = df["Class"]
        X_scaled, _ = scale_features(X, feature_cols=FEATURE_COLS, fit=True)
        _, _, X_test, _, _, y_test = split_data(X_scaled, y, test_size=0.15, val_size=0.15, random_state=42)
        with open(project_root / "models" / "xgboost.pkl", "rb") as f:
            xgb = pickle.load(f)
        X_test_np = X_test.values if hasattr(X_test, "values") else X_test
        y_test_np = y_test.values if hasattr(y_test, "values") else y_test
        st.header("Global feature importance")
        names, imp = global_importance(xgb, X_test_np[:500])
        imp_df = pd.DataFrame({"feature": names, "importance": imp}).sort_values("importance", ascending=False)
        st.bar_chart(imp_df.set_index("feature")["importance"])
        st.header("Local SHAP")
        idx = st.slider("Mẫu (test set)", 0, min(100, len(X_test_np) - 1), 0)
        sample = X_test_np[idx : idx + 1]
        shap_vals = local_shap_values(sample, model=xgb)[0]
        shap_df = pd.DataFrame({"feature": FEATURE_COLS, "SHAP": shap_vals}).sort_values("SHAP", key=abs, ascending=False)
        st.bar_chart(shap_df.set_index("feature")["SHAP"])
        st.caption(f"Nhãn thật: {'Gian lận' if y_test_np[idx] == 1 else 'Bình thường'}")
    except Exception as e:
        st.error(f"Không tải được dữ liệu/model: {e}")

with tab3:
    st.markdown("""
- **Dự đoán**: Nhập số tiền và tùy chọn V1–V28 (mặc định 0). Model dùng ngưỡng cost-sensitive.
- **SHAP**: Xem feature importance và lý do từng giao dịch được dự đoán.
- Train model: chạy notebook `01_EDA_and_Preprocessing.ipynb` và `02_Model_Training.ipynb`, lưu scaler và model vào `models/`.
    """)
