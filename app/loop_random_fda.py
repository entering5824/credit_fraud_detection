import sys
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import random

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import modules
from src.data_preprocessing import scale_features
from src.models import load_model

# ==============================
#  LOAD SCALER & MODELS
# ==============================
def load_scaler_and_models():
    models_dir = project_root / 'models'

    # Load scaler
    scaler_path = models_dir / 'scaler.pkl'
    if not scaler_path.exists():
        raise FileNotFoundError(f"Không tìm thấy scaler tại {scaler_path}.")

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    print("✅ Đã tải scaler thành công")

    # Load models
    models = {}
    model_names = ['Random Forest', 'AdaBoost', 'XGBoost']

    for model_name in model_names:
        try:
            model = load_model(model_name, model_dir=models_dir)
            models[model_name] = model
            print(f"✅ Đã tải model: {model_name}")
        except FileNotFoundError:
            print(f"⚠️ Không tìm thấy model: {model_name}")

    if not models:
        raise FileNotFoundError("Không tìm thấy bất kỳ model nào!")

    return scaler, models


# ==============================
#  METHOD 1 – NORMAL TRANSACTION
# ==============================
def generate_normal_transaction():
    """
    Tạo giao dịch bình thường giống dữ liệu thật (V1–V28 nhỏ, amount vừa).
    """
    print("\n📌 Tạo giao dịch bình thường...")

    transaction = {f"V{i}": round(random.uniform(-2, 2), 4) for i in range(1, 29)}
    transaction["Amount"] = round(random.uniform(1, 1000), 2)

    return transaction


# ==============================
#  METHOD 2 – RANDOM FRAUD (GIẢ LẬP)
# ==============================
def generate_random_fraud_transaction():
    """
    Giả lập giao dịch gian lận bằng cách random giá trị cực lớn.
    """
    print("\n🚨 TẠO GIAO DỊCH GIẢ LẬP GIAN LẬN!")

    transaction = {f"V{i}": round(random.uniform(-10, 10), 4) for i in range(1, 29)}
    transaction["Amount"] = round(random.uniform(2000, 5000), 2)

    return transaction


# ==============================
#  METHOD 3 – REAL FRAUD (TỪ DATASET)
# ==============================
def generate_realistic_fraud_transaction():
    """
    Lấy một giao dịch FRAUD thật từ dataset creditcard.csv.
    """
    csv_path = project_root / "data" / "creditcard.csv"
    if not csv_path.exists():
        raise FileNotFoundError("Không tìm thấy file creditcard.csv!")

    df = pd.read_csv(csv_path)
    fraud_df = df[df["Class"] == 1]

    if len(fraud_df) == 0:
        raise ValueError("Dataset không có giao dịch gian lận!")

    row = fraud_df.sample(1).iloc[0]

    transaction = {f"V{i}": float(row[f"V{i}"]) for i in range(1, 29)}
    transaction["Amount"] = float(row["Amount"])

    print("\n🔥 Đã lấy 1 giao dịch FRAUD thật từ dataset!")

    return transaction


# ==============================
#  PREDICT
# ==============================
def predict_transaction(scaler, models, transaction_data):

    df = pd.DataFrame([transaction_data])
    feature_cols = [f'V{i}' for i in range(1, 29)] + ['Amount']

    X_scaled, _ = scale_features(df, feature_cols=feature_cols, scaler=scaler, fit=False)

    predictions = {}
    probabilities = {}

    for model_name, model in models.items():
        pred = model.predict(X_scaled)[0]
        predictions[model_name] = pred

        proba = model.predict_proba(X_scaled)[0]
        probabilities[model_name] = proba[1]

    return predictions, probabilities


# ==============================
#  DISPLAY RESULTS
# ==============================
def display_results(transaction_data, predictions, probabilities):

    print("\n" + "=" * 60)
    print("KẾT QUẢ DỰ ĐOÁN")
    print("=" * 60)

    print(f"\n📊 Thông tin giao dịch:")
    print(f"   Số tiền: ${transaction_data['Amount']:,.2f}")

    print(f"\n🤖 Kết quả từ các models:")
    print("-" * 60)

    fraud_count = sum(1 for p in predictions.values() if p == 1)
    total_models = len(predictions)

    for model_name in predictions.keys():
        pred = predictions[model_name]
        proba = probabilities[model_name] * 100

        result_text = "🚨 GIAN LẬN" if pred == 1 else "✅ BÌNH THƯỜNG"

        print(f"\n{model_name}:")
        print(f"   Kết quả: {result_text}")
        print(f"   Xác suất gian lận: {proba:.2f}%")

    print("\n" + "-" * 60)

    avg_proba = np.mean(list(probabilities.values()))
    consensus = "Gian lận" if fraud_count > total_models / 2 else "Bình thường"

    print(f"\n📈 Tổng hợp:")
    print(f"   {fraud_count}/{total_models} models dự đoán gian lận")
    print(f"   Xác suất trung bình: {avg_proba * 100:.2f}%")
    print(f"   Kết luận chung: {consensus}")

    if fraud_count > 0:
        print("\n⚠️  CẢNH BÁO: Có dấu hiệu gian lận!")
    else:
        print("\n✅ Giao dịch có vẻ bình thường.")

    print("=" * 60)


# ==============================
#  MAIN APP
# ==============================
def main():
    print("=" * 60)
    print("AUTO TEST – PHÁT HIỆN GIAO DỊCH GIAN LẬN")
    print("=" * 60)

    try:
        print("\n🔄 Đang tải models...")
        scaler, models = load_scaler_and_models()

        print("\n🚀 Chạy 3 loại giao dịch:")
        print("1) Normal")
        print("2) Fraud giả lập")
        print("3) Fraud thật từ dataset\n")

        # TEST #1 – Normal
        print("\n==================== TEST #1 ====================")
        t1 = generate_normal_transaction()
        preds, probs = predict_transaction(scaler, models, t1)
        display_results(t1, preds, probs)

        # TEST #2 – Random Fraud
        print("\n==================== TEST #2 ====================")
        t2 = generate_random_fraud_transaction()
        preds, probs = predict_transaction(scaler, models, t2)
        display_results(t2, preds, probs)

        # TEST #3 – Real Fraud
        print("\n==================== TEST #3 ====================")
        t3 = generate_realistic_fraud_transaction()
        preds, probs = predict_transaction(scaler, models, t3)
        display_results(t3, preds, probs)

        print("\n🎉 Hoàn tất AUTO TEST!")

    except Exception as e:
        print(f"\n❌ Lỗi xảy ra: {e}")


if __name__ == "__main__":
    main()
