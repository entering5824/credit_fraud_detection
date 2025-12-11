"""
Ứng dụng Phát hiện Giao dịch Gian lận Thẻ Tín dụng

Ứng dụng này cho phép người dùng nhập thông tin giao dịch mới
và dự đoán xem giao dịch đó có phải là gian lận hay không.
rgvNKML 
Cách sử dụng:qrgvNKML 
1. Chạy file này: python app/fraud_detection_app.py
2. Nhập thông tin giao dịch khi được yêu cầu
3. Ứng dụng sẽ hiển thị kết quả dự đoán
"""

import sys
from pathlib import Path
import pickle
import pandas as pd
import numpy as np

# Thêm thư mục gốc của project vào đường dẫn để có thể import các module
# Điều này giúp Python tìm được các file trong thư mục src/
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import các module cần thiết từ thư mục src/
from src.data_preprocessing import scale_features
from src.models import load_model


def load_scaler_and_models():
    """
    Hàm này tải scaler và các models đã được train sẵn
    
    Scaler: Dùng để chuẩn hóa dữ liệu mới (giống như khi train)
    Models: Các mô hình đã được train để dự đoán
    
    Returns:
        scaler: Scaler đã được train
        models: Dictionary chứa các models (Random Forest, AdaBoost, XGBoost)
    """
    models_dir = project_root / 'models'
    
    # Tải scaler - dùng để chuẩn hóa dữ liệu
    scaler_path = models_dir / 'scaler.pkl'
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy scaler tại {scaler_path}. "
            "Vui lòng chạy notebook 01 để tạo scaler."
        )
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    print("✅ Đã tải scaler thành công")
    
    # Tải các models
    models = {}
    model_names = ['Random Forest', 'AdaBoost', 'XGBoost']
    
    for model_name in model_names:
        try:
            model = load_model(model_name, model_dir=models_dir)
            models[model_name] = model
            print(f"✅ Đã tải model: {model_name}")
        except FileNotFoundError:
            print(f"⚠️  Không tìm thấy model: {model_name}")
    
    if not models:
        raise FileNotFoundError(
            "Không tìm thấy model nào. Vui lòng chạy notebook 02 để train models."
        )
    
    return scaler, models


def get_transaction_input():
    """
    Hàm này yêu cầu người dùng nhập thông tin giao dịch
    
    Dataset có 29 features:
    - V1 đến V28: Các features đã được PCA transform (không thể giải thích trực tiếp)
    - Amount: Số tiền giao dịch
    
    Returns:
        transaction_data: Dictionary chứa thông tin giao dịch
    """
    print("\n" + "="*60)
    print("NHẬP THÔNG TIN GIAO DỊCH")
    print("="*60)
    print("\nLưu ý: V1-V28 là các features đã được mã hóa (PCA),")
    print("       bạn có thể nhập giá trị ngẫu nhiên hoặc từ dataset thật.")
    print()
    
    transaction = {}
    
    # Nhập các features V1-V28
    # Trong thực tế, các giá trị này thường nằm trong khoảng -5 đến 5
    print("Nhập các giá trị V1-V28 (có thể để trống để dùng giá trị mặc định):")
    for i in range(1, 29):
        while True:
            try:
                value = input(f"V{i} (mặc định: 0.0): ").strip()
                if value == "":
                    transaction[f'V{i}'] = 0.0
                else:
                    transaction[f'V{i}'] = float(value)
                break
            except ValueError:
                print("⚠️  Vui lòng nhập số hợp lệ!")
    
    # Nhập Amount (số tiền giao dịch)
    while True:
        try:
            amount = input("\nNhập số tiền giao dịch (Amount): ").strip()
            if amount == "":
                print("⚠️  Số tiền không được để trống!")
                continue
            transaction['Amount'] = float(amount)
            if transaction['Amount'] < 0:
                print("⚠️  Số tiền phải >= 0!")
                continue
            break
        except ValueError:
            print("⚠️  Vui lòng nhập số hợp lệ!")
    
    return transaction


def predict_transaction(scaler, models, transaction_data):
    """
    Hàm này dự đoán xem giao dịch có phải gian lận hay không
    
    Quy trình:
    1. Chuyển dữ liệu thành DataFrame (bảng dữ liệu)
    2. Chuẩn hóa dữ liệu bằng scaler (giống như khi train)
    3. Dự đoán bằng từng model
    4. Tổng hợp kết quả
    
    Parameters:
        scaler: Scaler để chuẩn hóa dữ liệu
        models: Dictionary chứa các models
        transaction_data: Dictionary chứa thông tin giao dịch
    
    Returns:
        predictions: Dictionary chứa kết quả dự đoán từ mỗi model
        probabilities: Dictionary chứa xác suất gian lận từ mỗi model
    """
    # Chuyển dữ liệu thành DataFrame (bảng dữ liệu) để xử lý
    # DataFrame giống như bảng Excel, có cột và hàng
    df = pd.DataFrame([transaction_data])
    
    # Lấy danh sách tên các cột (features) cần chuẩn hóa
    feature_cols = [f'V{i}' for i in range(1, 29)] + ['Amount']
    
    # Chuẩn hóa dữ liệu - bước quan trọng!
    # Dữ liệu mới phải được chuẩn hóa giống như dữ liệu train
    # Nếu không, model sẽ cho kết quả sai
    X_scaled, _ = scale_features(df, feature_cols=feature_cols, scaler=scaler, fit=False)
    
    # Dự đoán bằng từng model
    predictions = {}
    probabilities = {}
    
    for model_name, model in models.items():
        # Dự đoán nhãn (0 = Normal, 1 = Fraud)
        pred = model.predict(X_scaled)[0]
        predictions[model_name] = pred
        
        # Dự đoán xác suất (probability)
        # proba[0] = xác suất Normal, proba[1] = xác suất Fraud
        proba = model.predict_proba(X_scaled)[0]
        probabilities[model_name] = proba[1]  # Lấy xác suất Fraud
    
    return predictions, probabilities


def display_results(transaction_data, predictions, probabilities):
    """
    Hàm này hiển thị kết quả dự đoán một cách dễ hiểu
    
    Parameters:
        transaction_data: Thông tin giao dịch
        predictions: Kết quả dự đoán (0 hoặc 1)
        probabilities: Xác suất gian lận (0.0 đến 1.0)
    """
    print("\n" + "="*60)
    print("KẾT QUẢ DỰ ĐOÁN")
    print("="*60)
    
    print(f"\n📊 Thông tin giao dịch:")
    print(f"   Số tiền: ${transaction_data['Amount']:,.2f}")
    
    print(f"\n🤖 Kết quả từ các models:")
    print("-" * 60)
    
    # Đếm số model dự đoán là gian lận
    fraud_count = sum(1 for pred in predictions.values() if pred == 1)
    total_models = len(predictions)
    
    for model_name in predictions.keys():
        pred = predictions[model_name]
        proba = probabilities[model_name]
        
        # Chuyển đổi kết quả thành text dễ hiểu
        result_text = "🚨 GIAN LẬN" if pred == 1 else "✅ BÌNH THƯỜNG"
        proba_percent = proba * 100
        
        print(f"\n{model_name}:")
        print(f"   Kết quả: {result_text}")
        print(f"   Xác suất gian lận: {proba_percent:.2f}%")
    
    print("\n" + "-" * 60)
    
    # Tổng hợp kết quả
    avg_proba = np.mean(list(probabilities.values()))
    consensus = "Gian lận" if fraud_count > total_models / 2 else "Bình thường"
    
    print(f"\n📈 Tổng hợp:")
    print(f"   {fraud_count}/{total_models} models dự đoán là gian lận")
    print(f"   Xác suất trung bình: {avg_proba*100:.2f}%")
    print(f"   Kết luận chung: {consensus}")
    
    # Cảnh báo nếu có model dự đoán gian lận
    if fraud_count > 0:
        print(f"\n⚠️  CẢNH BÁO: Có {fraud_count} model(s) phát hiện giao dịch có thể là gian lận!")
        print("   Vui lòng kiểm tra kỹ giao dịch này.")
    else:
        print(f"\n✅ Giao dịch có vẻ bình thường.")
    
    print("="*60)


def main():
    """
    Hàm chính của ứng dụng
    
    Quy trình:
    1. Tải scaler và models
    2. Nhập thông tin giao dịch
    3. Dự đoán
    4. Hiển thị kết quả
    """
    print("="*60)
    print("ỨNG DỤNG PHÁT HIỆN GIAO DỊCH GIAN LẬN THẺ TÍN DỤNG")
    print("="*60)
    print("\nỨng dụng này sử dụng các mô hình Machine Learning")
    print("để phát hiện giao dịch gian lận.")
    print("\nCác models được sử dụng:")
    print("  - Random Forest")
    print("  - AdaBoost")
    print("  - XGBoost")
    
    try:
        # Bước 1: Tải scaler và models
        print("\n🔄 Đang tải models...")
        scaler, models = load_scaler_and_models()
        
        # Bước 2: Nhập thông tin giao dịch
        transaction_data = get_transaction_input()
        
        # Bước 3: Dự đoán
        print("\n🔄 Đang dự đoán...")
        predictions, probabilities = predict_transaction(scaler, models, transaction_data)
        
        # Bước 4: Hiển thị kết quả
        display_results(transaction_data, predictions, probabilities)
        
        # Hỏi người dùng có muốn tiếp tục không
        while True:
            continue_choice = input("\n\nBạn có muốn kiểm tra giao dịch khác? (y/n): ").strip().lower()
            if continue_choice in ['y', 'yes', 'có', 'c']:
                transaction_data = get_transaction_input()
                predictions, probabilities = predict_transaction(scaler, models, transaction_data)
                display_results(transaction_data, predictions, probabilities)
            elif continue_choice in ['n', 'no', 'không', 'k']:
                print("\n👋 Cảm ơn bạn đã sử dụng ứng dụng!")
                break
            else:
                print("⚠️  Vui lòng nhập 'y' hoặc 'n'")
    
    except FileNotFoundError as e:
        print(f"\n❌ Lỗi: {e}")
        print("\n💡 Hướng dẫn:")
        print("   1. Chạy notebook 01_EDA_and_Preprocessing.ipynb để tạo scaler")
        print("   2. Chạy notebook 02_Model_Training.ipynb để train models")
        print("   3. Sau đó chạy lại ứng dụng này")
    
    except Exception as e:
        print(f"\n❌ Đã xảy ra lỗi: {e}")
        print("   Vui lòng kiểm tra lại các bước trên.")


# Chạy ứng dụng khi file được execute trực tiếp
if __name__ == "__main__":
    main()

