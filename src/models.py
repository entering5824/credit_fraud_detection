"""
Module Training Models cho Phát hiện Gian lận Thẻ Tín dụng

Module này cung cấp các hàm để train (huấn luyện) các mô hình Ensemble Learning:
- Random Forest: Rừng ngẫu nhiên - tạo nhiều cây quyết định
- AdaBoost: Adaptive Boosting - kết hợp nhiều models yếu
- XGBoost: Extreme Gradient Boosting - gradient boosting tối ưu

Giải thích đơn giản:
- Train model = dạy máy tính học từ dữ liệu để nhận biết gian lận
- Ensemble = kết hợp nhiều models để có kết quả tốt hơn
- Lưu model = lưu lại model đã train để dùng sau (không cần train lại)
"""

import pickle
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn import __version__ as sklearn_version


def train_random_forest(X_train, y_train, n_estimators=200, max_depth=15, 
                        class_weight=None, random_state=42, n_jobs=-1, verbose=1):
    """
    Train (huấn luyện) mô hình Random Forest
    
    Random Forest là gì?
    - Tạo nhiều cây quyết định (decision trees)
    - Mỗi cây đưa ra một dự đoán
    - Lấy kết quả trung bình hoặc đa số từ tất cả cây
    - Giống như hỏi nhiều chuyên gia và lấy ý kiến đa số
    
    Tại sao dùng Random Forest?
    - Mạnh mẽ, ít bị overfitting
    - Xử lý tốt dữ liệu mất cân bằng (với class_weight)
    - Dễ hiểu và giải thích
    
    Parameters:
    -----------
    X_train : array-like
        Dữ liệu để train (features - các đặc trưng)
    y_train : array-like
        Nhãn (labels - 0 = bình thường, 1 = gian lận)
    n_estimators : int, mặc định=200
        Số cây trong rừng (nhiều hơn = tốt hơn nhưng chậm hơn)
    max_depth : int, mặc định=15
        Độ sâu tối đa của mỗi cây (sâu hơn = phức tạp hơn)
    class_weight : dict hoặc 'balanced', tùy chọn
        Trọng số cho các lớp (để xử lý mất cân bằng dữ liệu)
    random_state : int, mặc định=42
        Số ngẫu nhiên để đảm bảo kết quả giống nhau mỗi lần chạy
    n_jobs : int, mặc định=-1
        Số CPU cores để sử dụng (-1 = dùng tất cả)
    verbose : int, mặc định=1
        Mức độ hiển thị thông tin (0 = im lặng, 1 = hiển thị)
    
    Returns:
    --------
    model : RandomForestClassifier
        Model Random Forest đã được train (sẵn sàng để dự đoán)
    """
    if verbose >= 1:
        print("🔹 Training Random Forest model...")
        print(f"   Parameters: n_estimators={n_estimators}, max_depth={max_depth}")
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs
    )
    
    model.fit(X_train, y_train)
    
    if verbose >= 1:
        print("✅ Random Forest training completed!")
    
    return model


def train_adaboost(X_train, y_train, n_estimators=50, learning_rate=1.0,
                   base_estimator=None, random_state=42, verbose=1):
    """
    Train (huấn luyện) mô hình AdaBoost
    
    AdaBoost là gì?
    - Adaptive Boosting - Tăng cường thích ứng
    - Train nhiều models yếu (weak learners)
    - Models sau học từ lỗi của models trước
    - Kết hợp tất cả models để có kết quả tốt
    - Giống như học từ sai lầm và cải thiện dần
    
    Tại sao dùng AdaBoost?
    - Hiệu quả với dữ liệu phức tạp
    - Tự động điều chỉnh trọng số cho các mẫu khó
    - Thường cho kết quả tốt
    
    Parameters:
    -----------
    X_train : array-like
        Dữ liệu để train (features)
    y_train : array-like
        Nhãn (0 = bình thường, 1 = gian lận)
    n_estimators : int, mặc định=50
        Số models yếu (nhiều hơn = tốt hơn nhưng chậm hơn)
    learning_rate : float, mặc định=1.0
        Tốc độ học (nhỏ hơn = học chậm hơn nhưng ổn định hơn)
    base_estimator : object, tùy chọn
        Model cơ sở (mặc định: Decision Tree đơn giản)
    random_state : int, mặc định=42
        Số ngẫu nhiên để đảm bảo kết quả giống nhau
    verbose : int, mặc định=1
        Mức độ hiển thị thông tin
    
    Returns:
    --------
    model : AdaBoostClassifier
        Model AdaBoost đã được train
    """

    if verbose >= 1:
        print("🔹 Training AdaBoost model...")
        print(f"   Parameters: n_estimators={n_estimators}, learning_rate={learning_rate}")

    # Default weak learner
    if base_estimator is None:
        base_estimator = DecisionTreeClassifier(max_depth=1, random_state=random_state)

    # 🔥 Sửa lỗi: chỉ dùng 'estimator=' (bản đúng của sklearn mới)
    model = AdaBoostClassifier(
        estimator=base_estimator,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state
    )

    model.fit(X_train, y_train)

    if verbose >= 1:
        print("✅ AdaBoost training completed!")
    
    return model



def train_xgboost(X_train, y_train, n_estimators=100, max_depth=6, 
                  learning_rate=0.1, scale_pos_weight=None, random_state=42, 
                  verbose=1, use_label_encoder=False):
    """
    Train (huấn luyện) mô hình XGBoost
    
    XGBoost là gì?
    - Extreme Gradient Boosting
    - Gradient Boosting được tối ưu hóa rất mạnh
    - Train nhiều cây quyết định theo thứ tự
    - Cây sau sửa lỗi của cây trước
    - Rất nhanh và mạnh, thường cho kết quả tốt nhất
    
    Tại sao dùng XGBoost?
    - Thường cho kết quả tốt nhất trong các competitions
    - Xử lý tốt dữ liệu mất cân bằng (với scale_pos_weight)
    - Nhanh và hiệu quả
    
    Parameters:
    -----------
    X_train : array-like
        Dữ liệu để train (features)
    y_train : array-like
        Nhãn (0 = bình thường, 1 = gian lận)
    n_estimators : int, mặc định=100
        Số cây (boosting rounds) - nhiều hơn = tốt hơn
    max_depth : int, mặc định=6
        Độ sâu tối đa của mỗi cây
    learning_rate : float, mặc định=0.1
        Tốc độ học (nhỏ hơn = học chậm hơn nhưng ổn định hơn)
    scale_pos_weight : float, tùy chọn
        Trọng số cho lớp gian lận (để xử lý mất cân bằng)
        Ví dụ: nếu có 100 bình thường và 1 gian lận, scale_pos_weight = 100
    random_state : int, mặc định=42
        Số ngẫu nhiên để đảm bảo kết quả giống nhau
    verbose : int, mặc định=1
        Mức độ hiển thị thông tin
    use_label_encoder : bool, mặc định=False
        Có dùng label encoder không (không dùng trong phiên bản mới)
    
    Returns:
    --------
    model : XGBClassifier
        Model XGBoost đã được train
    """
    if verbose >= 1:
        print("🔹 Training XGBoost model...")
        print(f"   Parameters: n_estimators={n_estimators}, max_depth={max_depth}, learning_rate={learning_rate}")
    
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        use_label_encoder=use_label_encoder,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    if verbose >= 1:
        print("✅ XGBoost training completed!")
    
    return model


def save_model(model, model_name, save_dir='models'):
    """
    Save trained model to pickle file.
    
    Parameters:
    -----------
    model : object
        Trained model to save
    model_name : str
        Name of the model (will be used as filename)
    save_dir : str or Path, default='models'
        Directory to save the model
    
    Returns:
    --------
    save_path : Path
        Path where model was saved
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean model name for filename
    filename = model_name.lower().replace(' ', '_') + '.pkl'
    save_path = save_dir / filename
    
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"✅ Model saved to {save_path}")
    return save_path


def load_model(model_name, model_dir='models'):
    """
    Load trained model from pickle file.
    
    Parameters:
    -----------
    model_name : str
        Name of the model (filename without .pkl)
    model_dir : str or Path, default='models'
        Directory containing the model
    
    Returns:
    --------
    model : object
        Loaded model
    """
    model_dir = Path(model_dir)
    filename = model_name.lower().replace(' ', '_') + '.pkl'
    model_path = model_dir / filename
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"✅ Model loaded from {model_path}")
    return model


def evaluate_model_performance(model, X_test, y_test, model_name='Model'):
    """
    Evaluate model and print classification report.
    
    Parameters:
    -----------
    model : object
        Trained model with predict() method
    X_test : array-like
        Test features
    y_test : array-like
        Test labels
    model_name : str, default='Model'
        Name of the model for display
    """
    print(f"\n{'='*60}")
    print(f"Evaluation Results for {model_name}")
    print(f"{'='*60}")
    
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    return y_pred
