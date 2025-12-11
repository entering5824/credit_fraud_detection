<<<<<<< HEAD
"""
Data preprocessing functions for credit card fraud detection.

This module provides reusable functions for scaling, splitting, and handling
class imbalance in the credit card fraud detection dataset.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def scale_features(X, feature_cols=None, scaler=None, fit=True):
    """
    Scale features using StandardScaler.
    
    Parameters:
    -----------
    X : pandas.DataFrame or numpy.ndarray
        Features to scale
    feature_cols : list, optional
        List of column names to scale. If None, scales all columns.
    scaler : StandardScaler, optional
        Pre-fitted scaler. If None, creates a new one.
    fit : bool, default=True
        Whether to fit the scaler (True) or only transform (False)
    
    Returns:
    --------
    X_scaled : pandas.DataFrame or numpy.ndarray
        Scaled features
    scaler : StandardScaler
        Fitted scaler object
    """
    if scaler is None:
        scaler = StandardScaler()
    
    if isinstance(X, pd.DataFrame):
        if feature_cols is None:
            feature_cols = X.columns.tolist()
        
        X_to_scale = X[feature_cols].copy()
        
        if fit:
            X_scaled_array = scaler.fit_transform(X_to_scale)
        else:
            X_scaled_array = scaler.transform(X_to_scale)
        
        X_scaled = X.copy()
        X_scaled[feature_cols] = X_scaled_array
        
        return X_scaled, scaler
    else:
        if fit:
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = scaler.transform(X)
        return X_scaled, scaler


def split_data(X, y, test_size=0.15, val_size=0.15, random_state=42):
    """
    Split data into train, validation, and test sets with stratification.
    
    Parameters:
    -----------
    X : pandas.DataFrame or numpy.ndarray
        Features
    y : pandas.Series or numpy.ndarray
        Target variable
    test_size : float, default=0.15
        Proportion of data for test set
    val_size : float, default=0.15
        Proportion of data for validation set (from remaining data after test split)
    random_state : int, default=42
        Random seed for reproducibility
    
    Returns:
    --------
    X_train, X_val, X_test, y_train, y_val, y_test : tuple
        Split datasets
    """
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        stratify=y, 
        random_state=random_state
    )
    
    # Second split: separate train and validation from remaining data
    # Adjust val_size to account for the fact that we're splitting from X_temp
    val_size_adjusted = val_size / (1 - test_size)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted,
        stratify=y_temp,
        random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def apply_smote(X, y, random_state=42, k_neighbors=5):
    """
    Apply SMOTE (Synthetic Minority Oversampling Technique) to balance classes.
    
    SMOTE creates synthetic samples for the minority class by interpolating
    between existing minority class samples.
    
    Advantages:
    - Increases minority class samples without exact duplicates
    - Can improve model performance on minority class
    - Works well with continuous features
    
    Disadvantages:
    - Can create noisy samples if minority class is too small
    - May overfit if used improperly
    - Computationally expensive for large datasets
    - Should only be applied to training data, not validation/test
    
    Parameters:
    -----------
    X : pandas.DataFrame or numpy.ndarray
        Features
    y : pandas.Series or numpy.ndarray
        Target variable
    random_state : int, default=42
        Random seed for reproducibility
    k_neighbors : int, default=5
        Number of nearest neighbors to use for SMOTE
    
    Returns:
    --------
    X_resampled : numpy.ndarray
        Resampled features
    y_resampled : numpy.ndarray
        Resampled target variable
    """
    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    
    if isinstance(X, pd.DataFrame):
        X_array = X.values
    else:
        X_array = X
    
    if isinstance(y, pd.Series):
        y_array = y.values
    else:
        y_array = y
    
    X_resampled, y_resampled = smote.fit_resample(X_array, y_array)
    
    return X_resampled, y_resampled


def get_class_weights(y):
    """
    Calculate class weights for imbalanced datasets.
    
    Class weights can be used with models that support the class_weight parameter
    (e.g., RandomForest, XGBoost, LogisticRegression) to give more importance
    to the minority class during training.
    
    Advantages:
    - No need to modify training data
    - Faster than oversampling
    - Works well with tree-based models
    - Preserves original data distribution
    
    Disadvantages:
    - May not work as well as SMOTE for some models
    - Less effective when imbalance is extreme
    - Requires model support for class_weight parameter
    
    Parameters:
    -----------
    y : pandas.Series or numpy.ndarray
        Target variable
    
    Returns:
    --------
    class_weights : dict
        Dictionary mapping class labels to their weights
    """
    if isinstance(y, pd.Series):
        y_array = y.values
    else:
        y_array = y
    
    unique_classes = np.unique(y_array)
    n_samples = len(y_array)
    n_classes = len(unique_classes)
    
    # Calculate weights inversely proportional to class frequency
    class_weights = {}
    for cls in unique_classes:
        n_class = np.sum(y_array == cls)
        weight = n_samples / (n_classes * n_class)
        class_weights[cls] = weight
    
    return class_weights



=======
"""
Module Xử lý Dữ liệu cho Phát hiện Gian lận Thẻ Tín dụng

Module này cung cấp các hàm để:
1. Chuẩn hóa dữ liệu (Scaling) - Đưa các số về cùng một thang đo
2. Chia dữ liệu (Train/Validation/Test split) - Chia dữ liệu thành các phần để train và test
3. Xử lý mất cân bằng dữ liệu (SMOTE, Class Weights) - Xử lý khi có quá ít dữ liệu gian lận

Giải thích đơn giản:
- Scaling: Giống như đổi đơn vị (ví dụ: từ cm sang m) để các số có cùng thang đo
- Train/Test split: Giống như chia bài tập thành phần học và phần thi
- SMOTE: Tạo thêm dữ liệu gian lận giả để cân bằng với dữ liệu bình thường
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def scale_features(X, feature_cols=None, scaler=None, fit=True):
    """
    Chuẩn hóa dữ liệu (Scaling) - Đưa các số về cùng một thang đo
    
    Tại sao cần chuẩn hóa?
    - Các features có đơn vị khác nhau (ví dụ: V1 có giá trị -5 đến 5, Amount có giá trị 0 đến 10000)
    - Model machine learning hoạt động tốt hơn khi tất cả số đều ở cùng một thang đo
    - Giống như đổi tất cả về cùng một đơn vị (ví dụ: tất cả về mét)
    
    Ví dụ:
    - Trước: V1 = 2.5, Amount = 1000
    - Sau: V1 = 0.5, Amount = 0.3 (cả hai đều nằm trong khoảng -3 đến 3)
    
    Parameters:
    -----------
    X : pandas.DataFrame hoặc numpy.ndarray
        Dữ liệu cần chuẩn hóa (bảng dữ liệu hoặc mảng số)
    feature_cols : list, tùy chọn
        Danh sách tên các cột cần chuẩn hóa. Nếu None, chuẩn hóa tất cả cột.
    scaler : StandardScaler, tùy chọn
        Scaler đã được train sẵn. Nếu None, tạo scaler mới.
    fit : bool, mặc định=True
        True = train scaler mới (dùng khi train model)
        False = chỉ áp dụng scaler đã train (dùng khi predict dữ liệu mới)
    
    Returns:
    --------
    X_scaled : pandas.DataFrame hoặc numpy.ndarray
        Dữ liệu đã được chuẩn hóa
    scaler : StandardScaler
        Scaler đã được train (dùng để lưu và sử dụng sau)
    """
    if scaler is None:
        scaler = StandardScaler()
    
    if isinstance(X, pd.DataFrame):
        if feature_cols is None:
            feature_cols = X.columns.tolist()
        
        X_to_scale = X[feature_cols].copy()
        
        if fit:
            X_scaled_array = scaler.fit_transform(X_to_scale)
        else:
            X_scaled_array = scaler.transform(X_to_scale)
        
        X_scaled = X.copy()
        X_scaled[feature_cols] = X_scaled_array
        
        return X_scaled, scaler
    else:
        if fit:
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = scaler.transform(X)
        return X_scaled, scaler


def split_data(X, y, test_size=0.15, val_size=0.15, random_state=42):
    """
    Chia dữ liệu thành 3 phần: Train, Validation, và Test
    
    Tại sao cần chia dữ liệu?
    - Train set (70%): Dùng để train model (giống như học bài)
    - Validation set (15%): Dùng để kiểm tra model trong quá trình train (giống như làm bài tập)
    - Test set (15%): Dùng để đánh giá cuối cùng (giống như thi cuối kỳ)
    
    Stratification: Đảm bảo tỉ lệ gian lận/bình thường giống nhau trong cả 3 phần
    
    Parameters:
    -----------
    X : pandas.DataFrame hoặc numpy.ndarray
        Dữ liệu đầu vào (features - các đặc trưng)
    y : pandas.Series hoặc numpy.ndarray
        Nhãn (labels - 0 = bình thường, 1 = gian lận)
    test_size : float, mặc định=0.15
        Tỉ lệ dữ liệu cho test set (15%)
    val_size : float, mặc định=0.15
        Tỉ lệ dữ liệu cho validation set (15% của phần còn lại sau khi lấy test)
    random_state : int, mặc định=42
        Số ngẫu nhiên để đảm bảo kết quả giống nhau mỗi lần chạy
    
    Returns:
    --------
    X_train, X_val, X_test, y_train, y_val, y_test : tuple
        Dữ liệu đã được chia thành 6 phần:
        - X_train, y_train: Dữ liệu để train
        - X_val, y_val: Dữ liệu để validation
        - X_test, y_test: Dữ liệu để test cuối cùng
    """
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        stratify=y, 
        random_state=random_state
    )
    
    # Second split: separate train and validation from remaining data
    # Adjust val_size to account for the fact that we're splitting from X_temp
    val_size_adjusted = val_size / (1 - test_size)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted,
        stratify=y_temp,
        random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def apply_smote(X, y, random_state=42, k_neighbors=5):
    """
    Áp dụng SMOTE để tạo thêm dữ liệu gian lận (oversampling)
    
    SMOTE là gì?
    - Synthetic Minority Oversampling Technique
    - Tạo thêm dữ liệu gian lận giả (synthetic) để cân bằng với dữ liệu bình thường
    - Không copy y hệt, mà tạo dữ liệu mới dựa trên dữ liệu cũ
    
    Tại sao cần SMOTE?
    - Dataset có rất ít giao dịch gian lận (chỉ 0.17%)
    - Model sẽ học lệch về phía "bình thường" nếu không cân bằng
    - SMOTE giúp model học tốt hơn về gian lận
    
    Lưu ý quan trọng:
    - CHỈ áp dụng trên training data, KHÔNG áp dụng trên validation/test
    - Nếu áp dụng trên validation/test, sẽ làm sai lệch kết quả đánh giá
    
    Parameters:
    -----------
    X : pandas.DataFrame hoặc numpy.ndarray
        Dữ liệu đầu vào (features)
    y : pandas.Series hoặc numpy.ndarray
        Nhãn (0 = bình thường, 1 = gian lận)
    random_state : int, mặc định=42
        Số ngẫu nhiên để đảm bảo kết quả giống nhau
    k_neighbors : int, mặc định=5
        Số điểm gần nhất để tạo dữ liệu mới
    
    Returns:
    --------
    X_resampled : numpy.ndarray
        Dữ liệu đã được tăng thêm (có thêm dữ liệu gian lận)
    y_resampled : numpy.ndarray
        Nhãn tương ứng (có thêm nhãn gian lận)
    """
    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    
    if isinstance(X, pd.DataFrame):
        X_array = X.values
    else:
        X_array = X
    
    if isinstance(y, pd.Series):
        y_array = y.values
    else:
        y_array = y
    
    X_resampled, y_resampled = smote.fit_resample(X_array, y_array)
    
    return X_resampled, y_resampled


def get_class_weights(y):
    """
    Tính toán trọng số (weights) cho các lớp để xử lý mất cân bằng dữ liệu
    
    Class Weights là gì?
    - Thay vì tạo thêm dữ liệu (như SMOTE), ta tăng "trọng số" cho lớp thiểu số
    - Model sẽ chú ý nhiều hơn đến các mẫu gian lận khi học
    - Giống như cho điểm cao hơn cho câu hỏi khó trong bài thi
    
    Ưu điểm so với SMOTE:
    - Không cần thay đổi dữ liệu gốc
    - Nhanh hơn (không cần tạo dữ liệu mới)
    - Hoạt động tốt với tree-based models (Random Forest, XGBoost)
    
    Nhược điểm:
    - Có thể không hiệu quả bằng SMOTE cho một số models
    - Cần model hỗ trợ tham số class_weight
    
    Parameters:
    -----------
    y : pandas.Series hoặc numpy.ndarray
        Nhãn (0 = bình thường, 1 = gian lận)
    
    Returns:
    --------
    class_weights : dict
        Dictionary chứa trọng số cho mỗi lớp
        Ví dụ: {0: 0.5, 1: 289.0} nghĩa là lớp gian lận có trọng số cao hơn 578 lần
    """
    if isinstance(y, pd.Series):
        y_array = y.values
    else:
        y_array = y
    
    unique_classes = np.unique(y_array)
    n_samples = len(y_array)
    n_classes = len(unique_classes)
    
    # Calculate weights inversely proportional to class frequency
    class_weights = {}
    for cls in unique_classes:
        n_class = np.sum(y_array == cls)
        weight = n_samples / (n_classes * n_class)
        class_weights[cls] = weight
    
    return class_weights



>>>>>>> 7f1a985 (Update notebooks and models)
