# Credit Card Fraud Detection với Ensemble Learning

## 📌 Tên đề tài
Credit Card Fraud Detection with Ensemble Learning Predictive Models

## 🎯 Mô tả bài toán
Phát hiện giao dịch gian lận thẻ tín dụng sử dụng các phương pháp **Ensemble Learning** (Học tập Tập hợp).

## 📋 Tổng quan

Đây là dự án phát hiện giao dịch gian lận thẻ tín dụng sử dụng các phương pháp **Ensemble Learning** (Học tập Tập hợp).

### Mục tiêu
Xây dựng một ứng dụng Python có thể:
1. Phát hiện giao dịch gian lận bằng **ít nhất 3 phương pháp Ensemble Learning**:
   - Random Forest (Rừng Ngẫu nhiên)
   - AdaBoost (Adaptive Boosting)
   - XGBoost (Extreme Gradient Boosting)
2. Chạy trên Python hoặc Google Colab
3. So sánh và đánh giá độ chính xác giữa các models

---

## 📊 Dữ liệu

- **Dataset**: Credit Card Fraud Detection từ Kaggle
- **Mô tả**: Giao dịch thẻ tín dụng của người dùng châu Âu trong tháng 9/2013
- **Link**: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- **Đặc điểm**:
  - 284,807 giao dịch
  - 492 giao dịch gian lận (0.17%) - Dữ liệu rất mất cân bằng
  - 31 features: V1-V28 (đã được PCA transform), Time, Amount, Class

---

## 🚀 Hướng dẫn Cài đặt

### Bước 1: Cài đặt thư viện

```bash
pip install -r requirements.txt
```

Các thư viện cần thiết:
- `numpy` - Tính toán số học
- `pandas` - Xử lý dữ liệu dạng bảng
- `scikit-learn` - Machine Learning
- `xgboost` - XGBoost model
- `imbalanced-learn` - Xử lý dữ liệu mất cân bằng
- `matplotlib`, `seaborn` - Vẽ biểu đồ
- `scipy` - Thống kê và tính toán khoa học

### Bước 2: Tải dataset

1. Tải file `creditcard.csv` từ Kaggle
2. Đặt vào thư mục `data/creditcard.csv`

---

## 📁 Cấu trúc Project

```
credit_fraud_detection/
│
├── data/
│   └── creditcard.csv          # Dataset từ Kaggle
│
├── notebooks/
│   ├── 01_EDA_and_Preprocessing.ipynb      # Phân tích và xử lý dữ liệu
│   ├── 02_Model_Training.ipynb            # Train các models
│   └── 03_Evaluation_and_Comparison.ipynb  # Đánh giá và so sánh
│
├── src/
│   ├── data_preprocessing.py   # Hàm xử lý dữ liệu (scaling, split, SMOTE)
│   ├── models.py               # Hàm train các ensemble models
│   ├── evaluate.py             # Hàm đánh giá models (metrics, plots)
│   └── utils.py                # Các hàm tiện ích
│
├── app/
│   ├── fraud_detection_app.py  # Ứng dụng dự đoán giao dịch mới
│   └── loop_random_fda.py      # Ứng dụng test với dữ liệu ngẫu nhiên
│
├── models/                     # Thư mục lưu models đã train
│   ├── scaler.pkl
│   ├── random_forest.pkl
│   ├── adaboost.pkl
│   └── xgboost.pkl
│
├── results/                    # Kết quả đánh giá
│   ├── confusion_matrices/    # Confusion matrices của từng model
│   ├── metrics.csv            # Bảng metrics
│   └── roc_curves_comparison.png
│
├── requirements.txt           # Danh sách thư viện
└── README.md                  # File này
```

---

## 🔄 Quy trình Sử dụng

### Bước 1: Phân tích và Xử lý Dữ liệu

Chạy notebook `01_EDA_and_Preprocessing.ipynb`:

```python
# Notebook này sẽ:
# 1. Phân tích dữ liệu (EDA)
#    - Kiểm tra missing values
#    - Phát hiện outliers
#    - Xem phân phối các features
#    - Kiểm tra tỉ lệ gian lận/bình thường
# 
# 2. Xử lý dữ liệu
#    - Chuẩn hóa (Scaling) features
#    - Chia train/validation/test (70/15/15)
#    - Lưu scaler vào models/scaler.pkl
```

**Kết quả**: Dữ liệu đã được xử lý và sẵn sàng cho training

---

### Bước 2: Train Models

Chạy notebook `02_Model_Training.ipynb`:

```python
# Notebook này sẽ:
# 1. Load dữ liệu đã preprocess
# 2. Train 3 ensemble models:
#    - Random Forest (200 trees, max_depth=15)
#    - AdaBoost (100 estimators)
#    - XGBoost (100 estimators, max_depth=6)
# 3. Đánh giá trên validation set
# 4. Lưu models vào models/
```

**Kết quả**: 
- Models đã được train và lưu
- Metrics trên validation set

---

### Bước 3: Đánh giá và So sánh

Chạy notebook `03_Evaluation_and_Comparison.ipynb`:

```python
# Notebook này sẽ:
# 1. Load models đã train
# 2. Đánh giá trên test set (dữ liệu chưa từng thấy)
# 3. Tính metrics: Precision, Recall, F1, AUC
# 4. Vẽ confusion matrices và ROC curves
# 5. Xuất metrics.csv và các biểu đồ
```

**Kết quả**:
- `results/metrics.csv` - Bảng so sánh metrics
- `results/confusion_matrices/` - Confusion matrices
- `results/roc_curves_comparison.png` - So sánh ROC curves

---

### Bước 4: Sử dụng Ứng dụng

Chạy ứng dụng để dự đoán giao dịch mới:

```bash
python app/fraud_detection_app.py
```

**Cách sử dụng**:
1. Ứng dụng sẽ yêu cầu nhập thông tin giao dịch
2. Nhập các giá trị V1-V28 và Amount
3. Ứng dụng sẽ hiển thị kết quả dự đoán từ cả 3 models

---

## 📈 Metrics được Sử dụng

### Precision (Độ chính xác)
- Tỉ lệ các dự đoán "gian lận" thực sự là gian lận
- **Cao = tốt**: Ít báo động giả (false positives)

### Recall (Độ nhạy)
- Tỉ lệ các giao dịch gian lận thực sự được phát hiện
- **Cao = tốt**: Bắt được nhiều gian lận

### F1-Score
- Trung bình điều hòa của Precision và Recall
- **Cao = tốt**: Cân bằng giữa Precision và Recall

### AUC-ROC
- Diện tích dưới đường ROC curve
- **Cao = tốt**: Model phân biệt tốt giữa gian lận và bình thường
- Giá trị từ 0 đến 1 (1 = hoàn hảo)

---

## 🎯 Kết quả Mong đợi

Sau khi chạy đầy đủ pipeline, bạn sẽ có:

1. ✅ **3 models đã được train**:
   - Random Forest
   - AdaBoost
   - XGBoost

2. ✅ **Metrics comparison**:
   - Bảng so sánh Precision, Recall, F1, AUC
   - Biểu đồ so sánh hiệu năng

3. ✅ **Visualizations**:
   - Confusion matrices cho từng model
   - ROC curves comparison
   - Metrics comparison chart

4. ✅ **Ứng dụng sẵn sàng**:
   - Có thể dự đoán giao dịch mới
   - Hiển thị kết quả từ cả 3 models

---

## 💡 Giải thích Thuật ngữ

### Ensemble Learning là gì?
- Kết hợp nhiều models để tạo ra dự đoán tốt hơn
- Giống như hỏi nhiều chuyên gia thay vì một người

### Random Forest
- Tạo nhiều cây quyết định và lấy kết quả trung bình
- Giống như hỏi nhiều người và lấy đa số

### AdaBoost
- Train nhiều models yếu và kết hợp chúng
- Models sau học từ lỗi của models trước

### XGBoost
- Gradient Boosting tối ưu hóa
- Rất mạnh và nhanh, thường cho kết quả tốt nhất

### Class Imbalance
- Dữ liệu có quá ít mẫu gian lận (0.17%)
- Cần xử lý bằng SMOTE hoặc class weights

---

## 🛠️ Xử lý Lỗi Thường gặp

### Lỗi: "Model not found"
**Giải pháp**: Chạy notebook 02 để train models trước

### Lỗi: "Scaler not found"
**Giải pháp**: Chạy notebook 01 để tạo scaler trước

### Lỗi: "Module not found"
**Giải pháp**: 
```bash
pip install -r requirements.txt
```

### Lỗi khi chạy trên Colab
**Giải pháp**: 
- Upload toàn bộ project lên Google Drive
- Mount Drive trong Colab
- Chạy notebooks từ thư mục đã mount

---

## 📝 Lưu ý Quan trọng

1. **Chạy theo thứ tự**: Notebook 01 → 02 → 03
2. **Test set**: Chỉ dùng để đánh giá cuối cùng, không train trên test set
3. **Class imbalance**: Dataset rất mất cân bằng, cần xử lý cẩn thận
4. **Random state**: Dùng `random_state=42` để đảm bảo kết quả giống nhau

---

## 👥 Thành viên

Dự án này được thực hiện bởi nhóm 3-4 người:
- **Phạm Tú**: Data & Preprocessing, Evaluation
- **Minh Phú**: Modeling, Application

---

## 📚 Tài liệu Tham khảo

- [Kaggle Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

---

## 📄 License

Dự án này được tạo cho mục đích học tập và nghiên cứu.

---

**Chúc bạn thành công với dự án!** 🚀
