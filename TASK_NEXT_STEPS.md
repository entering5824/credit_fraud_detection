# Công việc Tiếp theo - Week 2 (Minh Phú)

## Tổng quan

Theo WORKFLOW.md, đây là công việc tiếp theo cho phần **Modeling setup** (Minh Phú) trong Week 2.

## Mục tiêu

1. Train các ensemble models đầy đủ với parameters tối ưu
2. Lưu models đã train vào thư mục `models/`
3. Đánh giá models trên validation set
4. Tạo bảng metrics và visualizations

## Công việc cần hoàn thành

### ✅ Đã hoàn thành (tích hợp từ codebase chính)

- Module `src/models.py` đã có đầy đủ functions:
  - `train_random_forest()` - Train Random Forest
  - `train_adaboost()` - Train AdaBoost  
  - `train_xgboost()` - Train XGBoost
  - `save_model()` - Lưu model
  - `load_model()` - Load model
  - `evaluate_model_performance()` - Đánh giá model

- Module `src/evaluate.py` đã có đầy đủ functions:
  - `calculate_metrics()` - Tính Precision, Recall, F1, AUC
  - `get_metrics_dict()` - Lấy metrics dạng dictionary
  - `print_metrics()` - In metrics
  - `plot_confusion_matrix()` - Vẽ confusion matrix
  - `plot_roc_curve()` - Vẽ ROC curve
  - `plot_metrics_comparison()` - So sánh metrics
  - `export_metrics_to_csv()` - Xuất metrics ra CSV

- Notebook `02_Model_Training.ipynb` đã được cập nhật với:
  - Sections để train 3 models
  - Evaluation với evaluate.py
  - Visualization metrics comparison
  - Lưu models

### 📋 Checklist công việc cần làm

#### 1. Chạy Notebook 02 để Train Models

```python
# Mở và chạy notebook: notebooks/02_Model_Training.ipynb
# Notebook sẽ:
# - Load và preprocess data
# - Train Random Forest, AdaBoost, XGBoost
# - Đánh giá trên validation set
# - Lưu models vào models/
```

**Các bước:**
- [ ] Mở `notebooks/02_Model_Training.ipynb`
- [ ] Chạy tất cả cells từ đầu đến cuối
- [ ] Kiểm tra models đã được train thành công
- [ ] Kiểm tra models đã được lưu vào `models/`:
  - `models/random_forest.pkl`
  - `models/adaboost.pkl`
  - `models/xgboost.pkl`

#### 2. Tối ưu Parameters (Optional nhưng khuyến nghị)

Nếu muốn cải thiện performance, có thể thử các parameters khác:

**Random Forest:**
```python
rf_model = train_random_forest(
    X_train, y_train,
    n_estimators=300,  # Tăng số trees
    max_depth=20,      # Tăng depth
    class_weight=class_weights,
    random_state=42
)
```

**AdaBoost:**
```python
ada_model = train_adaboost(
    X_train, y_train,
    n_estimators=150,   # Tăng số estimators
    learning_rate=0.3,  # Điều chỉnh learning rate
    random_state=42
)
```

**XGBoost:**
```python
xgb_model = train_xgboost(
    X_train, y_train,
    n_estimators=200,   # Tăng số rounds
    max_depth=8,        # Tăng depth
    learning_rate=0.05, # Giảm learning rate
    scale_pos_weight=scale_pos_weight,
    random_state=42
)
```

#### 3. Đánh giá Models trên Validation Set

Notebook 02 đã có code để đánh giá, nhưng có thể thêm:

```python
# Sử dụng evaluate.py để đánh giá chi tiết
from src.evaluate import evaluate_model, get_metrics_dict

# Đánh giá từng model
for model_name, model in trained_models.items():
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # Tính metrics
    metrics = get_metrics_dict(y_val, y_pred, y_pred_proba, model_name)
    
    # In metrics
    print_metrics(y_val, y_pred, y_pred_proba, model_name)
```

#### 4. Tạo Bảng Metrics

Notebook 02 đã tạo bảng metrics, nhưng có thể export ra CSV:

```python
from src.evaluate import export_metrics_to_csv

# Collect all metrics
all_metrics = []
for model_name in trained_models.keys():
    y_pred = model_predictions[model_name]
    y_pred_proba = model_probabilities[model_name]
    metrics = get_metrics_dict(y_val, y_pred, y_pred_proba, model_name)
    all_metrics.append(metrics)

# Export to CSV
export_metrics_to_csv(all_metrics, 'results/metrics_validation.csv')
```

#### 5. Vẽ Bar Chart So sánh Hiệu năng

Notebook 02 đã có visualization, nhưng có thể lưu:

```python
from src.evaluate import plot_metrics_comparison

# Plot và lưu
fig = plot_metrics_comparison(
    all_metrics, 
    save_path='results/metrics_comparison_validation.png',
    figsize=(14, 7)
)
plt.show()
```

#### 6. Lưu Confusion Matrices và ROC Curves

Có thể thêm vào notebook 02:

```python
from src.evaluate import plot_confusion_matrix, plot_roc_curve

# Lưu confusion matrices
for model_name in trained_models.keys():
    y_pred = model_predictions[model_name]
    save_path = f'results/confusion_matrices/{model_name.replace(" ", "_")}_validation.png'
    plot_confusion_matrix(y_val, y_pred, model_name, save_path=save_path)

# Lưu ROC curves
from src.evaluate import compare_models_roc

y_true_dict = {name: y_val for name in model_probabilities.keys()}
fig = compare_models_roc(
    y_true_dict, 
    model_probabilities, 
    save_path='results/roc_curves_validation.png'
)
plt.show()
```

## Hướng dẫn sử dụng

### Cách 1: Sử dụng Notebook (Khuyến nghị)

1. Mở `notebooks/02_Model_Training.ipynb`
2. Chạy tất cả cells
3. Models sẽ được train và lưu tự động
4. Metrics sẽ được hiển thị và có thể export

### Cách 2: Sử dụng Script Python

Có thể tạo script `train_all_models.py`:

```python
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.data_preprocessing import scale_features, split_data, get_class_weights
from src.models import train_random_forest, train_adaboost, train_xgboost, save_model
from src.evaluate import get_metrics_dict, print_metrics, export_metrics_to_csv

# Load và preprocess data
# ... (code từ notebook)

# Train models
# ... (code từ notebook)

# Save models
# ... (code từ notebook)
```

## Files cần kiểm tra sau khi hoàn thành

- [ ] `models/random_forest.pkl` - Random Forest model
- [ ] `models/adaboost.pkl` - AdaBoost model
- [ ] `models/xgboost.pkl` - XGBoost model
- [ ] `models/scaler.pkl` - Scaler (đã có từ notebook 01)
- [ ] `results/metrics_validation.csv` - Metrics table (optional)
- [ ] `results/metrics_comparison_validation.png` - Comparison chart (optional)

## Lưu ý quan trọng

1. **Sử dụng Validation Set**: Đánh giá trên validation set, không phải test set
2. **Test Set**: Giữ test set cho notebook 03 (final evaluation)
3. **Class Weights**: Đã được tính và sử dụng trong training
4. **Scale Pos Weight**: XGBoost cần `scale_pos_weight` để handle imbalance
5. **Random State**: Sử dụng `random_state=42` để đảm bảo reproducibility

## Kết quả mong đợi

Sau khi hoàn thành, bạn sẽ có:

1. ✅ 3 models đã được train và lưu
2. ✅ Metrics comparison cho tất cả models
3. ✅ Visualizations (confusion matrices, ROC curves)
4. ✅ Models sẵn sàng cho notebook 03 và application

## Công việc tiếp theo (sau khi hoàn thành)

Sau khi train xong models, công việc tiếp theo:

1. **Notebook 03** (Phạm Tú): Evaluation & Comparison cuối cùng trên test set
2. **Application** (Minh Phú): Xây dựng `fraud_detection_app.py` để sử dụng models

## Hỗ trợ

Nếu gặp vấn đề:

1. Kiểm tra `requirements.txt` đã cài đủ thư viện (đặc biệt `xgboost`)
2. Đảm bảo đã chạy notebook 01 trước để có preprocessed data
3. Kiểm tra paths và imports trong notebook
4. Xem lại code examples trong notebook 02

## Code Examples

### Example 1: Train và Save một Model

```python
from src.models import train_random_forest, save_model
from src.data_preprocessing import get_class_weights

# Get class weights
class_weights = get_class_weights(y_train)

# Train model
model = train_random_forest(
    X_train, y_train,
    n_estimators=200,
    max_depth=15,
    class_weight=class_weights,
    random_state=42
)

# Save model
save_model(model, 'Random Forest', save_dir='models')
```

### Example 2: Load và Sử dụng Model

```python
from src.models import load_model

# Load model
model = load_model('Random Forest', model_dir='models')

# Predict
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]
```

### Example 3: Đánh giá Model

```python
from src.evaluate import evaluate_model, get_metrics_dict

# Evaluate
metrics, figures = evaluate_model(
    y_test, y_pred, y_pred_proba,
    model_name='Random Forest',
    save_dir='results',
    plot_cm=True,
    plot_roc=True
)

# Get metrics dict
metrics_dict = get_metrics_dict(y_test, y_pred, y_pred_proba, 'Random Forest')
```

---

**Chúc bạn hoàn thành tốt công việc!** 🚀

