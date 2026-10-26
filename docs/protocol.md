# Warfarin QML — Experimental Protocol

## 1. Objective

Xây dựng giao thức thực nghiệm thống nhất để đánh giá và so sánh các mô hình Classical Machine Learning và Quantum Machine Learning trên IWPC dataset.

Đảm bảo các thí nghiệm công bằng, có khả năng tái lập và tránh Data Leakage.

## 2. Dataset and Target

- Dataset: International Warfarin Pharmacogenetics Consortium (IWPC).
- Input X: Các đặc trưng lâm sàng và di truyền hợp lệ sau khi kiểm tra Data Leakage.
- Target y: Nhóm liều Warfarin ổn định.
- Classification task: Multiclass Classification gồm ba lớp.

**Label Definition:**

| Label | Therapeutic Dose |
|---|---|
| Low | ≤21 mg/week |
| Intermediate | >21 và <49 mg/week |
| High | ≥49 mg/week |

Chỉ sử dụng bệnh nhân có stable dose hợp lệ theo quy tắc lọc được chốt ở GĐ3.

## 3. Dataset Splitting

Sử dụng Stratified Sampling để duy trì phân bố các lớp.

- Final Test Set: 20% dữ liệu hợp lệ.
- Development Set: 80% dữ liệu hợp lệ.

Final Test Set được tách và khóa sau khi hoàn thành kiểm tra tính hợp lệ của mẫu và tạo nhãn.

Không sử dụng Final Test Set để lựa chọn feature, tuning hyperparameters hoặc lựa chọn mô hình.

## 4. Development Protocol

**Matched Classical & Quantum Benchmark:**

- Tạo Quantum Development Pool cố định từ Development Set.
- Số lượng mẫu mục tiêu: 600.
- Có thể giảm xuống 400 nếu pilot runtime hoặc số lượng mẫu từng lớp yêu cầu.
- Khóa kích thước Quantum Development Pool trước khi xem model performance.
- Sử dụng Stratified 5-Fold Cross-Validation.
- Tất cả mô hình trong matched benchmark sử dụng cùng sample IDs và fold IDs.

**Strong Classical Reference:**

- Sử dụng toàn bộ Development Set.
- Sử dụng các feature hợp lệ, không giới hạn theo số qubit.
- Thực hiện Stratified 5-Fold Cross-Validation riêng.
- Kết quả được trình bày riêng với matched benchmark.

## 5. Preprocessing Protocol

Tất cả các thao tác preprocessing có học tham số từ dữ liệu phải được fit trên training fold.

- Numeric features: Missing value imputation và scaling phù hợp.
- Categorical/genotype features: Imputation và encoding phù hợp.
- Feature reduction: PCA(k) được fit trên training fold.
- Quantum input: Scale PCA components về [0, π].
- Matched Classical models: Sử dụng cùng PCA components với Quantum models, cho phép scaling riêng nếu cần.

Validation và Test chỉ được transform bằng preprocessing đã fit trên training data.

## 6. Feature and Qubit Budget

- Main experiment: k = 4 dimensions/qubits.
- Ablation: k = 2 dimensions/qubits.
- Optional experiment: k = 6 nếu tài nguyên tính toán cho phép.

k được định nghĩa là số chiều số học cuối cùng đưa vào Quantum Circuit, không phải số cột raw ban đầu.

Cần thực hiện PCA diagnostics và kiểm tra mất mát thông tin dự đoán trước khi khóa representation.

## 7. Model Comparison

**Classical Models:**

- Logistic Regression.
- Support Vector Machine (SVM).
- Multilayer Perceptron (MLP).

**Quantum Models:**

- Quantum Kernel Classifier.
- Variational Quantum Classifier (VQC).

Trong matched benchmark, các mô hình phải sử dụng cùng tập dữ liệu, cùng splits và cùng reduced representation.

Hyperparameter tuning budget phải được xác định trước và ghi lại.

## 8. Evaluation Metrics

**Primary Metric:**
- Macro-F1 Score.

**Mandatory Metric:**
- Accuracy.

**Secondary Metrics:**
- Balanced Accuracy.
- Per-class Recall.
- Confusion Matrix.
- Training and Prediction Runtime.
- Quantum Circuit Statistics.

Kết quả cross-validation được báo cáo dưới dạng mean ± standard deviation.

## 9. Experiment Protocol

**Experiment A — Main Benchmark**

So sánh Logistic Regression, SVM, MLP, Quantum Kernel và VQC trên cùng reduced representation k=4 và Quantum Development Pool.

**Experiment B — Strong Classical Reference**

Đánh giá Classical Models trên toàn bộ feature hợp lệ và full Development Set.

**Experiment C — Limited-data Learning Curve**

Khảo sát ba mức training size:

- 25%.
- 50%.
- 100%.

Sử dụng 3 stratified holdout repeats. Trong mỗi repeat, giữ validation cố định và dùng cùng training sample IDs cho các matched models.

**Experiment D — Feature/Qubit Ablation**

So sánh k=2 và k=4 bằng cùng Stratified 5-Fold splits.

k=6 chỉ thực hiện nếu tài nguyên cho phép.

## 10. Reproducibility

- Cố định và lưu random seeds.
- Lưu sample IDs, train/validation/test splits.
- Lưu preprocessing và reduction configurations.
- Lưu model hyperparameters và package versions.
- VQC sử dụng một primary seed và tối đa hai robustness seeds được khai báo trước nếu runtime cho phép.
- Không lựa chọn kết quả từ seed tốt nhất để thay thế kết quả primary.

Các giá trị seed cụ thể và danh sách cấu hình tuning sẽ được nhóm thống nhất trong `experiment_matrix` trước khi chạy mô hình.

## 11. Final Evaluation

- Chỉ thực hiện sau khi hoàn tất model selection và khóa cấu hình.
- Matched models được retrain trên toàn Quantum Development Pool.
- Strong Classical Models được retrain trên toàn Development Set.
- Tất cả mô hình được đánh giá trên cùng Final Test Set 20%.
- Thực hiện paired bootstrap cho các cặp mô hình chính được xác định trước để báo cáo chênh lệch metric và 95% confidence interval.
- Không điều chỉnh pipeline dựa trên Final Test performance.

## 12. Class Imbalance Policy

- Sử dụng Macro-F1 làm metric chính do khả năng mất cân bằng giữa ba lớp.
- Không sử dụng class weighting hoặc resampling trong core benchmark theo mặc định.
- Nếu cần xử lý imbalance, phải thiết kế thành experiment riêng và chỉ thực hiện trên training fold.

## 13. Research Limitations

Kết quả chỉ phục vụ mục đích nghiên cứu và đánh giá thuật toán, không sử dụng để khuyến nghị liều Warfarin thực tế.

Không giả định Quantum Machine Learning có hiệu năng cao hơn Classical Machine Learning.