# Warfarin QML — Project Specification

## 1. Project Title

Phân loại nhóm liều Warfarin bằng Quantum Machine Learning.

## 2. Problem Definition

Đề tài giải quyết bài toán phân loại đa lớp (Multiclass Classification), dự đoán nhóm liều Warfarin ổn định dựa trên các đặc trưng lâm sàng và di truyền của bệnh nhân.

## 3. Input Features (X)

Các đặc trưng lâm sàng và di truyền hợp lệ từ IWPC dataset, bao gồm các nhóm ứng viên:

- Tuổi, chiều cao, cân nặng.
- Thông tin thuốc sử dụng.
- Kiểu gen CYP2C9.
- Kiểu gen VKORC1.

Danh sách feature chính thức sẽ được xác định sau Feature Policy và Leakage Audit ở GĐ3.

## 4. Target Variable (y)

Target được tạo từ `Therapeutic Dose of Warfarin`, đơn vị mg/tuần.

Quy tắc phân loại:

| Class | Dose Range |
|---|---|
| Low | Dose ≤ 21 |
| Intermediate | 21 < Dose < 49 |
| High | Dose ≥ 49 |

## 5. Machine Learning Models

**Classical Machine Learning:**
- Logistic Regression
- Support Vector Machine (SVM)
- Multilayer Perceptron (MLP)

**Quantum Machine Learning:**
- Quantum Kernel
- Variational Quantum Classifier (VQC)

## 6. Research Questions

- RQ1: Quantum ML so sánh với Classical ML như thế nào khi sử dụng cùng dữ liệu và ngân sách đặc trưng?
- RQ2: Hiệu năng các mô hình thay đổi như thế nào khi giới hạn lượng dữ liệu huấn luyện?
- RQ3: Số lượng feature/qubit ảnh hưởng như thế nào đến hiệu năng và chi phí tính toán của Quantum ML?

## 7. Evaluation Metrics

- Primary metric: Macro-F1
- Mandatory metric: Accuracy
- Secondary metrics: Balanced Accuracy, per-class Recall, Confusion Matrix, Runtime

## 8. Experimental Scope

- Development set: 80%.
- Locked final test set: 20%.
- Stratified 5-Fold Cross-Validation.
- Main quantum representation: k = 4.
- Ablation: k = 2, k = 6 nếu tài nguyên cho phép.
- Matched Classical và Quantum sử dụng cùng quantum development pool, cùng folds và cùng k-dimensional representation.
- Strong Classical Reference sử dụng toàn bộ feature hợp lệ trên full development và được báo cáo riêng.

## 9. Research Limitations

Đây là nghiên cứu so sánh phương pháp Machine Learning trên dữ liệu IWPC, không phải công cụ kê đơn hoặc khuyến nghị liều thuốc cho bệnh nhân.

Đề tài không giả định Quantum Machine Learning luôn tốt hơn Classical Machine Learning.