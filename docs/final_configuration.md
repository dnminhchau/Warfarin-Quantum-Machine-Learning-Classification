# Cấu hình cuối cùng và kết quả Final Test

## Dữ liệu và tập chia

- 5.410 bệnh nhân đạt liều ổn định và có liều hợp lệ (> 0).
- Nhãn: Thấp ≤ 21 mg/tuần, Vừa 21–49 mg/tuần, Cao ≥ 49 mg/tuần.
- Development 4.328 / Final Test 1.082: chia ngẫu nhiên 80/20, phân tầng theo nhãn, seed 42
  (`configs/split_config.yaml`). Cả 21 trung tâm đều có mặt ở hai tập.
- Nhóm 600 người thuộc Development dành cho các thí nghiệm lượng tử ban đầu và learning curve trên nhóm nhỏ.

## Tiền xử lý (fit trên phần huấn luyện của từng fold)

1. Phục hồi VKORC1 -1639 bị thiếu từ SNP liên kết VKORC1 1173 (rs9934438).
2. Điền chiều cao và cân nặng bằng hồi quy tuyến tính (Height ~ Weight + Age + Male;
   Weight ~ Height + Age + Male); giá trị còn thiếu điền trung vị.
3. Mã hóa: VKORC1 theo số allele A, CYP2C9 theo điểm hoạt tính CPIC, tuổi theo thập niên, cân nặng.
4. Biểu diễn chính: domain với k = 4 đặc trưng (4 qubit). Mô hình lượng tử nhận góc trong [0, π];
   mô hình cổ điển nhận dữ liệu chuẩn hóa.

## Mô hình và lưới tham số

Tinh chỉnh bằng inner 3-fold theo Macro-F1, trọng số lớp cân bằng cho mọi mô hình trừ MLP phiên bản đầu.
MLP cân bằng lớp dùng trọng số mẫu theo cùng công thức `class_weight='balanced'` vì MLPClassifier không có tham số này.

| Mô hình | Lưới tham số |
|---|---|
| LR | C ∈ {0.01, 0.1, 1, 10, 100} |
| SVM (RBF) | C ∈ {0.1, 1, 10} × gamma ∈ {0.01, 0.1, scale, 1} |
| MLP (cân bằng lớp) | hidden ∈ {(16), (32, 16)} × alpha ∈ {1e-4, 1e-2} |
| MLP (không cân bằng lớp, phiên bản đầu) | như trên |
| HGB | max_depth ∈ {2, 3} × learning_rate ∈ {0.05, 0.1} |
| QSVC (ZZ feature map) | bandwidth ∈ {0.05, 0.1, 0.25, 0.5, 1} × C ∈ {0.1, 1, 10} |
| QSVC-product (Z feature map, không rối lượng tử) | như QSVC |
| VQC | layers ∈ {1, 2} × reps ∈ {1, 3} × bandwidth ∈ {0.5, 1}; statevector, L-BFGS-B, maxiter 250, seed 0 |

## Đánh giá

- Development: 5-fold phân tầng theo trung tâm (seed 42), inner 3-fold để tinh chỉnh.
- Chỉ số: Macro-F1 (chính), accuracy, recall từng nhóm, QWK, tỉ lệ nhầm Thấp ↔ Cao; với hồi quy: MAE, PW20, R².
- Final Test: tinh chỉnh trên Development, huấn luyện lại trên toàn bộ Development, đánh giá trên Final Test.
  So sánh cặp bằng paired bootstrap 2.000 lần.

## Kết quả Final Test

Nguồn: `results/final_test/metrics.csv` và `metrics_additional.csv`.

| Mô hình | Macro-F1 | Accuracy | Recall Thấp | Recall Vừa | Recall Cao | QWK | Tham số được chọn |
|---|---|---|---|---|---|---|---|
| LR | 0,587 | 0,599 | 0,779 | 0,450 | 0,768 | 0,586 | C = 100 |
| SVM | 0,583 | 0,591 | 0,700 | 0,477 | 0,790 | 0,569 | C = 10, gamma = 0,01 |
| QSVC-product | 0,582 | 0,591 | 0,697 | 0,480 | 0,783 | 0,572 | C = 10, bandwidth = 0,1 |
| HGB | 0,577 | 0,585 | 0,723 | 0,455 | 0,783 | 0,573 | max_depth = 2, learning_rate = 0,1 |
| MLP (cân bằng lớp) | 0,575 | 0,584 | 0,751 | 0,433 | 0,797 | 0,573 | hidden = (16), alpha = 1e-2 |
| MLP (không cân bằng lớp) | 0,575 | 0,665 | 0,588 | 0,806 | 0,261 | 0,517 | hidden = (32, 16), alpha = 1e-4 |
| QSVC | 0,568 | 0,575 | 0,703 | 0,443 | 0,804 | 0,570 | C = 10, bandwidth = 0,05 |
| VQC | 0,548 | 0,556 | 0,717 | 0,407 | 0,775 | 0,536 | layers = 2, reps = 3, bandwidth = 0,5 |
| Công thức IWPC | 0,541 | 0,644 | 0,412 | 0,876 | 0,261 | 0,442 | – |

Paired bootstrap trên Macro-F1 (`results/final_test/bootstrap.json`):

| Cặp (A − B) | Chênh lệch | KTC 95% | Kết luận |
|---|---|---|---|
| LR − QSVC-product | +0,006 | [−0,012; +0,022] | Không khác biệt có ý nghĩa |
| QSVC-product − QSVC | +0,014 | [+0,003; +0,024] | QSVC-product tốt hơn |
| QSVC-product − SVM | −0,001 | [−0,014; +0,010] | Không khác biệt có ý nghĩa |

Paired bootstrap cho các cặp khai báo trước trong `docs/protocol.md` (`results/final_test/bootstrap_predeclared.json`):

| Cặp (A − B) | Chênh lệch | KTC 95% | Kết luận |
|---|---|---|---|
| QSVC − SVM | −0,015 | [−0,029; −0,001] | QSVC thấp hơn SVM |
| VQC − LR | −0,039 | [−0,060; −0,017] | VQC thấp hơn LR |
| QSVC − QSVC-product | −0,014 | [−0,024; −0,003] | QSVC thấp hơn QSVC-product |

Dự đoán từng bệnh nhân và ma trận nhầm lẫn của các mô hình nằm trong `predictions.csv` và `confusion_matrices.csv`.

## Lịch sử sử dụng Final Test

Final Test đã được dùng năm lần:

1. Trong quá trình phát triển, với tham số cố định (chưa tinh chỉnh). Kết quả lần này không được báo cáo.
2. Sau khi bổ sung tinh chỉnh siêu tham số: LR, SVM, QSVC-product, QSVC và công thức IWPC
   (`metrics.csv`, `bootstrap.json`). Đây là kết quả chính.
3. Ngày 17/11/2026, bổ sung MLP, HGB và VQC bằng `scripts/evaluate_final_test_additional.py`
   (`metrics_additional.csv`). Pipeline và lưới tham số giữ nguyên, kết quả của lần 2 không thay đổi.
4. Ngày 18/11/2026, tái tạo dự đoán của cả 7 mô hình và công thức IWPC bằng
   `scripts/evaluate_final_test_predictions.py` để tính bootstrap cho các cặp khai báo trước và ma trận nhầm lẫn.
   Tham số được chọn và mọi chỉ số trùng khớp với các lần 2 và 3 (sai khác dưới 1e-16); 3 cặp bootstrap cũ
   tính lại cũng trùng khớp.

5. Ngày 22/11/2026, bổ sung MLP có cân bằng lớp (`MLP-balanced`) bằng `scripts/evaluate_final_test_additional.py`
   và tái tạo dự đoán bằng `scripts/evaluate_final_test_predictions.py` (thêm cột MLP-balanced vào `predictions.csv`
   và `confusion_matrices.csv`). Chỉ số của các mô hình đã có trùng khớp với các lần trước; bootstrap của các cặp
   khai báo trước không đổi.

Các lần 2–5 chỉ thêm mô hình hoặc tái tạo dự đoán với pipeline và lưới tham số không đổi, không dùng để chọn
cấu hình. Vì Final Test đã được xem trước khi hoàn tất thí nghiệm, kết quả trên Final Test không hoàn toàn độc lập.
