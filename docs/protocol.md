# Thiết kế nghiên cứu

Tài liệu ghi lại kế hoạch thí nghiệm được lập ngày 08/11/2026, trước khi chạy bộ thí nghiệm chính,
và các điểm thực tế đã thay đổi so với kế hoạch.

## 1. Lý do điều chỉnh so với thiết kế ban đầu

Các vấn đề phát hiện trên tập Development ở thiết kế ban đầu:

| Vấn đề | Bằng chứng |
|---|---|
| PCA k = 4 trên 40 cột one-hot mất nhiều thông tin | Chỉ giữ khoảng 30% phương sai; thành phần chính thứ nhất chủ yếu phản ánh kiểu thiếu dữ liệu theo trung tâm |
| VQC không học được | Loss cuối 1,607 so với mức đoán ngẫu nhiên log2(3) = 1,585; accuracy 0,38, thấp hơn mô hình luôn đoán nhóm Vừa (0,54). Nguyên nhân: COBYLA 60 vòng, huấn luyện với 1.024 shots, ansatz reps = 1, ánh xạ bitstring mod 3 bị lệch |
| Hàm nhân lượng tử bị tập trung | Trung bình phần tử ngoài đường chéo khoảng 0,09–0,12; chưa có tham số bandwidth |
| Ngân sách tinh chỉnh không đều | Mô hình cổ điển dùng tham số mặc định |
| Bỏ sót nhóm liều Cao | Recall nhóm Cao bằng 0 ở LR, SVM, MLP và QSVC |
| Thiếu mốc so sánh | Chưa có mô hình Dummy và công thức IWPC |

Việc chọn biểu diễn domain dựa trên các biến của công thức IWPC (NEJM 2009), không dựa trên các
kết quả chạy thử.

## 2. Giữ nguyên

- Dữ liệu IWPC gốc, quy tắc gán nhãn và 10 cột đầu vào trong `configs/feature_policy.yaml`.
  Chủng tộc, thuốc dùng kèm và mã trung tâm không đưa vào đặc trưng.
- Các tập chia trong `data/splits/`.
- Chỉ số chính Macro-F1, báo cáo trung bình ± độ lệch chuẩn qua 5 fold.

## 3. Kế hoạch

- **Biểu diễn:** domain (chính), PCA, mutual information, PLS-DA và đầy đủ 40 cột (tham chiếu).
- **Mất cân bằng lớp:** mọi mô hình dùng trọng số lớp cân bằng.
- **Tinh chỉnh:** inner 3-fold theo Macro-F1, mỗi mô hình tối đa 15 cấu hình.
- **VQC:** ZZ feature map, RealAmplitudes, data re-uploading khi layers = 2, đo ⟨Z⟩ và lớp đọc tuyến tính;
  huấn luyện bằng statevector, gradient parameter-shift, L-BFGS-B; 5 seed (0–4).
- **Learning curve:** 50, 100, 200, 400, 600 mẫu lấy từ nhóm 600 người, tập đánh giá là phần còn lại
  của Development, lặp 5 lần.
- **Thống kê:** trên Development dùng CV lặp 5×5 với kiểm định t hiệu chỉnh Nadeau–Bengio; trên Final Test
  dùng paired bootstrap 2.000 lần cho ba cặp QSVC – SVM, VQC – LR, QSVC – QSVC-product.
- **Phân tích phụ:** theo nhóm chủng tộc; bỏ 21 ca liều ≥ 100 mg/tuần; công thức IWPC có và không có các cột HOLD.

## 4. Giả thuyết khai báo trước

- H1: Với cùng 4 qubit, cách biểu diễn (PCA hay domain) ảnh hưởng tới Macro-F1 nhiều hơn việc chọn mô hình
  lượng tử hay cổ điển.
- H2: QSVC không vượt SVM đã tinh chỉnh trên cùng đầu vào (chênh lệch kỳ vọng trong ±0,03).
- H3: Rối lượng tử không cải thiện hàm nhân (QSVC ≈ QSVC-product).
- H4: Khi dữ liệu huấn luyện ít (≤ 200 mẫu), khoảng cách giữa mô hình lượng tử và cổ điển không đổi chiều.

## 5. Thay đổi so với kế hoạch

| Kế hoạch | Thực hiện |
|---|---|
| CV chính trên 5 fold cố định trong `data/splits/` | CV chính là 5-fold phân tầng theo trung tâm (seed 42) trong `scripts/run_classification_cv.py`; kết quả trên fold cố định nằm ở `results/benchmarks/` |
| Điền giá trị thiếu bằng trung bình | Phục hồi VKORC1 từ SNP liên kết và điền chiều cao, cân nặng bằng hồi quy |
| SVM gamma gồm cả giá trị 10; VQC tối đa 300 vòng | SVM gamma ∈ {0.01, 0.1, scale, 1}; VQC tối đa 250 vòng |
| VQC với 5 seed | Kết quả chính dùng seed 0; 5 seed (0–4) được đánh giá bổ sung trên lần lặp CV đầu tiên (`results/repeated_cv/vqc_seeds.csv`) |
| CV lặp 5×5 và kiểm định Nadeau–Bengio | Thực hiện bổ sung bằng `scripts/run_repeated_cv.py` (`results/repeated_cv/`), thêm hiệu chỉnh Holm cho 4 cặp so sánh |
| Bootstrap trên Final Test cho QSVC – SVM, VQC – LR, QSVC – QSVC-product | Lần đánh giá chính tính bootstrap cho LR – QSVC-product, QSVC-product – QSVC, QSVC-product – SVM. Ba cặp khai báo trước được tính bổ sung sau khi tái tạo dự đoán (`results/final_test/bootstrap_predeclared.json`) và được kiểm định thêm bằng Nadeau–Bengio trên Development |
| Learning curve trên nhóm 600 người | Có cả bản này (`results/learning_curve/`) và bản trên toàn bộ Development, mốc 50 đến khoảng 3.460 mẫu (`results/learning_curve_full/`) |
| Phân tích độ nhạy bỏ ca liều ≥ 100 mg/tuần; IWPC có cột HOLD | Thực hiện bổ sung bằng `scripts/run_protocol_checks.py`; tập Development có 15 trong 21 ca này. Ngoài ra có phân tích trên nhóm đủ kiểu gen và nhóm INR mục tiêu 2–3 |
| Mọi mô hình dùng trọng số lớp cân bằng | MLP phiên bản đầu không cân bằng lớp (recall nhóm Cao 0,19 trên CV). Bổ sung `MLP-balanced` ở CV chính, CV lặp 5×5, learning curve, khảo sát số qubit, chi phí và Final Test; bản cũ được giữ lại để tái tạo kết quả |
| Final Test chạy một lần | Final Test đã được dùng năm lần (xem `final_configuration.md`) |
| Không có phân tích shots và nhiễu | Bổ sung bằng `scripts/run_noise_analysis.py` (`results/noise/`): CV 5-fold trên Development, tham số cố định, không dùng Final Test |
