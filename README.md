# Warfarin Dose Group Classification with Quantum and Classical Machine Learning

Phân loại nhóm liều warfarin (Thấp / Vừa / Cao) trên bộ dữ liệu IWPC, so sánh mô hình học máy lượng tử
(QSVC, VQC) với mô hình cổ điển (LR, SVM, MLP, HGB) và công thức dược di truyền IWPC.

Đồ án môn học NT409 – Điện toán lượng tử và ứng dụng, Trường Đại học Công nghệ Thông tin, ĐHQG-HCM.
Dự án chỉ phục vụ nghiên cứu trên dữ liệu hồi cứu, **không dùng để chỉ định liều điều trị**.

## Kết quả chính

Biểu diễn đầu vào gồm 4 đặc trưng (VKORC1, cân nặng, tuổi, CYP2C9), tương ứng 4 qubit. Mô hình lượng tử
được mô phỏng bằng statevector trên Qiskit.

| Mô hình | CV trên Development (Macro-F1) | Final Test (Macro-F1) | Final Test (Recall nhóm Cao) |
|---|---|---|---|
| LR | 0,561 ± 0,011 | 0,587 | 0,768 |
| SVM | 0,561 ± 0,007 | 0,583 | 0,790 |
| HGB | 0,558 ± 0,014 | 0,577 | 0,783 |
| MLP | 0,558 ± 0,024 | 0,575 | 0,261 |
| QSVC-product (không rối lượng tử) | 0,558 ± 0,011 | 0,582 | 0,783 |
| QSVC (rối lượng tử, ZZ feature map) | 0,546 ± 0,013 | 0,568 | 0,804 |
| VQC | 0,528 ± 0,012 | 0,548 | 0,775 |
| Công thức IWPC | – | 0,541 | 0,261 |
| Dummy (luôn đoán nhóm Vừa) | 0,235 ± 0,004 | – | – |

- QSVC-product cho kết quả tương đương LR và SVM: chênh lệch với LR là +0,006 Macro-F1,
  khoảng tin cậy 95% của paired bootstrap là [-0,012; +0,022].
- QSVC có rối lượng tử thấp hơn QSVC-product có ý nghĩa thống kê (chênh lệch 0,014, KTC 95% [0,003; 0,024]).
- Theo learning curve, Macro-F1 của LR, SVM và QSVC-product gần như ổn định từ khoảng 100 mẫu huấn luyện,
  trong khi VQC, QSVC và HGB cần nhiều dữ liệu hơn.

Cấu hình chi tiết và lịch sử đánh giá Final Test: [docs/final_configuration.md](docs/final_configuration.md).

## Cấu trúc thư mục

```
configs/        feature_policy.yaml (68 cột và quyết định chọn/loại), split_config.yaml (tỉ lệ chia, seed)
data/
  raw/          dữ liệu IWPC gốc (.xls, giữ nguyên)
  interim/      labels.csv: nhãn nhóm liều
  splits/       tập chia Development / Final Test, nhóm 600 người cho mô hình lượng tử, các fold CV
  source_manifest.json, data_license.md
docs/           final_configuration.md, protocol.md
results/        kết quả các thí nghiệm (xem bảng bên dưới)
scripts/        các lệnh chạy thí nghiệm
src/            mã nguồn chính
```

| Module trong `src/` | Nội dung |
|---|---|
| `dataset.py` | Đọc dữ liệu, nhãn, tập chia; lọc nhóm bệnh nhân |
| `preprocessing.py` | Mã hóa đặc trưng dược di truyền, phục hồi VKORC1 từ SNP liên kết, điền thiếu bằng hồi quy |
| `policy_preprocessing.py` | Tiền xử lý 10 cột được duyệt, dùng khi tạo tập chia |
| `quantum_models.py` | QSVC (fidelity quantum kernel) và VQC (parameter-shift, L-BFGS-B) |
| `evaluation.py` | Danh sách mô hình và lưới tham số, chỉ số đánh giá, CV, Final Test, paired bootstrap |
| `baselines.py` | Công thức IWPC |

## Cài đặt

Yêu cầu Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.quantum_models     # đối chiếu với Qiskit, cần in ra 4 dòng OK
```

## Chạy thí nghiệm

Chạy từ thư mục gốc của repo.

| Bước | Lệnh | Kết quả |
|---|---|---|
| Kiểm tra dữ liệu | `python -m scripts.check_dataset` | `results/data_validation/dataset_check.json` |
| Thống kê tiền xử lý | `python -m scripts.analyze_preprocessing` | `results/data_validation/preprocessing_report.txt` |
| Phân loại (CV) | `python -m scripts.run_classification_cv --rep domain --k 4` | `results/classification_cv/` |
| Hồi quy liều | `python -m scripts.run_regression_cv --k 4` | `results/regression_cv/` |
| Learning curve | `python -m scripts.run_learning_curve_full` | `results/learning_curve_full/` |
| Số qubit, feature map | `python -m scripts.run_qubit_ablation --part both` | `results/ablation/` |
| Phân tích theo nhóm | `python -m scripts.run_subgroup_analysis --part subgroup` (hoặc `sensitivity`, `selection`) | `results/subgroup_analysis/` |
| Final Test | `python -m scripts.evaluate_final_test` | `results/final_test/` |

Các bước tạo nhãn (`scripts.prepare_labels`) và tập chia (`scripts.prepare_splits`) đã được chạy;
kết quả nằm trong `data/interim/` và `data/splits/`. Hai script này không ghi đè dữ liệu đã có.
Script Final Test cũng dừng lại nếu kết quả đã tồn tại, để tránh đánh giá lại trên tập kiểm thử.

Các thư mục kết quả khác:
- `results/learning_curve/`: learning curve trên nhóm 600 người (bản có và không tinh chỉnh tham số).
- `results/benchmarks/`: thí nghiệm ban đầu trên các fold cố định, gồm so sánh biểu diễn domain, PCA,
  mutual information và PLS.
- `results/exploratory/`: các phân tích thăm dò.

## Thiết kế đánh giá

- 5.410 bệnh nhân đạt liều ổn định. Nhóm liều: Thấp (≤ 21 mg/tuần), Vừa (21–49), Cao (≥ 49).
- Tập Development (4.328) và Final Test (1.082) được chia ngẫu nhiên 80/20, phân tầng theo nhóm liều.
- CV 5-fold trên Development, phân tầng theo trung tâm thu thập (Project Site) để mỗi fold có tỉ lệ
  bệnh nhân của các trung tâm tương tự nhau. Siêu tham số chọn bằng inner 3-fold theo Macro-F1.
- Mọi bước tiền xử lý có học tham số chỉ được fit trên phần huấn luyện của từng fold.
- Chủng tộc, thuốc dùng kèm và mã trung tâm không được dùng làm đặc trưng.

## Hạn chế

- Mô hình lượng tử được mô phỏng chính xác, chưa chạy trên phần cứng lượng tử thật và chưa xét nhiễu.
- Final Test đã được dùng nhiều lần trong quá trình phát triển (chi tiết trong `docs/final_configuration.md`).
- VQC chỉ được chạy với một seed khởi tạo.
- Nhóm Châu Á chỉ có 5 bệnh nhân liều Cao trong tập phân tích theo nhóm, nên kết quả của nhóm này không ổn định.

## Dữ liệu và giấy phép

Dữ liệu: Warfarin Consortium Combined Data Set (PharmGKB PS206767), International Warfarin
Pharmacogenetics Consortium, phân phối bởi PharmGKB/ClinPGx theo giấy phép CC BY-SA 4.0.
Các file phái sinh trong `data/interim/` và `data/splits/` phát hành theo cùng giấy phép.
Repo này không được PharmGKB/ClinPGx xác nhận. Chi tiết: [data/data_license.md](data/data_license.md).

International Warfarin Pharmacogenetics Consortium. Estimation of the warfarin dose with clinical and
pharmacogenetic data. *N Engl J Med* 2009;360:753–764.

## Thành viên

- Đoàn Ngọc Minh Châu – 23520168
- Võ Thị Hồng Phúc – 23521226

Giảng viên hướng dẫn: ThS. Phan Trung Phát.
