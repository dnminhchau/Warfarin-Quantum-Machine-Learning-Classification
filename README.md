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
| MLP (cân bằng lớp) | 0,548 ± 0,020 | 0,575 | 0,797 |
| MLP (không cân bằng lớp, phiên bản đầu) | 0,558 ± 0,024 | 0,575 | 0,261 |
| QSVC-product (không rối lượng tử) | 0,558 ± 0,011 | 0,582 | 0,783 |
| QSVC (rối lượng tử, ZZ feature map) | 0,546 ± 0,013 | 0,568 | 0,804 |
| VQC | 0,528 ± 0,012 | 0,548 | 0,775 |
| Công thức IWPC | – | 0,541 | 0,261 |
| Dummy (luôn đoán nhóm Vừa) | 0,235 ± 0,004 | – | – |

- QSVC-product cho kết quả tương đương LR và SVM: chênh lệch với LR là +0,006 Macro-F1,
  khoảng tin cậy 95% của paired bootstrap là [-0,012; +0,022].
- QSVC có rối lượng tử thấp hơn QSVC-product có ý nghĩa thống kê (chênh lệch 0,014, KTC 95% [0,003; 0,024]).
- Với các cặp khai báo trước, QSVC thấp hơn SVM (−0,015, KTC 95% [−0,029; −0,001]) và VQC thấp hơn LR
  (−0,039, KTC 95% [−0,060; −0,017]) trên Final Test. Các khoảng tin cậy này chưa hiệu chỉnh cho việc so sánh
  nhiều cặp; kết quả CV lặp 5×5 bên dưới là bằng chứng chính.
- Các mô hình có trọng số lớp cân bằng chủ yếu nhầm nhóm Vừa thành Thấp hoặc Cao. MLP không cân bằng lớp
  và công thức IWPC ngược lại, dự đoán phần lớn là nhóm Vừa và bỏ sót khoảng 3/4 bệnh nhân nhóm Cao.
  Khi cân bằng lớp, recall nhóm Cao của MLP tăng từ 0,261 lên 0,797 trên Final Test.
- Theo learning curve, Macro-F1 của LR, SVM và QSVC-product gần như ổn định từ khoảng 100 mẫu huấn luyện,
  trong khi VQC, QSVC, HGB và MLP cần nhiều dữ liệu hơn.

CV lặp 5×5 trên Development (25 fold, `results/repeated_cv/summary.csv`):

| Mô hình | Macro-F1 | Accuracy | Recall nhóm Cao |
|---|---|---|---|
| SVM | 0,564 ± 0,013 | 0,579 ± 0,012 | 0,70 |
| LR | 0,562 ± 0,013 | 0,578 ± 0,013 | 0,71 |
| QSVC-product | 0,561 ± 0,014 | 0,577 ± 0,014 | 0,71 |
| QSVC | 0,548 ± 0,014 | 0,561 ± 0,013 | 0,72 |
| MLP (cân bằng lớp) | 0,537 ± 0,032 | 0,549 ± 0,032 | 0,77 |
| VQC | 0,526 ± 0,018 | 0,537 ± 0,018 | 0,73 |

Kiểm định t hiệu chỉnh Nadeau–Bengio trên 25 fold, giá trị p hiệu chỉnh thêm bằng phương pháp Holm vì so sánh
4 cặp cùng lúc (`results/repeated_cv/nadeau_bengio.csv`):

| Cặp (A − B) | Chênh lệch Macro-F1 | p | p (Holm) | Kết luận |
|---|---|---|---|---|
| LR − QSVC-product | +0,001 | 0,85 | 0,85 | Không khác biệt |
| QSVC − QSVC-product | −0,013 | 0,054 | 0,11 | Không khác biệt có ý nghĩa |
| QSVC − SVM | −0,016 | 0,028 | 0,085 | QSVC có xu hướng thấp hơn, chưa đủ ý nghĩa sau hiệu chỉnh |
| VQC − LR | −0,036 | < 0,001 | 0,002 | VQC thấp hơn LR |

VQC qua 5 seed khởi tạo đạt 0,529 ± 0,005 Macro-F1, nên kết quả với seed 0 đại diện được cho mô hình.
Bỏ 15 ca liều ≥ 100 mg/tuần khỏi Development gần như không đổi kết quả (LR 0,559, QSVC-product 0,556).

Chi phí tính toán theo số qubit, đo trên fold 0 của CV (3.462 mẫu huấn luyện, 1 luồng CPU,
tham số đã chọn ở Final Test; `results/cost/`):

| k (qubit) | LR | SVM | MLP (cân bằng lớp) | QSVC | QSVC-product | VQC | Mạch VQC (độ sâu / CNOT / tham số) |
|---|---|---|---|---|---|---|---|
| 2 | 0,03 s | 0,25 s | 0,14 s | 1,5 s | 1,1 s | 5,8 s | 24 / 10 / 16 |
| 4 | 0,03 s | 0,25 s | 0,17 s | 3,4 s | 2,0 s | 43 s | 41 / 30 / 32 |
| 6 | 0,03 s | 0,21 s | 0,11 s | 4,2 s | 2,7 s | ≈ 59 phút (ước tính) | 51 / 50 / 48 |

- Thời gian mỗi vòng tối ưu của VQC tăng khoảng 8 lần từ 2 lên 4 qubit và khoảng 45 lần từ 4 lên 6 qubit,
  do statevector có 2^k phần tử và số tham số tăng theo k. Thời gian VQC ở k = 6 được ước tính từ 20 vòng
  (14,2 s/vòng × tối đa 250 vòng); ở k = 2 và 4, VQC dừng sau 154 và 137 vòng.
- Theo số mẫu huấn luyện (k = 4, `cost_by_train_size.csv`): từ 500 lên 3.462 mẫu, thời gian huấn luyện QSVC tăng
  từ 0,4 s lên 3,3 s và VQC từ 9,8 s lên 39 s; LR, SVM và MLP đều dưới 0,3 s.
- Mạch của QSVC-product không có cổng CNOT và có độ sâu cố định 4, nên hàm nhân của nó tách được thành tích
  của các hàm nhân trên từng qubit và có thể tính hiệu quả trên máy tính cổ điển.

### Ảnh hưởng của shots và nhiễu

Các kết quả trên dùng statevector: xác suất được tính chính xác và mạch không có lỗi. Trên máy lượng tử thật,
mỗi xác suất phải ước lượng bằng cách chạy mạch nhiều lần (shots), và mỗi cổng có thể bị lỗi. Thí nghiệm này
mô phỏng cả hai điều đó (`results/noise/`), trên CV 5-fold của Development với tham số cố định, không dùng Final Test.

- Mạch được transpile về các cổng gốc cx, rz, sx, x trên 4 qubit nối thành hàng. Nhiễu khử cực (depolarizing)
  sau mỗi cổng và lỗi đọc. Mức thấp: 0,3% sau mỗi CNOT (như ví dụ trong bài giảng W8), 0,03% sau mỗi cổng 1 qubit,
  đọc nhầm 1%. Mức cao: 1%, 0,1% và 3%.
- Bộ mô phỏng nhiễu (`src/noise_simulation.py`) được đối chiếu với Qiskit Aer, sai khác dưới 1e-15.
- VQC được huấn luyện không nhiễu như kết quả chính; shots và nhiễu chỉ áp dụng khi dự đoán.

Macro-F1 trung bình 5 fold (trong ngoặc: recall nhóm Cao):

| Điều kiện | QSVC | QSVC-product | VQC |
|---|---|---|---|
| Chính xác (statevector) | 0,546 (0,72) | 0,559 (0,71) | 0,528 (0,74) |
| 256 shots | 0,387 | 0,404 | 0,531 |
| 1.024 shots | 0,424 | 0,452 | 0,527 |
| 4.096 shots | 0,465 | 0,505 | 0,527 |
| 16.384 shots | 0,517 | 0,553 | 0,529 |
| 65.536 shots | 0,558 | 0,575 | 0,528 |
| Nhiễu thấp, đo vô hạn lần | 0,598 (0,35) | 0,592 (0,61) | 0,534 |
| Nhiễu thấp + 1.024 shots | 0,404 | 0,439 | 0,532 |
| Nhiễu cao, đo vô hạn lần | 0,547 (0,16) | 0,591 (0,27) | 0,542 |
| Nhiễu cao + 1.024 shots | 0,412 | 0,413 | 0,547 |

- QSVC rất nhạy với shots. Hàm nhân bị tập trung: trung bình phần tử ngoài đường chéo là 0,89 (QSVC) và 0,96
  (QSVC-product), nên thông tin nằm ở những khác biệt rất nhỏ giữa các bệnh nhân. Sai số shots trên từng phần tử
  nhỏ (khoảng 0,007 ở 1.024 shots) nhưng cộng dồn trên ma trận 3.462 × 3.462 thành sai số lớn (chuẩn phổ khoảng 1,2),
  lấn át các hướng thông tin nhỏ. QSVC cần khoảng 65.536 shots mỗi mạch, QSVC-product khoảng 16.384 shots,
  mới về lại mức của bản chính xác.
- Nhiễu cổng, khi đo vô hạn lần, gần như chỉ co hàm nhân lại: K_nhiễu ≈ 0,90·K + 0,01 với QSVC ở mức thấp.
  Vì đường chéo của hàm nhân huấn luyện đặt bằng 1 (mặc định của Qiskit), việc co này tương đương thêm điều chuẩn hóa
  (regularization) cho SVM. Macro-F1 vì vậy có thể tăng, nhưng recall nhóm Cao giảm mạnh (QSVC: 0,72 → 0,35 → 0,16).
  Huấn luyện bằng hàm nhân chính xác được co theo cùng cách cho đúng Macro-F1 này (cột `macro_f1_exact_kernel_shrunk`),
  nên đây không phải là nhiễu giúp mô hình tốt hơn.
- Khi có cả nhiễu và 1.024 shots, Macro-F1 của QSVC và QSVC-product giảm còn khoảng 0,40–0,44.
- Dự đoán của VQC gần như không đổi (chênh lệch trung bình không quá 0,02), vì mỗi bệnh nhân chỉ cần ước lượng 4 giá trị kỳ vọng ⟨Z⟩.
  Phần tốn kém của VQC là huấn luyện, chưa được mô phỏng với nhiễu.

Ước tính chi phí trên phần cứng nếu huấn luyện trên toàn bộ Development (4.328) và dự đoán Final Test (1.082),
chưa tính tinh chỉnh tham số (`results/noise/circuits.csv`, `hardware_cost.csv`):

| Mô hình | CNOT mỗi mạch (sau transpile) | Tỉ lệ lần chạy không lỗi, (1 − 0,3%)^CNOT | Số mạch | Tổng số lần chạy với 1.000 shots |
|---|---|---|---|---|
| QSVC | 24 | 93% | 14,0 triệu | 1,4 × 10¹⁰ |
| QSVC-product | 0 | 100% | 14,0 triệu | 1,4 × 10¹⁰ |
| VQC | 30 | 91% | 45,4 triệu | 4,5 × 10¹⁰ |

- Mạch hàm nhân của QSVC gồm U(x_i) rồi U(x_j)†; mỗi cặp bệnh nhân cần một mạch riêng.
- VQC cần 1 + 2 × 32 mạch cho mỗi bệnh nhân ở mỗi lần tính gradient (parameter-shift), trung bình 161 lần tính.
- Nếu dùng số shots cần thiết ở trên, QSVC cần khoảng 9,2 × 10¹¹ lần chạy và QSVC-product khoảng 2,3 × 10¹¹.

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
notebooks/      01_tong_hop_ket_qua.ipynb (bảng, biểu đồ), 02_demo.ipynb (demo trực tiếp)
results/        kết quả các thí nghiệm (xem bảng bên dưới); hình cho báo cáo và slide trong results/figures/
scripts/        các lệnh chạy thí nghiệm
src/            mã nguồn chính
```

| Module trong `src/` | Nội dung |
|---|---|
| `dataset.py` | Đọc dữ liệu, nhãn, tập chia; lọc nhóm bệnh nhân |
| `preprocessing.py` | Mã hóa đặc trưng dược di truyền, phục hồi VKORC1 từ SNP liên kết, điền thiếu bằng hồi quy |
| `policy_preprocessing.py` | Tiền xử lý 10 cột được duyệt, dùng khi tạo tập chia |
| `quantum_models.py` | QSVC (fidelity quantum kernel) và VQC (parameter-shift, L-BFGS-B) |
| `noise_simulation.py` | Mô phỏng mạch có nhiễu bằng ma trận mật độ: transpile, nhiễu khử cực, lỗi đọc, shots |
| `evaluation.py` | Danh sách mô hình và lưới tham số, chỉ số đánh giá, CV, Final Test, paired bootstrap |
| `baselines.py` | MLP có cân bằng lớp, công thức IWPC |

## Cài đặt

Các kết quả trong repo được chạy bằng Python 3.13 với các phiên bản thư viện trong `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.quantum_models     # đối chiếu với Qiskit, cần in ra 4 dòng OK
python -m src.noise_simulation   # đối chiếu bộ mô phỏng nhiễu với Qiskit Aer, cần in ra 4 dòng OK
```

## Notebook và demo

- `notebooks/01_tong_hop_ket_qua.ipynb`: đọc kết quả trong `results/`, vẽ bảng và biểu đồ chính, lưu hình vào
  `results/figures/`. Không huấn luyện lại, chạy trong vài giây.
- `notebooks/02_demo.ipynb`: demo để trình bày: mã hóa bệnh nhân thành góc quay, vẽ mạch lượng tử, hàm nhân
  lượng tử và ảnh hưởng của shots, huấn luyện nhanh trên một fold của Development, dự đoán cho bệnh nhân nhập tay.
  Chạy trong khoảng 1–2 phút, không dùng Final Test.

Mở bằng VS Code hoặc Jupyter (`pip install notebook`, rồi `jupyter notebook notebooks/`).

## Chạy thí nghiệm

Chạy từ thư mục gốc của repo.

| Bước | Lệnh | Kết quả |
|---|---|---|
| Kiểm tra dữ liệu | `python -m scripts.check_dataset` | `results/data_validation/dataset_check.json` |
| Thống kê tiền xử lý | `python -m scripts.analyze_preprocessing` | `results/data_validation/preprocessing_report.txt` |
| Phân loại (CV) | `python -m scripts.run_classification_cv --rep domain --k 4` | `results/classification_cv/` |
| MLP cân bằng lớp ở các thí nghiệm | thêm `--models MLP-balanced` vào lệnh CV, learning curve, số qubit; CV lặp: `--models MLP-balanced --tag mlp_balanced` | như trên |
| Hồi quy liều | `python -m scripts.run_regression_cv --k 4` | `results/regression_cv/` |
| Learning curve | `python -m scripts.run_learning_curve_full` | `results/learning_curve_full/` |
| Số qubit, feature map | `python -m scripts.run_qubit_ablation --part both` | `results/ablation/` |
| Chi phí tính toán | `OMP_NUM_THREADS=1 python -m scripts.run_cost_analysis` | `results/cost/` |
| Shots và nhiễu | `python -m scripts.run_noise_analysis` | `results/noise/` |
| Phân tích theo nhóm | `python -m scripts.run_subgroup_analysis --part subgroup` (hoặc `sensitivity`, `selection`) | `results/subgroup_analysis/` |
| CV lặp 5×5, Nadeau–Bengio | `python -m scripts.run_repeated_cv` rồi `python -m scripts.run_repeated_cv --summarize` | `results/repeated_cv/` |
| Bỏ ca liều ≥ 100, IWPC có cột HOLD | `python -m scripts.run_protocol_checks --part outliers` (hoặc `iwpc_hold`) | `results/subgroup_analysis/`, `results/regression_cv/` |
| Final Test | `python -m scripts.evaluate_final_test` | `results/final_test/` |
| Dự đoán Final Test, ma trận nhầm lẫn | `python -m scripts.evaluate_final_test_predictions` | `results/final_test/` |

Các bước tạo nhãn (`scripts.prepare_labels`) và tập chia (`scripts.prepare_splits`) đã được chạy;
kết quả nằm trong `data/interim/` và `data/splits/`. Hai script này không ghi đè dữ liệu đã có.
Script Final Test cũng dừng lại nếu kết quả đã tồn tại, để tránh đánh giá lại trên tập kiểm thử.

Các thư mục kết quả khác:
- `results/learning_curve/`: learning curve trên nhóm 600 người (bản có và không tinh chỉnh tham số).
- `results/benchmarks/`: thí nghiệm ban đầu trên các fold cố định, gồm so sánh biểu diễn domain, PCA,
  mutual information, PLS và đủ 10 cột. Kết quả này được tạo bằng phiên bản code trước khi sắp xếp lại repo;
  script tạo ra chúng không còn trong repo.
- `results/exploratory/`: các phân tích thăm dò, cũng từ phiên bản code trước.

## Thiết kế đánh giá

- 5.410 bệnh nhân đạt liều ổn định. Nhóm liều: Thấp (≤ 21 mg/tuần), Vừa (21–49), Cao (≥ 49).
- Tập Development (4.328) và Final Test (1.082) được chia ngẫu nhiên 80/20, phân tầng theo nhóm liều.
- CV 5-fold trên Development, phân tầng theo trung tâm thu thập (Project Site) để mỗi fold có tỉ lệ
  bệnh nhân của các trung tâm tương tự nhau. Siêu tham số chọn bằng inner 3-fold theo Macro-F1.
- Mọi bước tiền xử lý có học tham số chỉ được fit trên phần huấn luyện của từng fold.
- Chủng tộc, thuốc dùng kèm và mã trung tâm không được dùng làm đặc trưng.

## Hạn chế

- Mô hình lượng tử chưa chạy trên phần cứng thật. Shots và nhiễu chỉ được mô phỏng với mô hình nhiễu đơn giản
  (nhiễu khử cực và lỗi đọc), và VQC chưa được huấn luyện có nhiễu.
- Final Test đã được dùng nhiều lần: một lần trong quá trình phát triển, các lần sau để bổ sung mô hình và tái tạo
  dự đoán với pipeline không đổi (chi tiết trong `docs/final_configuration.md`). Vì vậy CV trên Development là
  bằng chứng chính.
- Nhiều mô hình chọn tham số ở mép lưới (ví dụ QSVC chọn bandwidth nhỏ nhất 0,05 ở 18/25 fold của CV lặp,
  SVM chọn gamma nhỏ nhất 0,01 ở 19/25 fold); lưới rộng hơn có thể thay đổi kết quả của cả hai phía.
- Biểu diễn 4 đặc trưng điền trung vị cho kiểu gen bị thiếu và không giữ thông tin "không rõ kiểu gen". Trên cùng
  bộ fold cố định, LR dùng đủ 10 cột đạt 0,581 so với 0,548 khi dùng 4 cột (`results/benchmarks/`).
- Final Test chỉ có kết quả VQC với seed 0; độ ổn định theo seed được đánh giá trên Development.
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
