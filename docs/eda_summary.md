# IWPC Dataset — Exploratory Data Analysis & Quality Audit Summary

**Project:** Phân loại nhóm liều Warfarin bằng Quantum Machine Learning

**Stage:** GĐ2 — Data Collection & Audit

**Dataset:** Warfarin Consortium Combined Data Set (IWPC)

**Raw file:** `PS206767-553247439.xls`

**Status:** Audit tổng hợp, còn các vấn đề cần theo dõi ở GĐ3.

---

## 1. Dataset Overview

Dữ liệu được sử dụng từ International Warfarin Pharmacogenetics Consortium (IWPC), bao gồm các đặc trưng lâm sàng, thuốc sử dụng, thông tin di truyền CYP2C9/VKORC1 và liều Warfarin được báo cáo.

File Excel có hai sheet:

- `Metadata`: Mô tả các cột và quy ước dữ liệu.
- `Subject Data`: Chứa thông tin của các bệnh nhân.

Kết quả kiểm kê:

| Nội dung | Kết quả |
|---|---:|
| Số bản ghi | 5.700 |
| Số cột | 68 |
| Số sheet | 2 |
| Dòng trùng hoàn toàn | 0 |
| Subject ID trùng | 0 |
| Sample ID trùng | 0 |

File raw được giữ nguyên. Thông tin nguồn, phiên bản, quyền sử dụng và SHA-256 được quản lý trong `source_manifest.json`.

Data Dictionary đã được xây dựng cho 68 cột, bao gồm tên cột, mô tả, kiểu dữ liệu, đơn vị và thống kê missing.

## 2. Missing Values Audit

Kết quả kiểm tra:

| Nội dung | Kết quả |
|---|---:|
| Tổng số cột | 68 |
| Cột có missing | 63 |
| Cột missing >50% | 33 |
| Cột missing >90% | 12 |

Các đặc trưng quan trọng:

| Feature | Missing count | Missing rate |
|---|---:|---:|
| Age | 42 | 0,74% |
| Height (cm) | 1.146 | 20,11% |
| Weight (kg) | 287 | 5,04% |
| CYP2C9 consensus | 144 | 2,53% |
| VKORC1 -1639 consensus | 1.499 | 26,30% |
| Target INR | 4.441 | 77,91% |
| Therapeutic Dose of Warfarin | 172 | 3,02% |

### Complete-case Impact

Nếu chỉ giữ những bản ghi có đầy đủ dữ liệu của các nhóm feature đang xét:

| Nhóm feature | Mẫu còn lại | Mẫu mất | Tỷ lệ mất |
|---|---:|---:|---:|
| CYP2C9 + VKORC1 | 4.144 | 1.556 | 27,30% |
| Age, Height, Weight, CYP2C9, VKORC1 | 3.620 | 2.080 | 36,49% |

**Nhận xét:**

Dataset có mức độ thiếu dữ liệu đáng kể, đặc biệt ở VKORC1 -1639 và một số cột QC.

Complete-case deletion có thể khiến dataset mất 27,30%–36,49% số mẫu raw trong hai kịch bản đã kiểm tra.

Vì vậy, không áp dụng complete-case deletion mặc định. Phương pháp xử lý missing được quyết định trong preprocessing ở GĐ4 và chỉ fit trên training fold.

## 3. Duplicate Audit

Kiểm tra trùng lặp trên toàn bộ raw dataset cho kết quả:

- Duplicate full rows: 0.
- Duplicate Subject ID: 0.
- Duplicate Sample ID: 0.

**Nhận xét:**

Không phát hiện trùng lặp theo ba tiêu chí đã kiểm tra. Chưa cần thực hiện thao tác loại bản ghi trùng.

## 4. Therapeutic Dose & Outlier Audit

Kết quả kiểm tra `Therapeutic Dose of Warfarin`:

| Nội dung | Kết quả |
|---|---:|
| Missing dose | 172 |
| Non-numeric dose ngoài missing | 0 |
| Non-positive dose | 0 |
| Dose ≥100 mg/tuần | 21 |
| Minimum positive dose | 2,1 mg/tuần |
| Maximum dose | 315 mg/tuần |

Cả 21 trường hợp có liều ≥100 mg/tuần đều được đánh dấu đã đạt stable dose.

**Nhận xét:**

Có 21 trường hợp liều cao cần được rà soát. Tuy nhiên, giá trị cao không tự động đồng nghĩa dữ liệu sai.

Đặc biệt, trường hợp liều 315 mg/tuần cần xác minh nguồn nếu có thể, vì một lỗi vị trí dấu thập phân giả định có thể thay đổi nhãn phân loại.

Ở GĐ2, nhóm không tự động chỉnh sửa hoặc loại bỏ các outlier. Danh sách trường hợp cần rà soát được lưu trong `dose_outlier_cases.csv`.

Các quyết định xử lý và phân tích độ nhạy, nếu cần, sẽ được ghi nhận trong GĐ3.

## 5. Clinical Data Quality

### 5.1. Age

Age được lưu theo nhóm tuổi, ví dụ `60 - 69`, thay vì tuổi chính xác.

Kết quả:

- Số bản ghi thiếu: 42.
- Số giá trị không đúng định dạng đã kiểm tra: 0.

Các giá trị tuổi không thiếu phù hợp với quy ước định dạng đã dùng trong audit.

### 5.2. Height and Weight

| Chỉ số | Height (cm) | Weight (kg) |
|---|---:|---:|
| Minimum | 124,968 | 30 |
| Median | 167,894 | 75 |
| Maximum | 202 | 237,7 |
| IQR outlier count | 8 | 118 |
| Non-positive count | 0 |
| Non-numeric count | 0 |

**Nhận xét:**

Không phát hiện chiều cao hoặc cân nặng không dương hay không chuyển được sang số trong các giá trị hiện có.

Tuy nhiên, có 8 outlier chiều cao và 118 outlier cân nặng theo quy tắc IQR.

Đây là các cảnh báo thống kê, không phải bằng chứng xác định dữ liệu sai. Không tự động xóa các trường hợp này.

## 6. Categorical Data Quality

### 6.1. Indication for Warfarin Treatment

Kết quả:

- Missing values: 698.
- Bản ghi chứa nhiều mã phân cách bằng dấu `;`: 648.
- Bản ghi có cú pháp không chuẩn: 104.

Trong 104 giá trị không chuẩn:

- `1 or 2`: 103 bản ghi.
- `1,2`: 1 bản ghi.

**Nhận xét:**

Trường Indication có thể chứa nhiều mã trong một ô.

Các giá trị `1 or 2` và `1,2` cần được kiểm tra riêng. Không tự động chuyển chúng thành `1; 2`, vì biểu diễn lựa chọn không chắc chắn có thể khác với biểu diễn nhiều chỉ định đồng thời.

### 6.2. Binary Clinical & Medication Flags

Các cột nhị phân được kiểm tra không phát hiện giá trị ngoài tập {0, 1} ở những ô không missing.

Trường `Medications` dạng chuỗi tổng hợp đã được loại khỏi phép kiểm tra nhị phân.

Các cờ thuốc và thông tin liên quan vẫn cần kiểm tra tính phù hợp và thời điểm thu thập trước khi được đưa vào tập X chính thức.

## 7. Genotype Data Quality

### 7.1. CYP2C9 Consensus

| Genotype | Count |
|---|---:|
| *1/*1 | 4.157 |
| *1/*2 | 737 |
| *1/*3 | 498 |
| *2/*3 | 69 |
| *2/*2 | 56 |
| *3/*3 | 22 |
| Missing | 144 |

Có thêm năm nhóm genotype có tần suất dưới 10 mẫu:

- *1/*5: 6.
- *1/*11: 6.
- *1/*6: 3.
- *1/*13: 1.
- *1/*14: 1.

Tổng cộng có 17 bản ghi trong năm nhóm genotype hiếm.

**Nhận xét:**

CYP2C9 consensus có tỷ lệ missing tương đối thấp, khoảng 2,53%.

Các genotype hiếm được ghi nhận nhưng chưa gộp hoặc loại bỏ.

Genotype `*1/*14` cần được đối chiếu thêm với tài liệu mô tả allele của dataset.

### 7.2. VKORC1 Consensus

| VKORC1 Consensus | Missing rate |
|---|---:|
| -1639 | 26,30% |
| 497 | 66,96% |
| 1173 | 37,00% |
| 1542 | 38,23% |
| 3730 | 41,33% |
| 2255 | 55,51% |
| -4451 | 84,00% |

Đối với VKORC1 -1639:

- A/A: 1.485.
- A/G: 1.470.
- G/G: 1.246.
- Missing: 1.499.

**Nhận xét:**

Tỷ lệ missing giữa các cột VKORC1 consensus khác nhau đáng kể.

Nhóm cần xây dựng chính sách lựa chọn các cột genotype hợp lệ, tránh biểu diễn trùng lặp cùng một thông tin di truyền khi xây dựng Feature Policy ở GĐ3.

## 8. Metadata Quality Issues

Trong quá trình xây dựng Data Dictionary, có hai vấn đề cần ghi nhận:

**Target INR:** Metadata ghi đơn vị `mg/week`, trong khi INR là chỉ số không có đơn vị.

**Anti-fungal Azoles:** Phần mô tả trong Metadata có điểm chưa nhất quán liên quan đến metronidazole và danh sách thuốc được tính.

Các vấn đề này đã được ghi trong Data Dictionary. Không chỉnh sửa file raw.

## 9. Outstanding Issues & Handover to GĐ3

Các công việc cần tiếp tục:

1. Xác định điều kiện lọc các bệnh nhân đạt stable dose và có therapeutic dose hợp lệ.
2. Xác minh hoặc ghi nhận quyết định đối với 21 trường hợp liều ≥100 mg/tuần.
3. Tạo nhãn Low, Intermediate, High từ therapeutic dose bằng code và kiểm tra các ngưỡng biên.
4. Kiểm tra sự mất cân bằng giữa ba nhóm liều sau khi lọc mẫu.
5. Rà soát 104 giá trị Indication có cú pháp không chuẩn.
6. Hoàn thiện Feature Policy, đặc biệt các trường hậu nghiệm và genotype có thông tin trùng lặp.
7. Kiểm tra Data Leakage trước khi bàn giao dataset.
8. Giữ nguyên missing values của các feature để thành viên phụ trách GĐ4 xử lý trong pipeline.

## 10. Conclusion

Giai đoạn 2 đã thực hiện kiểm kê cấu trúc, missing values, duplicate records, outliers và chất lượng các nhóm đặc trưng lâm sàng, categorical, genotype của IWPC dataset.

Các kiểm tra đã phát hiện nhiều vấn đề cần xử lý ở những giai đoạn tiếp theo, đặc biệt là missingness, outlier của therapeutic dose, định dạng Indication và chính sách lựa chọn genotype.

Ở giai đoạn này, dữ liệu raw được giữ nguyên và chưa thực hiện imputation, encoding, feature reduction hoặc tạo nhãn phân loại.

Các kết quả audit cung cấp cơ sở để chuyển sang GĐ3 — Label Construction & Leakage Audit, với điều kiện các vấn đề còn tồn đọng được theo dõi và xử lý có ghi nhận quyết định.

---

## 11. Supporting Audit Files

- `source_manifest.json`
- `data_dictionary.csv`
- `missingness_report.csv`
- `complete_case_impact.csv`
- `duplicate_summary.csv`
- `dose_quality_summary.csv`
- `dose_outlier_cases.csv`
- `numeric_quality_checks.csv`
- `feature_quality_flags.csv`
- `clinical_numeric_summary.csv`
- `categorical_value_counts.csv`

> Note: CSV reports not retained as standalone files under `docs/` are preserved in `archive/supporting_materials.zip`. Core result CSVs and validated inputs remain in `docs/results/` and `docs/reference/`.
