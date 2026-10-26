# Warfarin QML — Research decisions and assumptions

**Current decision:** Research-approved with a documented temporal assumption; **not clinical approval**. This document combines five source documents without removing their historical wording.

## Document map

| Section | Original document | How to interpret |
|---|---|---|
| Prediction time contract | `prediction_time_contract.md` | Policy definition; still applies |
| Research core proposal (historical) | `research_core_decision.md` | Historical PENDING stage; superseded by amendment |
| Temporal evidence review conclusion | `temporal_evidence_conclusion.md` | UNVERIFIED evidence; continues to apply |
| Outlier decision | `dose_outlier_decision.md` | Retain for primary and sensitivity analysis |
| Research protocol amendment — effective decision | `research_protocol_amendment.md` | CURRENT retrospective research approval; overrides old approval status |

**Policy of record:** `configs/feature_policy.yaml`. Ten research features are allowed, temporal evidence remains `UNVERIFIED`, high-dose outliers are retained for a separately reported sensitivity analysis. The older PENDING statements below are retained as review history, not current research approval.

---

## Source: prediction_time_contract.md
*Policy definition; still applies*

# Prediction Time Contract — Warfarin QML

## 1. Prediction Objective

Mô hình Quantum Machine Learning (QML) được xây dựng nhằm phân loại nhóm liều Warfarin ổn định của bệnh nhân thành ba nhóm: Low, Intermediate và High.

Mục tiêu là dự đoán nhóm liều dựa trên những thông tin lâm sàng và di truyền có sẵn trước khi liều Warfarin ổn định được xác định.

## 2. Prediction Time Definition

Thời điểm dự đoán (prediction time) được định nghĩa là thời điểm mô hình được sử dụng để hỗ trợ dự đoán nhóm liều Warfarin, trước khi biết liều điều trị ổn định của bệnh nhân.

Chỉ những đặc trưng đã có sẵn tại thời điểm này mới đủ điều kiện được xem xét đưa vào tập đầu vào X.

## 3. Feature Availability Policy

Các nhóm thông tin có thể được xem xét bao gồm:

- Thông tin nhân khẩu học: Gender, Age.
- Thông tin cơ thể: Height, Weight.
- Thông tin bệnh nền và tiền sử điều trị, nếu có sẵn trước thời điểm dự đoán.
- Thông tin thuốc đang sử dụng, nếu xác minh được thời điểm ghi nhận.
- Thông tin di truyền CYP2C9 và VKORC1, nếu kết quả xét nghiệm có sẵn trước thời điểm dự đoán.

Những feature chưa xác minh được thời điểm thu thập phải được đánh dấu `UNVERIFIED` và không được phê duyệt vào core cho đến khi có căn cứ phù hợp.

## 4. Target and Leakage Restrictions

Các cột sau không được sử dụng làm đầu vào X:

| Column | Reason |
|---|---|
| Subject Reached Stable Dose of Warfarin | Chỉ dùng xác định bệnh nhân đủ điều kiện |
| Therapeutic Dose of Warfarin | Nguồn dữ liệu để tạo nhãn y |
| INR on Reported Therapeutic Dose of Warfarin | Thông tin hậu nghiệm tại liều điều trị đã được báo cáo |

Các biến được suy ra trực tiếp từ target hoặc chỉ hình thành sau thời điểm dự đoán cũng không được phép đưa vào X.

## 5. Dataset Limitations

IWPC là bộ dữ liệu tổng hợp hồi cứu. Metadata hiện được sử dụng trong project chưa cung cấp đầy đủ bằng chứng về thời điểm thu thập của tất cả các đặc trưng.

Vì vậy, tính sẵn có tại thời điểm dự đoán của một số biến vẫn là giả định cần kiểm chứng, không được xem là đã xác nhận.

Dự án phân biệt rõ:

- Feature đã có căn cứ phù hợp để sử dụng.
- Feature có tiềm năng nhưng còn thiếu bằng chứng.
- Feature không được sử dụng do chứa target, thông tin hậu nghiệm hoặc không phù hợp với core.

## 6. Implementation Rules

1. Không đưa nguồn tạo target hoặc thông tin hậu nghiệm vào X.
2. Không tự động phê duyệt feature chỉ vì có tương quan với nhãn.
3. Không dùng final test để lựa chọn feature.
4. Các thao tác imputation, scaling, encoding và feature reduction phụ thuộc dữ liệu phải được fit trên training fold ở GĐ4–GĐ5.
5. Giữ lại lý do lựa chọn hoặc loại bỏ từng feature trong `feature_policy.yaml` và `leakage_audit.md`.

## 7. Status

**Status:** Proposed prediction-time contract, pending final feature policy approval.

Tài liệu này xác định giả định thiết kế phục vụ Feature Policy và Leakage Audit. Nó không chứng minh rằng tất cả feature trong IWPC đã được thu thập đúng thời điểm dự đoán.

---

## Source: research_core_decision.md
*Historical PENDING stage; superseded by amendment*

# Research Core Feature Decision — Warfarin QML

> **LƯU Ý (bản lịch sử):** Văn bản này ghi nhận trạng thái đề xuất tại thời điểm rà soát. Quyết định áp dụng hiện hành là `docs/research_protocol_amendment.md` và `configs/feature_policy.yaml`. Bằng chứng temporal vẫn `UNVERIFIED`.


## 1. Objective

Xác định danh sách feature ứng viên phục vụ nghiên cứu phân loại nhóm liều Warfarin ổn định bằng Quantum Machine Learning.

Tài liệu này ghi nhận lựa chọn feature ở mức nghiên cứu, chưa xác nhận tính phù hợp để triển khai dự đoán lâm sàng.

## 2. Prediction Scenario

Mô hình hướng đến việc dự đoán nhóm liều Warfarin ổn định từ thông tin lâm sàng và di truyền có sẵn trước khi liều ổn định được xác định.

Thời điểm dự đoán tuân theo `docs/prediction_time_contract.md`.

## 3. Proposed Research Core

Danh sách 10 feature ứng viên:

1. Gender
2. Age
3. Height (cm)
4. Weight (kg)
5. Diabetes
6. Congestive Heart Failure and/or Cardiomyopathy
7. Valve Replacement
8. Current Smoker
9. CYP2C9 consensus
10. VKORC1 -1639 consensus

**Selection status:** RESEARCH_CANDIDATE

**Temporal evidence:** UNVERIFIED

**Approval status:** PENDING

Không feature nào được xem là đã phê duyệt chỉ dựa trên danh sách này.

## 4. Scientific Rationale

Các biến nhân khẩu học, chỉ số cơ thể, thông tin lâm sàng và genotype được lựa chọn vì có tiềm năng liên quan đến nhu cầu liều Warfarin.

Nghiên cứu IWPC năm 2009 và các hướng dẫn dược di truyền cung cấp cơ sở để xem xét nhiều nhóm biến trong danh sách.

Tuy nhiên, việc một biến có ý nghĩa về mặt nghiên cứu không đủ để chứng minh rằng nó được ghi nhận trước thời điểm dự đoán trong dataset đang sử dụng.

## 5. Exclusion and Hold Policy

Không đưa các biến sau vào X:

- `Therapeutic Dose of Warfarin`: nguồn tạo target.
- `Subject Reached Stable Dose of Warfarin`: điều kiện lọc bệnh nhân.
- `INR on Reported Therapeutic Dose of Warfarin`: thông tin hậu nghiệm.

Các cột định danh, Project Site, genotype gốc/QC và VKORC1 consensus bổ sung không thuộc core đề xuất ban đầu vì các lý do đã ghi trong Feature Policy Review.

Những feature đang HOLD vì thiếu bằng chứng về thời điểm ghi nhận không được đưa vào training pipeline trước khi hoàn thành bước xác minh.

## 6. Conditions for Approval

Một feature chỉ được phê duyệt vào core khi:

1. Có mô tả rõ ý nghĩa và cách hình thành.
2. Không chứa target hoặc thông tin được suy ra trực tiếp từ target.
3. Có đủ căn cứ cho tính sẵn có tại thời điểm dự đoán theo protocol.
4. Có quy tắc xử lý missing và encoding có thể tái lập.
5. Không được lựa chọn dựa trên hiệu năng final test.

Những điều kiện chưa đáp ứng phải được ghi nhận là unresolved, không tự động coi là đã đạt.

## 7. Feature Reduction Policy

Số feature ban đầu không đồng nghĩa số qubit.

Đối với các feature đã được phê duyệt, GĐ4–GĐ5 sẽ thực hiện preprocessing và giảm chiều xuống k=4 hoặc k=2 theo Guide v4.

Mọi phép biến đổi cần học từ dữ liệu phải được fit trên training fold. Không sử dụng final test để chọn feature hoặc tham số giảm chiều.

## 8. Current Decision

**Research candidates:** 10

**Temporally verified core features:** Chưa xác nhận.

**Next action:** Thu thập và đánh giá bằng chứng về temporal availability, cập nhật quyết định cho từng feature, sau đó xây dựng `feature_policy.yaml` và `leakage_audit.md`.

Tài liệu này không cấp phép sử dụng các feature PENDING trong huấn luyện.

---

## Source: temporal_evidence_conclusion.md
*UNVERIFIED evidence; continues to apply*

# Temporal Evidence Conclusion — Warfarin QML

> **LƯU Ý (bản lịch sử):** Văn bản này ghi nhận trạng thái đề xuất tại thời điểm rà soát. Quyết định áp dụng hiện hành là `docs/research_protocol_amendment.md` và `configs/feature_policy.yaml`. Bằng chứng temporal vẫn `UNVERIFIED`.


## 1. Review Results

- Total candidate features: 10
- Scientific context reviewed: 10
- Temporal evidence verified: 0
- Approval status: 10 PENDING

## 2. Evidence Assessment

IWPC Metadata mô tả ý nghĩa và quy tắc tạo các feature, bao gồm hai cột genotype consensus.

Nghiên cứu IWPC năm 2009 hỗ trợ cơ sở khoa học cho việc dự đoán liều Warfarin từ dữ liệu lâm sàng và di truyền.

Tuy nhiên, nguồn hiện có chưa xác minh đầy đủ thời điểm sẵn có của các feature tại từng bản ghi trong dataset đang sử dụng.

## 3. Proposed Research Scenario

Đề xuất nghiên cứu hồi cứu nhằm so sánh QML và classical ML trên cùng tập dữ liệu IWPC.

Việc sử dụng các feature ứng viên sẽ phụ thuộc vào sự chấp thuận của protocol đối với giả định rằng những thông tin này có sẵn trước khi liều ổn định được xác định.

Không được diễn giải kết quả đánh giá hồi cứu là bằng chứng về khả năng triển khai lâm sàng thực tế.

## 4. Approval Gate

**Current status: PENDING PROTOCOL REVIEW**

- Chưa phê duyệt feature nào dựa trên temporal evidence.
- Chưa sử dụng các feature PENDING trong training pipeline.
- Chưa thay đổi trạng thái UNVERIFIED thành VERIFIED.
- Cần có quyết định của nhóm và giảng viên hướng dẫn về kịch bản nghiên cứu và các giả định được phép sử dụng.

## 5. Next Deliverables

Sau khi thống nhất protocol:

- Hoàn thiện `feature_policy.yaml`.
- Hoàn thiện `leakage_audit.md`.
- Ghi rõ danh sách feature được phép dùng và những giới hạn của nghiên cứu.

---

## Source: dose_outlier_decision.md
*Retain for primary and sensitivity analysis*

# Therapeutic Dose Outlier Decision — Warfarin QML

**Stage:** GĐ3 — Label Construction & Leakage Audit

**Status:** RESEARCH APPROVED — Retain with sensitivity analysis

## 1. Audit Findings

Dataset IWPC ban đầu có 5.700 bản ghi. Sau khi áp dụng điều kiện stable dose và therapeutic dose hợp lệ, có 5.410 bệnh nhân đủ điều kiện sơ bộ.

Trong đó, có 21 bệnh nhân có therapeutic dose ≥100 mg/tuần.

Cả 21 bệnh nhân đều được đánh dấu đã đạt stable dose. Liều lớn nhất được ghi nhận là 315 mg/tuần.

Hiện chưa có bằng chứng để kết luận những giá trị này là lỗi nhập liệu.

## 2. Proposed Handling Policy

Đề xuất giữ nguyên 21 trường hợp liều cao trong tập dữ liệu chính, với các điều kiện:

- Không tự động thay đổi hoặc loại bỏ therapeutic dose.
- Không tự động điều chỉnh dấu thập phân.
- Giữ nguyên nhãn được tạo theo ngưỡng Low ≤21, Intermediate >21 và <49, High ≥49 mg/tuần.
- Duy trì danh sách 21 trường hợp để truy vết.
- Không đưa therapeutic dose hoặc cờ đánh dấu outlier của target vào tập feature X.

Việc giữ mẫu là quyết định nghiên cứu tạm thời, không đồng nghĩa xác nhận tất cả giá trị đều chính xác về mặt lâm sàng.

## 3. Sensitivity Analysis

Đề xuất thực hiện phân tích độ nhạy ở giai đoạn đánh giá nhằm xác định mức ảnh hưởng của các trường hợp liều cao.

Phân tích chính sử dụng tập dữ liệu theo chính sách giữ mẫu đã chốt.

Có thể bổ sung phân tích độ nhạy sử dụng tập dữ liệu không bao gồm các trường hợp có therapeutic dose ≥100 mg/tuần, với điều kiện thiết kế và cách xử lý dữ liệu được xác định trước, không dùng final test để lựa chọn phương án có kết quả tốt hơn.

Kết quả phân tích độ nhạy được báo cáo riêng, không thay thế kết quả chính một cách hậu nghiệm.

## 4. Decision Status

**Research decision:** RETAIN_WITH_SENSITIVITY_ANALYSIS

**Approval:** APPROVED FOR RETROSPECTIVE STUDY — see `docs/research_protocol_amendment.md`

**Raw data modification:** NONE

**Current labeled sample count:** 5.410 (preliminary)

## 5. Supporting Files

- `docs/dose_quality_summary.csv`
- `docs/dose_outlier_cases.csv`
- `docs/eligibility_summary.csv`
- `data/interim/labels.csv`
- `docs/class_balance.csv`

---

## Source: research_protocol_amendment.md
*CURRENT retrospective research approval; overrides old approval status*

# Research Protocol Amendment — Warfarin QML

**Status:** SELF-APPROVED FOR RETROSPECTIVE RESEARCH (reconstructed from the user's confirmed decisions in this conversation)

**Dataset:** IWPC, 5,700 raw records and 68 columns

**User-provided source page:** https://www.clinpgx.org/downloads

## Research feature decision

The researcher chose to use the 10 features listed under `approved_X_columns` in `configs/feature_policy.yaml` for retrospective QML/classical ML comparison. The study explicitly assumes feature availability at prediction time without claiming it has been demonstrated for individual IWPC records.

**Approval:** RESEARCH_APPROVED_WITH_TEMPORAL_ASSUMPTION  
**Temporal evidence:** UNVERIFIED  
**Clinical deployment:** NOT APPROVED

Features: Gender; Age; Height (cm); Weight (kg); Diabetes; Congestive Heart Failure and/or Cardiomyopathy; Valve Replacement; Current Smoker; CYP2C9 consensus; VKORC1 -1639 consensus.

## Outlier decision

Retain the 21 records with reported therapeutic dose >=100 mg/week in the primary, preliminary eligible cohort (n=5,410). Record a pre-specified sensitivity analysis excluding these records, reported separately. This decision does not verify that extreme doses are clinically accurate. Never include target-derived flags, dose, or post-outcome INR among the model features.

**Outlier policy:** RETAIN_WITH_SENSITIVITY_ANALYSIS

## Reproducibility and leakage restrictions

Use stratified 20% locked test split, with fold-local imputation/encoding/scaling/PCA and no final-test tuning. Restrict X to approved columns, compare methods under the locked protocol, and treat all results as retrospective research, not medical dosing guidance. 

This file was reconstructed for a CLEANED ARCHIVE because the uploaded ZIP contained an older DRAFT policy, although the user previously confirmed the approved research policy tests passed on their local machine. If the user's local copy is newer, compare before replacing it.

---
