# Data Leakage Audit — Warfarin QML

**Project:** Phân loại nhóm liều Warfarin bằng Quantum Machine Learning

**Stage:** GĐ3 — Label Construction & Leakage Audit

**Dataset:** International Warfarin Pharmacogenetics Consortium (IWPC)

**Status:** RESEARCH APPROVED WITH TEMPORAL ASSUMPTION — NOT FOR CLINICAL DEPLOYMENT

## 1. Audit Objective

Mục tiêu của Data Leakage Audit là xác định và ngăn chặn việc sử dụng thông tin không hợp lệ trong quá trình dự đoán nhóm liều Warfarin ổn định.

Quá trình audit tập trung vào:

- Target leakage: thông tin chứa trực tiếp hoặc được suy ra từ nhãn.
- Temporal leakage: thông tin chưa có tại thời điểm dự đoán.
- Feature redundancy: các cột trùng lặp hoặc biểu diễn cùng thông tin.
- Preprocessing leakage: thông tin từ validation/test được sử dụng trong quá trình xây dựng mô hình.

## 2. Prediction Scenario

Mô hình hướng đến dự đoán ba nhóm liều Warfarin ổn định dựa trên dữ liệu lâm sàng và di truyền có sẵn trước khi biết liều ổn định.

Thời điểm dự đoán và các giả định được mô tả trong `docs/research_decisions.md`.

IWPC là dữ liệu được tổng hợp phục vụ nghiên cứu hồi cứu. Metadata hiện có chưa xác minh đầy đủ thời điểm sẵn có của tất cả các feature tại từng bệnh nhân.

## 3. Label Construction

Dataset ban đầu có 5.700 bản ghi.

Các điều kiện lọc:

1. `Subject Reached Stable Dose of Warfarin = 1`.
2. `Therapeutic Dose of Warfarin` là giá trị số hữu hạn và lớn hơn 0.

Kết quả:

- Eligible preliminary records: 5.410.
- Excluded records: 290.

Quy tắc tạo nhãn:

| Label | Therapeutic Dose |
|---|---|
| Low | ≤21 mg/week |
| Intermediate | >21 và <49 mg/week |
| High | ≥49 mg/week |

Phân bố nhãn:

| Label | Count |
|---|---:|
| Low | 1.786 |
| Intermediate | 2.936 |
| High | 688 |
| Total | 5.410 |

Hàm tạo nhãn đã vượt qua 4 Unit Test, bao gồm kiểm tra ngưỡng 21 và 49 mg/tuần.

Các nhãn hiện mang tính sơ bộ vì 21 trường hợp liều ≥100 mg/tuần còn chờ xác minh.

## 4. Target and Post-outcome Leakage

Ba cột quan trọng không được đưa vào X:

| Column | Role | Decision |
|---|---|---|
| Subject Reached Stable Dose of Warfarin | Eligibility filter | EXCLUDE |
| Therapeutic Dose of Warfarin | Target source | EXCLUDE |
| INR on Reported Therapeutic Dose of Warfarin | Post-outcome information | EXCLUDE |

**Findings:**

`Therapeutic Dose of Warfarin` là nguồn tạo nhãn. Đưa cột này hoặc biến được suy ra trực tiếp từ nó vào X sẽ làm lộ đáp án.

`Subject Reached Stable Dose of Warfarin` chỉ được dùng để xác định tập bệnh nhân hợp lệ, không được sử dụng làm feature.

`INR on Reported Therapeutic Dose of Warfarin` phản ánh thông tin liên quan đến mức liều đã được báo cáo, không phù hợp với kịch bản dự đoán trước khi biết liều ổn định.

Các cột này bị loại khỏi X trong Feature Policy hiện hành.

## 5. Identifier and Non-core Features

Các cột PharmGKB Subject ID, PharmGKB Sample ID và Comments được loại khỏi core để tránh sử dụng thông tin định danh hoặc nội dung không phù hợp làm đặc trưng dự đoán.

`Project Site` cũng không được đề xuất sử dụng trong core ban đầu do nguy cơ mô hình học khác biệt riêng của các trung tâm nghiên cứu.

Những quyết định này không đồng nghĩa tất cả các cột đều chứa target leakage.

## 6. Temporal Leakage Review

Nhóm đã kiểm tra 10 feature ứng viên:

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

Kết quả rà soát:

| Status | Count |
|---|---:|
| CONTEXT_REVIEWED | 10 |
| UNVERIFIED | 10 |
| PENDING APPROVAL | 10 |
| APPROVED | 0 |

Metadata đã xác nhận ý nghĩa và quy tắc biểu diễn của các feature. Đối với CYP2C9 và VKORC1 consensus, Metadata mô tả việc tổng hợp genotype từ Original và QC mà không sử dụng trực tiếp Therapeutic Dose.

Tuy nhiên, bằng chứng hiện có chưa xác nhận đầy đủ thời điểm các giá trị được ghi nhận hoặc có sẵn để dự đoán tại từng bệnh nhân.

Theo temporal evidence nghiêm ngặt, chưa feature nào được xác minh thời điểm. Tuy nhiên, 10 feature đã được chấp thuận riêng cho nghiên cứu hồi cứu dưới giả định temporal theo `docs/research_decisions.md`.

## 7. Feature Redundancy Review

### 7.1. Genotype

IWPC chứa các cột genotype gốc, QC và consensus.

Nhóm đề xuất ưu tiên xem xét:

- CYP2C9 consensus.
- VKORC1 -1639 consensus.

Các cột genotype gốc/QC và sáu VKORC1 consensus bổ sung không được đưa vào core ban đầu.

Lý do là tránh biểu diễn dư thừa cùng locus hoặc sử dụng thêm locus khi chưa có cơ sở cần thiết.

Trùng lặp thông tin genotype không tự động được xem là data leakage.

### 7.2. Medication

Cột `Medications` dạng chuỗi có thể trùng thông tin với các cờ thuốc riêng biệt.

Nhóm chưa đưa các cột thuốc vào core khi chưa xác minh thời điểm ghi nhận.

Các thuốc tương tác như Amiodarone, Carbamazepine, Phenytoin và Rifampin/Rifampicin được giữ trong danh sách xem xét bổ sung khi có căn cứ phù hợp.

## 8. Data Quality Considerations

Các vấn đề được ghi nhận từ GĐ2:

- 21 trường hợp therapeutic dose ≥100 mg/tuần cần rà soát.
- 104 giá trị Indication có cú pháp không chuẩn.
- Một số đặc trưng lâm sàng và di truyền có missing đáng kể.
- Một số genotype CYP2C9 hiếm cần xem xét khi thiết kế encoding.

Các vấn đề này không tự động cấu thành data leakage.

Không tự động xóa outlier, biến missing thành 0, gộp genotype hiếm hoặc sửa các giá trị Indication chưa rõ ý nghĩa.

## 9. Preprocessing and Evaluation Leakage Prevention

Đối với GĐ4–GĐ5, nhóm cần bảo đảm:

1. Chỉ sử dụng những feature được phê duyệt trong Feature Policy.
2. Khóa final test theo protocol trước khi thực hiện lựa chọn mô hình.
3. Không sử dụng nhãn hoặc thông tin từ final test để chọn feature hay điều chỉnh hyperparameter.
4. Imputation, scaling, encoding có học tham số và PCA phải được fit trên training fold.
5. Validation fold chỉ được transform bằng pipeline đã fit trên training fold tương ứng.
6. Quantum models và classical baselines phải tuân theo quy trình đánh giá thống nhất của Guide v4.
7. Không sử dụng final test nhiều lần để tối ưu quyết định nghiên cứu.

Các nội dung trên là yêu cầu kiểm soát cho giai đoạn tiếp theo, chưa phải xác nhận pipeline đã thực hiện thành công.

## 10. Feature Policy Status

File cấu hình hiện hành: `configs/feature_policy.yaml`. Quyết định tại `docs/research_decisions.md`.

| Policy Field | Current Value |
|---|---|
| policy_status | RESEARCH_APPROVED_WITH_TEMPORAL_ASSUMPTION |
| training_allowed | true (retrospective research only) |
| research_training_allowed | true |
| clinical_deployment_allowed | false |
| approved_X_columns | 10 features |
| timing_evidence | UNVERIFIED |
| outlier_policy | RETAIN_WITH_SENSITIVITY_ANALYSIS |

Tổng cộng 10/68 cột được duyệt làm X trong nghiên cứu hồi cứu; 58 cột khác không được đưa vào X. Việc phê duyệt không xác nhận tính sẵn có về thời gian hoặc tính hợp lệ để kê đơn. Pipeline GĐ4 phải thực thi danh sách `approved_X_columns`.

## 11. Outstanding Issues and Handover Conditions

1. Temporal evidence cho cả 10 feature vẫn UNVERIFIED; chỉ sử dụng trong phạm vi nghiên cứu hồi cứu và khai báo giới hạn này.
2. Đánh giá độ nhạy 21 trường hợp liều cao, không hiệu chỉnh hay loại bỏ ngầm khỏi dữ liệu chính.
3. GĐ4 phải chỉ sử dụng 10 cột được phê duyệt và kiểm tra policy tại runtime.
4. Chốt và lưu stratified 20% final test split trước modeling, tránh mọi fit preprocessing trên validation/test.
5. Xác minh điều kiện quyền sử dụng/chia sẻ dữ liệu trước khi công bố hay đưa raw lên GitHub.

## 12. Conclusion

Các nguy cơ target leakage trực tiếp và post-outcome leakage đã được nhận diện và các cột liên quan bị cấm xuất hiện trong X. Đã đối chiếu nhãn và đã thực hiện một chính sách feature có thể đọc bằng YAML. Nhóm cho phép 10 feature trong nghiên cứu hồi cứu với giả định temporal **chưa xác minh** và không chấp thuận triển khai lâm sàng.

**Audit status: RESEARCH APPROVED WITH TEMPORAL ASSUMPTION — temporal evidence remains UNVERIFIED.**

## 13. Supporting Files

- `docs/eda_summary.md`
- `docs/reference/feature_policy_review.csv`
- `archive/supporting_materials.zip (docs/hold_review.csv)`
- `archive/supporting_materials.zip (docs/core_feature_candidates.csv)`
- `docs/reference/temporal_evidence_review.csv`
- `docs/research_decisions.md`
- `docs/research_decisions.md`
- `docs/research_decisions.md`
- `configs/feature_policy.yaml`
- `data/interim/labels.csv`
- `docs/results/class_balance.csv`
## 14. Research Approval Amendment

Quyết định hiện hành tại `docs/research_decisions.md` được ghi sau giai đoạn review ban đầu. Các thống kê PENDING trong bảng temporal ở mục 6 là trạng thái **bằng chứng**, không phải quyết định research approval cuối. 21 liều cao được giữ cho phân tích chính và phân tích độ nhạy, không được biến target thành feature.
