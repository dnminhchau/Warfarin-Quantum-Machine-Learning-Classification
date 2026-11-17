# Nguồn dữ liệu và giấy phép

**Dữ liệu:** Warfarin Consortium Combined Data Set, March 2008 (PharmGKB dataset PS206767),
International Warfarin Pharmacogenetics Consortium (IWPC), phân phối bởi PharmGKB/ClinPGx.

- Trang tải dữ liệu: https://www.clinpgx.org/downloads
- Trang IWPC: https://www.clinpgx.org/page/iwpc
- Chính sách sử dụng dữ liệu: https://www.clinpgx.org/page/dataUsagePolicy

## Giấy phép

Dữ liệu được cấp phép theo Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0):
https://creativecommons.org/licenses/by-sa/4.0/

| Điều kiện | Cách đáp ứng |
|---|---|
| Ghi nguồn và nêu rõ thay đổi | File này và mục "Dữ liệu và giấy phép" trong README |
| Dữ liệu phái sinh phát hành cùng giấy phép | Các file trong `data/interim/` và `data/splits/` phát hành theo CC BY-SA 4.0 |
| Không ngụ ý được PharmGKB/ClinPGx xác nhận | Repo không được PharmGKB/ClinPGx xác nhận |

Giấy phép CC BY-SA 4.0 áp dụng cho dữ liệu; mã nguồn có thể dùng giấy phép riêng.

## Thay đổi so với bản gốc

- `data/raw/PS206767-553247439.xls`: giữ nguyên; mã SHA-256 lưu trong `data/source_manifest.json`.
- `data/interim/labels.csv`: nhãn Low / Intermediate / High tạo từ cột "Therapeutic Dose of Warfarin".
- `data/splits/*.csv`: chỉ số dòng của các tập Development, Final Test và các fold.

## Trích dẫn

International Warfarin Pharmacogenetics Consortium. Estimation of the warfarin dose with clinical and
pharmacogenetic data. *N Engl J Med* 2009;360:753–764.

Dữ liệu đã được ẩn danh; không thực hiện bất kỳ thao tác nào nhằm định danh lại bệnh nhân.
