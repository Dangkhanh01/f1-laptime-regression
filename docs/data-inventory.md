# Kiểm kê dữ liệu F1-02

Nguồn: [Mendeley Data V1](https://data.mendeley.com/datasets/djr8rnjtjp/1), DOI
`10.17632/djr8rnjtjp.1`, giấy phép CC BY 4.0. File được chọn là
`f1_research_dataset_with_risk_safetycar.parquet` trong ZIP “Download All”.
`data/manifest.json` ghi tên, kích thước và SHA-256 **tự tính từ ZIP/file local**;
chưa có checksum do nhà xuất bản công bố để đối chiếu độc lập.

## Chạy lại

Sau khi nhóm duyệt và đưa dữ liệu lên GitHub, clone repo sẽ có sẵn các file đã trích,
không cần ZIP. Chạy lệnh dưới đây để kiểm tra số file, hash hai bản Parquet tổng hợp và
tái tạo báo cáo. Nếu cần khôi phục dữ liệu, tải **Download All** trên trang nguồn rồi chạy
`uv run python -m scripts.download_data --archive "<đường dẫn ZIP>"`. Lệnh kiểm tra hash/kích
thước ZIP, trích `datasets/` (96 file Parquet) và `paper_outputs/` (11 file tham khảo) vào
`data/raw/f1_safetycar/`, rồi đặt thêm file Parquet tổng hợp ở `data/raw/`. ZIP chỉ được đọc,
không chép vào repo. Lệnh không trích `fastf1_cache/` (~653 MB). Không dùng model/kết quả trong
`paper_outputs/` để huấn luyện hoặc báo cáo thay cho mô hình của nhóm. Có thể dùng
`--url` nếu có URL HTTPS trực tiếp đến ZIP.
Đường dẫn trích ngắn hơn tên thư mục gốc trong ZIP để tránh lỗi đường dẫn dài trên Windows.

```sh
uv sync --locked
uv run python -m scripts.download_data
uv run python -m scripts.inspect_data
```

Kết quả: [schema.csv](../reports/data-inventory/schema.csv) ghi dtype, missing,
chuỗi giả thiếu, ý nghĩa/đơn vị và nguồn đối chiếu; [quality.json](../reports/data-inventory/quality.json)
ghi số dòng/cột, khóa ứng viên, giá trị nghi vấn và coverage. Ô chưa xác minh để trống.
Script đọc file từ manifest và xác minh SHA-256 trước khi thống kê; không sửa raw. Manifest
giữ SHA-256 của ZIP làm thông tin nguồn, nhưng lệnh kiểm kê không xác minh ZIP khi ZIP vắng mặt.

Ý nghĩa các cột lap và thời tiết được đối chiếu với [FastF1 lap timing](https://github.com/theOehrly/Fast-F1/blob/main/docs/api_reference/timing_data.rst)
và [FastF1 weather data](https://github.com/theOehrly/Fast-F1/blob/main/fastf1/_api.py).
`Time` trong file đã là số; đơn vị giây là suy luận từ giá trị và định nghĩa FastF1, chưa có mã
chuyển đổi của tác giả để xác minh. `PitInTime`/`PitOutTime` thiếu toàn bộ nên chưa xác minh
đơn vị sau chuyển đổi. `RaceProgress`, `HasPitIn`, `HasPitOut` chỉ có nghĩa khái quát;
công thức tạo cột chưa được công bố. Không dùng nghĩa suy luận để chốt feature.

## Phát hiện cần bàn giao cho Danh và Khang

- Một dòng là một vòng của một tay đua trong một sự kiện/phiên. File có **89.980 dòng,
  35 cột**, bốn mùa 2022–2025, **95 sự kiện**, chỉ có mã phiên `R`. Khóa ứng viên
  `(Year, EventName, Session, Driver, LapNumber)` không trùng và không thiếu trên file này.
- `LapTimeSec` là target ứng viên tính bằng giây; khoảng quan sát **67,012–149,321 giây**.
  `LapTime` trùng hoàn toàn với `LapTimeSec`; tổng ba sector cũng khớp target.
  Không dùng các cột này làm feature nếu dự đoán lap time trước khi vòng kết thúc.
- Mô tả nguồn nói chỉ có phiên đua chính thức, nhưng file chứa **4.526 dòng Pre-Season**.
  Dữ liệu đã qua lọc: `IsAccurate` đều true. Không tự loại Pre-Season trước khi Danh
  thống nhất scope/filter/split.
- `PitInTime` và `PitOutTime` thiếu toàn bộ; `HasPitIn`/`HasPitOut` đều bằng 0.
  Không thể suy ra pit stops từ các cột này trong file tổng hợp. `TyreLife` thiếu
  887 dòng, `Stint` thiếu 382 dòng. `IsSafetyCar` có 621 dòng dương tính,
  `IsVSC` có 243; nhãn được tác giả suy ra từ race control.
- `Compound` không có giá trị null theo pandas nhưng có **765 chuỗi giả thiếu**:
  383 giá trị `"None"` và 382 giá trị `"nan"`. Bảng schema báo riêng `placeholder_count`;
  chưa thay đổi raw hay quyết định cách xử lý cho F1-05.
- `Risk_SafetyCar` là đầu ra mô hình theo `dataset_card.json` của tác giả. `Time`,
  `Position`, `RaceProgress`, nhãn SC/VSC và các cột sau vòng cần được Danh xét theo
  **thời điểm dự đoán**. Chưa chốt feature, target, lọc, split hay diễn giải mã
  `TrackStatus`; không dùng final test để quyết định các việc đó.

Theo FastF1, `TrackStatus` là chuỗi mã trạng thái xảy ra trong vòng. Các giá trị như `"12"`
và `"21"` không phải số thứ tự 12/21; cần xác minh cách tác giả giữ/chuyển mã trước khi dùng
chúng làm feature.

File tổng hợp khớp số dòng và số sự kiện trong `dataset_card.json` đi kèm ZIP. Có 95
file Parquet theo sự kiện trong ZIP; cache FastF1 và model/báo cáo của tác giả không
phải dữ liệu đầu vào để chạy kiểm kê. Trước EDA/model, nhóm cần thống nhất
liệu Pre-Season thuộc phạm vi và liệu có thể dùng nhãn SC/VSC hay biến sau vòng.
