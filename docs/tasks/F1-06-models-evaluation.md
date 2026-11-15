# F1-06 — Xây dựng và đánh giá các mô hình

**Phụ trách:** cả nhóm, mỗi người một Linear Regression model; Khánh điều phối.
**Issue:** #5.
Chờ dữ liệu nền và split từ F1-05. Chi tiết thiết kế sẽ bổ sung sau khi nhóm bàn bạc.

Mục tiêu: xây dựng và so sánh ít nhất bốn model trên cùng dữ liệu nền, target và tập đánh giá.
Mỗi người dùng lại đầu ra F1-05, không tự làm sạch, lọc dòng hoặc chia tập riêng.

## Làm ở file nào

| File/vị trí | Trách nhiệm dự kiến khi bắt đầu |
|---|---|
| `docs/eda-findings.md`, `docs/experiment-protocol.md`, `configs/experiment.json` | Đầu vào dùng chung từ F1-03/04 |
| `data/processed/base.csv`, `data/processed/splits.csv`, `data/processed/metadata.json` | Đầu ra local từ F1-05; dùng đúng phiên bản dữ liệu và danh sách dòng của từng tập |
| `src/f1_lap_time/data_preparation.py` | Dùng lại phần chuẩn bị dữ liệu của Nguyên khi cần; không sao chép thành bốn bản |
| `scripts/check_prepared_data.py`, `reports/data-preparation/checks.json` | Chạy kiểm tra đầu vào F1-05; đối chiếu kết quả đạt và hash phiên bản bàn giao |
| `src/f1_lap_time/` | Pipeline biến đổi feature, train và evaluation; tên module chốt khi có thiết kế |
| `configs/models/` | Cấu hình cho ít nhất 4 thiết kế LR, chưa tạo trước khi thống nhất |
| `scripts/` | Lệnh chạy train/evaluate và tổng hợp; tên file bổ sung sau khi chốt cấu trúc |
| `artifacts/` | Model và kết quả từng run ở local, không commit binary |
| `reports/models/` | Bảng/ảnh so sánh cần chia sẻ, có run ID và provenance |
| `tests/` | Kiểm tra tính đúng pipeline, split và tránh leakage |

## Cách làm khi đủ đầu vào

1. Chạy kiểm tra đầu vào của F1-05; chỉ bắt đầu khi đạt và hash dữ liệu/split/config khớp
   phiên bản bàn giao. Không cần làm lại inventory/EDA. Nạp dữ liệu theo mã dòng để X/y/split
   khớp nhau. Call để thống nhất bốn thiết kế LR,
   lý do chọn features/biến đổi và người phụ trách phần train/evaluation dùng chung.
2. Mỗi người xây một model; có thể chọn feature hoặc biến đổi khác nhau nhưng giữ nguyên
   target, dữ liệu nền và các dòng đánh giá. Không tự bỏ dòng để làm đẹp metrics.
   Điền thiếu có học từ dữ liệu, scaling, encoding và chọn feature có học phải nằm trong
   Pipeline, chỉ fit trên train rồi transform validation/test; không fit trước trên toàn bộ data.
   Dùng chính sách missing/category đã chốt ở F1-05. Kiểm tra Pipeline trên train/validation
   theo protocol và dữ liệu giả cho tình huống thiếu/category mới; không dùng final test
   để thử và sửa thiết kế. Biến đổi do model tạo ra NaN/vô cực là phần cần xử lý trong model.
3. Lưu config/code/data version, artifact và metrics thật của từng run; diễn giải hệ số/sai số.
4. So sánh trên cùng tập đánh giá, khóa thiết kế rồi mới đánh giá final test theo protocol.
5. Bàn giao pipeline/artifact và cách dùng cho F1-07, bảng kết quả cho F1-08.

Nếu phát hiện dữ liệu cần sửa, trao đổi với Nguyên/Danh để cập nhật F1-05/protocol chung,
phân biệt lỗi dữ liệu chung với lỗi Pipeline riêng. Lỗi dữ liệu chung phải sửa một nơi,
chạy lại kiểm tra, cập nhật hash/metadata và chạy lại các model bị ảnh hưởng trước khi so sánh.
PR sửa tiếp vẫn liên kết Issue F1-05, mở lại Issue đó nếu cần; không tạo Issue con.
Không tạo dataset riêng cho từng người hoặc âm thầm sửa file đã bàn giao.
Diễn giải hệ số phải theo quy tắc F1-03: dựa trên run thật, hiểu dấu/đơn vị và ảnh hưởng
của biến đổi, không copy nguyên văn lời giải thích tự động từ AI.

Chỉ có **một Issue #5**, không tách Issue cho từng model. Chốt thiết kế và kết luận bằng
2–3 gạch sau trao đổi. Mỗi PR phần việc ghi `Refs #5`; PR tổng hợp hoàn tất task mới `Closes #5`.

**Xong khi:** ít nhất 4 LR models có thiết kế được giải thích, kết quả tái tạo/so sánh được
và pipeline dùng được cho demo. Không điền sẵn features hoặc kết quả khi nhóm chưa quyết định/chạy thật.
