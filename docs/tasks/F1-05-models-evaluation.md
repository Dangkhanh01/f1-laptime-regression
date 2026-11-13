# F1-05 — Xây dựng và đánh giá các mô hình

**Phụ trách:** cả nhóm, mỗi người một Linear Regression model; Khánh điều phối.
**Issue:** #5. Chờ EDA và protocol chung. Chi tiết thiết kế sẽ bổ sung vào trang này sau khi nhóm bàn bạc.

## Làm ở file nào

| File/vị trí | Trách nhiệm dự kiến khi bắt đầu |
|---|---|
| `docs/eda-findings.md`, `docs/experiment-protocol.md`, `configs/experiment.json` | Đầu vào dùng chung từ F1-03/04 |
| `src/f1_lap_time/` | Tiền xử lý, train và evaluation dùng chung; tên module chốt khi có thiết kế |
| `configs/models/` | Cấu hình cho ít nhất 4 thiết kế LR, chưa tạo trước khi thống nhất |
| `scripts/` | Lệnh chạy train/evaluate và tổng hợp; tên file bổ sung sau khi chốt cấu trúc |
| `artifacts/` | Model và kết quả từng run ở local, không commit binary |
| `reports/models/` | Bảng/ảnh so sánh cần chia sẻ, có run ID và provenance |
| `tests/` | Kiểm tra tính đúng pipeline, split và tránh leakage |

## Cách làm khi đủ đầu vào

1. Call để thống nhất bốn thiết kế LR, lý do chọn features/biến đổi và người phụ trách.
2. Dùng chung target, bộ lọc, split và quy tắc fit preprocessing; train/validation trước.
3. Lưu config/code/data version, artifact và metrics thật của từng run; diễn giải hệ số/sai số.
4. So sánh trên cùng tập đánh giá, khóa thiết kế rồi mới đánh giá final test theo protocol.
5. Bàn giao pipeline/artifact và cách dùng cho task demo, bảng kết quả cho báo cáo.

Chỉ có **một Issue #5**, không tách Issue cho từng model. Chốt thiết kế và kết luận bằng
2–3 gạch sau trao đổi. Mỗi PR phần việc ghi `Refs #5`; PR tổng hợp hoàn tất task mới `Closes #5`.

**Xong khi:** ít nhất 4 LR models có thiết kế được giải thích, kết quả tái tạo/so sánh được
và pipeline dùng được cho demo. Không điền sẵn features hoặc kết quả khi nhóm chưa quyết định/chạy thật.
