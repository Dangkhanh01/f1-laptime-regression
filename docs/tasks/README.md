# Giao việc

**Nói miệng để tranh luận — Issue chỉ để chốt hạ và lưu vết.**

Mỗi mã task lớn có đúng một Issue cho cả giai đoạn. Trao đổi, bàn giao và góp ý bằng
nói trực tiếp/voice call. Không cần Issue con, story point, báo cáo tiến độ hay biểu đồ quản lý.
Các trang dưới đây là hướng dẫn kỹ thuật để biết làm ở file nào, không phải thủ tục phải điền.

## Bảy task của đồ án

| Issue | Task | Phụ trách | Khi bắt đầu |
|---|---|---|---|
| [#1](https://github.com/Dangkhanh01/f1-laptime-regression/issues/1) | [F1-01 — Khởi động nhóm](F1-01-team-kickoff.md) | Khánh | Ngay |
| [#2](https://github.com/Dangkhanh01/f1-laptime-regression/issues/2) | [F1-02 — Kiểm tra dataset](F1-02-dataset-inventory.md) | Nguyên | Khi môi trường của Nguyên chạy được |
| [#3](https://github.com/Dangkhanh01/f1-laptime-regression/issues/3) | [F1-03 — Protocol chung](F1-03-experiment-protocol.md) | Danh, đề xuất | Đọc yêu cầu ngay; chốt từ inventory |
| [#4](https://github.com/Dangkhanh01/f1-laptime-regression/issues/4) | [F1-04 — EDA](F1-04-exploratory-analysis.md) | Khang | Sau inventory và ranh giới split |
| [#5](https://github.com/Dangkhanh01/f1-laptime-regression/issues/5) | [F1-05 — Mô hình và đánh giá](F1-05-models-evaluation.md) | Cả nhóm, mỗi người một model | Sau EDA/protocol |
| [#6](https://github.com/Dangkhanh01/f1-laptime-regression/issues/6) | [F1-06 — Demo và video](F1-06-demo-video.md) | Nguyên, đề xuất | Sau khi có pipeline/artifact dùng được |
| [#7](https://github.com/Dangkhanh01/f1-laptime-regression/issues/7) | [F1-07 — Báo cáo và nộp bài](F1-07-report-outline.md) | Khang tổng hợp, cả nhóm viết | Backlog; sau EDA và có định hướng model |

Khánh là leader; các phân công ghi “đề xuất” được nhóm thống nhất khi nói chuyện.
F1-05/06 hiện chỉ mô tả phạm vi và vị trí đầu ra; chi tiết bổ sung trong chính trang task
khi đủ dữ liệu/quyết định, không mở thêm Issue để chia vụn.

## Chỉ ghi lại kết luận

Khi chốt một quyết định quan trọng, người phụ trách ghi **2–3 gạch đầu dòng** trong Issue
của task đó. Không chép lại diễn biến tranh luận. Ví dụ về cách ghi, không phải quyết định
kỹ thuật đã chốt cho dataset hiện tại:

> - Đã thống nhất: target là `<cột đã xác minh>`, scope là `<phạm vi đã chọn>`.
> - Split dùng `<ranh giới đã thống nhất>`; cấu hình ở `configs/experiment.json`.

Nếu quyết định thay đổi, nói chuyện rồi ghi kết luận mới trong cùng Issue. Các giá trị thực
thi trong code/config cần khớp với kết luận; số liệu và nguồn vẫn lưu cùng kết quả để tái tạo được.
Không bắt buộc ghi comment cho mọi thao tác hoặc cập nhật hằng ngày.

## PR và đóng task

Làm trên nhánh riêng, trao đổi/góp ý trực tiếp và mở PR. PR hoàn tất toàn bộ task ghi
`Closes #<số Issue>` trong mô tả; Issue tự đóng khi PR được merge vào nhánh mặc định `main`.
PR mới hoàn thành một phần chỉ ghi `Refs #...`, để Issue lớn còn mở.
Riêng F1-05: bốn người có thể có nhiều PR cùng `Refs #5`; PR tổng hợp cuối mới `Closes #5`.
Không cần mở Issue riêng cho mỗi model. [Cách GitHub liên kết PR và Issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/linking-a-pull-request-to-an-issue).

## Tìm và lưu file

Đường dẫn trong mọi trang task tính từ **gốc repo**. File ghi “tạo khi làm” chưa tồn tại;
owner tự thiết kế code, trang task chỉ chốt trách nhiệm và đầu ra.

- `docs/tasks/`: hướng dẫn giao việc; `docs/`: giải thích chuyên môn khi thực sự cần.
- `scripts/`, `src/f1_lap_time/`: lệnh chạy và logic dùng chung; `configs/`: cấu hình thực thi.
- `data/manifest.json`: nguồn/version/hash; `data/raw/`, `data/processed/`: dữ liệu local, không commit.
- `reports/`: bảng/ảnh/báo cáo gọn cần chia sẻ; `artifacts/`: model và run local, không commit.
- `tests/`: kiểm chứng logic cần thiết khi triển khai, không phải số liệu nghiên cứu.

Nguồn: [README dự án](../../README.md). Hạn nộp **11/12/2026**, ngân sách mục tiêu **80 giờ công**.
