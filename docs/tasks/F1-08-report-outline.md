# F1-08 — Báo cáo và hoàn thiện gói nộp

**Phụ trách:** Khang tổng hợp, cả nhóm viết phần mình thực hiện; Khánh điều phối nộp bài.
**Issue hiện có:** #7 (cần cập nhật mã task trong tiêu đề trên GitHub).
**Backlog:** bắt đầu sau EDA và khi có định hướng model, không ưu tiên trước F1-04.
Đề cương, viết báo cáo và hoàn thiện gói nộp đều nằm trong Issue này.

## Làm ở file nào

| File/vị trí | Cần làm khi task bắt đầu |
|---|---|
| `Lab 01 - Linear Regression.pdf` | Đã có; đối chiếu yêu cầu báo cáo và sản phẩm nộp |
| `docs/data-inventory.md`, `docs/experiment-protocol.md`, `docs/eda-findings.md` | Đọc kết quả chuyên môn đã có, không viết lại từ suy đoán |
| `reports/eda/`, `reports/models/`, `reports/demo/` | Lấy bảng/ảnh có nguồn và kết quả thật |
| `docs/report-outline.md` | Tạo khi hữu ích để nhóm phân phần viết; chỉ là đề cương, không sao chép cả báo cáo |
| Word/Google Docs chung | Bản nội dung chính; trao đổi quyền truy cập trực tiếp |
| `reports/final-report.pdf` | Bản PDF cuối cùng |
| `README.md`, `requirements.txt` | Hoàn thiện hướng dẫn chạy và dependency để bàn giao |
| `Group [Group-ID].zip` | Gói nộp cuối khi có Group-ID thật; không commit ZIP/video lớn |

## Cách làm

1. Đọc Lab, call để chia phần viết và thống nhất cấu trúc: bài toán/động lực, dữ liệu,
   mô hình, đánh giá/so sánh, app, kết luận/giới hạn và references.
2. Mỗi người viết phần mình đã làm, dẫn tới dữ liệu/run/bảng/ảnh thật. Khang tổng hợp
   bố cục và phần thiếu; không thay người khác giải thích kết quả họ thực hiện.
3. Sửa nội dung qua trao đổi trực tiếp; không cần bảng người duyệt từng mục, lịch báo cáo
   hoặc comment Issue xác nhận mỗi bước. Chưa có kết quả thì để trống phần đó.
4. Khi hoàn thiện, xuất PDF và kiểm tra gói nộp theo Lab: dataset/link, source/link,
   environment/requirements, video demo, report và references. Link dữ liệu ngoài cần đáp ứng
   yêu cầu duy trì truy cập; code/report nộp theo quy định Lab.
5. Mở thử gói, chạy lại hướng dẫn và kiểm tra link; thống nhất đại diện nộp trước **11/12/2026**.

## Quy tắc viết và giải thích số liệu

- **Mọi phân tích hệ số (coefficients) và nhận xét dữ liệu phải dựa hoàn toàn trên kết quả
  chạy thực tế của mô hình.** Dẫn tới đúng run, bảng hệ số/metrics hoặc biểu đồ làm căn cứ;
  không viết kết luận theo kỳ vọng, kiến thức suy đoán hoặc lời AI khi output chưa chứng minh.
- Thành viên viết báo cáo phải hiểu và tự giải thích được từng con số mình sử dụng:
  dấu/đơn vị hệ số, tác động của scaling/encoding, ý nghĩa metric và tập train/validation/test
  tương ứng. Nếu chưa rõ, trao đổi trực tiếp với người làm model trước khi đưa vào báo cáo.
- **Không copy nguyên văn lời giải thích tự động từ AI.** Tự đối chiếu từng nhận định với
  kết quả thật và viết lại bằng hiểu biết của mình. Nhận định chưa có căn cứ thì bỏ hoặc
  chờ kiểm chứng, không trình bày như kết luận của nhóm.

Issue #7 lưu 2–3 gạch chốt cách chia phần hoặc quyết định nộp bài quan trọng.
**Xong khi:** báo cáo phản ánh công việc thật, người viết giải thích được hệ số và số liệu
đã dùng, đủ yêu cầu Lab, gói nộp truy cập/chạy được.
PR hoàn tất task ghi `Closes #7`; phần nộp ngoài Git được kiểm tra xong trước khi kết thúc task.
