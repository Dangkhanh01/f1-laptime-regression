# F1-03 — Thống nhất bài toán và protocol chung

**Phụ trách đề xuất:** Danh. **Issue:** #3. Đọc Lab ngay; chốt từ inventory của Nguyên.

Mục tiêu: cả nhóm giải cùng một bài toán và so sánh model trên cùng quy tắc dữ liệu.
Ngồi cùng/call với Nguyên, Khang và Khánh để tranh luận, không cần PR riêng cho từng quyết định.

## Làm ở file nào

| File/vị trí | Cần làm |
|---|---|
| `Lab 01 - Linear Regression.pdf`, `README.md` | Đã có; đọc mục tiêu và yêu cầu |
| `docs/data-inventory.md`, `data/manifest.json` | Chờ F1-02; đọc nguồn/schema và điểm chưa rõ |
| `reports/data-inventory/schema.csv`, `reports/data-inventory/quality.json` | Chờ F1-02; kiểm tra căn cứ trước khi chọn quy tắc |
| `configs/experiment.json` | Tạo khi đủ căn cứ; cấu hình dùng chung về target/unit, source, scope/filter và split |
| `docs/experiment-protocol.md` | Tạo khi làm; giải thích quy tắc thực nghiệm và cách dùng config, không ghi biên bản tranh luận |
| `docs/eda-findings.md`, `reports/eda/` | Chờ F1-04; dùng phát hiện EDA để hoàn thiện protocol |

## Cách làm

1. Từ Lab và inventory, viết rõ dự đoán gì, cho đơn vị quan sát nào, vào thời điểm nào.
   Xác minh target/đơn vị; trao đổi với Nguyên nếu tên cột hoặc ý nghĩa chưa đủ chắc chắn.
2. Call để thống nhất scope, bộ lọc và ranh giới train/validation/test dựa trên coverage
   thực tế. Cân nhắc thời gian/event, overlap và thay đổi phân phối; chưa có căn cứ thì chưa chốt.
3. Ghi quy tắc thực thi vào `configs/experiment.json`; giải thích ngắn trong protocol.
   Chỉ rõ phần nào Khang được khám phá, phần test nào giữ lại. Sau đó Khang bắt đầu EDA,
   không cần đợi thiết kế model hoặc mọi chi tiết preprocessing.
4. Xác định feature nào biết trước thời điểm dự đoán, feature nào có leakage/chưa rõ.
   Chốt fit preprocessing trên train và áp dụng nhất quán cho tập còn lại/app.
5. Khi có EDA, nói chuyện với Khang để hoàn thiện missing/category/bộ lọc/features.
   Nếu đổi config, chỉ rõ phần EDA nào cần chạy lại; không dùng final test để chọn thiết kế.
6. Thống nhất metrics MAE/MSE/R², có thể thêm RMSE; quy ước lưu config, data/split,
   code version, metrics và artifact cho từng run để nhóm chạy lại và so sánh được.
7. Bàn giao protocol/config cho Nguyên thực thi tại [F1-05](F1-05-shared-data-preparation.md).
   Phân biệt quy tắc làm sạch/split chung với các biến đổi học từ dữ liệu trong Pipeline
   của [F1-06](F1-06-models-evaluation.md); không để mỗi người tự lọc hoặc chia dataset riêng.

## Quy tắc diễn giải kết quả

Đưa các quy tắc sau vào `docs/experiment-protocol.md` để áp dụng khi chạy model và viết báo cáo:

- **Phân tích hệ số (coefficients) và nhận xét dữ liệu phải dựa hoàn toàn trên kết quả
  chạy thực tế của mô hình.** Mỗi diễn giải phải đối chiếu được với đúng run, cấu hình,
  bảng hệ số và kết quả đánh giá; không suy đoán hoặc thêm kết luận ngoài bằng chứng đã có.
- Người viết phải hiểu ý nghĩa từng con số: dấu và đơn vị của hệ số, ảnh hưởng của scaling/
  encoding, ý nghĩa metrics và tập dữ liệu được đánh giá. Chưa hiểu thì trao đổi với người
  làm model trước khi kết luận; không diễn giải tương quan thành quan hệ nhân quả.
- AI chỉ hỗ trợ tham khảo. Thành viên phải tự kiểm chứng lời giải thích với output thật
  và viết lại bằng hiểu biết của mình; **không copy nguyên văn phần diễn giải tự động từ AI**.

Sau mỗi lần chốt vấn đề quan trọng, ghi 2–3 gạch đầu dòng trong Issue #3. Config là nơi
code đọc giá trị thực thi; protocol giải thích cách dùng. Không cần bảng trạng thái quyết định,
phân vai người duyệt từng dòng hoặc ghi lại mọi ý kiến trong call.

**Xong khi:** nhóm hiểu thống nhất target, scope, feature availability, split, preprocessing
và metrics, cùng quy tắc diễn giải dựa trên kết quả thật; các model có thể dùng cùng protocol.
Điểm chưa đủ căn cứ được trao đổi để
giải quyết, không điền giả định như một quyết định đã chốt. PR hoàn tất task ghi `Closes #3`.
