# F1-04 — EDA phục vụ thiết kế model

**Phụ trách:** Khang. **Issue:** #4. Ưu tiên trước việc lập báo cáo.
Chuẩn bị câu hỏi từ inventory; chạy phân tích sau khi nhóm chốt ranh giới split trong F1-03.

Mục tiêu: hiểu phân phối target và quan hệ các biến để thiết kế mô hình có căn cứ.
Nguyên làm kiểm kê nguồn/schema; Khang dùng kết quả đó để phân tích, không lặp lại toàn bộ inventory.

## Làm ở file nào

| File/vị trí | Cần làm |
|---|---|
| `docs/data-inventory.md`, `data/manifest.json`, `reports/data-inventory/` | Chờ F1-02; lấy đúng nguồn/version và xem kiểm tra đã có |
| `docs/experiment-protocol.md`, `configs/experiment.json` | Chờ F1-03 chốt phần cần cho EDA; đọc target/scope/split |
| `scripts/run_eda.py` | Tạo khi làm; chạy EDA từ manifest/config, không sửa raw |
| `src/f1_lap_time/eda.py` | Chỉ tạo nếu cần tách logic tái sử dụng |
| `reports/eda/metadata.json` | Kết quả chạy; nguồn/hash, config/code version, tập dữ liệu và số dòng trước/sau lọc |
| `reports/eda/target-summary.csv`, `reports/eda/group-summary.csv` | Kết quả chạy; thống kê target/nhóm thực sự cần phân tích |
| `reports/eda/figures/` | Biểu đồ PNG từ Matplotlib/Seaborn, tên file theo nội dung |
| `docs/eda-findings.md` | Tạo khi làm; phát hiện, dẫn chứng và đề xuất cho protocol/model |
| `tests/test_eda.py` | Viết khi cần kiểm tra chọn partition, giữ nguyên raw và tính đúng bảng |

## Cách làm

1. Đọc inventory/config và nói chuyện với Nguyên/Danh để hiểu dữ liệu, phạm vi được phân tích.
   Không tự chọn split riêng. Mặc định khám phá train; validation chỉ dùng khi protocol cho phép,
   không khám phá final test để chọn mô hình/bộ lọc.
2. Viết script chạy từ gốc repo; ghi lệnh tái tạo trong findings, metadata ở `reports/eda/`.
   Đối chiếu nguồn/hash/config và ghi số dòng thực sự dùng để mọi bảng/ảnh cùng phạm vi.
3. Phân tích target: đơn vị, phân phối, độ phân tán, giá trị bất thường. Xuất bảng và biểu đồ;
   nêu lý do nếu muốn lọc outlier, không tự xóa chỉ để phân phối đẹp hơn.
4. Phân tích theo event/track/mùa hoặc nhóm có ý nghĩa trong dữ liệu. Ghi số mẫu khi so sánh,
   xem nhóm ít dữ liệu và khác biệt phân phối ảnh hưởng tổng quát hóa.
5. Xem quan hệ feature ứng viên–target bằng thống kê/biểu đồ thích hợp. Không dùng bản sao
   target, sector hoặc thông tin sau vòng chỉ vì có tương quan cao.
6. Kiểm tra ảnh hưởng missing/làm sạch khi cần. Phát hiện lỗi mới thì trao đổi với Nguyên;
   đề xuất biến đổi/feature thì call với Danh để thống nhất, không tự đổi protocol chung.
7. Tổng hợp phát hiện trong `docs/eda-findings.md`: câu hỏi, dẫn chứng, giới hạn và đề xuất.
   Không suy quan hệ nhân quả từ tương quan. Lưu nguồn/bảng/ảnh để sau này dùng cho báo cáo.

Thiết kế code, tên hàm và cách vẽ do Khang quyết định. Trao đổi trực tiếp khi cần góp ý;
Issue #4 chỉ giữ 2–3 gạch cho kết luận quan trọng và hướng mô hình/bộ lọc đã thống nhất.

**Xong khi:** EDA chạy lại được trên đúng phần dữ liệu, phát hiện có bằng chứng, và Danh/cả
nhóm dùng được kết quả để hoàn thiện protocol/thiết kế model. PR hoàn tất task ghi `Closes #4`.
