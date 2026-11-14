# F1-02 — Kiểm tra dataset

**Phụ trách:** Nguyên. **Issue:** #2. **Bắt đầu:** khi môi trường của Nguyên chạy được.

Mục tiêu: lấy đúng dữ liệu, xác minh schema/chất lượng và bàn giao đủ căn cứ cho Danh
chốt protocol, Khang làm EDA. Đây là kiểm kê, không phải toàn bộ phân tích khám phá.
Nguyên thực thi làm sạch và xuất dữ liệu nền/split chung ở [F1-05](F1-05-shared-data-preparation.md)
sau khi có protocol và EDA; F1-02 giữ nguyên raw, chưa xuất dataset sẵn sàng train.

## Làm ở file nào

| File/vị trí | Cần làm |
|---|---|
| `README.md`, `Lab 01 - Linear Regression.pdf` | Đã có; đọc nguồn dataset và yêu cầu bài toán |
| `scripts/download_data.py` | Tạo khi làm; tải file đã chọn và kiểm tra hash/kích thước |
| `data/manifest.json` | Tạo khi làm; ghi DOI/version/license, URL, tên file, kích thước, SHA-256 và đường dẫn local |
| `data/raw/` | Chia sẻ các file đã trích cần dùng qua Git; không đưa ZIP/cache vào repo |
| `scripts/inspect_data.py` | Tạo khi làm; chạy các kiểm tra cấu trúc/chất lượng từ manifest |
| `reports/data-inventory/schema.csv` | Kết quả chạy; dtype, missing và ý nghĩa/đơn vị đã xác minh |
| `reports/data-inventory/quality.json` | Kết quả chạy; số dòng/cột, duplicate/key, coverage mùa/event/session |
| `docs/data-inventory.md` | Tạo khi làm; giải thích nguồn, cách chạy, phát hiện và giới hạn; dẫn tới các bảng thay vì chép lại |
| `tests/test_data_inventory.py` | Viết nếu có logic hash/khóa/schema cần kiểm chứng bằng dữ liệu nhỏ |

## Cách làm

1. Đọc nguồn Mendeley trong README, đối chiếu DOI/version/license và danh sách file.
   Chọn file cần dùng, kiểm tra dung lượng; trao đổi với Khánh nếu nguồn/phạm vi chưa rõ.
2. Viết downloader và manifest, tải vào `data/raw/`. Đối chiếu hash nhà xuất bản nếu có;
   phân biệt hash tự tính với hash nguồn công bố. Không tải cả cache/model của tác giả nếu không cần.
3. Viết script kiểm kê: xác định đơn vị quan sát, khóa ứng viên, schema, missing, duplicate,
   giá trị không hợp lệ và coverage. Xuất bảng vào `reports/data-inventory/`.
4. Kiểm tra target ứng viên/đơn vị; chú ý cột trùng target, biến sau vòng, nhãn tương lai
   hoặc đầu ra model có sẵn. Ghi chỗ chưa rõ vào inventory, không tự đoán nghĩa cột.
5. So sánh file thực tế với mô tả nguồn: loại session/vòng, các nhóm đã bị lọc sẵn,
   khả năng nhận diện pit/SC/VSC. Mâu thuẫn cần được chỉ ra, không âm thầm xóa/sửa raw.
6. Call với Danh/Khang để bàn giao manifest, bảng kiểm tra, lệnh tái tạo và điểm chưa rõ.
   Nói chuyện để giải quyết; chỉ chốt kết luận quan trọng bằng 2–3 gạch trong Issue #2.

**Xong khi:** người khác lấy lại đúng dữ liệu và tái tạo kiểm kê được; Danh/Khang biết rõ
dữ liệu có gì và giới hạn nào ảnh hưởng bước tiếp theo. Nếu phát sinh quá nhiều kiểm tra,
trao đổi để thu hẹp phần đang làm trong chính task này, không mở hàng loạt Issue con.

Code và số liệu cần có nguồn tái tạo; không cần nhật ký thao tác, lịch báo cáo hay mốc
xác nhận bằng comment. PR hoàn tất toàn bộ task ghi `Closes #2`.
