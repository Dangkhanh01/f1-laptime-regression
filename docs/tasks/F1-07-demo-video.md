# F1-07 — Ứng dụng demo và video

**Phụ trách đề xuất:** Nguyên, các thành viên phối hợp giải thích model.
**Issue:** #6.
Chờ pipeline/artifact và đầu vào đã thống nhất từ F1-06.
Trang này mới chốt phạm vi; màn hình/input cụ thể bổ sung sau khi có model.

## Làm ở file nào

| File/vị trí | Trách nhiệm dự kiến khi bắt đầu |
|---|---|
| `app/main.py` | Tạo ứng dụng Streamlit local khi làm task |
| `src/f1_lap_time/` | Dùng lại logic prediction/pipeline chung, không viết preprocessing khác với train |
| `artifacts/` | Nạp model nhóm đã train; đường dẫn cụ thể chốt lúc bàn giao model |
| `README.md` | Thêm lệnh chạy demo và cách lấy/tạo artifact |
| `reports/demo/` | Ảnh minh họa gọn cho báo cáo; video local hoặc link được ghi khi bàn giao |
| `tests/` | Kiểm tra input/prediction và các tình huống quan trọng của app |

## Cách làm khi đủ đầu vào

1. Call với người phụ trách model để xác định feature nhập được, đơn vị và giới hạn giá trị.
2. Làm giao diện nhập/chỉnh feature, nạp pipeline và hiển thị dự đoán kèm biểu đồ diễn giải.
3. Kiểm tra cách chạy trên máy nhóm, tình huống thiếu model/input sai và sự nhất quán với train.
4. Quay video chạy app, giải thích input/output và giới hạn dự đoán; bàn giao cho F1-08.

Issue #6 chỉ lưu 2–3 gạch chốt phạm vi demo/kịch bản sau khi nói chuyện.
**Xong khi:** app chạy local theo README, dự đoán từ pipeline thật, có biểu đồ và video demo.
PR hoàn tất toàn bộ task ghi `Closes #6`.
