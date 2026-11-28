# F1-07 — Ứng dụng demo và video

**Phụ trách đề xuất:** Nguyên, các thành viên phối hợp giải thích model.
**Issue:** #6.
App Streamlit dùng một form và một nút **Dự đoán** để chạy trực tiếp M1–M4 trên cùng
một vòng đua. Nhận bốn pipeline, kết quả vòng mẫu và bảng MAE validation từ [F1-06](F1-06-models-evaluation.md).

## Làm ở file nào

| File/vị trí | Trách nhiệm dự kiến khi bắt đầu |
|---|---|
| `app/main.py` | Tạo ứng dụng Streamlit local khi làm task |
| `src/f1_lap_time/` | Dùng lại logic prediction/pipeline chung, không viết preprocessing khác với train |
| `artifacts/` | Nạp bốn pipeline đã fit theo đúng `run_id` bàn giao |
| `README.md` | Thêm lệnh chạy demo và cách lấy/tạo artifact |
| `reports/demo/` | Ảnh minh họa gọn cho báo cáo; video local hoặc link được ghi khi bàn giao |
| `tests/` | Kiểm tra input/prediction và các tình huống quan trọng của app |

## Cách làm khi đủ đầu vào

1. **Nhập một vòng.** Form nhận chặng, số vòng, loại lốp, lốp mới hay đã dùng, số thứ tự
   stint, tuổi lốp đầu vòng, tay đua và đội. Với ô “Số vòng bộ lốp đã chạy trước vòng này”,
   giá trị 0 ứng với `TyreLife=1`. Khi có giá trị hợp lệ, cộng 1 để tạo đầu vào gốc cho
   pipeline. Khi thiếu tuổi lốp, truyền giá trị khuyết. Pipeline thực hiện phép trừ và điền thiếu như lúc train.
2. **Chạy bốn dự đoán.** Mỗi lần bấm nút, lấy đúng các cột gốc mà từng pipeline yêu cầu,
   gọi `predict` và hiển thị bốn kết quả cạnh nhau. Mỗi kết quả ghi tên model, lap time
   dự đoán (s) và MAE validation 2024 (s) của run tương ứng. Đánh dấu model thắng theo
   bảng F1-06. Dùng biểu đồ so sánh bốn dự đoán cho vòng đang nhập.
3. **Kiểm tra trên máy demo.** Nhập vòng mẫu F1-06 và đối chiếu từng kết quả trước khi
   làm tròn với dự đoán bàn giao. Thử tuổi lốp bằng 0, để trống, nhập giá trị âm và
   chọn category mới. Báo lỗi nhập liệu nếu tuổi lốp âm. Khi thiếu hoặc nạp lỗi một
   artifact, báo rõ tên model bị ảnh hưởng và vẫn hiển thị dự đoán của các pipeline chạy được.
4. **Quay video.** Dự đoán một vòng, sau đó chỉ đổi tay đua và bấm lại. Chỉ ra vì sao
   M1 và M2 giữ nguyên kết quả, rồi mô tả kết quả thực tế của M3 và M4. Giải thích phạm
   vi vòng đua theo F1-03 và ý nghĩa của nhãn model thắng. Bàn giao video cho F1-08.

Issue #6 chỉ lưu 2–3 gạch chốt phạm vi demo/kịch bản sau khi nói chuyện.
**Xong khi** app chạy local theo README, cả bốn pipeline dự đoán trực tiếp và khớp kết quả
vòng mẫu, có biểu đồ và video minh họa việc đổi tay đua.
PR hoàn tất toàn bộ task ghi `Closes #6`.
