# F1-05 — Chuẩn bị dữ liệu chung

**Phụ trách:** Nguyên. **Issue:** chưa tạo; xem bảng liên kết trong [Giao việc](README.md).
**Bắt đầu:** sau khi F1-03 chốt protocol từ inventory và EDA F1-04.

Mục tiêu: thực thi protocol để tạo một bộ dữ liệu nền và thông tin chia tập duy nhất
cho cả bốn model. Task này không train model và không tự quyết định lại target/filter/split.

Đầu ra phải đạt các điều kiện nghiệm thu dưới đây trước khi bàn giao. Mục tiêu là để
F1-06 tập trung thiết kế/train model; không cam kết dữ liệu không bao giờ phát sinh lỗi mới.

## Làm ở file nào

| File/vị trí | Cần làm |
|---|---|
| `data/manifest.json`, `data/raw/`, `docs/data-inventory.md` | Đầu vào F1-02; xác minh đúng nguồn/hash và giữ nguyên raw |
| `docs/experiment-protocol.md`, `configs/experiment.json`, `docs/eda-findings.md` | Đọc quy tắc đã thống nhất và căn cứ từ F1-03/04 |
| `src/f1_lap_time/data_preparation.py` | Tạo khi làm; logic làm sạch chung, lọc scope, chuẩn hóa đơn vị/kiểu và gán split theo protocol |
| `scripts/prepare_data.py` | Tạo khi làm; lệnh chạy chuẩn bị dữ liệu từ manifest/config |
| `scripts/check_prepared_data.py` | Tạo khi làm; kiểm tra đầu ra đã lưu, dùng lại logic kiểm tra chung, trả mã lỗi khi không đạt |
| `data/processed/base.csv` | Xuất local; dữ liệu nền với target, feature ứng viên được phép và mã dòng ổn định |
| `data/processed/splits.csv` | Xuất local; ánh xạ mã dòng sang train/validation/test đúng protocol |
| `data/processed/metadata.json` | Xuất local; hash nguồn/đầu ra, config thực tế, code version, số dòng trước/sau từng bước và theo split |
| `tests/test_data_preparation.py` | Kiểm tra quy tắc làm sạch/split bằng dữ liệu nhỏ, không cần dataset thật trong CI |
| `reports/data-preparation/checks.json` | Kết quả chạy; từng kiểm tra đạt/không đạt, số vi phạm và cảnh báo được phép theo protocol; gắn với hash đầu ra |
| `README.md` | Thêm lệnh chuẩn bị và cách tái tạo dữ liệu/split cho các thành viên |

Các file trong `data/processed/` không commit. Cả nhóm chạy cùng lệnh/config để tái tạo
cùng phiên bản dữ liệu; dùng metadata/hash kiểm tra kết quả khớp nhau.

## Cách làm

1. Đọc inventory, EDA và protocol; kiểm tra nguồn/hash, các cột bắt buộc, đơn vị target.
   Quy tắc còn mơ hồ thì trao đổi với Danh/Khang trước khi code, không tự chọn cách xử lý.
2. Viết logic chung cho dòng trùng/lỗi, scope/filter và chuyển kiểu/đơn vị theo protocol.
   Ghi số dòng bị loại và lý do để nhóm giải thích được thay đổi; không sửa file raw.
3. Gán mã dòng ổn định và chia tập theo đúng ranh giới đã chốt. Mỗi dòng thuộc đúng một tập;
   kiểm tra không có overlap event/thời gian theo yêu cầu protocol, không random split riêng.
4. Xuất dữ liệu nền, ánh xạ split và metadata. Chỉ giữ feature được phép dùng; mã dòng,
   khóa chia tập và cột phục vụ truy vết không tự động trở thành feature của model.
5. Viết và chạy kiểm tra đầu ra theo bảng nghiệm thu; lưu kết quả thật vào
   `reports/data-preparation/checks.json`. Kiểm tra lại các file đã lưu, không chỉ DataFrame
   trong bộ nhớ. Chạy Ruff và tests liên quan; ghi lệnh tái tạo trong README.
6. Nhờ một thành viên khác tái tạo dữ liệu và chạy kiểm tra trên máy họ; đối chiếu hash,
   số dòng và split với metadata. Bàn giao cách nạp và phân biệt feature với khóa truy vết.
   Khi protocol thay đổi, cập nhật
   đầu ra chung và báo các model bị ảnh hưởng cần chạy lại, không sửa riêng bản của một người.

## Điều kiện nghiệm thu trước khi sang F1-06

| Nhóm kiểm tra | Phải đạt |
|---|---|
| Nguồn và khả năng tái tạo | Đúng hash/version nguồn; metadata chứa config thực tế và code version; chạy lại tạo cùng dữ liệu/split |
| Schema và ý nghĩa cột | Đủ cột, kiểu/đơn vị đúng protocol; có danh sách feature được phép và cột chỉ phục vụ truy vết; ý nghĩa cột dùng cho model đã được xác minh |
| Target và giá trị lỗi | Target không thiếu/NaN/vô cực; giá trị thỏa điều kiện đã chốt. Giá trị số không hợp lệ ở feature được xử lý hoặc khai báo rõ theo protocol |
| Dòng trùng và làm sạch | Mã dòng không thiếu/trùng; duplicate theo khóa nghiệp vụ được xử lý đúng quy tắc; số dòng trước/sau và lý do loại đối chiếu được |
| Split | Mỗi dòng dữ liệu nền có đúng một nhãn tập và ngược lại; các tập không rỗng, không giao nhau; ranh giới event/thời gian thỏa protocol, thứ tự dòng không làm lệch X/y |
| Leakage | Loại các cột bị cấm, bản sao target và thông tin chưa có khi dự đoán; không học tham số xử lý từ validation/test. Cần cả test tự động lẫn đối chiếu ý nghĩa cột với protocol |
| Missing và category | Mỗi cột được phép dùng có quy tắc rõ: không thiếu hoặc cách Pipeline xử lý missing. Chốt cách xử lý cột toàn missing trên train và category mới khi dự đoán; không để người train tự đoán |
| Bàn giao | Một máy khác tái tạo/kiểm tra đạt; README có lệnh chuẩn bị, kiểm tra và cách nạp dữ liệu/split; kết quả kiểm tra khớp hash phiên bản bàn giao |

Thiếu dữ liệu ở feature hoặc xuất hiện category mới không nhất thiết là lỗi: có thể bàn giao
nếu protocol đã quy định cách Pipeline xử lý và ghi rõ giới hạn. Không điền sẵn bằng trung
bình toàn dataset hoặc xóa tùy ý chỉ để bảng kiểm tra đạt.

**Không đóng F1-05 khi còn kiểm tra bắt buộc không đạt hoặc quy tắc xử lý chưa chốt.**
Test bằng dữ liệu nhỏ phải bắt được các lỗi như trùng mã dòng, thiếu split, overlap nhóm,
target không hợp lệ và dữ liệu đầu vào không khớp metadata. Kết quả kiểm tra dataset thật
được lưu riêng; test bằng dữ liệu nhỏ không thay thế việc kiểm tra đầu ra thật.

Các kiểm tra cấu trúc trên validation/test chỉ xác nhận quy tắc đã chốt. Không xem phân phối,
tương quan hoặc metrics final test để điều chỉnh feature/bộ lọc. Bảng kiểm tra đạt xác nhận
các điều kiện đã kiểm chứng, không chứng minh dữ liệu hoàn toàn không còn lỗi ngữ nghĩa.

## Ranh giới với preprocessing trong model

F1-05 thực thi các quy tắc chung đã chốt; không fit imputer, scaler, encoder hoặc chọn
feature dựa trên toàn bộ dataset. Nếu một ngưỡng làm sạch cần ước lượng từ dữ liệu,
phải chốt cách học chỉ trên train trong protocol trước khi áp dụng cho các tập còn lại.

Trong F1-06, các biến đổi có học từ dữ liệu nằm trong scikit-learn Pipeline: **fit trên train,
transform validation/test**. Khi có cross-validation, fit lại trong từng fold train.
Các model có thể dùng feature/biến đổi khác nhau, nhưng phải giữ cùng target và cùng các
dòng đánh giá; xử lý missing theo protocol thay vì mỗi người tự bỏ dòng.

**Xong khi:** cả nhóm tái tạo được cùng dữ liệu nền/split, metadata truy được nguồn/config,
toàn bộ điều kiện nghiệm thu đã đạt, không còn lỗi chặn đã biết, và bốn model có thể dùng
chung đầu vào này. Khi bàn giao, giữ cố định phiên bản dữ liệu/config/split để so sánh model.
Trao đổi bằng miệng/call; chỉ ghi 2–3 gạch chốt trong Issue của task.
PR hoàn tất dùng `Closes #<số Issue thực tế của F1-05>` sau khi Issue được tạo.
