# F1-06 — Huấn luyện và đánh giá bốn mô hình hồi quy tuyến tính

**Khánh điều phối Issue #5.** Mỗi thành viên chịu trách nhiệm một mô hình dự đoán
`LapTimeSec` (giây) trước khi vòng đua bắt đầu. Phạm vi là các vòng đua chính thức
đã lọc theo F1-03. Cả nhóm dùng chung dữ liệu F1-05 và một hàm chấm điểm. Bàn giao cả bốn
pipeline cho demo dự đoán trực tiếp. Model thắng theo validation được đánh dấu trên app
và đánh giá một lần trên final test. [Giao thức F1-03](../experiment-protocol.md) · [Cấu hình thực nghiệm](../../configs/experiment.json)

## 1. Mỗi người kiểm chứng một câu hỏi

| Người, mô hình | Thiết kế đã chốt | Câu hỏi khi đọc kết quả validation 2024 |
|---|---|---|
| **Khánh · M1** | OLS (`LinearRegression` có intercept) với `EventName`, `LapNumber` | Khi chỉ biết chặng và số vòng, dự đoán lệch trung bình bao nhiêu giây? Chặng nào có sai số lớn? |
| **Nguyên · M2** | OLS như M1, thêm `Compound`, `FreshTyre`, `Stint`, `TyreLife_start` | So với M1, thêm cả nhóm thông tin lốp và stint làm MAE thay đổi bao nhiêu? Mức thay đổi có đồng đều giữa các chặng? |
| **Khang · M3** | OLS như M2, thêm `Driver` và `Team` | So với M2, thông tin tay đua và đội giúp dự đoán thế nào? Đo riêng các vòng có tay đua hoặc đội chưa xuất hiện trong train và ghi số vòng của nhóm này. |
| **Danh · M4** | `Ridge` có intercept, dùng đúng các đầu vào của M3 | So với M3, thiết kế Ridge cho MAE và RMSE thế nào trên cùng các vòng? Ghi các giá trị `alpha` vào config trước khi thử, lưu kết quả validation của từng giá trị và lý do chọn. |

EDA ghi nhận `LapNumber` tương quan âm với lap time ở 42/43 phiên train, còn quan hệ giữa tuổi
lốp và lap time thay đổi theo phiên. Đây là căn cứ để đưa số vòng vào M1 và kiểm tra M2 theo
chặng. Chênh lệch M2–M1 đo đóng góp chung của bốn biến được thêm. M3–M4 so sánh hai pipeline
hoàn chỉnh vì F1-03 quy định cách mã hóa category khác nhau cho OLS và Ridge. [Kết quả EDA F1-04](../eda-findings.ipynb)

## 2. Bắt đầu từ dữ liệu bàn giao

Từ thư mục gốc repo, đồng bộ môi trường. Nếu máy chưa có dữ liệu trong `data/processed/`,
chạy bước chuẩn bị trước khi đối chiếu với [báo cáo chuẩn F1-05](../../reports/data-preparation/checks.json).

```sh
uv sync --locked
uv run python -m scripts.prepare_data
uv run python -m scripts.check_prepared_data --reference reports/data-preparation/checks.json --report tmp/f1-05-checks.json
```

Sau khi kiểm tra đạt, nạp train 2022–2023 (39.433 vòng) và validation 2024 (23.426 vòng).
Ví dụ lấy đầu vào của M1 bằng [hàm nạp chung](../../src/f1_lap_time/data_preparation.py).

```python
from f1_lap_time.data_preparation import load_prepared_data

train, metadata = load_prepared_data(split="train")
val, _ = load_prepared_data(split="validation")
features = ["EventName", "LapNumber"]
X_train, y_train = train[features], train["LapTimeSec"]
X_val, y_val = val[features], val["LapTimeSec"]
val_ids = val["row_id"]  # Gắn lại vào dự đoán để so sánh cùng vòng.
```

Mỗi config khai báo rõ các cột đầu vào thuộc `features.allowed_candidate_features`.
Với M2–M4, cột đọc từ dữ liệu là `TyreLife`. Giữ `row_id`, `split`, `Year`, `Session`,
`EventKey` và `IsAccurate` ở ngoài ma trận đặc trưng. Giữ nguyên các vòng có feature
thiếu hoặc được EDA đánh dấu ngoại lệ. Nếu kiểm tra dữ liệu chung thất bại, đối chiếu với
Nguyên/Danh, sửa tại F1-05 hoặc protocol và chạy lại các model bị ảnh hưởng.

## 3. Train độc lập, chấm điểm chung

**Khánh, Nguyên, Khang và Danh có thể bắt đầu song song ngay khi dữ liệu F1-05 trên máy
của mình đã đạt kiểm tra.** Mỗi người dùng bộ nạp ở mục 2, tự xây, fit và kiểm thử
pipeline từ đầu. Ví dụ, “M2 thêm thông tin lốp” nghĩa là Nguyên dùng các cột của M1
cùng các cột lốp trong dữ liệu F1-05. M2 có thể train ngay cả khi M1 chưa có artifact.
Đặt config từng model ở `configs/models/`, logic ở `src/f1_lap_time/` và lệnh train ở
`scripts/`. Mỗi người ghi lại lệnh tái tạo run của mình.

Mỗi người lưu dự đoán train và validation cùng `row_id`, `split`, `y_true` và `y_pred`,
rồi tự chấm CSV bằng lệnh chung ở mục 6.

1. **Biến đổi đầu vào.** Với M2–M4, kiểm tra các giá trị `TyreLife` quan sát được là
   số hữu hạn ≥ 1, rồi thay cột này bằng `TyreLife_start = TyreLife - 1` trong pipeline.
   Giữ NaN để điền ở bước sau. Giá trị sai kiểu, vô cực hoặc nhỏ hơn 1 phải báo lỗi.
2. **Tiền xử lý.** Loại cột toàn khuyết trên tập dùng để fit. Điền số bằng trung vị,
   chuẩn hóa biến số theo F1-03. Điền category thiếu bằng `UNKNOWN`, giữ nguyên nhãn đội
   và loại lốp. Dùng `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`,
   với `drop="first"` cho M1–M3 và `drop=None` cho M4. Dùng cùng cách xử lý các đầu vào
   chung để có thể đối chiếu từng cặp thiết kế.
3. **Fit và dự đoán.** Gói toàn bộ biến đổi và estimator vào một scikit-learn `Pipeline`.
   Gọi `fit(X_train, y_train)` rồi `predict` cho train và validation. Nếu dùng fold trong
   train, tách theo thời gian và fit lại cả pipeline trong từng fold. Tập 2025 dành riêng
   cho lần đánh giá cuối ở mục 5.
4. **Kiểm thử trong `tests/`.** Kiểm tra `TyreLife=1` thành 0 và giá trị thiếu được
   điền sau phép trừ. Thử category mới và cột toàn khuyết ở train nhưng có giá trị ở
   validation. Kiểm tra thống kê imputer/scaler chỉ dựa trên train và pipeline sau khi
   lưu, nạp lại cho cùng dự đoán trên cùng đầu vào.

**Khi validation có nhãn chưa gặp trong train.** Áp dụng cho từng cột phân loại mà model
sử dụng. Giữ nguyên nhãn quan sát được và gọi `predict` bằng pipeline đã fit trên train.
Chỉ dùng `UNKNOWN` để điền giá trị thiếu. Với nhãn mới, encoder tạo toàn số 0 trong
nhóm cột của riêng đặc trưng đó. Ở M1–M3, nhãn mới vì thế cho cùng dự đoán với nhãn
tham chiếu khi giữ cố định các đầu vào khác. Ở M4, đóng góp của nhóm cột đó bằng 0,
còn intercept và các đặc trưng khác vẫn tham gia dự đoán. Giữ cách mã hóa đã chốt
cho cả lượt chạy, kể cả khi một nhóm có sai số validation lớn.

## 4. Chấm validation và giải thích kết quả

- **Ghép theo vòng.** Lưu bảng dự đoán từng run gồm `row_id`, `split`, `y_true`,
  `y_pred`. Hàm chấm đối chiếu đúng tập ID theo F1-05, mỗi ID có đúng một dự đoán hữu hạn
  và target khớp theo ID. Nếu thiếu, trùng hoặc lệch dòng, sửa run rồi chấm lại.
- **Tổng hợp.** Mỗi dòng bảng điểm ghi `model`, `run_id`, `split`, số vòng,
  MAE (s), RMSE (s), MSE (s²), R². Tính trên toàn bộ vòng của từng tập train và
  validation. Khi so hai model, lấy MAE validation của model mới trừ MAE của đối chứng.
  Giá trị âm cho biết sai số tuyệt đối trung bình giảm bao nhiêu giây.
- **Tìm nơi model sai.** Trên validation, nhóm dự đoán theo chặng, ghi số vòng, MAE và
  sai số có dấu trung bình (`y_pred - y_true`, dương nghĩa là dự đoán vòng chậm hơn).
  Chọn một chặng để giải thích chênh lệch với đối chứng trên cùng các vòng. Điểm tổng
  được tính trực tiếp từ từng vòng để mỗi vòng có trọng số như nhau.
- **Kiểm tra sai số ở nhãn mới.** Đối chiếu từng cột phân loại với các nhãn quan sát
  được trên train. Nếu có nhãn mới, ghi số vòng, MAE và sai số có dấu trung bình của
  nhóm vòng liên quan. Khi so hai model trên nhóm đó, dùng cùng danh sách `row_id`.
  Nếu xét nhiều cột cùng lúc, mỗi vòng có nhãn mới ở ít nhất một cột chỉ tính một lần.
  Tất cả các vòng này vẫn có đủ trọng số trong bốn metric validation.
- **Đọc hệ số.** Xuất tên feature sau biến đổi cùng hệ số và intercept. Mỗi người chọn
  hai hoặc ba hệ số từ chính run của mình, giải thích dấu và đơn vị khi giữ các biến
  còn lại cố định. Với `StandardScaler`, hệ số số ứng với tăng một độ lệch chuẩn train.
  Với OLS, ghi category tham chiếu. Với Ridge, có thể lấy hiệu hai hệ số category để
  giải thích chênh lệch dự đoán. Khi `Driver` và `Team` đi cùng nhau, nêu giới hạn của
   việc tách riêng hai hệ số và tránh diễn giải chúng như tác động nhân quả. Tự kiểm
   chứng từng nhận xét với số liệu trước khi viết báo cáo.

## 5. Khóa model, đánh giá 2025 và bàn giao

1. **Mỗi người chốt đúng một phiên bản cho model mình phụ trách.** Có thể thử nhiều
   phiên bản trong phạm vi thiết kế ở mục 1. Gán mỗi lượt train một `run_id` riêng.
   Dùng quy tắc validation dưới đây để chọn một run, ghi lý do chọn và bàn giao đúng
   một pipeline đã fit. Ví dụ, dù Danh thử nhiều giá trị `alpha`, M4 chỉ có một
   `run_id` trong bảng so sánh cuối và một file `.joblib` cho demo.
2. **Khánh chọn model thắng trong bốn run đã chốt.** Chọn MAE thấp nhất. Trong nhóm cách MAE
   tốt nhất ≤ 0,005 giây, ưu tiên RMSE thấp nhất. Nếu RMSE cũng chênh ≤ 0,005 giây,
   ưu tiên mô hình đơn giản hơn, ít feature hơn. So bằng số chưa làm tròn, ghi lý do chọn
   và khóa `run_id`, config, pipeline đã fit trên train cùng siêu tham số.
3. **Chỉ đánh giá final test một lần.** Nạp `split="final_test"` (21.782 vòng mùa 2025)
   và gọi `predict` từ artifact đã khóa. Báo cáo đủ bốn metric và số vòng cho run đó.
   Nếu sai số cao hơn validation, ghi nhận mức chênh và giới hạn khi dự đoán mùa mới.
4. **Lưu vết các lần thử.** Trong khi huấn luyện, lưu pipeline `.joblib` và đầu ra chi tiết ở `artifacts/` local.
   Đưa config, lệnh tái tạo và các bảng/ảnh dùng trong báo cáo vào Git. Mỗi bảng/ảnh
   trong `reports/models/` dẫn về `run_id`. Metadata ghi đủ các trường của
   `evaluation.provenance_metadata`, gồm cả hash dữ liệu/split/config, Git commit và
   trạng thái working tree. [Danh sách trường bắt buộc](../../configs/experiment.json)
5. **Cho demo và báo cáo.** Khánh, Nguyên, Khang và Danh mỗi người bàn giao cho Nguyên
   pipeline đã fit của mình, `run_id`, lệnh tái tạo, tên, kiểu và đơn vị của các cột đầu vào.
   Nhóm chọn cùng một `row_id` trên validation làm vòng mẫu. Mỗi người nạp lại pipeline,
   dự đoán vòng đó và ghi kết quả theo giây để Nguyên đối chiếu với app.
   Khánh gửi bảng MAE validation 2024 của bốn run bàn giao và đánh dấu run thắng.
   Khang nhận bảng so sánh validation, kết quả test của model đã khóa, bảng hệ số
   và nhận xét sai số của từng người. [F1-07 · Demo](F1-07-demo-video.md) · [F1-08 · Báo cáo](F1-08-report-outline.md)
6. **Dọn phiên bản thử trước khi kết thúc.** Sau khi Nguyên nạp đúng bốn run và đối
   chiếu dự đoán vòng mẫu, mỗi người xóa các binary của phiên bản cũ, bản sao và đầu
   ra tạm do mình tạo trong F1-06. Giữ config, metadata và bảng điểm của các lần thử
   để truy lại quyết định, cùng các đầu ra dùng trong báo cáo. Đối chiếu lần cuối:
   gói bàn giao có đúng bốn file model, mỗi mã M1–M4 đúng một file, và `run_id`
   của mỗi file khớp bảng điểm cùng hướng dẫn nạp cho F1-07.

**Hoàn tất F1-06 khi** một thành viên khác chạy lại được kết quả từ config đã lưu, cả bốn
model có đối chiếu trên cùng vòng validation và Nguyên nạp được cả bốn pipeline để tái tạo
bốn dự đoán trên vòng mẫu. Chốt thiết kế và kết luận trong 2–3 gạch ở Issue #5. PR từng phần ghi `Refs #5`,
PR tổng hợp hoàn tất task ghi `Closes #5`.

## 6. Chạy M1 và chấm CSV của từng thành viên

Sau khi đối chiếu F1-05 ở mục 2, chạy từ gốc repo với [config M1](../../configs/models/m1.json).
Lệnh sinh `run_id` mới, in đường dẫn báo cáo và lưu pipeline cùng CSV dự đoán vào
`artifacts/M1/<run_id>/` trên máy chạy.

```sh
uv run python -m scripts.train_m1
```

Nguyên, Khang và Danh tự fit pipeline của mình trên train. Với `frames = [train, val]`,
`pipeline` đã fit và `features` là danh sách cột đầu vào gốc của model, có thể xuất cả hai tập
bằng đoạn dưới. Đổi thư mục và mã lượt chạy theo model thực tế.

```python
from pathlib import Path
import pandas as pd

out = Path("artifacts/M2/m2-001/predictions.csv")
out.parent.mkdir(parents=True, exist_ok=True)
tables = []
for frame in frames:
    table = frame[["row_id", "split", "LapTimeSec"]].rename(columns={"LapTimeSec": "y_true"})
    table["y_pred"] = pipeline.predict(frame[features])
    tables.append(table)
pd.concat(tables).to_csv(out, index=False)
```

Chấm CSV đã lưu ở bất kỳ thứ tự dòng nào. Dùng target trực tiếp từ bộ nạp F1-05 và giữ
đủ độ chính xác khi xuất. Lệnh đối chiếu ID, split, target và giá trị hữu hạn trước khi
tính điểm từng vòng. Bỏ `--output` để chỉ in điểm, hoặc truyền đường dẫn CSV mới để lưu.

```sh
uv run python -m scripts.score_predictions --model M2 --run-id m2-001 --predictions artifacts/M2/m2-001/predictions.csv --output reports/models/M2/m2-001/scores.csv
```

### Bàn giao M1 của Khánh

Khánh giữ run chuẩn `m1-20261128T100841Z-eeeb6586`. Đây là run M1 duy nhất, nên chưa
cần phá hòa theo RMSE. OLS có intercept, số vòng chuẩn hóa bằng `StandardScaler`, chặng
mã hóa với `drop="first"`. Khánh giữ cách xử lý chặng mới đã chốt để M2–M3 có cùng
mốc so sánh. Điểm dưới đây làm tròn để đọc.
[Bảng điểm gốc](../../reports/models/M1/m1-20261128T100841Z-eeeb6586/scores.csv) ·
[Metadata, hash và vòng mẫu](../../reports/models/M1/m1-20261128T100841Z-eeeb6586/metadata.json)

| Tập | Số vòng | MAE (s) | RMSE (s) | MSE (s²) | R² |
|---|---:|---:|---:|---:|---:|
| Train 2022–2023 | 39.433 | 2,128571 | 3,838956 | 14,737583 | 0,892070 |
| Validation 2024 | 23.426 | 3,434873 | 4,757563 | 22,634410 | 0,779230 |

- **Sai số theo chặng.** Trên 694 vòng có nhãn chặng mới, M1 dự đoán nhanh hơn thực
  tế trung bình 10,776 s. Trong nhóm chặng đã thấy ở train, chặng có MAE cao nhất
  gồm 1.109 vòng, với dự đoán chậm hơn thực tế trung bình 8,907 s.
  [Sai số của 24 chặng](../../reports/models/M1/m1-20261128T100841Z-eeeb6586/validation_by_event.csv)
- **Hệ số** cho thấy ở cùng chặng, tăng 18,108 vòng (một độ lệch chuẩn train) ứng với
  dự đoán giảm 0,952 s. Mỗi hệ số chặng cho biết dự đoán chênh bao nhiêu giây so với
  chặng tham chiếu ở cùng số vòng.
  [Hệ số và intercept](../../reports/models/M1/m1-20261128T100841Z-eeeb6586/coefficients.csv)

F1-07 truyền một `DataFrame` có `EventName` (chuỗi tên chặng) và `LapNumber`
(số nguyên hoặc thực, đơn vị vòng). Trường `validation_example` trong metadata chứa
`row_id` chung, đầu vào và dự đoán mẫu theo giây để đối chiếu trên máy demo.

```python
from pathlib import Path
import json
import joblib
import pandas as pd

run = Path("reports/models/M1/m1-20261128T100841Z-eeeb6586")
meta = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
model = joblib.load(meta["artifact_path"])
sample = meta["validation_example"]
lap = pd.DataFrame([sample["input"]])
assert abs(model.predict(lap)[0] - sample["y_pred_s"]) < 1e-9
```
