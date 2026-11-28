# Giao thức thực nghiệm (Experiment Protocol)

Giao thức này quy định bài toán, nguồn dữ liệu, bộ lọc phạm vi, phân chia tập dữ liệu (train/validation/test), danh mục đặc trưng và quy chuẩn đánh giá chung cho toàn bộ dự án **F1 Lap Time Regression**.

Cấu hình thực thi bằng máy đọc được lưu tại [configs/experiment.json](../configs/experiment.json). Mọi thành viên và các task tiếp theo ([F1-04](tasks/F1-04-exploratory-analysis.md), [F1-05](tasks/F1-05-shared-data-preparation.md), [F1-06](tasks/F1-06-models-evaluation.md)) phải tuân thủ nghiêm ngặt các quy định dưới đây.

---

## 1. Định nghĩa bài toán và mục tiêu nghiên cứu

- **Bài toán**: Dự đoán thời gian hoàn thành một vòng đua của một tay đua (tính bằng giây), **trước khi vòng đua đó bắt đầu**.
- **Loại bài toán**: Hồi quy (Regression).
- **Đơn vị quan sát (Observation Unit)**: Một dòng dữ liệu tương ứng với **một vòng đua của một tay đua trong một chặng đua chính thức** (`Session = 'R'`).
- **Khóa định danh quan sát (Candidate Key)**: Bộ 5 trường `(Year, EventName, Session, Driver, LapNumber)`. Trên dữ liệu chuẩn, khóa này không trùng lặp và không có giá trị khuyết.
- **Biến mục tiêu (Target)**:
  - Cột mục tiêu: `LapTimeSec`.
  - Đơn vị: **Giây (seconds)**.
  - Căn cứ xác minh: `LapTimeSec` trùng khớp 100% về mặt số học với cột `LapTime` và bằng tổng của ba sector (`Sector1Time + Sector2Time + Sector3Time`). Giá trị quan sát dao động trong khoảng hợp lệ từ **67,012s đến 149,321s**.
- **Giới hạn bài toán**:
  - Mô hình được huấn luyện và đánh giá trên các vòng đua sau đó được xác nhận là **vòng đua thông thường** (các vòng đua chính thức không có pit và không có SC/VSC).
  - Trước khi vòng đua bắt đầu, trên thực tế không thể đảm bảo chắc chắn vòng đó có xảy ra sự cố SC/VSC hay tay đua đột ngột vào pit hay không. Vì vậy, mô hình dự đoán hiệu năng xe và tay đua trong điều kiện đua thông thường, không quảng bá như một mô hình dự đoán mọi tình huống bất thường.

---

## 2. Nguồn dữ liệu và căn cứ lựa chọn

- **Nguồn dữ liệu chính**: File Parquet tổng hợp `data/raw/f1_research_dataset_with_risk_safetycar.parquet` từ bộ dữ liệu Mendeley Data (DOI: `10.17632/djr8rnjtjp.1`, CC BY 4.0).
- **Kích thước nguồn ban đầu**: 89.980 dòng, 35 cột, bao gồm 4 mùa giải (2022–2025).
- **Căn cứ xác minh đối chiếu với 95 file sự kiện raw**:
  1. Trong 95 file sự kiện raw (tổng cộng 104.140 dòng), quan sát thấy có **3.554 dòng vào pit (`PitInTime`)** và **3.548 dòng ra pit (`PitOutTime`)**.
  2. Toàn bộ **100% các dòng vào pit và ra pit quan sát được này đều mang cờ `IsAccurate = False`** theo phân loại của FastF1 (thời gian vòng bị kéo dài do di chuyển qua đường pit lane).
  3. Tập khóa của file tổng hợp khớp tập khóa `IsAccurate = True` trong 95 file sự kiện được đối chiếu. Do đó, **toàn bộ các vòng vào pit và ra pit quan sát được đã không xuất hiện trong file tổng hợp**. Điều này giải thích vì sao trong file tổng hợp `PitInTime`/`PitOutTime` khuyết 100% và `HasPitIn`/`HasPitOut` toàn bộ bằng 0.
  4. File tổng hợp đã được tác giả bổ sung nhãn SC/VSC chuẩn hóa từ Race Control Messages: có 621 dòng `IsSafetyCar = 1` và 243 dòng `IsVSC = 1`. Trong khi đó, ở 95 file sự kiện trên các dòng `IsAccurate = True`, các cờ này chưa được cập nhật (bằng 0).
  5. Vì các lý do trên, việc sử dụng file tổng hợp là lựa chọn nhất quán, chuẩn xác và đầy đủ thông tin nhãn sự cố nhất.

---

## 3. Phạm vi nghiên cứu và bộ lọc dữ liệu (Scope & Filtering)

Mẫu nghiên cứu chỉ bao gồm các **vòng đua chính thức ngoài Pre-Season, không ghi nhận Safety Car hoặc Virtual Safety Car theo nhãn Race Control của tập dữ liệu**. Quy tắc lọc dữ liệu cụ thể như sau:

| Tiêu chí | Quy tắc lọc | Số dòng loại trừ | Căn cứ và lý giải |
|---|---|---|---|
| **Phiên đua chính thức** | `Session == 'R'` | 0 dòng | Toàn bộ 89.980 dòng trong file tổng hợp đều thuộc phiên đua chính thức `R`. |
| **Loại bỏ Pre-Season** | `~EventName.str.startswith('Pre-Season')` | **4.526 dòng** | Loại 3 sự kiện thử nghiệm trước mùa: `Pre-Season Test` (778 dòng), `Pre-Season Track Session` (674 dòng) và `Pre-Season Testing` (3.074 dòng). |
| **Loại bỏ Safety Car & VSC** | `IsSafetyCar == 0 and IsVSC == 0` | **813 dòng** (ngoài Pre-Season) | Loại bỏ các vòng bị trung hòa tốc độ bởi xe an toàn (SC) hoặc xe an toàn ảo (VSC) theo nhãn Race Control. |
| **Loại bỏ vòng Pit (Inlap/Outlap)** | Tự động loại từ nguồn | 0 dòng bổ sung | Các vòng vào pit và ra pit quan sát được đều có `IsAccurate = False` nên đã không xuất hiện trong file tổng hợp. |
| **Vòng xuất phát (Standing Start)** | Giới hạn dữ liệu nguồn | 0 dòng bổ sung | Trên bộ dữ liệu được khảo sát, không có bản ghi nào có `LapNumber = 1.0` (dữ liệu bắt đầu từ `LapNumber = 2.0`). Đây là đặc điểm của dữ liệu nguồn do tập khóa file tổng hợp khớp tập khóa `IsAccurate = True` trong các file sự kiện. |
| **Kiểm tra tính hợp lệ của target** | `LapTimeSec` phải là số hữu hạn, lớn hơn 0. Dừng nếu có vi phạm trên các dòng trong phạm vi. | 0 dòng | Dữ liệu hiện tại đạt kiểm tra này. |

- **Tổng số dòng hợp lệ sau lọc**: **84.641 dòng**.
- **Lưu ý về trạng thái cờ**: Sau khi loại SC/VSC, dữ liệu vẫn còn 2.612 dòng có cờ vàng cục bộ (`TrackStatus` là `'12'`, `'21'`, hoặc `'2'`). Do đó, không đồng nhất mẫu nghiên cứu này với trạng thái "green flag toàn phần".
- **Rà soát sau EDA**. Giữ bộ lọc trong config. Trên 39.433 vòng train, cờ IQR tính riêng
  theo mùa, chặng và phiên đánh dấu 115 vòng dưới ngưỡng và 466 vòng trên ngưỡng để rà soát.
  Giữ các vòng được đánh dấu trong bộ dữ liệu; quy tắc IQR chỉ phục vụ chẩn đoán.
  [Bảng thống kê target](../reports/eda/target-summary.csv)

---

## 4. Phân chia tập dữ liệu (Split Protocol)

Dữ liệu được phân chia theo **thời gian (Temporal Split) theo mùa giải** nhằm phản ánh đúng bài toán thực tế: mô hình học từ các mùa giải đã qua và dự đoán cho các mùa giải tương lai mà không bị rò rỉ thông tin (data leakage).

```
Dữ liệu hợp lệ (84.641 dòng)
├── Train Set (2022–2023):       39.433 dòng (46,59%) - 24 tên EventName, 43 nhóm sự kiện (Year, EventName, Session), 25 tay đua, 10 đội
├── Validation Set (2024):       23.426 dòng (27,68%) - 24 tên EventName, 24 nhóm sự kiện, 24 tay đua, 10 đội
└── Final Test Set (2025):       21.782 dòng (25,73%) - 23 tên EventName, 23 nhóm sự kiện, 21 tay đua, 10 đội
```

### Nguyên tắc sử dụng các tập dữ liệu:
1. **Train (Mùa 2022–2023)**:
   - Dùng để khám phá phân tích dữ liệu (EDA trong F1-04).
   - Dùng để fit các bộ tiền xử lý (Imputer, Scaler, Encoder) và huấn luyện các mô hình Linear Regression trong F1-06.
   - Khi áp dụng Cross-Validation, việc fit tiền xử lý phải thực hiện độc lập bên trong từng fold huấn luyện.
2. **Validation (Mùa 2024)**:
   - Dùng để đánh giá độc lập giữa các mô hình, tinh chỉnh siêu tham số (hyperparameters), và **chọn ra mô hình tốt nhất**.
   - Các phép biến đổi trên tập này chỉ là `transform` từ pipeline đã fit trên Train.
3. **Final Test (Mùa 2025)**:
   - **TẬP KIỂM ĐỊNH CUỐI CÙNG DUY NHẤT**: Chỉ được đánh giá một lần sau khi đã khóa cứng toàn bộ thiết kế mô hình, pipeline tiền xử lý và siêu tham số.
   - **TUYỆT ĐỐI CẤM**: Không sử dụng tập Final Test để chọn mô hình, chọn đặc trưng, điều chỉnh bộ lọc, chọn ngưỡng hoặc fit bất kỳ bước tiền xử lý nào.

---

## 5. Danh mục đặc trưng và chống rò rỉ dữ liệu (Feature Availability & Leakage Prevention)

Nguyên tắc cốt lõi: **Chỉ những thông tin thực sự có sẵn trước khi vòng đua bắt đầu mới được phép làm đặc trưng đầu vào**.

### 5.1. Bảng phân loại chi tiết toàn bộ 35 cột dữ liệu raw

| Tên cột | Ý nghĩa và đơn vị | Thời điểm sẵn có | Phân loại | Căn cứ và quyết định |
|---|---|---|---|---|
| `LapTimeSec` | Thời gian hoàn thành vòng (giây) | Sau khi vòng kết thúc | **Target** | Biến mục tiêu dự đoán. Cấm làm feature. |
| `LapTime` | Bản sao thời gian vòng (giây) | Sau khi vòng kết thúc | **Bị cấm (Forbidden)** | Trùng khớp 100% với target. Leakage. |
| `Sector1Time`, `Sector2Time`, `Sector3Time` | Thời gian từng sector (giây) | Trong và sau vòng | **Bị cấm (Forbidden)** | Diễn biến diễn ra trong vòng đua. Leakage. |
| `Time` | Mốc thời gian kết thúc vòng (giây) | Sau khi vòng kết thúc | **Bị cấm (Forbidden)** | Thời điểm xe chạm vạch đích cuối vòng. Leakage. |
| `Position` | Thứ hạng xe khi qua vạch đích | Sau khi vòng kết thúc | **Bị cấm (Forbidden)** | Vị trí cuối vòng L. Cấm dùng trực tiếp cho vòng L. |
| `TrackStatus` | Chuỗi mã trạng thái cờ đường đua | Trong vòng | **Bị cấm (Forbidden)** | Diễn biến cờ xảy ra trong vòng. Leakage. |
| `IsSafetyCar`, `IsVSC` | Cờ SC / VSC từ Race Control | Trong vòng | **Bị cấm (Forbidden)** | Diễn biến trong vòng; chỉ dùng để lọc phạm vi. |
| `PitInTime`, `PitOutTime`, `HasPitIn`, `HasPitOut` | Thông tin pit lane | Trong/sau vòng | **Bị cấm (Forbidden)** | Thuộc tính sự cố pit (khuyết 100% trên clean data). |
| `RaceProgress` | Tiến độ cuộc đua | Chưa xác minh | **Bị cấm (Forbidden)** | Công thức tác giả tự sinh chưa công bố; nguy cơ leakage từ tổng số vòng thực tế. |
| `Risk_SafetyCar` | Điểm rủi ro SC của tác giả | Mô hình ngoài | **Bị cấm (Forbidden)** | Đầu ra mô hình của nghiên cứu trước, không có trong thực tế. |
| `AirTemp`, `TrackTemp`, `Humidity`, `Pressure`, `Rainfall`, `WindSpeed`, `WindDirection` | Các thông số khí tượng trạm đo | Chưa xác minh | **Chưa được phép dùng (Pending)** | Khảo sát trên Train cho thấy TrackTemp khác nhau giữa các xe cùng vòng trong 62,52% nhóm vòng. Chưa rõ thời điểm lấy mẫu và cách ghép từ nguồn. Tạm thời chưa được phép dùng để tránh rò rỉ thông tin. |
| `Driver` | Mã tay đua (str, ví dụ: VER, HAM) | Trước vòng đua | **Được phép (Allowed)** | Biết rõ tay đua nào chuẩn bị chạy vòng L. |
| `Team` | Đội đua (str, ví dụ: Red Bull Racing) | Trước vòng đua | **Được phép (Allowed)** | Thuộc tính xe và đội đua. |
| `EventName` | Tên chặng đua Grand Prix | Trước vòng đua | **Được phép (Allowed)** | Đặc trưng đường đua / địa điểm thi đấu. |
| `LapNumber` | Số thứ tự vòng đua (float/int) | Trước vòng đua | **Được phép (Allowed)** | Khi chuẩn bị chạy vòng $L$, số $L$ đã được xác định. |
| `Compound` | Loại lốp đang lắp (HARD, MEDIUM, ...) | Trước vòng đua | **Được phép (Allowed)** | Bộ lốp đã được gắn vào xe trước khi rời pit. |
| `FreshTyre` | Cờ lốp mới khi bắt đầu stint (bool) | Trước vòng đua | **Được phép (Allowed)** | Biết rõ trạng thái lốp khi xuất phát stint. |
| `Stint` | Số thứ tự stint trong chặng (float) | Trước vòng đua | **Được phép (Allowed)** | Biết rõ tay đua đang ở stint thứ mấy. |
| `TyreLife` | Số vòng đã chạy trên bộ lốp | Trước/cuối vòng | **Được phép (Allowed)** | Trong FastF1, `TyreLife` là số vòng lốp đã chạy tính đến cuối vòng $L$. Cần chuyển đổi thành `TyreLife_start`. |
| `Year`, `Session`, `EventKey`, `IsAccurate` | Thuộc tính mùa giải và kỹ thuật | Quản lý hệ thống | **Chỉ dùng Split/Truy vết** | Không đưa vào ma trận đặc trưng huấn luyện. |

Giữ danh sách đặc trưng ứng viên trong config. F1-06 có thể dùng `EventName` làm mô hình mốc,
rồi đo mức cải thiện khi thêm `LapNumber`, `Compound` và `TyreLife_start` trên validation 2024.
Trên train, `LapNumber` tương quan âm với lap time ở 42/43 phiên. Chênh lệch giữa các loại lốp
và tương quan của tuổi lốp thay đổi theo phiên. Đây là căn cứ để so sánh các thiết kế bằng MAE.
[Notebook F1-04](eda-findings.ipynb)

### 5.2. Đặc tả đặc trưng dẫn xuất trong Pipeline
- **`TyreLife_start`**:
  - **Công thức**: $\text{TyreLife\_start} = \text{TyreLife} - 1.0$ (khi $\text{TyreLife} \ge 1.0$).
  - **Ý nghĩa**: Số vòng bộ lốp đã hoàn thành **trước khi** vòng đua hiện tại bắt đầu.
  - **Phân công và ranh giới**:
    + F1-05 giữ raw `TyreLife` làm thuộc tính đầu vào chuẩn trong file `data/processed/base.csv`.
    + Khi mô hình dùng tuổi lốp, Pipeline phải chuyển raw `TyreLife` thành `TyreLife_start`
      và chỉ đưa cột dẫn xuất vào ma trận đặc trưng.
  - **Xử lý khuyết**. Giữ 428 dòng thiếu `TyreLife` trên train. Giá trị `TyreLife_start`
    tương ứng vẫn khuyết cho tới bước điền trung vị trong Pipeline ở Mục 6.2.
  - **Giá trị không hợp lệ và thứ tự xử lý**: Giá trị không phải số, vô cực hoặc hữu hạn nhỏ hơn 1 phải báo lỗi và dừng để xác minh; không tự ép kiểu, đổi thành missing hoặc loại dòng. Với giá trị hợp lệ, trừ 1 trước, giữ NaN là missing, sau đó fit imputer trên `TyreLife_start` của Train và chỉ transform các tập còn lại/app. Đây là đặc tả cho F1-05/F1-06, chưa triển khai biến đổi mô hình trong F1-03.
- **Chính sách về Lagged Features (`PrevLapTime`, `PrevPosition`)**:
  - Khảo sát trên Train (2022–2023) cho thấy có 2.838 dòng (7,20%) không có vòng liền kề nghiêm ngặt $L-1$: 832 dòng đầu tiên quan sát được theo tay đua/sự kiện/phiên và 2.006 dòng có khoảng cách số vòng lớn hơn 1 sau lọc. Phép đo này chưa xác định nguyên nhân từng khoảng trống, không quy tất cả cho pit/SC.
  - Đồng thời, việc sử dụng biến trễ đòi hỏi cơ chế truyền lịch sử tuần tự chuyên biệt và có nguy cơ rò rỉ target trong scikit-learn Pipeline khi inference từng dòng đơn lẻ.
  - **Quyết định**: **Chưa cho phép dùng Lag features** ở phiên bản giao thức này. Các mô hình F1-06 sẽ tập trung khai thác các đặc trưng bối cảnh đầu vòng tĩnh và trạng thái lốp trước vòng.
- **Lưu ý giải trình về khảo sát bổ sung**:
  - Trong khảo sát ban đầu, thống kê biến thiên thời tiết nội vòng từng bị tính nhầm trên toàn bộ 84.641 dòng dữ liệu sạch (bao gồm cả mùa 2024 và 2025).
  - Nay đã được sửa hoàn toàn: 100% các đo đạc khảo sát bổ sung phục vụ thiết kế giao thức (tỷ lệ khuyết `TyreLife`, giá trị giả khuyết `'None'`/`'nan'` trong `Compound`, độ hở của lag features, và biến thiên nhiệt độ `TrackTemp` nội vòng) đều được đo đạc độc quyền trên tập Train (2022–2023, 39.433 dòng). Kết quả được lưu tại [reports/protocol_survey.json](../reports/protocol_survey.json). Không có bất kỳ thống kê thiết kế nào được lấy từ tập Validation hay Final Test.
  - Tái tạo từ gốc repo bằng `uv run python -m scripts.survey_protocol_evidence`. Script bắt buộc đối chiếu file nguồn với config và manifest trước khi khảo sát. Báo cáo lưu config path/hash và snapshot, code commit/dirty flag, hash các file code khảo sát, khóa nhóm có `Session` và lệnh tái tạo. Nếu không đọc được Git, commit/dirty flag là `null`, không được hiểu là working tree sạch. Thống kê phải tái tạo giống nhau khi đầu vào/phương thức không đổi; provenance có thể đổi khi config/code thay đổi. Dirty flag chỉ ghi nhận trạng thái tại lúc khảo sát, không thay thế việc lưu phiên bản code.

---

## 6. Quy tắc tiền xử lý dữ liệu và Scikit-Learn Pipeline

Để đảm bảo tính độc lập và khả năng tái lập, dự án phân định rõ ràng giữa **tiền xử lý dữ liệu chung (F1-05)** và **tiền xử lý có học trong mô hình (F1-06)**.

### 6.1. Trách nhiệm của F1-05 (Chuẩn bị dữ liệu chung)
- Thực hiện lọc phạm vi theo đúng Mục 3 trên file raw để tạo ra đúng **84.641 dòng** sạch.
- **Chuẩn hóa giá trị giả thiếu**. Đổi `'None'` và `'nan'` trong `Compound` thành giá trị
  khuyết. Giữ các dòng thiếu đặc trưng. Trên train có 327 dòng thiếu `Compound` sau chuẩn hóa
  và 428 dòng thiếu `TyreLife`. F1-05 đối chiếu các số này, kể cả tại Austrian GP 2022
  (259 và 307 trên 1.176 vòng). [Bảng thống kê theo phiên](../reports/eda/group-summary.csv)
- **Kiểm tra trùng lặp khóa định danh**: Nếu phát hiện trùng lặp khóa `(Year, EventName, Session, Driver, LapNumber)` mà giá trị xung đột nhau thì phải báo lỗi và dừng; nếu trùng lặp hoàn toàn thì loại bỏ bản sao. (Trên raw hiện tại có 0 dòng trùng lặp).
- Giữ nguyên tên đội đua gốc (`Team`) từ dữ liệu FastF1 để không áp đặt giả định chủ quan.
- Xuất các file local:
  + `data/processed/base.csv` chứa `row_id` ổn định dựa trên khóa ở Mục 1, các cột khóa, target
    và đặc trưng ứng viên từ config. Giữ `LapNumber`, `Stint`, `TyreLife` ở kiểu số,
    `FreshTyre` ở kiểu boolean và `LapTimeSec` ở đơn vị giây.
  + `data/processed/splits.csv` ánh xạ mỗi `row_id` sang đúng một nhãn `train`,
    `validation` hoặc `test` (`final_test` trong config).
  + `data/processed/metadata.json` lưu config đã dùng, phiên bản code, hash dữ liệu nguồn và
    đầu ra, số dòng từng bước và từng tập.

### 6.2. Trách nhiệm của F1-06 (Biến đổi có học trong Scikit-Learn Pipeline)
Tất cả các bước biến đổi có học từ dữ liệu **bắt buộc phải nằm bên trong `sklearn.pipeline.Pipeline`**:
1. **Xử lý giá trị khuyết (Missing Values)**:
   - Với các biến số như `Stint` và `TyreLife_start`, dùng `SimpleImputer(strategy='median')`
     nếu cần điền thiếu. Áp dụng phép trừ ở Mục 5.2 trước khi điền `TyreLife_start`.
   - Với các biến phân loại như `Compound`, dùng
     `SimpleImputer(strategy='constant', fill_value='UNKNOWN')`.
   - Fit imputer trên train hoặc fold train, rồi transform validation, final test và đầu vào app.
   - **Quy tắc cột toàn khuyết**: Nếu một đặc trưng bị khuyết 100% trên tập Train, đặc trưng đó phải bị loại bỏ khỏi không gian huấn luyện của mô hình; không được lấy giá trị từ Validation/Test để điền.
2. **Mã hóa biến phân loại (Categorical Encoding) và xử lý Category mới**:
   - Sử dụng `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` cho các cột phân loại (`Driver`, `Team`, `EventName`, `Compound`).
   - Giữ nhãn loại lốp gốc, kể cả `INTERMEDIATE` và `WET`. Giới hạn tối thiểu 30 vòng mỗi
     nhóm trong EDA chỉ áp dụng cho thống kê so sánh theo phiên.
   - Cấu hình `handle_unknown='ignore'` là **bắt buộc** để pipeline không bị lỗi khi gặp các tay đua mới xuất hiện ở mùa 2024 (`BEA`, `COL`, `DOO`) và mùa 2025 (`ANT`, `BOR`, `HAD`), cũng như chặng đua `Chinese Grand Prix` (chỉ xuất hiện từ 2024) hoặc khi người dùng nhập giá trị mới trên ứng dụng Streamlit.
3. **Chuẩn hóa thang đo (Scaling)**:
   - Các biến số được chuẩn hóa bằng `StandardScaler()` (hoặc `RobustScaler()`) fit trên Train, transform trên Validation/Test.
4. **Nguyên tắc ranh giới**:
   - `pipeline.fit(X_train, y_train)`
   - `pipeline.predict(X_val)` / `pipeline.predict(X_test)`
   - Tuyệt đối không được gọi `fit` hay `fit_transform` trên bất kỳ dữ liệu nào ngoài Train.

---

## 7. Thước đo đánh giá, tiêu chí chọn mô hình và quy tắc phá hoà

### 7.1. Thước đo đánh giá (Evaluation Metrics)
Mọi thí nghiệm và báo cáo kết quả đều phải tính toán và công bố đầy đủ 4 thước đo sau trên từng tập dữ liệu được đánh giá:

| Thước đo | Ký hiệu | Đơn vị | Ý nghĩa trong bài toán F1 |
|---|---|---|---|
| **Mean Absolute Error** | **MAE** | **Giây (s)** | Sai số tuyệt đối trung bình giữa lap time dự đoán và lap time thực tế. Trực quan, dễ giải thích với chuyên gia F1. |
| **Root Mean Squared Error** | **RMSE** | **Giây (s)** | Căn bậc hai sai số toàn phương trung bình. Phạt nặng hơn các vòng dự đoán sai lệch lớn. |
| **Mean Squared Error** | **MSE** | **Giây² (s²)** | Sai số toàn phương trung bình, phục vụ tối ưu hóa hàm mất mát của Linear Regression. |
| **Coefficient of Determination** | **R²** | **Không đơn vị** | Tỷ lệ phương sai của lap time được giải thích bởi mô hình. **Tuyệt đối không gọi R² là "phần trăm độ chính xác"**. |

### 7.2. Tiêu chí chọn mô hình tốt nhất (Model Selection Criterion)
- **Tiêu chí chính** là MAE thấp nhất trên tập validation 2024, với quy tắc phá hòa bên dưới. App demo chạy trực tiếp cả bốn mô hình và đánh dấu mô hình thắng theo validation. Chỉ mô hình thắng đã khóa được đánh giá một lần trên final test 2025.
- **Quy tắc phá hoà (Tie-breaking Rule)**:
  - Nếu hai hoặc nhiều mô hình có giá trị Validation MAE chênh lệch rất nhỏ ($\le 0,005$ giây):
    1. Ưu tiên mô hình có **Validation RMSE thấp hơn**.
    2. Nếu RMSE vẫn tương đương (chênh lệch $\le 0,005$ giây), ưu tiên **mô hình có cấu trúc đơn giản hơn / ít số lượng đặc trưng hơn** (tuân thủ nguyên lý Occam's Razor, giảm nguy cơ quá khớp và dễ giải thích hơn).

---

## 8. Quản lý vết thực nghiệm (Provenance) và Quy tắc diễn giải kết quả

### 8.1. Metadata bắt buộc cho mỗi lượt chạy (Run Provenance)
Mỗi mô hình khi huấn luyện và đánh giá ở F1-06 phải lưu kèm metadata có thể truy xuất nguồn gốc:
- `run_id`: Định danh duy nhất của lượt chạy (ví dụ: `ridge_baseline_v1_20261116`).
- `timestamp_utc`: Thời gian thực thi chuẩn UTC.
- `data_source_sha256`: Checksum SHA-256 của file raw `data/raw/f1_research_dataset_with_risk_safetycar.parquet`.
- `processed_base_sha256`: Checksum SHA-256 của file `data/processed/base.csv`.
- `processed_splits_sha256`: Checksum SHA-256 của file `data/processed/splits.csv`.
- `config_path`: Đường dẫn file config được sử dụng (`configs/experiment.json`).
- `config_sha256`: Checksum SHA-256 của file config tại thời điểm chạy.
- `split_version`: Định danh phiên bản phân chia tập dữ liệu.
- `git_commit_hash`: Mã commit Git tại thời điểm chạy.
- `git_dirty_flag`: Cờ boolean ghi nhận working tree có thay đổi chưa commit hay không.
- `model_class`: Tên lớp thuật toán (`LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`...).
- `hyperparameters`: Từ điển siêu tham số chi tiết (`alpha`, `l1_ratio`...).
- `random_seed`: Seed ngẫu nhiên (nếu có chia fold cross-validation).
- `features_list`: Danh sách đặc trưng đầu vào thực tế.
- `metrics`: Kết quả MAE, RMSE, MSE, R² ghi rõ tập Train và tập Validation.
- `artifact_path`: Đường dẫn lưu trữ model binary (`.joblib`).

### 8.2. Bốn quy tắc diễn giải kết quả bắt buộc (Interpretation Rules)
Theo yêu cầu của Lab 01 và dự án, toàn bộ thành viên khi viết báo cáo hoặc phân tích hệ số mô hình phải tuân thủ nghiêm ngặt:
1. **Dựa hoàn toàn trên kết quả thực tế**: Mọi nhận xét về dữ liệu và phân tích hệ số (coefficients) phải đối chiếu trực tiếp với một `run_id` thật, bảng hệ số thật và các độ đo thực tế; không suy diễn theo cảm tính hoặc đưa ra kết luận không có số liệu chứng minh.
2. **Nắm vững bản chất toán học, scaling và encoding**:
   - Giải thích rõ **dấu (+ / -)** và **đơn vị** của từng hệ số sau khi chuẩn hóa.
   - Đối với mô hình OLS thuần có intercept, OneHotEncoder bắt buộc dùng `drop='first'` để tránh bẫy biến giả (dummy variable trap); khi đó hệ số thể hiện độ lệch so với nhóm tham chiếu (reference category). Cần lưu ý `drop='first'` chỉ xử lý hoàn toàn đa cộng tuyến hoàn hảo trong từng biến phân loại riêng lẻ; nếu có quan hệ phụ thuộc chéo giữa các biến (ví dụ Driver và Team có thể phụ thuộc chặt chẽ), hiện tượng đa cộng tuyến vẫn có thể tồn tại và cần cẩn trọng khi diễn giải t-statistic hoặc p-value.
   - Đối với mô hình Regularized (Ridge, Lasso) dùng `drop=None`, mỗi category có hệ số riêng đo lường độ lệch trực tiếp so với intercept dưới tác động của hàm phạt chuẩn (regularization penalty).
3. **Không suy diễn tương quan thành nhân quả**: Mối quan hệ giữa các biến đầu vào và thời gian vòng trong mô hình tuyến tính chỉ phản ánh tương quan thống kê quan sát được trên dữ liệu đua, không khẳng định quan hệ nhân quả vật lý trực tiếp.
4. **Không sao chép nguyên văn từ AI**: Các công cụ AI chỉ đóng vai trò gợi ý tham khảo. Từng thành viên phải tự mình đối chiếu từng câu chữ với số liệu chạy thật và viết lại bằng hiểu biết của chính mình; tuyệt đối không đưa vào báo cáo các giải thích tự động chưa được kiểm chứng.

---

## 9. Hướng dẫn bàn giao cho các task tiếp theo

- [F1-04](eda-findings.ipynb) khảo sát **train 2022–2023**. Các bảng EDA dùng cùng nguồn,
  bộ lọc, split và cách chuẩn hóa `Compound` với bản bàn giao. [Metadata EDA](../reports/eda/metadata.json)
- **Bàn giao cho Nguyên ([F1-05 — Chuẩn bị dữ liệu chung](tasks/F1-05-shared-data-preparation.md))**:
  + Đọc config để áp dụng bộ lọc và split ở Mục 3–4, sau đó chuẩn hóa và xuất dữ liệu theo Mục 6.1.
  + Kiểm tra `base.csv` và `splits.csv` cùng có **84.641 dòng** và khớp từng `row_id`.
    Đối chiếu **39.433 / 23.426 / 21.782** dòng cho train/validation/test cùng các số đếm
    giá trị thiếu trên train ở Mục 6.1.
- **Bàn giao cho cả nhóm ([F1-06 — Mô hình và đánh giá](tasks/F1-06-models-evaluation.md))**:
  + Bốn mô hình Linear Regression dùng chung đúng một bộ dữ liệu nền và ánh xạ split từ F1-05.
  + Mọi biến đổi đặc trưng phải nằm trong `Pipeline` và tuân thủ nguyên tắc không rò rỉ dữ liệu (chỉ fit trên Train, xử lý category mới qua `OneHotEncoder(handle_unknown='ignore')`).
  + Chọn mô hình xuất sắc nhất theo MAE trên tập Validation (2024).
