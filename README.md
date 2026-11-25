# F1 Lap Time Regression

Dự án Machine Learning nghiên cứu và dự đoán thời gian hoàn thành một vòng đua Formula 1
bằng Linear Regression. Dự án tập trung vào thiết kế đặc trưng, đánh giá công bằng giữa
các mô hình và diễn giải kết quả trên dữ liệu nhiều mùa giải.

**Trạng thái:** đã có cấu trúc, môi trường, kiểm kê dữ liệu F1-02, giao thức thực nghiệm F1-03,
[notebook EDA F1-04 trên tập train](docs/eda-findings.ipynb) và pipeline chuẩn bị dữ liệu chung F1-05.
Mô hình và ứng dụng demo sẽ được triển khai trong các giai đoạn tiếp theo.

## Mục tiêu

- Xác định bài toán dự đoán lap time, đơn vị target và thời điểm các thông tin đầu vào có sẵn.
- Khảo sát chất lượng dữ liệu, phân tích khám phá và xây dựng quy trình tiền xử lý có thể tái tạo.
- Xây dựng, giải thích và so sánh ít nhất **4 mô hình Linear Regression**.
- Đánh giá bằng **MAE, MSE, R²**, bổ sung RMSE khi phù hợp; kiểm soát data leakage.
- Phát triển demo **Streamlit** cho phép nhập đặc trưng, dự đoán và xem biểu đồ diễn giải.

## Dữ liệu

**Multi-Season Formula 1 Lap Dataset with Race-Control-Based Safety Car Labels (2022–2025)**
do Tanvinur Rahman Siam công bố trên Mendeley Data.

- Nguồn: [Mendeley Data, phiên bản 1](https://data.mendeley.com/datasets/djr8rnjtjp/1).
- DOI: [10.17632/djr8rnjtjp.1](https://doi.org/10.17632/djr8rnjtjp.1).
- Giấy phép: **CC BY 4.0**.

Target là `LapTimeSec` tính bằng giây. [Giao thức F1-03](docs/experiment-protocol.md)
và [configs/experiment.json](configs/experiment.json) quy định bài toán, bộ lọc phạm vi
(loại Pre-Season, vòng pit và SC/VSC), temporal split (Train 2022–2023, Validation 2024,
Final Test 2025) và chống rò rỉ dữ liệu. Các mô hình dùng chung giao thức này.

Repo chia sẻ trực tiếp các file cần dùng trong `data/raw/` để cả nhóm clone về là có dữ liệu.
ZIP nguồn và cache FastF1 đã được xóa vì không cần cho kiểm kê; không cần tải lại để chạy
`uv run python -m scripts.download_data` và `uv run python -m scripts.inspect_data`.
[Kiểm kê F1-02](docs/data-inventory.md) ghi checksum, cách kiểm tra và các giới hạn cần
chốt trước khi làm EDA.

## Công nghệ

| Hạng mục | Công cụ |
|---|---|
| Ngôn ngữ và môi trường | Python 3.11, uv, `.venv` |
| Xử lý dữ liệu | pandas, NumPy |
| Mô hình và tiền xử lý | scikit-learn Pipeline |
| Lưu mô hình | joblib |
| Biểu đồ | Matplotlib, Seaborn |
| Giao diện demo | Streamlit chạy local |
| Chất lượng code | Ruff, pytest, GitHub Actions |

## Cài đặt và sử dụng môi trường

Cài [uv](https://docs.astral.sh/uv/getting-started/installation/), sau đó chạy từ thư mục repo để đồng bộ môi trường:

```sh
uv sync --locked
uv run python scripts/check_environment.py
```

Áp dụng trên Windows, macOS và Linux. `uv` sử dụng Python 3.11 và tạo `.venv` riêng cho project.

### 2 cách chạy lệnh

* **Cách 1 (Khuyên dùng):** Dùng tiền tố `uv run`, không cần kích hoạt môi trường thủ công:
  ```sh
  uv run python -m scripts.prepare_data
  uv run pytest -q
  ```
* **Cách 2 (Kích hoạt thủ công):** Bật `.venv` vào terminal rồi gõ lệnh trực tiếp:
  * Windows PowerShell: `.venv\Scripts\Activate.ps1`
  * Windows CMD: `.venv\Scripts\activate.bat`
  * macOS / Linux / Git Bash: `source .venv/bin/activate` (hoặc `source .venv/Scripts/activate` trên Git Bash)
  *(Khi thấy `(.venv)` xuất hiện ở đầu dòng lệnh, có thể gõ trực tiếp `python ...`, `pytest`, `ruff`. Để thoát gõ `deactivate`).*

### Quản lý thư viện (Dependencies)

* Khi bạn muốn **cài thêm thư viện mới** vào dự án:
  ```sh
  uv add <tên_thư_viện>
  ```
  `uv` sẽ tự động tải thư viện vào `.venv`, ghi nhận vào `pyproject.toml` và cập nhật phiên bản chính xác vào `uv.lock`.
* Khi **pull code mới từ nhóm về** (có thư viện mới từ người khác):
  ```sh
  uv sync --locked
  ```
* `pyproject.toml` khai báo dependencies, `uv.lock` khóa phiên bản dùng chung. `requirements.txt` là bản export phục vụ bàn giao và nộp bài. Khi thay dependencies:

```sh
uv export --frozen --no-dev --no-emit-project --output-file requirements.txt
```

## Cấu trúc thư mục

```text
src/f1_lap_time/   Mã nguồn Python dùng chung
scripts/          Các lệnh hỗ trợ phát triển
configs/          Cấu hình dữ liệu và thí nghiệm
data/             Dữ liệu local
app/              Ứng dụng Streamlit
reports/          Báo cáo, bảng và biểu đồ
tests/            Kiểm thử
.github/          Workflow CI và mẫu pull request
docs/tasks/       Hướng dẫn giao việc và bàn giao theo từng task
```

Git theo dõi dữ liệu nguồn trong `data/raw/`. Lệnh chuẩn bị tái tạo `base.csv`,
`splits.csv` và `metadata.json` trong `data/processed/` trên máy từng thành viên.

## Phát triển và kiểm tra

```sh
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
uv run python scripts/check_experiment_config.py
uv run python -m scripts.prepare_data
uv run python -m scripts.check_prepared_data --reference reports/data-preparation/checks.json --report tmp/f1-05-checks.json
```

CI chạy lint/format và pytest trên Linux; kiểm tra môi trường, checksum toàn bộ dữ liệu,
chuẩn bị dữ liệu và nghiệm thu trên Windows/Linux/macOS.

### Nạp dữ liệu nền cho mô hình (F1-06)

Sau khi chạy `scripts.prepare_data`, các thành viên nạp cùng phiên bản dữ liệu và split bằng hàm dùng chung:

```python
from f1_lap_time.data_preparation import load_prepared_data

# Nạp từng tập, kiểm tra checksum và reset index 0-based:
train_df, meta = load_prepared_data(split="train")  # 39.433 dòng (2022–2023)
val_df, _ = load_prepared_data(split="validation")  # 23.426 dòng (2024)
test_df, _ = load_prepared_data(split="test")  # 21.782 dòng (2025, hoặc dùng split="final_test")
```

Chỉ dùng các cột trong `features.allowed_candidate_features` của `configs/experiment.json`
làm đầu vào mô hình. `LapTimeSec` là target; `row_id`, `split`, `Year`, `Session`, `EventKey`
và `IsAccurate` chỉ dùng để chia tập/truy vết, không đưa vào ma trận đặc trưng.

Theo giao thức F1-03, nếu dùng tuổi lốp, Pipeline phải đổi `TyreLife` thành
`TyreLife_start = TyreLife - 1` trước khi điền thiếu. Với feature đã chọn, điền số bằng
trung vị, điền category bằng `UNKNOWN` và mã hóa với `handle_unknown="ignore"`. Chỉ fit
trên train hoặc fold train. Loại feature toàn khuyết trên train của lượt fit.

Lệnh kiểm tra ở trên đối chiếu hai CSV với SHA-256 trong báo cáo bàn giao và ghi kết quả
vào `tmp/f1-05-checks.json`. Khi chốt phiên bản dữ liệu mới, chạy
`uv run python -m scripts.check_prepared_data` để ghi báo cáo bàn giao. Metadata lưu
thời điểm và phiên bản code của từng lượt chạy nên hash của metadata có thể khác giữa các máy.

**Nói miệng để tranh luận — Issue chỉ để chốt hạ và lưu vết.** Nhóm dùng đúng một Issue
cho mỗi mã task lớn F1-01 đến F1-08. F1-05 chuẩn bị dữ liệu/split dùng chung; F1-06 xây
bốn model từ đầu vào đó, các biến đổi có học nằm trong Pipeline và chỉ fit trên train.
Trao đổi và góp ý trực tiếp/qua call; sau quyết định
quan trọng, người phụ trách ghi 2–3 gạch đầu dòng trong Issue tương ứng.
Làm trên nhánh riêng, mở **pull request**; PR hoàn tất task ghi `Closes #...` để Issue đóng
khi merge vào `main`. PR một phần ghi `Refs #...`.
Tài liệu giao việc chi tiết nằm tại [Hướng dẫn khởi động nhóm](docs/tasks/README.md).
Mỗi thành viên đọc trang task của mình để nắm phạm vi, các bước làm, đầu ra và tiêu chí
nghiệm thu. Tài liệu kỹ thuật và các task tiếp theo được bổ sung khi có kết quả và quyết định
từ bước trước.

## Thành viên và sản phẩm bàn giao

**Khánh, Khang, Danh, Nguyên.** Hạn nộp: **11/12/2026**.

Sản phẩm gồm mã nguồn và môi trường tái tạo được, dataset hoặc đường dẫn truy cập,
ứng dụng demo, video và báo cáo PDF. Yêu cầu chi tiết nằm trong
[Lab 01 – Linear Regression](<Lab 01 - Linear Regression.pdf>).
