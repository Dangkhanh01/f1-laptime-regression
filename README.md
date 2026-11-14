# F1 Lap Time Regression

Dự án Machine Learning nghiên cứu và dự đoán thời gian hoàn thành một vòng đua Formula 1
bằng Linear Regression. Dự án tập trung vào thiết kế đặc trưng, đánh giá công bằng giữa
các mô hình và diễn giải kết quả trên dữ liệu nhiều mùa giải.

**Trạng thái:** đã có cấu trúc, môi trường và kiểm kê dữ liệu F1-02. Pipeline tiền xử lý,
mô hình và ứng dụng demo sẽ được triển khai trong các giai đoạn tiếp theo.

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

Target dự kiến là lap time tính bằng giây. Các cột sử dụng, quy tắc lọc và cách chia
train/validation/test sẽ được xác định sau khi khảo sát dữ liệu. Các mô hình phải dùng chung
giao thức đánh giá; không sử dụng thông tin chưa có tại thời điểm dự đoán.

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

## Cài đặt

Cài [uv](https://docs.astral.sh/uv/getting-started/installation/), sau đó chạy từ thư mục repo:

```sh
uv sync --locked
uv run python scripts/check_environment.py
```

Áp dụng trên Windows, macOS và Linux. `uv` sử dụng Python 3.11 và tạo `.venv` riêng cho
project; không cần activate môi trường khi dùng `uv run`.

`pyproject.toml` khai báo dependencies, `uv.lock` khóa phiên bản dùng chung.
`requirements.txt` là bản export phục vụ bàn giao và nộp bài. Khi thay dependencies:

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

Hiện các thư mục nghiệp vụ mới là bộ khung. `.gitignore` không loại trừ thư mục nào trong
`data/`; nhóm có thể chia sẻ dữ liệu cần thiết qua Git mà không phải tải riêng. Model artifacts
và môi trường Python vẫn bị loại khỏi Git.

## Phát triển và kiểm tra

```sh
uv run ruff check .
uv run ruff format --check .
```

CI kiểm tra lint/format và môi trường trên Windows/Linux/macOS. pytest đã được cấu hình;
khi bổ sung tests, chạy `uv run pytest -q` và bật bước pytest trong workflow.

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
