# F1-01 — Khởi động nhóm

**Phụ trách:** Khánh. **Issue:** #1. **Bắt đầu:** ngay.

Mục tiêu: mọi người truy cập repo, chạy được môi trường và bắt đầu nhiệm vụ của mình.
Đọc [cách làm việc chung](README.md); trao đổi lỗi và hướng xử lý bằng miệng/call.

## Làm ở file nào

| File/vị trí | Cần làm |
|---|---|
| `README.md` | Đã có; cả nhóm làm theo hướng dẫn cài đặt, Khánh sửa nếu hướng dẫn thiếu/sai |
| `scripts/check_environment.py` | Đã có; mỗi người chạy trên máy mình |
| `pyproject.toml`, `uv.lock`, `.python-version` | Đã có; dùng chung môi trường, chỉ sửa nếu cần đổi dependency/Python |
| `.github/workflows/ci.yml` | Đã có; kiểm tra job lint/environment; bật pytest khi có tests thực tế |
| `.github/pull_request_template.md` | Đã có; dùng mẫu PR gọn |
| `docs/tasks/README.md` | Đã có; cập nhật phân công nếu nhóm đổi người |

## Cách làm

1. Mời các thành viên vào repo; mỗi người clone và chạy các lệnh sau từ gốc repo:

   ```sh
   uv sync --locked
   uv run python scripts/check_environment.py
   uv run ruff check .
   uv run ruff format --check .
   ```

2. Ngồi cùng/call để xử lý lỗi. Nếu lỗi do hướng dẫn hoặc cấu hình chung, sửa đúng file
   trong bảng trên để máy khác dùng lại được; không cần bảng xác nhận môi trường từng người.
3. Kiểm tra kết quả Actions trên GitHub, thống nhất người phụ trách từng task và cách làm
   nhánh/PR. Ai chạy được môi trường có thể bắt đầu phần mình, không phải đợi cả nhóm.

**Xong khi:** cả nhóm dùng được repo/môi trường và biết phần việc của mình.
Issue #1 chỉ cần 2–3 gạch đầu dòng chốt cách làm hoặc vướng mắc quan trọng còn lại.
Nếu có PR hoàn tất task, ghi `Closes #1`; không cần tạo thêm tài liệu setup.
