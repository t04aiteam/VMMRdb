# Postman — VMMRdb Classifier (Tiếng Việt)

## Import (Nhập)

1. Postman → **Import** → kéo thả file `VMMRdb.postman_collection.json`.
2. Collection **VMMRdb Classifier** xuất hiện kèm biến `baseUrl`.

## Biến

| Biến | Mặc định | Công dụng |
|------|----------|-----------|
| `baseUrl` | `http://localhost:8000` | Đổi sang host/port của bạn. Sửa ở cấp collection. |

## Danh sách request

| Tên | Method | Path | Ghi chú |
|-----|--------|------|---------|
| Health | GET | `/health` | Số lớp (class) + thiết bị (device). |
| Predict — image | POST | `/predict` | `form-data` `files` = file ảnh. |
| Predict — image + detect | POST | `/predict?detect=true` | Thêm bbox YOLO + make/model cho từng xe. |
| Predict — video | POST | `/predict` | `form-data` `files` = video; kết quả theo từng frame. |
| Predict — zip batch | POST | `/predict` | `form-data` `files` = file zip chứa ảnh. |
| Predict — stream URL | POST | `/predict` | `form-data` `urls` = stream rtsp/http. |
| Predict — multi (files + url) | POST | `/predict?topk=5&detect=false` | Hỗn hợp: 2 file + 1 url trong cùng request. |

## Request gửi file

Các dòng `files` trong `form-data` có kiểu **File** — Postman hiện nút **Select Files**; chọn file cục bộ trước khi gửi. Lặp lại key `files` (nhiều dòng cùng tên) để gửi nhiều file trong một request. Dòng `urls` có kiểu **Text** với giá trị là stream URL.

## Query params

`topk` (số nguyên, mặc định 5) và `detect` (bool, mặc định false) là query param, đã điền sẵn ở request multi và bật/tắt được qua tab Params.

## Kiểm tra nhanh

Gửi **Health** trước — xác nhận server đang chạy và trả về `{"classes": N, "device": "..."}`.
