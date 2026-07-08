# Postman — VMMRdb Classifier (Tiếng Việt)

## Import (Nhập)

1. Postman → **Import** → kéo thả file `VMMRdb.postman_collection.json`.
2. Collection **VMMRdb Classifier** xuất hiện kèm biến `baseUrl`.

## Biến

| Biến | Mặc định | Công dụng |
|------|----------|-----------|
| `baseUrl` | `http://100.111.0.111:8100` | Đổi sang host/port của bạn. Sửa ở cấp collection. |

## Danh sách request

| Tên | Method | Path | Ghi chú |
|-----|--------|------|---------|
| Health | GET | `/health` | Số lớp (class) + thiết bị (device). |
| Predict — image | POST | `/predict` | `form-data` `files` = file ảnh. |
| Predict — image + detect | POST | `/predict?detect=true` | Thêm bbox YOLO + make/model cho từng xe (car/bus/truck). |
| Predict — image + annotate | POST | `/predict?annotate=true` | Thêm ảnh `annotated` (base64) đã vẽ bbox + nhãn make/model cao nhất. Tự bật `detect`. |
| Predict — image (raw JPEG) | POST | `/predict?image=true` | Trả về ảnh **đầu tiên** đã vẽ dưới dạng `image/jpeg` thô (hiển thị trực tiếp trong Postman). Tự bật `detect`. |
| Predict — video | POST | `/predict` | `form-data` `files` = video; kết quả theo từng frame. |
| Predict — zip batch | POST | `/predict` | `form-data` `files` = file zip chứa ảnh. |
| Predict — stream URL | POST | `/predict` | `form-data` `urls` = stream rtsp/http. |
| Predict — multi (files + url) | POST | `/predict?topk=3&detect=false` | Hỗn hợp: 2 file + 1 url trong cùng request. |

## Request gửi file

Các dòng `files` trong `form-data` có kiểu **File** — Postman hiện nút **Select Files**; chọn file cục bộ trước khi gửi. Lặp lại key `files` (nhiều dòng cùng tên) để gửi nhiều file trong một request. Dòng `urls` có kiểu **Text** với giá trị là stream URL.

## Query params

`topk` (số nguyên, mặc định 3), `detect` (bool, mặc định false), `annotate` (bool, mặc định false; tự bật `detect`), và `image` (bool, mặc định false; tự bật `detect`; trả về JPEG thô) là query param, đã điền sẵn ở các request liên quan và bật/tắt được qua tab Params. Các dự đoán make/model có độ tin cậy dưới `0.25` sẽ bị loại.

## Kiểm tra nhanh

Gửi **Health** trước — xác nhận server đang chạy và trả về `{"classes": N, "device": "..."}`.
