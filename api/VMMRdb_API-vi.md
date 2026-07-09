# API Nhận Diện Xe — VMMRdb Classifier (vmmrdb-api)

Phân loại hãng xe/mẫu xe/năm sản xuất, và phát hiện + gán thuộc tính (loại xe, màu
sắc, kiểu dáng thân xe) cho từng phương tiện trong ảnh, video, file zip, hoặc luồng
stream trực tiếp. **Không lưu trạng thái, không xác thực, không giới hạn tần suất** —
mọi trường dữ liệu ở trên đều được tính toán mới hoàn toàn ở mỗi lần gọi; không có gì
được cache, lưu trữ, hay gắn với danh tính người gọi. Nếu cần lưu trữ, khử trùng lặp,
hay giới hạn tần suất, backend của bạn phải tự đảm nhiệm việc đó.

## Nguyên Tắc Cốt Lõi — KHÔNG LƯU TRẠNG THÁI / CHỈ ĐỒNG BỘ

Dịch vụ không giữ trạng thái phiên (session) và không lưu trữ gì cả. Mỗi lệnh gọi
`POST /predict` là độc lập — không có khái niệm "job", không có kho lưu kết quả, và
không có lịch sử. Dịch vụ cũng không xác thực người gọi (không API key, không token) —
được thiết kế để chạy trong mạng nội bộ mà thôi. Việc retry, xếp hàng request, và
giới hạn tần suất là trách nhiệm của backend gọi đến, không phải của dịch vụ này.

## Cách Hoạt Động

Không phải kiểu pipeline nhiều bước như một số API suy luận khác — mọi thứ chạy qua
**một** endpoint duy nhất, `POST /predict`, tự động định tuyến theo hai yếu tố độc
lập:

- **Loại input** (tự nhận diện từ dữ liệu/tên file tải lên): ảnh → 1 dự đoán; video →
  dự đoán theo từng frame lấy mẫu; zip → 1 dự đoán cho mỗi ảnh bên trong; URL stream →
  dự đoán theo frame lấy mẫu, giống video.
- **Query flags**: `detect=false` (mặc định) → chỉ trả nhãn make/model top-K dạng
  phẳng. `detect=true` → trả đầy đủ thông tin phát hiện theo từng xe kèm mọi thuộc
  tính. `image=true` → bỏ qua JSON hoàn toàn, trả về ảnh đầu tiên đã tải lên với box +
  nhãn được vẽ sẵn dưới dạng JPEG thô (tự bật `detect=true`).

### Xử lý nội bộ

Theo thứ tự khi `detect=true`:

1. **Phát hiện bằng YOLO** (`vehicle_yolov9s_640_30oct2025.pt`, một model tùy chỉnh
   5 lớp — `bicycle, bus, car, motorbike, truck`, **không phải** 80 lớp của COCO gốc)
   tìm mọi box chứa phương tiện. `vehicle_type` chính là tên lớp của detector này —
   miễn phí, không cần gọi thêm model nào khác.
2. Với các box được phân loại là `car`/`bus`/`truck`/`motorbike` (`CLASSIFY_CLS`), box
   sẽ được cắt ra và đưa qua:
   - **Bộ phân loại make/model** — ResNet50, fine-tune trên tập gộp VMMRdb + vn_vmmr +
     DVM_vmmr (dữ liệu thị trường Mỹ + Việt Nam + Anh). Trả về top-K nhãn dạng
     `honda_civic_2015`.
   - **`year`/`make`/`model`** — được tách ra từ chuỗi text của nhãn make/model cao
     nhất (không phải một lệnh gọi model riêng). Một danh sách nhỏ đã biết trước
     (`mercedes benz`, `land rover`, `alfa romeo`, ...) xử lý các hãng xe tên nhiều
     từ; còn lại thì tách tại dấu `_` đầu tiên.
   - **Màu sắc** — một heuristic dựa trên pixel (`code/color/color_heuristic.py`): cắt
     ảnh, bỏ 1/3 phía trên (tránh chói kính chắn gió/mái xe), chọn nhóm màu HSV chiếm
     ưu thế nhất trong 14 tên màu có sẵn. Không cần model, không cần dữ liệu huấn
     luyện.
   - **Kiểu dáng thân xe** — ResNet18, huấn luyện trên nhãn lấy từ tên thư mục của
     boxcars116k + stanfordcars (8 lớp: sedan/suv/coupe/hatchback/convertible/wagon/
     van/pickup). Chỉ chạy nếu file `models/bodystyle_model.pt` tồn tại (`GET
     /health` báo cáo điều này).

### Logic quyết định

- Các dự đoán make/model có độ tin cậy dưới **0.25** (`CONF_MIN` trong `api.py`) sẽ bị
  loại khỏi danh sách top-K. Kiểu dáng thân xe dùng cùng ngưỡng 0.25
  (`BODYSTYLE_CONF_MIN`) và trả về `null` nếu dưới ngưỡng.
- **Lưu ý quan trọng:** `topk` chọn ứng viên *trước khi* áp ngưỡng 0.25, nên
  `topk=10` không đảm bảo trả về đủ 10 kết quả — bạn có thể nhận được từ 0 đến `topk`
  kết quả tùy vào độ tin cậy thực tế của model. Nó có nghĩa là "tối đa K", không phải
  "chính xác K".

### Ràng buộc đầu vào

- Ảnh: `.jpg .jpeg .png .webp .bmp .gif .tif .tiff`. Bất kỳ file nào khác được tải lên
  sẽ được coi là video và chuyển cho ffmpeg xử lý (qua PyAV) — input lỗi sẽ fail theo
  từng mục, không làm fail toàn bộ request.
- Việc lấy mẫu video/stream giới hạn tối đa `MAX_FRAMES=16` frame, cứ mỗi
  `FRAME_STRIDE=15` frame lấy 1 (~1 fps với nguồn 15 fps).
- URL stream: chỉ chấp nhận `rtsp rtsps http https`. Hostname sau khi resolve sẽ được
  kiểm tra so với dải private/loopback/link-local/reserved trước khi ffmpeg mở kết nối
  (chống SSRF) — xem `safe_stream_url()`.
- Độ phủ make/model phụ thuộc vào checkpoint nào đang được load làm `model.pt` — kiểm
  tra số `classes` ở `GET /health` nếu thấy kết quả dự đoán thưa thớt cho một thị
  trường mà bạn kỳ vọng có độ phủ tốt.

## Luồng Sử Dụng Điển Hình

Tải lên một hoặc nhiều file và/hoặc URL stream tới `POST /predict` trong cùng một
request — file và URL có thể trộn lẫn. Với `detect=false` (mặc định) bạn nhận về
`predictions` dạng phẳng: danh sách `{label, confidence}`, rẻ và nhanh nhất, không có
tọa độ box. Với `detect=true` bạn nhận về danh sách `vehicles` cho mỗi input, mỗi mục
kèm box và mọi thuộc tính mô tả ở trên — dùng cái này khi cần biết *vị trí* từng xe,
không chỉ *có gì* trong khung hình. Với `image=true` bạn bỏ qua hoàn toàn định dạng
JSON và nhận về ảnh JPEG đã vẽ sẵn của ảnh đầu tiên tải lên, hữu ích để xem nhanh kết
quả trên trình duyệt hoặc Postman mà không cần viết code vẽ ở phía client.

## Thông Tin Kết Nối

- **Base URL:** `http://0.0.0.0:8100` (bind mọi interface — thay bằng host thực tế bạn
  dùng để kết nối)
- **Content-Type:** `multipart/form-data` cho request; `application/json` cho response
  — ngoại trừ `image=true` trả trực tiếp `image/jpeg`, và `/health` là `GET` thuần
  không có body.
- **Mạng:** chỉ dùng nội bộ — không xác thực, không expose ra ngoài.
- **Routes:** `POST /predict`, `GET /health` (không theo quy ước `/{entity}/{action}` —
  chỉ có 2 route này).
- **Docs tương tác:** Swagger UI tại `GET /docs` · OpenAPI JSON tại `GET /openapi.json`
  (mặc định của FastAPI, không phải tài liệu tự viết tay).

## Trường Dữ Liệu Chung (xuất hiện ở mọi dạng response khi `detect=true`)

| Trường | Kiểu | Bắt buộc/Mặc định | Mô tả |
|---|---|---|---|
| `bbox` | `[x1,y1,x2,y2]` | — | Box pixel từ detector YOLO. |
| `det_class` | str | — | Lớp thô từ detector: `car`, `bus`, `truck`, `motorbike`, hoặc `bicycle`. |
| `det_conf` | float | — | Độ tin cậy detector cho `bbox`. |
| `vehicle_type` | str | — | Trùng với `det_class` — miễn phí, không cần phân loại thêm. |
| `make_model` | list hoặc `null` | `null` nếu dưới 0.25 conf hoặc không thuộc `CLASSIFY_CLS` | Top-K dự đoán `{label, confidence}`, ví dụ `honda_civic_2015`. |
| `make` | str hoặc `null` | `null` nếu `make_model` là `null` | Tách từ nhãn cao nhất trong `make_model`. |
| `model` | str hoặc `null` | `null` nếu `make_model` là `null` | Tách từ nhãn cao nhất trong `make_model`. |
| `year` | int hoặc `null` | `null` nếu `make_model` là `null` hoặc nhãn không có năm ở cuối | Tách từ nhãn cao nhất trong `make_model`. |
| `color` | `{color, confidence}` hoặc `null` | `null` nếu không thuộc `CLASSIFY_CLS` | Màu chủ đạo theo heuristic, 1 trong 14 tên màu. |
| `bodystyle` | `{label, confidence}` hoặc `null` | `null` nếu dưới 0.25 conf, không thuộc `CLASSIFY_CLS`, hoặc model chưa load | 1 trong 8 kiểu dáng thân xe. |
| `track_id` | int hoặc `null` | chỉ video/stream | ID xuyên suốt các frame (YOLO tracking). |

---

## Endpoints

### 1. Phân Loại hoặc Phát Hiện Xe

**POST `/predict`**

Endpoint thực sự duy nhất. Nhận thông minh nhiều file **và** URL stream trong cùng
một request; mỗi input tự động định tuyến theo loại (xem "Cách Hoạt Động" ở trên).

#### Request (`multipart/form-data`)

| Trường | Kiểu | Bắt buộc/Mặc định | Mô tả |
|---|---|---|---|
| `files` | file (lặp lại được) | `[]` | Ảnh, video, hoặc zip. Lặp lại field để gửi nhiều file. |
| `urls` | string (lặp lại được) | `[]` | URL stream. Có chống SSRF (xem Ràng buộc đầu vào). |
| `topk` | int (query) | `3` | Số nhãn tối đa cho mỗi dự đoán make/model — xem lưu ý về topk ở trên. |
| `detect` | bool (query) | `false` | `true` → chạy phát hiện + phân loại đầy đủ thuộc tính. |
| `image` | bool (query) | `false` | `true` → trả ảnh đầu tiên đã vẽ box dưới dạng JPEG thô. Tự bật `detect`. |

**Ví dụ — curl**

```bash
curl -F 'files=@car.jpg' \
     -F 'files=@clip.mp4' \
     -F 'files=@batch.zip' \
     -F 'urls=rtsp://cam.example.com/stream' \
     'http://0.0.0.0:8100/predict?topk=3&detect=true'
```

Trả về ảnh đã vẽ thay vì JSON:

```bash
curl -F 'files=@street.jpg' 'http://0.0.0.0:8100/predict?image=true' -o out.jpg
```

**Postman:** `POST http://0.0.0.0:8100/predict` · Body → form-data: `files` (file, lặp
lại được), `urls` (text, lặp lại được) · Params → `topk`, `detect`, `image`

#### Response (JSON)

Cấp cao nhất là `{"results": [...]}`, mỗi phần tử tương ứng 1 input (file trước, url
sau). Hình dạng thay đổi theo loại input và theo `detect`:

| Loại | `detect=false` | `detect=true` |
|---|---|---|
| Ảnh | `predictions: [{label, confidence}, ...]` | `vehicles: [...]` (xem Trường Dữ Liệu Chung) |
| Video / stream | `frames: [[{label, confidence}, ...], ...]` | `frames: [{frame: N, vehicles: [...]}, ...]`, có thêm `track_id` |
| Zip | `predictions: [[...], ...]`, mỗi ảnh 1 danh sách | `images: [{vehicles: [...]}, ...]` |

**Ví dụ — ảnh, `detect=true`:**

```json
{
  "results": [
    {
      "name": "car.jpg",
      "type": "image",
      "vehicles": [
        {
          "bbox": [34, 50, 220, 180],
          "det_class": "car",
          "det_conf": 0.91,
          "vehicle_type": "car",
          "make_model": [{"label": "honda_civic_2015", "confidence": 0.81}],
          "make": "honda",
          "model": "civic",
          "year": 2015,
          "color": {"color": "silver", "confidence": 0.62},
          "bodystyle": {"label": "sedan", "confidence": 0.84}
        }
      ]
    }
  ]
}
```

**Lỗi riêng của endpoint này:** không trả HTTP status riêng biệt nào — xem Mã Lỗi
Thường Gặp bên dưới, dịch vụ này khá "phẳng" về mặt status code.

---

### 2. Kiểm Tra Trạng Thái

**GET `/health`**

Xác nhận server đang chạy và báo cáo model nào đang được load — gọi cái này đầu tiên
khi tích hợp, và gọi lại bất cứ khi nào kết quả dự đoán có vẻ bất thường (ví dụ sau
khi đổi model).

#### Response (JSON)

| Trường | Kiểu | Mô tả |
|---|---|---|
| `classes` | int | Số lớp make/model trong `model.pt` đang được load. |
| `device` | str | `cuda` hoặc `cpu`. |
| `bodystyle_available` | bool | File `models/bodystyle_model.pt` có tồn tại không — nếu `false`, `bodystyle` luôn là `null`. |

```json
{"classes": 15288, "device": "cuda", "bodystyle_available": true}
```

**Lỗi riêng của endpoint này:** không có.

---

## Luồng Tích Hợp (cho backend implementer)

```
A. Chỉ phân loại, rẻ nhất (không box, không thuộc tính):
1) POST /predict?detect=false (files=[image.jpg]) -> predictions: [{label, confidence}, ...]
# dùng khi đã biết chắc trong khung hình chỉ có 1 xe và không cần tọa độ box
```

```
B. Phát hiện đầy đủ + thuộc tính:
1) POST /predict?detect=true (files=[image.jpg]) -> vehicles: [...] (bbox + mọi thuộc tính)
2) backend đọc vehicle_type/make/model/year/color/bodystyle cho từng phần tử
# dùng cho trường hợp nhiều xe, hoặc khi cần tọa độ box
```

```
C. Kiểm tra nhanh bằng mắt (không cần code vẽ ở client):
1) POST /predict?image=true (files=[image.jpg]) -> ảnh image/jpeg thô, đã vẽ box + nhãn
# chỉ áp dụng cho ảnh đầu tiên tải lên -- dùng để debug/demo, không dùng cho tích hợp production
```

## Mã Lỗi Thường Gặp

Dịch vụ này không trả HTTP error có cấu trúc cho các trường hợp lỗi đã được xử lý —
khá "phẳng" về status code, cần lưu ý rõ:

| HTTP | Ý nghĩa | Khi nào |
|---|---|---|
| 200 | OK | Thành công, **và** hầu hết lỗi đã được xử lý — kiểm tra payload: lỗi từng file/url trả về dạng `{"name": ..., "error": "processing failed"}` bên trong `results`, request vẫn 200; `image=true` mà không có file tải lên trả về `{"error": "..."}`, cũng vẫn 200. |
| 422 | Unprocessable | Chỉ do FastAPI tự động validate request (ví dụ multipart body sai định dạng) — không phải do code ứng dụng raise. |
| 5xx | Server Error | Crash chưa được xử lý (ví dụ GPU hết bộ nhớ) — hiếm gặp, vì việc xử lý từng file/url đã được bọc trong try/except. |

## Chế Độ Async / Callback

Không hỗ trợ — mọi request đều đồng bộ; kết nối của bên gọi giữ mở trong suốt thời
gian phát hiện + phân loại.
