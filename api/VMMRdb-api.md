# VMMRdb Classifier API

Vehicle make/model classifier (ResNet50, 9169 VMMRdb classes) with optional YOLO vehicle detection.

## Run

```bash
uv run uvicorn api:app --host 0.0.0.0 --port 8100
```

Base URL: `http://100.111.0.111:8100`

---

## `POST /predict`

Smart-accepts mixed files **and** stream URLs in one request. Each input is auto-routed by type:

| Input | Routing | Output shape |
|-------|---------|--------------|
| Image (`.jpg .jpeg .png .webp .bmp .gif .tif .tiff`) | 1 prediction | `predictions` |
| Video (`.mp4 .mov`, any non-image/non-zip file) | sample up to 16 frames (stride 15), predict per frame | `frames` |
| Zip | predict every image inside | `predictions` (list) |
| Stream URL (`rtsp rtsps http https`) | sample frames from live stream | `frames` |

### Request — `multipart/form-data`

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `files` | file (repeatable) | `[]` | Images, videos, or zips. Repeat field for multiple. |
| `urls` | string (repeatable) | `[]` | Stream URLs. SSRF-guarded: scheme allowlist + internal/loopback/link-local hosts rejected. |
| `topk` | int (query) | `3` | Top-K labels per prediction. |
| `detect` | bool (query) | `false` | `true` → run YOLO vehicle detection, then crop+classify make/model for car/bus/truck. |
| `annotate` | bool (query) | `false` | `true` → also return an annotated image (boxes + top make/model label drawn) as a base64 data-URI. Implies `detect`. |
| `image` | bool (query) | `false` | `true` → return the annotated **first** image as a raw `image/jpeg` body (renders in Postman/browser) instead of JSON. Implies `detect`. |

`topk`, `detect`, `annotate`, `image` are query params; `files`/`urls` are form fields.

Make/model labels below `confidence` **0.25** are dropped (`make_model` becomes `null` / pred list omits them). Annotated boxes: green = classified car (shows `make_model 0.xx`), orange = other/unclassified (shows `det_class 0.xx`).

### Example

```bash
curl -F 'files=@car.jpg' \
     -F 'files=@clip.mp4' \
     -F 'files=@batch.zip' \
     -F 'urls=rtsp://cam.example.com/stream' \
     'http://100.111.0.111:8100/predict?topk=3&detect=false'
```

Annotated image (boxes + labels drawn) as base64 in JSON:

```bash
curl -F 'files=@street.jpg' 'http://100.111.0.111:8100/predict?annotate=true' -o out.json
```

Annotated image as a raw JPEG you can open directly:

```bash
curl -F 'files=@street.jpg' 'http://100.111.0.111:8100/predict?image=true' -o out.jpg
```

### Response — `200 OK`

Top-level `{"results": [...]}`, one entry per input (files first, then urls). Shape varies by type.

**Image (classify):**
```json
{
  "name": "car.jpg",
  "type": "image",
  "predictions": [
    {"label": "honda_civic_2015", "confidence": 0.8123},
    {"label": "honda_accord_2014", "confidence": 0.0456}
  ]
}
```

**Image (`detect=true`):**
```json
{
  "name": "car.jpg",
  "type": "image",
  "vehicles": [
    {
      "bbox": [34, 50, 220, 180],
      "det_class": "car",
      "det_conf": 0.91,
      "make_model": [{"label": "honda_civic_2015", "confidence": 0.81}]
    }
  ]
}
```
`make_model` is `null` for non-car/bus/truck detections (bicycle/motorbike have no VMMRdb make/model), and also when every pred is below the 0.25 confidence floor.

With `annotate=true`, an `"annotated"` field is added alongside `vehicles`:
```json
{"name": "car.jpg", "type": "image", "vehicles": [...],
 "annotated": "data:image/jpeg;base64,/9j/4AAQ..."}
```
For video/stream, `annotated` is per-frame; for zip, per-image.

**Video / stream (classify):**
```json
{"name": "clip.mp4", "type": "video", "frames": [[{"label": "...", "confidence": 0.7}], ...]}
```

**Video / stream (`detect=true`):** frames carry detections; tracking IDs added (`track_id`):
```json
{"name": "clip.mp4", "type": "video",
 "frames": [{"frame": 0, "vehicles": [{"bbox": [...], "det_class": "car", "det_conf": 0.9,
                                       "track_id": 3, "make_model": [...]}]}]}
```

**Zip:**
```json
{"name": "batch.zip", "type": "zip", "predictions": [[{"label": "...", "confidence": 0.6}], ...]}
```
With `detect=true`: `{"name": "...", "type": "zip", "images": [{"vehicles": [...]}]}` (each image also gets `annotated` when `annotate=true`).

**Per-input error** (one bad input doesn't fail the request):
```json
{"name": "broken.xyz", "error": "processing failed"}
```

---

## `GET /health`

```bash
curl http://100.111.0.111:8100/health
```

```json
{"classes": 9169, "device": "cuda"}
```

`device` is `cuda` or `cpu`.

---

## Notes

- **Frame sampling:** up to `MAX_FRAMES=16` frames per video/stream, every `FRAME_STRIDE=15`th frame (~1 fps at 15 fps source).
- **Detection weights:** `weights/vehicle/vehicle_yolov9s_640_30oct2025.pt` (lazy-loaded on first `detect=true`).
- **Tracking:** persistent IDs only for video/stream detection (not single images).
- **SSRF guard:** stream hostnames resolving to private/loopback/link-local/reserved IPs are rejected; ffmpeg protocol allowlist blocks `file/concat/data` to prevent arbitrary file read.
