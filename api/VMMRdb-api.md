# VMMRdb Classifier API

Vehicle make/model classifier (ResNet50, trained on merged VMMRdb + vn_vmmr + DVM_vmmr; class count via `GET /health`) with optional YOLO vehicle detection. `detect=true` also returns `vehicle_type`, `year`, `color`, and `bodystyle` per detected vehicle.

## Run

```bash
uv run uvicorn api:app --host 0.0.0.0 --port 8100
```

Base URL: `http://0.0.0.0:8100`

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
| `detect` | bool (query) | `false` | `true` → run YOLO vehicle detection, then crop+classify make/model/year/color/bodystyle for car/bus/truck/motorbike. |
| `image` | bool (query) | `false` | `true` → return the detected **first** image, boxes drawn, as a raw `image/jpeg` body (renders in Postman/browser) instead of JSON. Implies `detect`. |

`topk`, `detect`, `image` are query params; `files`/`urls` are form fields.

Make/model labels below `confidence` **0.25** are dropped (`make_model` becomes `null` / pred list omits them); `bodystyle` uses the same 0.25 floor and is `null` below it. Boxes in the `image=true` render: green = classified vehicle (shows `make_model 0.xx`), orange = other/unclassified (shows `det_class 0.xx`).

### Example

```bash
curl -F 'files=@car.jpg' \
     -F 'files=@clip.mp4' \
     -F 'files=@batch.zip' \
     -F 'urls=rtsp://cam.example.com/stream' \
     'http://0.0.0.0:8100/predict?topk=3&detect=false'
```

Detected image (boxes + labels drawn) as a raw JPEG you can open directly:

```bash
curl -F 'files=@street.jpg' 'http://0.0.0.0:8100/predict?image=true' -o out.jpg
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
      "vehicle_type": "car",
      "make_model": [{"label": "honda_civic_2015", "confidence": 0.81}],
      "year": 2015,
      "color": {"color": "silver", "confidence": 0.62},
      "bodystyle": {"label": "sedan", "confidence": 0.84}
    }
  ]
}
```
`vehicle_type` mirrors `det_class` for every detection (free from the YOLO pass, no classification needed). `make_model`/`year`/`color`/`bodystyle` are only filled for car/bus/truck/motorbike detections (`CLASSIFY_CLS` — note `DET_WEIGHTS` uses `motorbike`, not COCO's `motorcycle`) — `null` for other YOLO classes (bicycle, etc.), and also `null` when every make/model pred (or the bodystyle pred) is below the 0.25 confidence floor. `year` is `null` when `make_model` is `null` or its top label has no trailing year.

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
With `detect=true`: `{"name": "...", "type": "zip", "images": [{"vehicles": [...]}]}`.

**Per-input error** (one bad input doesn't fail the request):
```json
{"name": "broken.xyz", "error": "processing failed"}
```

---

## `GET /health`

```bash
curl http://0.0.0.0:8100/health
```

```json
{"classes": 15288, "device": "cuda", "bodystyle_available": true}
```

`device` is `cuda` or `cpu`. `bodystyle_available` reflects whether `models/bodystyle_model.pt` is present — when `false`, `bodystyle` is always `null` in `/predict` responses.

---

## Notes

- **Frame sampling:** up to `MAX_FRAMES=16` frames per video/stream, every `FRAME_STRIDE=15`th frame (~1 fps at 15 fps source).
- **Detection weights:** `weights/vehicle/vehicle_yolov9s_640_30oct2025.pt` (lazy-loaded on first `detect=true`).
- **Tracking:** persistent IDs only for video/stream detection (not single images).
- **SSRF guard:** stream hostnames resolving to private/loopback/link-local/reserved IPs are rejected; ffmpeg protocol allowlist blocks `file/concat/data` to prevent arbitrary file read.
