# Vehicle Recognition API — VMMRdb Classifier (vmmrdb-api)

Classifies vehicle make/model/year, and detects+attributes (type, color, body style)
per vehicle in images, videos, zips, or live streams. **Stateless, no auth, no rate
limiting** — every field above is computed fresh on every call; nothing is cached,
stored, or tied to a caller identity. If you need persistence, dedup, or throttling,
your backend owns that.

## Core Principle — STATELESS / SYNC-ONLY

The service holds no session state and does no storage. Every `POST /predict` call is
independent — there is no "job", no result store, and no history. It also does not
authenticate callers (no API key, no token) — it is meant to sit on an internal
network only. Retries, request queuing, and rate limiting are the caller's backend's
responsibility, not this service's.

## How It Works

Not a multi-step pipeline like some inference APIs — everything runs behind **one**
endpoint, `POST /predict`, which auto-routes by two independent things:

- **Input type** (auto-detected from the uploaded bytes/filename): image → single
  prediction; video → sampled per-frame predictions; zip → one prediction per image
  inside; stream URL → sampled per-frame predictions, same as video.
- **Query flags**: `detect=false` (default) → flat top-K make/model labels only.
  `detect=true` → full per-vehicle detection with all attributes. `image=true` →
  skip JSON entirely, return the first uploaded image with boxes+labels drawn on it
  as a raw JPEG (implies `detect=true`).

### Internal processing

In `detect=true` order:

1. **YOLO detection** (`vehicle_yolov9s_640_30oct2025.pt`, a custom 5-class model —
   `bicycle, bus, car, motorbike, truck`, **not** stock COCO's 80 classes) finds every
   vehicle box. `vehicle_type` is just this detector's class name — free, no further
   model call.
2. For boxes classified as `car`/`bus`/`truck`/`motorbike` (`CLASSIFY_CLS`), the box is
   cropped and passed to:
   - **Make/model classifier** — ResNet50, fine-tuned on merged VMMRdb + vn_vmmr +
     DVM_vmmr (US + Vietnam + UK market data). Returns top-K labels like
     `honda_civic_2015`.
   - **`year`/`make`/`model`** — parsed out of the top make/model label's text (not a
     separate model call). A small known-list (`mercedes benz`, `land rover`, `alfa
     romeo`, ...) handles multi-word makes; everything else splits on the first `_`.
   - **Color** — a pixel heuristic (`code/color/color_heuristic.py`): crop, drop the
     top third (windshield/roof glare), snap the dominant HSV bucket to one of 14
     named colors. No model, no training data needed.
   - **Body style** — ResNet18, trained on boxcars116k + stanfordcars folder-name
     labels (8 classes: sedan/suv/coupe/hatchback/convertible/wagon/van/pickup). Only
     runs if `models/bodystyle_model.pt` exists (`GET /health` reports this).

### Decision logic

- Make/model predictions below confidence **0.25** (`CONF_MIN` in `api.py`) are
  dropped from the top-K list. Body style uses the same 0.25 floor
  (`BODYSTYLE_CONF_MIN`) and returns `null` below it.
- **Known caveat:** `topk` selects candidates *before* the 0.25 floor is applied, so
  `topk=10` does not guarantee 10 results — you can get anywhere from 0 to `topk`
  depending on how confident the model actually is. It means "at most K", not "exactly
  K".

### Input constraints

- Images: `.jpg .jpeg .png .webp .bmp .gif .tif .tiff`. Anything else uploaded as a
  file is assumed to be a video and handed to ffmpeg (via PyAV) — malformed input
  fails per-item, not for the whole request.
- Video/stream sampling is capped at `MAX_FRAMES=16` frames, one every
  `FRAME_STRIDE=15`th frame (~1 fps at a 15 fps source).
- Stream URLs: `rtsp rtsps http https` only. Resolved hostnames are checked against
  private/loopback/link-local/reserved ranges before ffmpeg ever opens them (SSRF
  guard) — see `safe_stream_url()`.
- Make/model coverage depends on whichever checkpoint is currently loaded as
  `model.pt` — check `GET /health`'s `classes` count if predictions look sparse for a
  market you expect coverage for.

## Typical End-to-End Flow

Upload one or more files and/or stream URLs to `POST /predict` in a single request —
files and URLs can be mixed. With `detect=false` (default) you get back flat
`predictions`: a plain list of `{label, confidence}` guesses, cheapest and fastest,
no bounding boxes. With `detect=true` you get a `vehicles` list per input, one entry
per detected vehicle with its box and every attribute described above — this is what
you want if you need to know *where* each vehicle is, not just *what's in the frame*.
With `image=true` you skip the JSON response shape entirely and get back a rendered
JPEG of the first uploaded image, useful for eyeballing results in a browser or
Postman without writing any client-side drawing code.

## Connection Info

- **Base URL:** `http://0.0.0.0:8100` (binds all interfaces — substitute your actual
  reachable host)
- **Content-Type:** `multipart/form-data` for the request; `application/json` for the
  response — except `image=true`, which returns `image/jpeg` directly, and `/health`,
  which is a plain `GET` with no body.
- **Network:** internal only — no auth, do not expose publicly.
- **Routes:** `POST /predict`, `GET /health` (no `/{entity}/{action}` convention here —
  just the two routes).
- **Interactive docs:** Swagger UI at `GET /docs` · OpenAPI JSON at `GET /openapi.json`
  (FastAPI defaults, not manually maintained).

## Common Fields (appear across every `detect=true` response shape)

| Field | Type | Required/Default | Description |
|---|---|---|---|
| `bbox` | `[x1,y1,x2,y2]` | — | Pixel box from the YOLO detector. |
| `det_class` | str | — | Raw detector class: `car`, `bus`, `truck`, `motorbike`, or `bicycle`. |
| `det_conf` | float | — | Detector confidence for `bbox`. |
| `vehicle_type` | str | — | Mirrors `det_class` — free, no extra classification. |
| `make_model` | list or `null` | `null` if below 0.25 conf or not in `CLASSIFY_CLS` | Top-K `{label, confidence}` guesses, e.g. `honda_civic_2015`. |
| `make` | str or `null` | `null` if `make_model` is `null` | Parsed from `make_model`'s top label. |
| `model` | str or `null` | `null` if `make_model` is `null` | Parsed from `make_model`'s top label. |
| `year` | int or `null` | `null` if `make_model` is `null` or label has no trailing year | Parsed from `make_model`'s top label. |
| `color` | `{color, confidence}` or `null` | `null` if not in `CLASSIFY_CLS` | Heuristic dominant color, one of 14 names. |
| `bodystyle` | `{label, confidence}` or `null` | `null` if below 0.25 conf, not in `CLASSIFY_CLS`, or model not loaded | One of 8 body shapes. |
| `track_id` | int or `null` | video/stream only | Persistent ID across frames (YOLO tracking). |

---

## Endpoints

### 1. Classify or Detect Vehicles

**POST `/predict`**

The only real endpoint. Smart-accepts multiple files **and** stream URLs in one
request; each is auto-routed by type (see "How It Works" above).

#### Request (`multipart/form-data`)

| Field | Type | Required/Default | Description |
|---|---|---|---|
| `files` | file (repeatable) | `[]` | Images, videos, or zips. Repeat the field for multiple. |
| `urls` | string (repeatable) | `[]` | Stream URLs. SSRF-guarded (see Input constraints). |
| `topk` | int (query) | `3` | Max labels per make/model prediction — see the topk caveat above. |
| `detect` | bool (query) | `false` | `true` → run detection + full attribute classification. |
| `image` | bool (query) | `false` | `true` → return the first image with boxes drawn as raw JPEG. Implies `detect`. |

**Example — curl**

```bash
curl -F 'files=@car.jpg' \
     -F 'files=@clip.mp4' \
     -F 'files=@batch.zip' \
     -F 'urls=rtsp://cam.example.com/stream' \
     'http://0.0.0.0:8100/predict?topk=3&detect=true'
```

Rendered image instead of JSON:

```bash
curl -F 'files=@street.jpg' 'http://0.0.0.0:8100/predict?image=true' -o out.jpg
```

**Postman:** `POST http://0.0.0.0:8100/predict` · Body → form-data: `files` (file,
repeatable), `urls` (text, repeatable) · Params → `topk`, `detect`, `image`

#### Response (JSON)

Top-level `{"results": [...]}`, one entry per input (files first, then urls). Shape
varies by input type and by `detect`:

| Type | `detect=false` | `detect=true` |
|---|---|---|
| Image | `predictions: [{label, confidence}, ...]` | `vehicles: [...]` (Common Fields above) |
| Video / stream | `frames: [[{label, confidence}, ...], ...]` | `frames: [{frame: N, vehicles: [...]}, ...]`, adds `track_id` |
| Zip | `predictions: [[...], ...]`, one list per image | `images: [{vehicles: [...]}, ...]` |

**Example — image, `detect=true`:**

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

**Errors specific to this endpoint:** none return a distinct HTTP status — see
Common Error Codes below, this service is unusually flat about it.

---

### 2. Health Check

**GET `/health`**

Confirms the server is up and reports what's currently loaded — call this first when
integrating, and again any time predictions look off (e.g. after a model swap).

#### Response (JSON)

| Field | Type | Description |
|---|---|---|
| `classes` | int | Make/model class count in the currently loaded `model.pt`. |
| `device` | str | `cuda` or `cpu`. |
| `bodystyle_available` | bool | Whether `models/bodystyle_model.pt` exists — if `false`, `bodystyle` is always `null`. |

```json
{"classes": 15288, "device": "cuda", "bodystyle_available": true}
```

**Errors specific to this endpoint:** none.

---

## Integration Flows (for backend implementers)

```
A. Cheapest classify-only (no boxes, no attributes):
1) POST /predict?detect=false (files=[image.jpg]) -> predictions: [{label, confidence}, ...]
# use when you already know there's exactly one vehicle in frame and don't need its box
```

```
B. Full detection + attributes:
1) POST /predict?detect=true (files=[image.jpg]) -> vehicles: [...] (bbox + every attribute)
2) your backend reads vehicle_type/make/model/year/color/bodystyle per entry
# use for anything multi-vehicle, or where you need box coordinates
```

```
C. Visual spot-check (no client-side drawing code needed):
1) POST /predict?image=true (files=[image.jpg]) -> raw image/jpeg, boxes + labels drawn
# first uploaded image only -- for debugging/demos, not production integration
```

## Common Error Codes

This service does not raise structured HTTP errors for handled failure cases — it is
unusually flat about status codes, worth calling out explicitly:

| HTTP | Meaning | When |
|---|---|---|
| 200 | OK | Success, **and** most handled failures — check the payload: per-file/per-url failures come back as `{"name": ..., "error": "processing failed"}` inside `results` with the request itself still 200; `image=true` with no uploaded file returns `{"error": "..."}`, also 200. |
| 422 | Unprocessable | FastAPI's own automatic request-validation only (e.g. malformed multipart body) — not raised by application code. |
| 5xx | Server Error | Unhandled crash (e.g. GPU OOM) — rare, since per-file/per-url processing is already wrapped in try/except. |

## Async / Callback Mode

Not supported — every request is synchronous; the caller's connection stays open for
the full duration of detection + classification.
