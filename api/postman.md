# Postman — VMMRdb Classifier

## Import

1. Postman → **Import** → drop `VMMRdb.postman_collection.json`.
2. Collection **VMMRdb Classifier** appears with a `baseUrl` variable.

## Variables

| Variable | Default | Use |
|----------|---------|-----|
| `baseUrl` | `http://localhost:8000` | Change to your host/port. Edit at collection level. |

## Requests

| Name | Method | Path | Notes |
|------|--------|------|-------|
| Health | GET | `/health` | Class count + device. |
| Predict — image | POST | `/predict` | `form-data` `files` = image file. |
| Predict — image + detect | POST | `/predict?detect=true` | Adds YOLO bbox + per-vehicle make/model. |
| Predict — video | POST | `/predict` | `form-data` `files` = video; per-frame results. |
| Predict — zip batch | POST | `/predict` | `form-data` `files` = zip of images. |
| Predict — stream URL | POST | `/predict` | `form-data` `urls` = rtsp/http stream. |
| Predict — multi (files + url) | POST | `/predict?topk=5&detect=false` | Mixed: 2 files + 1 url in one call. |

## File-body requests

`form-data` `files` rows are typed **File** — Postman shows **Select Files**; pick a local file before sending. Repeat the `files` key (multiple rows, same name) to send several files in one request. `urls` rows are typed **Text** with a stream URL value.

## Query params

`topk` (int, default 5) and `detect` (bool, default false) are query params, pre-filled on the multi-request and toggleable via the Params tab.

## Quick test

Send **Health** first — confirms the server is up and returns `{"classes": N, "device": "..."}`.
