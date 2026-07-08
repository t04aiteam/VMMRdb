"""Simple inference API. Run: uv run uvicorn api:app --host 0.0.0.0 --port 8100

One endpoint. Smart-accepts many files at once + stream URLs:
  - images (jpg/png/webp/...)         -> 1 prediction
  - videos (mp4/mov/...) & stream URLs -> sample frames, prediction per frame
  - zips                              -> predict every image inside

  curl -F 'files=@a.jpg' -F 'files=@clip.mp4' -F 'files=@batch.zip' \
       -F 'urls=rtsp://cam/stream' http://0.0.0.0:8100/predict
"""
import io
import base64
import zipfile
import socket
import logging
import ipaddress
from urllib.parse import urlparse, urlunparse
from typing import List
import torch
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
import av
from ultralytics import YOLO
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response

CKPT = "model.pt"
DET_WEIGHTS = "weights/vehicle/vehicle_yolov9s_640_30oct2025.pt"
CLASSIFY_CLS = {"car", "bus", "truck"}  # crop+classify these; bicycle/motorbike have no VMMRdb make/model
MAX_FRAMES = 16          # ponytail: cap frames/video so a long clip or live stream can't run forever
FRAME_STRIDE = 15        # ~1 fps at 15fps source; raise to sample sparser
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
ALLOWED_SCHEMES = {"rtsp", "rtsps", "http", "https"}
# ffmpeg protocol allowlist — excludes file/concat/subfile/data to block arbitrary file read
PROTO_WHITELIST = "rtsp,rtsps,tcp,udp,tls,http,https"
CONF_MIN = 0.25          # drop make/model preds below this; empty list -> no classification

log = logging.getLogger("api")

TF = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

dev = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(CKPT, map_location=dev, weights_only=True)
classes = ck["classes"]
model = models.resnet50(num_classes=len(classes))
model.load_state_dict(ck["state_dict"])
model.eval().to(dev)

app = FastAPI(title="VMMRdb classifier")


def classify(imgs: List[Image.Image], topk: int):
    """Batch-classify PIL images -> list of topk pred dicts (one list per image)."""
    if not imgs:
        return []
    x = torch.stack([TF(i.convert("RGB")) for i in imgs]).to(dev)
    with torch.no_grad():
        probs = model(x).softmax(1)
    out = []
    for prob in probs:
        p, idx = prob.topk(min(topk, len(classes)))
        out.append([{"label": classes[i], "confidence": round(v.item(), 4)}
                    for v, i in zip(p, idx) if v.item() >= CONF_MIN])
    return out


_yolo = None
def yolo():  # lazy: only load detector when detect=true is first requested
    global _yolo
    if _yolo is None:
        _yolo = YOLO(DET_WEIGHTS)
    return _yolo


def _font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()  # ponytail: tiny on huge imgs; bundle a ttf if it matters


def _draw_bytes(img: Image.Image, vehicles) -> bytes:
    """Draw det boxes + top make/model label per vehicle -> JPEG bytes."""
    im = img.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    w = max(2, im.width // 600)
    font = _font(max(14, im.width // 70))
    for v in vehicles:
        x1, y1, x2, y2 = v["bbox"]
        mm = v.get("make_model")
        label = (f'{mm[0]["label"]} {mm[0]["confidence"]:.2f}' if mm
                 else f'{v["det_class"]} {v["det_conf"]:.2f}')
        color = (0, 200, 0) if mm else (255, 140, 0)  # green=classified car, orange=other
        d.rectangle([x1, y1, x2, y2], outline=color, width=w)
        tb = d.textbbox((x1, y1), label, font=font)
        d.rectangle([tb[0], tb[1], tb[2], tb[3]], fill=color)
        d.text((x1, y1), label, fill=(0, 0, 0), font=font)
    b = io.BytesIO(); im.save(b, "JPEG")
    return b.getvalue()


def _annotated(img: Image.Image, vehicles) -> str:
    """JPEG bytes -> base64 data-URI."""
    return "data:image/jpeg;base64," + base64.b64encode(_draw_bytes(img, vehicles)).decode()


def detect_frame(img: Image.Image, track: bool, topk: int, annotate: bool = False):
    """Detect vehicles, crop+classify make/model for car/bus/truck. track=True -> persistent IDs.
    Returns {"vehicles": [...], "annotated": <data-uri>?}."""
    m = yolo()
    res = (m.track(img, persist=True, verbose=False) if track else m.predict(img, verbose=False))[0]
    vehicles, crops, to_fill = [], [], []
    for b in res.boxes:
        xyxy = [round(v) for v in b.xyxy[0].tolist()]
        v = {"bbox": xyxy, "det_class": m.names[int(b.cls)], "det_conf": round(float(b.conf), 4),
             "make_model": None}
        if track:
            v["track_id"] = int(b.id) if b.id is not None else None
        if v["det_class"] in CLASSIFY_CLS:
            crops.append(img.crop(xyxy)); to_fill.append(v)
        vehicles.append(v)
    for v, p in zip(to_fill, classify(crops, topk)):
        v["make_model"] = p or None  # None when all preds below CONF_MIN
    out = {"vehicles": vehicles}
    if annotate:
        out["annotated"] = _annotated(img, vehicles)
    return out


def safe_stream_url(u: str) -> str:
    """Reject SSRF / file-read vectors: scheme allowlist + block internal/loopback/link-local hosts."""
    p = urlparse(u)
    if p.scheme not in ALLOWED_SCHEMES:
        raise ValueError("url scheme not allowed")
    if not p.hostname:
        raise ValueError("url has no host")
    for info in socket.getaddrinfo(p.hostname, p.port or None):
        ip = ipaddress.ip_address(info[4][0])
        # normalize IPv4-mapped IPv6 (::ffff:a.b.c.d) so the embedded v4 addr is classified
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
            ip = ip.ipv4_mapped
        # is_global rejects private/loopback/link-local/multicast/reserved/unspecified +
        # NAT64 / 6to4 / IPv4-translated ranges in one check (also covers pre-3.12 gaps)
        if not ip.is_global:
            raise ValueError("url host not allowed")
    # ponytail: validates every resolved IP, but ffmpeg re-resolves at connect time, so a
    # DNS-rebind to another *public* IP remains possible (rebind to private is blocked above).
    # Real fix if it ever matters: front outbound fetches through an egress proxy.
    return u


def _redact_url(u: str) -> str:
    """Strip userinfo (user:pass@) from a URL so creds don't reach logs."""
    try:
        q = urlparse(u)
        netloc = q.hostname or ""
        if q.port:
            netloc += f":{q.port}"
        return urlunparse(q._replace(netloc=netloc))
    except Exception:
        return "<unparseable url>"


def sample_frames(source, options=None):
    """Decode up to MAX_FRAMES frames (strided) from a file-like or stream URL via PyAV."""
    frames = []
    with av.open(source, options=options or {}) as c:
        for n, frame in enumerate(c.decode(video=0)):
            if n % FRAME_STRIDE == 0:
                frames.append(frame.to_image())
                if len(frames) >= MAX_FRAMES:
                    break
    return frames


def handle(name: str, data: bytes, topk: int, detect: bool, annotate: bool = False):
    ext = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if zipfile.is_zipfile(io.BytesIO(data)):
        imgs = []
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for m in z.namelist():
                if "." + m.rsplit(".", 1)[-1].lower() in IMG_EXT and not m.endswith("/"):
                    imgs.append(Image.open(io.BytesIO(z.read(m))).convert("RGB"))
        if detect:
            return {"name": name, "type": "zip",
                    "images": [detect_frame(i, False, topk, annotate) for i in imgs]}
        return {"name": name, "type": "zip", "predictions": classify(imgs, topk)}
    if ext in IMG_EXT:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        if detect:
            return {"name": name, "type": "image", **detect_frame(img, False, topk, annotate)}
        return {"name": name, "type": "image", "predictions": classify([img], topk)[0]}
    # else: treat as video. pipe-only protocol allowlist: a malicious container can't make
    # ffmpeg open file:/concat:/http: external resources (arbitrary file read / SSRF).
    frames = sample_frames(io.BytesIO(data), options={"protocol_whitelist": "pipe"})
    if detect:
        return {"name": name, "type": "video",
                "frames": [{"frame": i, **detect_frame(f, True, topk, annotate)} for i, f in enumerate(frames)]}
    return {"name": name, "type": "video", "frames": classify(frames, topk)}


@app.post("/predict")
async def predict(files: List[UploadFile] = File(default=[]),
                  urls: List[str] = Form(default=[]), topk: int = 3,
                  detect: bool = False, annotate: bool = False,
                  image: bool = False):
    detect = detect or annotate  # annotation needs boxes; imply detect
    if image:  # return the annotated first image as raw JPEG (so Postman/browser renders it)
        if not files:
            return {"error": "image=true needs an uploaded image file"}
        img = Image.open(io.BytesIO(await files[0].read())).convert("RGB")
        out = detect_frame(img, False, topk)  # classifies cars -> make_model
        return Response(_draw_bytes(img, out["vehicles"]), media_type="image/jpeg")
    results = []
    for f in files:
        try:
            results.append(handle(f.filename or "upload", await f.read(), topk, detect, annotate))
        except Exception:
            log.exception("failed processing file %s", f.filename)
            results.append({"name": f.filename, "error": "processing failed"})
    for u in urls:
        try:
            frames = sample_frames(safe_stream_url(u), options={"protocol_whitelist": PROTO_WHITELIST})
            if detect:
                out = [{"frame": i, **detect_frame(f, True, topk, annotate)} for i, f in enumerate(frames)]
            else:
                out = classify(frames, topk)
            results.append({"name": u, "type": "stream", "frames": out})
        except Exception as e:
            log.error("failed processing stream %s: %s", _redact_url(u), type(e).__name__)
            results.append({"name": u, "error": "processing failed"})
    return {"results": results}


@app.get("/health")
def health():
    return {"classes": len(classes), "device": dev}


if __name__ == "__main__":  # `uv run api.py` -> serves here; CLI uvicorn flags still override
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)
