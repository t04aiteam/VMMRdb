"""Smoke test api.handle/classify on image, zip, video bytes. No network stream. assert runs."""
import io, os, sys, tempfile, zipfile
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torchvision import models
import av

tmp = Path(tempfile.mkdtemp())
# fake 2-class checkpoint at tmp/model.pt
m = models.resnet50(num_classes=2)
torch.save({"state_dict": m.state_dict(), "classes": ["ford_focus_2010", "honda_civic_2012"]}, tmp / "model.pt")
os.chdir(tmp)
sys.path.insert(0, str(Path(__file__).parent))
import api  # loads tmp/model.pt

assert api._parse_year("honda_civic_2012") == 2012
assert api._parse_year("aito_ask_world_m5") is None

def jpg_bytes(color):
    b = io.BytesIO(); Image.new("RGB", (300, 300), color).save(b, "JPEG"); return b.getvalue()

# image
r = api.handle("a.jpg", jpg_bytes((200, 30, 30)), 3, False)
assert r["type"] == "image" and len(r["predictions"]) == 2, r  # 2 classes -> topk caps at 2

# zip of 2 images
zb = io.BytesIO()
with zipfile.ZipFile(zb, "w") as z:
    z.writestr("x.jpg", jpg_bytes((10, 200, 10)))
    z.writestr("y.png", jpg_bytes((10, 10, 200)))
r = api.handle("batch.zip", zb.getvalue(), 3, False)
assert r["type"] == "zip" and len(r["predictions"]) == 2, r

# tiny mp4 (30 frames) -> sampled
vb = io.BytesIO()
with av.open(vb, "w", format="mp4") as c:
    st = c.add_stream("libx264", rate=15); st.width, st.height, st.pix_fmt = 320, 240, "yuv420p"
    for i in range(30):
        arr = np.full((240, 320, 3), (i * 8 % 255, 100, 50), dtype=np.uint8)
        for pkt in st.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
            c.mux(pkt)
    for pkt in st.encode():
        c.mux(pkt)
r = api.handle("clip.mp4", vb.getvalue(), 3, False)
assert r["type"] == "video" and len(r["frames"]) >= 1, r

# SSRF guard: reject file://, loopback, link-local, private; allow public host
for bad in ["file:///etc/passwd", "http://127.0.0.1/x", "http://169.254.169.254/latest",
            "http://10.0.0.5/x", "concat:/etc/passwd", "rtsp://localhost/s"]:
    try:
        api.safe_stream_url(bad); raise SystemExit(f"FAIL: {bad} not rejected")
    except ValueError:
        pass
assert api.safe_stream_url("https://example.com/stream") == "https://example.com/stream"

# detect path on a REAL car image (synthetic solids have no vehicles to find)
import glob
repo = Path(__file__).parent
api.DET_WEIGHTS = str(repo / "weights/vehicle/vehicle_yolov9s_640_30oct2025.pt")
real = sorted(glob.glob(str(repo / "data/*/*.jpg")))[:1]
if real:
    rb = Path(real[0]).read_bytes()
    r = api.handle("car.jpg", rb, 3, detect=True)
    assert r["type"] == "image" and isinstance(r["vehicles"], list), r
    assert all("bbox" in v and "make_model" in v for v in r["vehicles"]), r
    assert all(v["vehicle_type"] == v["det_class"] for v in r["vehicles"]), r
    classified = [v for v in r["vehicles"] if v["det_class"] in api.CLASSIFY_CLS]
    assert all(isinstance(v["color"], dict) and "color" in v["color"] for v in classified), classified
    assert all("year" in v for v in classified), classified
    assert "annotated" not in r, "annotated should be absent when annotate=False"
    # annotate path
    ra = api.handle("car.jpg", rb, 3, detect=True, annotate=True)
    assert ra["annotated"].startswith("data:image/jpeg;base64,") and len(ra["annotated"]) > 100, ra.get("annotated", "")[:50]
    print(f"DETECT OK: {len(r['vehicles'])} vehicle(s) on {Path(real[0]).parent.name}; "
          f"annotated={len(ra['annotated'])//1024}KB b64")
else:
    print("DETECT SKIP: no data/*/*.jpg found")

print(f"API SMOKE OK: image={len(api.handle('a.jpg', jpg_bytes((1,2,3)),5,False)['predictions'])}top "
      f"zip=2imgs video+detect plumbing verified")
