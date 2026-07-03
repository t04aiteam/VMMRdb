# Sanity-check vn_vmmr with a vehicle detector: for every class, run the
# vehicle_yolov9s detector on all N images. If <=1 image in the class gets a
# vehicle detection, flag the class as suspect (interior shots, wrong-photo
# scrapes, non-vehicle junk that slipped past the pHash dedup/poison filters).
#
# Non-destructive: writes flagged classes to a review file, does NOT delete
# anything -- deletion happens only after the user reviews the list.
#
# Run:
#   uv run --with ultralytics code/detect_junk_classes.py \
#       [--src data/vn_vmmr] [--conf 0.25] [--out meta/vn_vmmr_suspect_classes.txt]

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "weights" / "vehicle" / "vehicle_yolov9s_640_30oct2025.pt"


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "data" / "vn_vmmr"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default=str(ROOT / "meta" / "vn_vmmr_suspect_classes.txt"))
    a = ap.parse_args()
    src = Path(a.src)

    model = YOLO(str(MODEL_PATH))
    device = pick_device()
    print("device:", device)

    classes = sorted(d for d in src.iterdir() if d.is_dir())
    print(f"{len(classes)} classes in {src}")

    suspects = []
    for i, d in enumerate(classes, 1):
        imgs = sorted(p for p in d.glob("*.*"))
        n = len(imgs)
        if n == 0:
            continue

        detected = 0
        for j in range(0, n, a.batch):
            batch = [str(p) for p in imgs[j:j + a.batch]]
            results = model.predict(batch, conf=a.conf, device=device, verbose=False)
            for r in results:
                if r.boxes is not None and len(r.boxes) > 0:
                    detected += 1

        if detected <= 1:
            suspects.append((d.name, n, detected))

        if i % 100 == 0:
            print(f"[{i}/{len(classes)}] ... {len(suspects)} suspects so far")

    suspects.sort(key=lambda t: t[1])  # smallest N first

    out_path = Path(a.out)
    lines = [f"{name}\tN={n}\tdetected={det}" for name, n, det in suspects]
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    print(f"classes checked: {len(classes)}")
    print(f"suspect classes (<=1 detection): {len(suspects)}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
