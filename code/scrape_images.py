# Phase 2 — scrape images per make/model/year class into a VMMR-style tree.
#
# Consumes meta/vn_classes.json (built by scrape_taxonomy.py) and, for every
# class, runs a keyless DuckDuckGo image search and downloads up to N images to
# data/vn_vmmr/<class>/. Resumable: a class whose folder already holds >= target
# images is skipped, so the crawl can run in chunks / be re-run after failures.
#
# Run:
#   uv run --with ddgs --with requests --with pillow code/scrape_images.py \
#       [--per 80] [--kind car|moto] [--limit N] [--out data/vn_vmmr]
#
# Add --verify to reject non-vehicle results (the keyless DDG image API is
# fragile under sustained automated queries and can silently backfill with
# unrelated "trending" images for obscure/rare classes -- fetch_image() only
# checks size/format, not content). --verify runs each candidate through the
# same vehicle_yolov9s detector used by detect_junk_classes.py before saving:
#   uv run --with ddgs --with requests --with pillow --with ultralytics \
#       code/scrape_images.py --verify [--conf 0.25] ...
#
# ponytail: DuckDuckGo image search (no API key) — same engine as the legacy
#   motorcycle scraper. Upgrade path: add Chotot listing-photo harvest as a
#   second source if per-class recall is thin for rare EV models.

import argparse, hashlib, io, json, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

try:
    from ddgs import DDGS
except ImportError:                       # older package name
    from duckduckgo_search import DDGS

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scrape_taxonomy import slug          # reuse exact slug() so folders match classes[]

ROOT = Path(__file__).resolve().parents[1]
TAX = ROOT / "meta" / "vn_classes.json"
HEADERS = {"User-Agent": "Mozilla/5.0 (research; VN VMMR image scrape)"}
MIN_BYTES = 4_000          # drop thumbnails / 1px trackers
MIN_SIDE = 160             # drop tiny images
QUERY_DELAY = 2.5          # politeness between DDG queries (it rate-limits hard)
DL_WORKERS = 8


def classes(kind=None):
    """Yield (slug, query) for every class in the taxonomy trees."""
    doc = json.loads(TAX.read_text(encoding="utf-8"))
    trees = {"car": doc["cars"], "moto": doc["motorbikes"],
             "truck": doc.get("trucks", {}), "bus": doc.get("buses", {})}
    for k, tree in trees.items():
        if kind and k != kind:
            continue
        for brand, models in tree.items():
            for model, rec in models.items():
                for year in rec["years"]:
                    cslug = f"{slug(brand)}_{slug(model)}_{year}"
                    query = f"{brand} {model} {year}"
                    yield cslug, query


def ddg_images(query, n):
    """Return up to n image URLs for a query, with simple backoff."""
    for attempt in range(3):
        try:
            with DDGS() as d:
                return [r["image"] for r in d.images(query, max_results=n) if r.get("image")]
        except Exception as e:
            wait = 5 * (attempt + 1)
            print(f"    ddg retry ({e}) in {wait}s", file=sys.stderr)
            time.sleep(wait)
    return []


def fetch_image(url):
    """Download + validate one image; return (bytes, ext, PIL.Image) or None."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        if r.status_code != 200 or len(r.content) < MIN_BYTES:
            return None
        if not r.headers.get("content-type", "").startswith("image"):
            return None
        im = Image.open(io.BytesIO(r.content))
        im.verify()
        if min(im.size) < MIN_SIDE:
            return None
        ext = (im.format or "jpg").lower().replace("jpeg", "jpg")
        # verify() leaves the handle unusable for pixel access -- reopen for --verify
        im = Image.open(io.BytesIO(r.content)).convert("RGB")
        return r.content, ext, im
    except Exception:
        return None


def have(folder):
    return len(list(folder.glob("*.*"))) if folder.exists() else 0


def scrape_class(cslug, query, target, out, verifier=None):
    folder = out / cslug
    if have(folder) >= target:
        return cslug, have(folder), True, 0        # already done -> skipped
    folder.mkdir(parents=True, exist_ok=True)
    seen = {f.stem for f in folder.glob("*.*")}    # content-hash filenames -> dedup across reruns
    urls = ddg_images(query, target * 2)           # overfetch; many fail validation
    saved = have(folder)
    rejected = 0
    with ThreadPoolExecutor(max_workers=DL_WORKERS) as ex:
        futs = {ex.submit(fetch_image, u): u for u in urls}
        for fut in as_completed(futs):
            if saved >= target:
                break
            res = fut.result()
            if not res:
                continue
            data, ext, im = res
            h = hashlib.md5(data).hexdigest()[:16]
            if h in seen:
                continue
            if verifier and not verifier(im):
                rejected += 1
                continue
            seen.add(h)
            (folder / f"{h}.{ext}").write_bytes(data)
            saved += 1
    return cslug, saved, False, rejected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=80, help="target images per class")
    ap.add_argument("--kind", choices=["car", "moto", "truck", "bus"], help="limit to one vehicle kind")
    ap.add_argument("--limit", type=int, help="process only first N classes (testing)")
    ap.add_argument("--out", default=str(ROOT / "data" / "vn_vmmr"))
    ap.add_argument("--verify", action="store_true",
                     help="reject non-vehicle images via the vehicle_yolov9s detector before saving")
    ap.add_argument("--conf", type=float, default=0.25, help="--verify detection confidence threshold")
    a = ap.parse_args()
    out = Path(a.out)

    verifier = None
    if a.verify:
        from detect_junk_classes import pick_device, MODEL_PATH
        from ultralytics import YOLO
        model = YOLO(str(MODEL_PATH))
        device = pick_device()
        print("verify device:", device)

        def verifier(im):
            r = model.predict([im], conf=a.conf, device=device, verbose=False)[0]
            return r.boxes is not None and len(r.boxes) > 0

    items = list(classes(a.kind))
    if a.limit:
        items = items[: a.limit]
    print(f"{len(items)} classes -> {out} (target {a.per}/class)")

    done = skipped = total_rejected = 0
    for i, (cslug, query) in enumerate(items, 1):
        c, saved, was_skip, rejected = scrape_class(cslug, query, a.per, out, verifier)
        total_rejected += rejected
        if was_skip:
            skipped += 1
        else:
            done += 1
            time.sleep(QUERY_DELAY)
        suffix = " (skip)" if was_skip else (f" ({rejected} rejected)" if rejected else "")
        print(f"[{i}/{len(items)}] {c}: {saved} imgs{suffix}")
    print(f"done: {done} scraped, {skipped} already complete, {total_rejected} rejected by verifier")


if __name__ == "__main__":
    main()
