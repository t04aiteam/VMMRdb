#!/usr/bin/env python3
"""Rebuild of `data/2 - Finding Motorcycle Data.ipynb` scraper.

The notebook used Azure Bing Image Search (retired by Microsoft Aug 2025) and a
private `randomdatautilities` package (source lost), seeded by a `motorcycle_data`
folder that no longer exists. This is a keyless, self-contained replacement:

  seed   crawl totalmotorcycle.com model guides -> list of "YEAR Make Model" classes
  images DuckDuckGo image search (no API key) -> data/motorcycle_data/<class>/

Run with uv so deps stay out of the project venv:
  uv run --with requests --with beautifulsoup4 --with ddgs \
      code/scrape_motorcycles.py seed   --years 2018 --out data/motorcycle_classes.txt
  uv run --with requests --with beautifulsoup4 --with ddgs \
      code/scrape_motorcycles.py images --classes data/motorcycle_classes.txt --per-class 20
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from uuid import uuid4

import requests

BASE = "https://www.totalmotorcycle.com"
HEADERS = {"User-Agent": "Mozilla/5.0 (research dataset builder; respectful crawl)"}
# Links appear as relative (/motorcycles/..) on year pages and absolute on mfg pages,
# so match the path portion regardless of host.
# mfg:   /motorcycles/2018/Indian-models
# model: /motorcycles/2018/2018-indian-chief-classic-review/
MFG_RE = re.compile(r"/motorcycles/\d{4}/[\w+%-]+-models/?$")
MODEL_RE = re.compile(r"/motorcycles/\d{4}/(\d{4}-[\w+%-]+)-review/?$")


def _get(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.text


def class_from_slug(slug: str) -> str | None:
    """URL slug '2018-indian-chief-classic' -> '2018 indian chief classic'.

    Hyphens become spaces (matches the notebook's `re.sub('-', ' ', ...)` convention).
    Requires a 4-digit year start and >=3 tokens (year + make + model).
    """
    if not re.match(r"\d{4}-", slug):
        return None
    name = slug.replace("-", " ").replace("+", " ").lower()
    name = re.sub(r"\s+", " ", name).strip()
    return name if len(name.split()) >= 3 else None


# ---------------------------------------------------------------- seed stage
def seed(years: list[int], out: Path, delay: float) -> None:
    from bs4 import BeautifulSoup

    classes: set[str] = set()
    for year in years:
        try:
            html = _get(f"{BASE}/{year}-motorcycle-models")
        except requests.RequestException as e:
            print(f"  [skip] {year} year page: {e}", file=sys.stderr)
            continue
        soup = BeautifulSoup(html, "html.parser")
        mfg_links = sorted({a["href"] for a in soup.find_all("a", href=MFG_RE)})
        print(f"{year}: {len(mfg_links)} manufacturers")
        for href in mfg_links:
            try:
                mfg_html = _get(BASE + href)
            except requests.RequestException as e:
                print(f"  [skip] {href}: {e}", file=sys.stderr)
                continue
            msoup = BeautifulSoup(mfg_html, "html.parser")
            before = len(classes)
            for a in msoup.find_all("a", href=MODEL_RE):
                m = MODEL_RE.search(a["href"])
                name = class_from_slug(m.group(1)) if m else None
                if name:
                    classes.add(name)
            print(f"  {href.split('/')[-1]}: +{len(classes) - before}")
            time.sleep(delay)
        time.sleep(delay)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(sorted(classes)) + "\n")
    print(f"\nWrote {len(classes)} classes -> {out}")


# -------------------------------------------------------------- images stage
def _download(url: str, dest: Path) -> bool:
    try:
        r = requests.get(url, headers=HEADERS, timeout=30, stream=True)
        r.raise_for_status()
        if "image" not in r.headers.get("Content-Type", ""):
            return False
        dest.write_bytes(r.content)
        return True
    except requests.RequestException:
        return False


def images(classes_file: Path, out_dir: Path, per_class: int, workers: int) -> None:
    from ddgs import DDGS

    names = [ln.strip() for ln in classes_file.read_text().splitlines() if ln.strip()]
    print(f"{len(names)} classes, target {per_class} imgs each -> {out_dir}/")
    for i, name in enumerate(names, 1):
        folder = out_dir / name.replace(" ", "_")
        folder.mkdir(parents=True, exist_ok=True)
        have = len(list(folder.glob("*.jpg")))
        if have >= per_class:
            print(f"[{i}/{len(names)}] {name}: have {have}, skip")
            continue
        try:
            with DDGS() as ddgs:
                hits = list(ddgs.images(name + " motorcycle", max_results=per_class * 2))
        except Exception as e:  # ddgs raises various rate-limit errors
            print(f"[{i}/{len(names)}] {name}: search failed ({e})", file=sys.stderr)
            continue
        urls = [h["image"] for h in hits if h.get("image")]
        got = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {}
            for u in urls:
                if got + len(futs) >= per_class - have:
                    break
                dest = folder / f"{uuid4()}.jpg"
                futs[ex.submit(_download, u, dest)] = dest
            for f in as_completed(futs):
                if f.result():
                    got += 1
                else:
                    futs[f].unlink(missing_ok=True)
        print(f"[{i}/{len(names)}] {name}: +{got} (total {have + got})")
    print("Done.")


def selfcheck() -> None:
    assert class_from_slug("2018-indian-chief-classic") == "2018 indian chief classic"
    assert class_from_slug("2019-yamaha-yzf-r1") == "2019 yamaha yzf r1"
    assert class_from_slug("2018-suzuki-burgman-200-abs") == "2018 suzuki burgman 200 abs"
    assert class_from_slug("cruiser") is None  # no year
    assert class_from_slug("2020-honda") is None  # only 2 tokens
    print("selfcheck ok")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("seed", help="crawl class list from totalmotorcycle.com")
    ps.add_argument("--years", type=int, nargs="+", required=True)
    ps.add_argument("--out", type=Path, default=Path("data/motorcycle_classes.txt"))
    ps.add_argument("--delay", type=float, default=1.0, help="politeness delay (s) between requests")

    pi = sub.add_parser("images", help="download images per class via DuckDuckGo")
    pi.add_argument("--classes", type=Path, default=Path("data/motorcycle_classes.txt"))
    pi.add_argument("--out", type=Path, default=Path("data/motorcycle_data"))
    pi.add_argument("--per-class", type=int, default=20)
    pi.add_argument("--workers", type=int, default=6)

    sub.add_parser("selfcheck", help="run assertions on the class-name parser")

    a = p.parse_args()
    if a.cmd == "seed":
        seed(a.years, a.out, a.delay)
    elif a.cmd == "images":
        images(a.classes, a.out, a.per_class, a.workers)
    elif a.cmd == "selfcheck":
        selfcheck()


if __name__ == "__main__":
    main()
