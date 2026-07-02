# /// script
# dependencies = ["scipy"]
# ///
"""Convert CompCars' surveillance-nature set (sv_data), keeping only classes
NOT already covered by any other VMMRdb-style dataset dir in DATA (including
compcars_vmmr itself) -- novelty-filtered variant of compcars_sv.py, same
idea as idd_fgvd_novel.py.

Source/label parsing is identical to compcars_sv.py (sv_make_model_name.mat,
no year -- surveillance captures aren't dated); see that script for the
mat-format details and strip_make_prefix rationale.

Overlap check: every other VMMRdb-style dir in DATA (VMMRdb itself, every
*_vmmr sibling except this script's own output) contributes its class names
to an "already covered" set, with any trailing "_<year>" suffix stripped
(e.g. "audi_a3_2016" -> "audi_a3") so year-tagged datasets compare on the
same make_model key as year-less sv_data classes. Same exact-string-match
limitation as idd_fgvd_novel.py applies (body-style-suffixed classes like
Stanford Cars' can slip through as false "novel").

Output:
  DATA/compcars_sv_vmmr/<slug>/<slug>_<idx>.jpg  (symlinks, no year)

Usage:
  uv run code/preprocess/compcars_sv_novel.py --sample 30
  uv run code/preprocess/compcars_sv_novel.py
"""
import argparse
import re
import sys

sys.path.insert(
    0,
    "/Users/logan/Developer/vibes/WORK/CLS/VMMRdb/.claude/worktrees/vmmr-multi-dataset-prep/code/preprocess",
)
from naming import slug, DATA  # noqa: E402
from compcars import cell_str, strip_make_prefix  # noqa: E402

SRC = DATA / "CompCars"
SV_ROOT = SRC / "extracted" / "sv_data"
MAT_PATH = SV_ROOT / "sv_make_model_name.mat"
IMAGE_ROOT = SV_ROOT / "image"
OUT = DATA / "compcars_sv_vmmr"

YEAR_SUFFIX_RE = re.compile(r"_(19|20)\d{2}$")


def strip_year(class_name):
    return YEAR_SUFFIX_RE.sub("", class_name)


def other_dataset_dirs():
    dirs = []
    for d in DATA.iterdir():
        if not d.is_dir() or d.name == OUT.name:
            continue
        if d.name == "VMMRdb" or d.name.endswith("_vmmr"):
            dirs.append(d)
    return dirs


def load_names(mat_path):
    from scipy.io import loadmat

    m = loadmat(str(mat_path))
    return m["sv_make_model_name"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None, help="process only first N images")
    args = ap.parse_args()

    if not MAT_PATH.is_file():
        print(f"ERROR: missing {MAT_PATH} -- extract sv_data.zip first.")
        return
    if not IMAGE_ROOT.is_dir():
        print(f"ERROR: missing {IMAGE_ROOT} -- extract sv_data.zip first.")
        return

    covered = set()
    other_dirs = other_dataset_dirs()
    for d in other_dirs:
        for cls_dir in d.iterdir():
            if cls_dir.is_dir():
                covered.add(strip_year(cls_dir.name))

    names = load_names(MAT_PATH)
    make_col = names[:, 0]
    model_col = names[:, 1]

    records = []
    for sv_dir in IMAGE_ROOT.iterdir():
        if not sv_dir.is_dir() or not sv_dir.name.isdigit():
            continue
        sv_id = int(sv_dir.name)
        for img in sv_dir.iterdir():
            if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            records.append((sv_id, img))
    records.sort(key=lambda r: (r[0], r[1].name))

    if args.sample:
        records = records[: args.sample]

    OUT.mkdir(parents=True, exist_ok=True)

    class_slug_cache = {}
    novel_ids = set()
    overlap_ids = set()

    def class_slug_for(sv_id):
        if sv_id in class_slug_cache:
            return class_slug_cache[sv_id]
        make = cell_str(make_col.reshape(-1, 1), sv_id) or f"svmake{sv_id}"
        model_raw = cell_str(model_col.reshape(-1, 1), sv_id)
        model = strip_make_prefix(make, model_raw) if model_raw else f"svmodel{sv_id}"
        if not model:
            model = f"svmodel{sv_id}"
        s = slug(make, model)
        class_slug_cache[sv_id] = s
        return s

    n_images = 0
    slug_counters = {}

    for sv_id, img in records:
        cls = class_slug_for(sv_id)
        if cls in covered:
            overlap_ids.add(sv_id)
            continue
        novel_ids.add(sv_id)

        out_dir = OUT / cls
        out_dir.mkdir(parents=True, exist_ok=True)

        idx = slug_counters.get(cls, 1)
        out_img = out_dir / f"{cls}_{idx}{img.suffix.lower()}"
        if not out_img.exists():
            out_img.symlink_to(img.resolve())
        idx += 1
        n_images += 1
        slug_counters[cls] = idx

    all_slugs = {class_slug_for(sv_id) for sv_id, _ in records}
    print(f"compared against {len(other_dirs)} other dataset dirs: {sorted(d.name for d in other_dirs)}")
    print(f"sv_data classes total: {len(all_slugs)}")
    print(f"novel classes (kept): {len(slug_counters)}")
    print(f"overlapping classes (dropped): {len(all_slugs) - len(slug_counters)}")
    print(f"images linked: {n_images}")


if __name__ == "__main__":
    main()
