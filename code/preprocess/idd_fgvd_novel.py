#!/usr/bin/env -S uv run --script
"""Filter idd_fgvd_vmmr down to only classes NOT already covered by any other
dataset in DATA, so idd_fgvd_novel_vmmr adds purely novel make/models rather
than duplicating vehicles already represented elsewhere.

Overlap check: every other VMMRdb-style dataset dir in DATA (VMMRdb itself
plus every *_vmmr sibling, excluding idd_fgvd's own outputs and color_vmmr
which isn't a make/model dataset) contributes its class names to an "already
covered" set, with any trailing "_<year>" suffix stripped first (e.g.
"audi_a3_2016" -> "audi_a3") so year-tagged datasets (VMMRdb, vn_vmmr,
compcars_vmmr, stanfordcars_vmmr) compare on the same make_model key as
year-less ones (boxcars116k_vmmr, car1000_vmmr, idd_fgvd_vmmr itself).
idd_fgvd_vmmr's own classes have no year suffix already, so no stripping
needed on that side.

KNOWN LIMITATION: this is exact string matching on the make_model key, not
fuzzy/semantic matching. A class like Stanford Cars' body-style-suffixed
"mercedes_benz_c_class_sedan_2012" strips to "mercedes_benz_c_class_sedan",
which will NOT match idd_fgvd's plain "mercedes_benz_c_class" -- so some
real-world overlaps (same make/model, different naming granularity between
source datasets) can slip through as "novel" when they aren't. Good enough
for the common case (most datasets use plain make_model), not perfect.

Output:
  DATA/idd_fgvd_novel_vmmr/<class>/...  (symlinks to idd_fgvd_vmmr's own
  real cropped files -- idd_fgvd_vmmr is itself already real files, not
  symlinks, so this is a filtered *view* of it, not a re-crop)

Usage:
  uv run code/preprocess/idd_fgvd_novel.py
"""
import re
import sys

sys.path.insert(
    0,
    "/Users/logan/Developer/vibes/WORK/CLS/VMMRdb/.claude/worktrees/vmmr-multi-dataset-prep/code/preprocess",
)
from naming import DATA  # noqa: E402

IDD_SRC = DATA / "idd_fgvd_vmmr"
OUT = DATA / "idd_fgvd_novel_vmmr"
EXCLUDE_PREFIXES = ("idd_fgvd", "color_vmmr")

YEAR_SUFFIX_RE = re.compile(r"_(19|20)\d{2}$")


def strip_year(class_name):
    return YEAR_SUFFIX_RE.sub("", class_name)


def other_dataset_dirs():
    dirs = []
    for d in DATA.iterdir():
        if not d.is_dir():
            continue
        if d.name.startswith(EXCLUDE_PREFIXES):
            continue
        if d.name == "VMMRdb" or d.name.endswith("_vmmr"):
            dirs.append(d)
    return dirs


def main():
    if not IDD_SRC.is_dir():
        print(f"ERROR: missing {IDD_SRC}")
        return

    covered = set()
    other_dirs = other_dataset_dirs()
    for d in other_dirs:
        for cls_dir in d.iterdir():
            if cls_dir.is_dir():
                covered.add(strip_year(cls_dir.name))

    idd_classes = sorted(d.name for d in IDD_SRC.iterdir() if d.is_dir())
    novel = [c for c in idd_classes if c not in covered]
    dropped = [c for c in idd_classes if c in covered]

    OUT.mkdir(parents=True, exist_ok=True)

    n_images = 0
    for cls in novel:
        src_dir = IDD_SRC / cls
        out_dir = OUT / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        for img in src_dir.glob("*.*"):
            out_img = out_dir / img.name
            if not out_img.exists():
                out_img.symlink_to(img.resolve())
            n_images += 1

    print(f"compared against {len(other_dirs)} other dataset dirs: {sorted(d.name for d in other_dirs)}")
    print(f"idd_fgvd_vmmr classes: {len(idd_classes)}")
    print(f"novel (kept): {len(novel)}")
    print(f"overlapping (dropped): {len(dropped)}  {sorted(dropped)[:10]}")
    print(f"images linked: {n_images}")


if __name__ == "__main__":
    main()
