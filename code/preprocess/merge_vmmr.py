"""Merge the 3 make/model/year sources into one VMMRdb-style class tree for
training: VMMRdb (base, US, 9169 classes) + vn_vmmr (VN market, 2355 classes)
+ DVM_vmmr (UK market, 4889 classes, via code/preprocess/dvm.py). All three
already use the same "<make>_<model>_<year>" folder-name convention (see
naming.py), so merging is a straight symlink union keyed by class name --
same-named class dirs across sources combine into one output class with a
continuous image index.

Exclusions (vn_vmmr only, from the earlier junk-detection pass):
  - meta/vn_vmmr_suspect_classes.txt -- whole classes flagged by
    code/detect_junk_classes.py (tab-separated, class name is the first
    field). Currently empty (0 suspects at the final ratio threshold used).
  - meta/vn_vmmr_suspect_classes_corrupt_images.txt -- specific image paths
    that crashed the detector (repo-relative, one per line).

DVM_vmmr and VMMRdb are not re-screened here (see plan: DVM-CAR is a
curated marketing dataset, VMMRdb has been used as-is since the original
model.pt training run).

Output:
  DATA/merged_vmmr/<class>/<class>_<idx>.jpg  (symlinks to the ultimate
  real file -- resolve() so a class merged from DVM_vmmr's own symlinks
  doesn't chain through an extra hop)

Usage:
  uv run code/preprocess/merge_vmmr.py --sample 30
  uv run code/preprocess/merge_vmmr.py            # full run
"""
import argparse

from naming import DATA, MAIN_REPO

SOURCES = ["VMMRdb", "vn_vmmr", "DVM_vmmr"]  # fixed order -> deterministic idx assignment
OUT = DATA / "merged_vmmr"
SUSPECT_CLASSES_FILE = MAIN_REPO / "meta" / "vn_vmmr_suspect_classes.txt"
CORRUPT_IMAGES_FILE = MAIN_REPO / "meta" / "vn_vmmr_suspect_classes_corrupt_images.txt"
IMG_EXT = (".jpg", ".jpeg", ".png")


def load_suspect_classes():
    if not SUSPECT_CLASSES_FILE.is_file():
        return set()
    names = set()
    for line in SUSPECT_CLASSES_FILE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            names.add(line.split("\t", 1)[0])
    return names


def load_corrupt_images():
    if not CORRUPT_IMAGES_FILE.is_file():
        return set()
    paths = set()
    for line in CORRUPT_IMAGES_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            paths.add(str((MAIN_REPO / line).resolve()))
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None, help="process only first N images")
    args = ap.parse_args()

    suspect_classes = load_suspect_classes()
    corrupt_images = load_corrupt_images()
    print(f"vn_vmmr suspect classes to skip: {len(suspect_classes)}")
    print(f"corrupt images to skip: {len(corrupt_images)}")

    records = []
    for source in SOURCES:
        src_root = DATA / source
        if not src_root.is_dir():
            print(f"ERROR: missing source {src_root}, skipping")
            continue
        class_dirs = sorted(d for d in src_root.iterdir() if d.is_dir())
        for class_dir in class_dirs:
            if source == "vn_vmmr" and class_dir.name in suspect_classes:
                continue
            imgs = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in IMG_EXT)
            for img in imgs:
                real = img.resolve()
                if source == "vn_vmmr" and str(real) in corrupt_images:
                    continue
                records.append((class_dir.name, real))

    if args.sample:
        records = records[: args.sample]

    OUT.mkdir(parents=True, exist_ok=True)

    n_images = 0
    slug_counters = {}  # class name -> next output index

    for cls, real in records:
        out_dir = OUT / cls
        out_dir.mkdir(parents=True, exist_ok=True)

        idx = slug_counters.get(cls, 1)
        out_img = out_dir / f"{cls}_{idx}{real.suffix.lower()}"
        if not out_img.exists():
            out_img.symlink_to(real)
        slug_counters[cls] = idx + 1
        n_images += 1

    print(f"images processed: {n_images}")
    print(f"unique output class folders: {len(slug_counters)}")
    print(f"example class folders: {sorted(slug_counters)[:5]}")


if __name__ == "__main__":
    main()
