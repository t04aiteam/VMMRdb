# Delete cross-class "poison" images from data/vn_vmmr in place.
#
# Unlike build_dataset.py (which is deliberately non-destructive -- it only
# writes a manifest, raw tree untouched), this physically removes files whose
# pHash appears in more than one class folder. Investigated via
# investigate_poison.py: ~65% of poison hashes are the same generic stock
# photo repeated across year-variants of one model (DDG can't tell years
# apart), ~35% are genuine cross-model collisions (shared dealer/listing
# thumbnails, interior shots, etc returned for loosely-matching queries).
# Both are the same failure mode (an ambiguous image that can't honestly
# belong to just one class) so both get deleted the same way.
#
# Run:
#   uv run --with pillow --with imagehash code/clean_poison.py [--src data/vn_vmmr] [--dry-run]

import argparse
from collections import defaultdict
from pathlib import Path

from PIL import Image
import imagehash

ROOT = Path(__file__).resolve().parents[1]
HASH_SIZE = 8


def phash(p):
    try:
        with Image.open(p) as im:
            return imagehash.phash(im.convert("RGB"), hash_size=HASH_SIZE)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "data" / "vn_vmmr"))
    ap.add_argument("--dry-run", action="store_true", help="report only, don't delete")
    a = ap.parse_args()
    src = Path(a.src)

    classes = sorted(d for d in src.iterdir() if d.is_dir())
    print(f"{len(classes)} class folders in {src}")

    hash_classes = defaultdict(set)
    path_hash = {}
    bad = 0
    for d in classes:
        for p in d.glob("*.*"):
            h = phash(p)
            if h is None:
                bad += 1
                continue
            hash_classes[h].add(d.name)
            path_hash[p] = h

    poison = {h for h, cs in hash_classes.items() if len(cs) > 1}
    print(f"poison hashes: {len(poison)}")

    to_delete = [p for p, h in path_hash.items() if h in poison]
    print(f"poisoned files to delete: {len(to_delete)}")

    if a.dry_run:
        print("dry run -- nothing deleted")
        return

    deleted = 0
    for p in to_delete:
        p.unlink()
        deleted += 1

    # report classes left empty by this cleanup (informational, not pruned here)
    empty = [d.name for d in classes if not any(d.iterdir())]
    print(f"deleted: {deleted}")
    print(f"classes now empty: {len(empty)}")
    print("sample empty classes:", sorted(empty)[:10])


if __name__ == "__main__":
    main()
