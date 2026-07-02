# /// script
# dependencies = ["scipy"]
# ///
"""Build an ImageFolder-style class tree for car color classification from
CompCars' sv_data color labels (the only source in this repo with per-image
color ground truth).

Source:
  data/CompCars/extracted/sv_data/color_list.mat -- MATLAB cell array
  'color_list', shape (44481, 2): col0 = path relative to sv_data/image/
  (e.g. "1/d5dcefddcde927.jpg"), col1 = color id.
  IDs: -1 unrecognized (dropped), 0 black, 1 white, 2 red, 3 yellow, 4 blue,
  5 green, 6 purple, 7 brown, 8 champagne, 9 silver.

Output:
  DATA/color_vmmr/<color_name>/<color_name>_<idx>.jpg  (symlinks)

Usage:
  uv run code/color/prepare_color_data.py
"""
import sys

sys.path.insert(
    0,
    "/Users/logan/Developer/vibes/WORK/CLS/VMMRdb/.claude/worktrees/vmmr-multi-dataset-prep/code/preprocess",
)
from naming import DATA  # noqa: E402

SV_ROOT = DATA / "CompCars" / "extracted" / "sv_data"
MAT_PATH = SV_ROOT / "color_list.mat"
IMAGE_ROOT = SV_ROOT / "image"
OUT = DATA / "color_vmmr"

COLOR_NAMES = {
    0: "black", 1: "white", 2: "red", 3: "yellow", 4: "blue",
    5: "green", 6: "purple", 7: "brown", 8: "champagne", 9: "silver",
}


def load_rows(mat_path):
    from scipy.io import loadmat

    m = loadmat(str(mat_path))
    return m["color_list"]


def main():
    if not MAT_PATH.is_file():
        print(f"ERROR: missing {MAT_PATH} -- extract sv_data.zip first.")
        return

    rows = load_rows(MAT_PATH)
    OUT.mkdir(parents=True, exist_ok=True)

    counters = {name: 0 for name in COLOR_NAMES.values()}
    dropped_unrecognized = 0
    dropped_missing = 0

    for row in rows:
        rel_path = str(row[0][0])
        color_id = int(row[1][0][0])
        if color_id not in COLOR_NAMES:
            dropped_unrecognized += 1
            continue

        src = IMAGE_ROOT / rel_path
        if not src.is_file():
            dropped_missing += 1
            continue

        name = COLOR_NAMES[color_id]
        out_dir = OUT / name
        out_dir.mkdir(parents=True, exist_ok=True)

        counters[name] += 1
        out_img = out_dir / f"{name}_{counters[name]}{src.suffix.lower()}"
        if not out_img.exists():
            out_img.symlink_to(src.resolve())

    print(f"images linked: {sum(counters.values())}")
    print(f"dropped (unrecognized id -1): {dropped_unrecognized}")
    print(f"dropped (source file missing): {dropped_missing}")
    print("per-color counts:", counters)


if __name__ == "__main__":
    main()
