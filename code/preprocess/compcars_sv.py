# /// script
# dependencies = ["scipy"]
# ///
"""Convert CompCars' surveillance-nature set (sv_data) into a flat VMMRdb-style
class tree. Sibling of compcars.py, which only converts the main web-nature
"image/" tree and explicitly skips sv_data (no per-model labels were found in
that script's first pass) -- sv_data DOES carry make/model labels via
sv_make_model_name.mat, just no year (street-camera captures aren't dated).

Source (data/CompCars/, extracted from sv_data.zip + sv_data.z01-z03, same
AES password as the main archive: d89551fd190e38):
  extracted/sv_data/sv_make_model_name.mat -- MATLAB cell array 'sv_make_model_name',
    shape (N, 3), 1-indexed by surveillance_model_id:
      col0 = make name (e.g. "Audi")
      col1 = model name, ROUTINELY prefixed with the make (e.g. "Audi Q5") --
        reuses compcars.py's strip_make_prefix so slug() doesn't double up.
      col2 = corresponding model_id in the *web-nature* data (unused here).
  extracted/sv_data/image/<surveillance_model_id>/<hash>.jpg -- the images.

Output:
  DATA/compcars_sv_vmmr/<slug>/<slug>_<idx>.jpg  (symlinks to originals, no year)

EXTRACT COMMANDS (absolute paths, copy-pasteable from any cwd):
  7z x /Users/logan/Developer/vibes/WORK/CLS/VMMRdb/data/CompCars/sv_data.zip \\
      -o/Users/logan/Developer/vibes/WORK/CLS/VMMRdb/data/CompCars/extracted \\
      -pd89551fd190e38 -y

Usage:
  uv run code/preprocess/compcars_sv.py --sample 30
  uv run code/preprocess/compcars_sv.py            # full run
"""
import argparse
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


def load_names(mat_path):
    from scipy.io import loadmat

    m = loadmat(str(mat_path))
    return m["sv_make_model_name"]  # shape (N, 3): make, model, web_model_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None, help="process only first N images")
    args = ap.parse_args()

    if not MAT_PATH.is_file():
        print(f"ERROR: missing {MAT_PATH} -- extract sv_data.zip first (see module docstring).")
        return
    if not IMAGE_ROOT.is_dir():
        print(f"ERROR: missing {IMAGE_ROOT} -- extract sv_data.zip first (see module docstring).")
        return

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
    fallback_makes = set()
    fallback_models = set()

    def class_slug_for(sv_id):
        if sv_id in class_slug_cache:
            return class_slug_cache[sv_id]

        make = cell_str(make_col.reshape(-1, 1), sv_id) or f"svmake{sv_id}"
        if make.startswith("svmake"):
            fallback_makes.add(sv_id)

        model_raw = cell_str(model_col.reshape(-1, 1), sv_id)
        if model_raw is None:
            model = f"svmodel{sv_id}"
            fallback_models.add(sv_id)
        else:
            model = strip_make_prefix(make, model_raw)
            if not model:
                model = f"svmodel{sv_id}"
                fallback_models.add(sv_id)

        s = slug(make, model)  # no year: surveillance captures aren't dated
        class_slug_cache[sv_id] = s
        return s

    n_images = 0
    slug_counters = {}
    example_class_dirs = []

    for sv_id, img in records:
        cls = class_slug_for(sv_id)

        out_dir = OUT / cls
        out_dir.mkdir(parents=True, exist_ok=True)

        idx = slug_counters.get(cls, 1)
        if cls not in example_class_dirs:
            example_class_dirs.append(cls)

        out_img = out_dir / f"{cls}_{idx}{img.suffix.lower()}"
        if not out_img.exists():
            out_img.symlink_to(img.resolve())
        idx += 1
        n_images += 1

        slug_counters[cls] = idx

    print(f"images processed: {n_images}")
    print(f"unique output class folders: {len(slug_counters)}")
    print(f"example class folders: {sorted(example_class_dirs)[:5]}")
    print(f"surveillance_model_ids with no make in .mat: {len(fallback_makes)} {sorted(fallback_makes)[:10]}")
    print(f"surveillance_model_ids with no model in .mat: {len(fallback_models)} {sorted(fallback_models)[:10]}")


if __name__ == "__main__":
    main()
