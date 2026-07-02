# /// script
# dependencies = ["scipy"]
# ///
"""Convert CompCars into a flat VMMRdb-style class tree.

Source (data/CompCars/):
  data.z01..data.z22 + data.zip -- a SPLIT zip archive (~15.7GB compressed,
  ~16.6GB uncompressed across image/label/part/misc/train_test_split). It is
  ALSO AES-encrypted -- 7z reports "Wrong password" until you pass -p with the
  CompCars dataset password (the widely-circulated one shipped by the CUHK
  authors on request: d89551fd190e38). This script does NOT extract the
  archive itself; it only reads a pre-extracted tree at
  data/CompCars/extracted/data/... which preserves the archive's internal
  paths verbatim (leading "data/" prefix kept). See EXTRACT COMMANDS below.

  data/misc/make_model_name.mat -- MATLAB cell arrays, 1-indexed:
    make_names[make_id-1]  -> e.g. "ABT"
    model_names[model_id-1] -> e.g. "ABT A3" (NOTE: CompCars model names
      routinely embed the make name as a prefix, e.g. "Audi A3 hatchback",
      "ABT A3" -- this script strips a leading "<make> " token so slug()
      does not double up into "abt_abt_a3"). ~288/2004 model_names entries
      are EMPTY (missing label) -- these fall back to "model<id>" so no
      class is silently dropped and no image is skipped for a naming gap.

  data/image/<make_id>/<model_id>/<released_year>/<image_name>.jpg -- the
  paper's full-car-photo set (per README.txt). <released_year> is almost
  always a 4-digit year but is NOT validated by the dataset -- if a segment
  doesn't match ^(19|20)\\d{2}$ the year is DROPPED from the slug rather than
  embedded verbatim (avoids e.g. "..._0" or "..._unknown" class name noise).

  SKIPPED by this script: data/sv_data* (surveillance images -- these DO
  have make/model labels via sv_make_model_name.mat, just no year; converted
  separately by the sibling compcars_sv.py -> data/compcars_sv_vmmr/),
  data/part (car part crops, not whole-car classification), data/label
  (bbox annotations -- unused, whole image is symlinked to match VMMRdb
  style, same choice as stanfordcars.py).

Output:
  DATA/compcars_vmmr/<slug>/<slug>_<idx>.jpg  (symlinks to originals)

EXTRACT COMMANDS (already run for the make_id=1 subtree used by --sample;
paths are absolute so these are copy-pasteable from any cwd):
  # misc mat file (needed for names):
  7z x /Users/logan/Developer/vibes/WORK/CLS/VMMRdb/data/CompCars/data.zip \\
      -o/Users/logan/Developer/vibes/WORK/CLS/VMMRdb/data/CompCars/extracted \\
      "data/misc/make_model_name.mat" -pd89551fd190e38 -y
  # sample subtree only (~11MB, one make, all 133 images under make_id=1):
  7z x /Users/logan/Developer/vibes/WORK/CLS/VMMRdb/data/CompCars/data.zip \\
      -o/Users/logan/Developer/vibes/WORK/CLS/VMMRdb/data/CompCars/extracted \\
      "data/image/1/*" -pd89551fd190e38 -y
  # FULL run (NOT executed by this agent -- ~13.2GB uncompressed, image/ only,
  # ~143k files/dirs, single continuous compressed stream across 23 volumes
  # so it always reads start-to-finish, no partial-resume):
  7z x /Users/logan/Developer/vibes/WORK/CLS/VMMRdb/data/CompCars/data.zip \\
      -o/Users/logan/Developer/vibes/WORK/CLS/VMMRdb/data/CompCars/extracted \\
      "data/image/*" -pd89551fd190e38 -y

Usage:
  uv run code/preprocess/compcars.py --sample 30
  uv run code/preprocess/compcars.py            # full run (not run by this agent)
"""
import argparse
import re
import sys

sys.path.insert(
    0,
    "/Users/logan/Developer/vibes/WORK/CLS/VMMRdb/.claude/worktrees/vmmr-multi-dataset-prep/code/preprocess",
)
from naming import slug, DATA  # noqa: E402

SRC = DATA / "CompCars"
EXTRACTED = SRC / "extracted" / "data"
MAT_PATH = EXTRACTED / "misc" / "make_model_name.mat"
IMAGE_ROOT = EXTRACTED / "image"
OUT = DATA / "compcars_vmmr"

YEAR_RE = re.compile(r"^(19|20)\d{2}$")


def cell_str(arr, idx1based):
    """1-indexed MATLAB cell-array lookup -> python str, or None if empty/OOB."""
    i = idx1based - 1
    if i < 0 or i >= arr.shape[0]:
        return None
    cell = arr[i]
    val = cell[0] if len(cell) else None
    if val is None or (hasattr(val, "size") and val.size == 0):
        return None
    return str(val[0]).strip() or None


def strip_make_prefix(make: str, model: str) -> str:
    """'ABT', 'ABT A3' -> 'A3'; leaves model unchanged if it doesn't start
    with the make name (many model_names entries don't repeat the make)."""
    make_low = make.lower()
    model_low = model.lower()
    if model_low == make_low:
        return model  # degenerate: model name IS the make name, keep as-is
    if model_low.startswith(make_low + " "):
        return model[len(make):].strip()
    return model


def load_names(mat_path):
    from scipy.io import loadmat

    m = loadmat(str(mat_path))
    return m["make_names"], m["model_names"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None, help="process only first N images")
    args = ap.parse_args()

    if not MAT_PATH.is_file():
        print(f"ERROR: missing {MAT_PATH} -- run the misc extract command from the module docstring first.")
        return
    if not IMAGE_ROOT.is_dir():
        print(f"ERROR: missing {IMAGE_ROOT} -- run an image extract command from the module docstring first.")
        return

    make_names, model_names = load_names(MAT_PATH)

    # Deterministic walk: sort by relative path so idx assignment (and thus
    # output filenames) is stable across re-runs regardless of filesystem
    # iteration order -- required for the skip-if-exists resume check below.
    records = []
    for make_dir in IMAGE_ROOT.iterdir():
        if not make_dir.is_dir() or not make_dir.name.isdigit():
            continue
        make_id = int(make_dir.name)
        for model_dir in make_dir.iterdir():
            if not model_dir.is_dir() or not model_dir.name.isdigit():
                continue
            model_id = int(model_dir.name)
            for year_dir in model_dir.iterdir():
                if not year_dir.is_dir():
                    continue
                year_tok = year_dir.name
                for img in year_dir.iterdir():
                    if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                        continue
                    records.append((make_id, model_id, year_tok, img))
    records.sort(key=lambda r: (r[0], r[1], r[2], r[3].name))

    if args.sample:
        records = records[: args.sample]

    OUT.mkdir(parents=True, exist_ok=True)

    class_slug_cache = {}
    fallback_makes = set()
    fallback_models = set()
    bad_years = set()

    def class_slug_for(make_id, model_id, year_tok):
        key = (make_id, model_id, year_tok)
        if key in class_slug_cache:
            return class_slug_cache[key]

        make = cell_str(make_names, make_id) or f"make{make_id}"
        if make.startswith("make") and make[4:].isdigit():
            fallback_makes.add(make_id)

        model_raw = cell_str(model_names, model_id)
        if model_raw is None:
            model = f"model{model_id}"
            fallback_models.add(model_id)
        else:
            model = strip_make_prefix(make, model_raw)
            if not model:
                model = f"model{model_id}"
                fallback_models.add(model_id)

        year = year_tok if YEAR_RE.match(year_tok) else None
        if year is None:
            bad_years.add(year_tok)

        s = slug(make, model, year)
        class_slug_cache[key] = s
        return s

    n_images = 0
    slug_counters = {}  # slug -> next output index
    example_class_dirs = []

    for make_id, model_id, year_tok, img in records:
        cls = class_slug_for(make_id, model_id, year_tok)

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
    print(f"makes with no name in .mat (fell back to 'make<id>'): {len(fallback_makes)} {sorted(fallback_makes)[:10]}")
    print(f"models with no/empty name in .mat (fell back to 'model<id>'): {len(fallback_models)} {sorted(fallback_models)[:10]}")
    print(f"year segments dropped from slug (non 4-digit 19xx/20xx): {sorted(bad_years)[:10]}")


if __name__ == "__main__":
    main()
