"""Convert DVM-CAR (Deep Visual Marketing) into a flat VMMRdb-style class tree.

Source (dataset/DVM/resized_DVM/):
  <Automaker>/<Genmodel>/<Year>/<Color>/<image>.jpg -- 84 automakers, 1,451,890
  images total. Color is a real per-image label here (23 named colors: Beige,
  Black, Blue, Bronze, Brown, Burgundy, Gold, Green, Grey, Indigo, Magenta,
  Maroon, Multicolour, Navy, Orange, Pink, Purple, Red, Silver, Turquoise,
  Unlisted, White, Yellow) but this script does NOT use it -- VMMRdb-style
  classes are make/model/year only, matching every other *_vmmr converter in
  this repo. Color-training is a separate, potential future use of this same
  source tree (richer than CompCars' sv_data: 23 classes vs 10, and actually
  present on this machine, unlike sv_data).

  Year is almost always a 4-digit folder name but not always (e.g. a literal
  "Unko" folder shows up under some models) -- same handling as compcars.py:
  a non ^(19|20)\\d{2}$ folder name drops the year from the slug rather than
  embedding it verbatim.

  A handful of images (97 out of 1,451,890) have no Color label at all (an
  empty "$$$$" segment in their filename) and sit directly under the Year
  folder instead of a Color subfolder -- handled as a color-less image, not
  dropped.

  There is also dataset/DVM/confirmed_fronts/ -- a smaller (61,827 image),
  curated single-front-view subset with color/genmodel encoded in the
  filename instead of the directory tree (quality-checked for the DVM
  paper's viewpoint task). NOT used here: resized_DVM is the full multi-angle
  set, matching how compcars.py treats CompCars' equivalent image/ tree.

Output:
  DATA/DVM_vmmr/<slug>/<slug>_<idx>.jpg  (symlinks to originals)

Usage:
  uv run code/preprocess/dvm.py --sample 30
  uv run code/preprocess/dvm.py            # full run (~1.45M symlinks)
"""
import argparse
import re

from naming import slug, DATA

SRC = DATA / "DVM" / "resized_DVM"
OUT = DATA / "DVM_vmmr"

YEAR_RE = re.compile(r"^(19|20)\d{2}$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None, help="process only first N images")
    args = ap.parse_args()

    if not SRC.is_dir():
        print(f"ERROR: missing {SRC}")
        return

    # Deterministic walk: sort by relative path so idx assignment (and thus
    # output filenames) is stable across re-runs regardless of filesystem
    # iteration order -- required for the skip-if-exists resume check below.
    records = []
    for make_dir in SRC.iterdir():
        if not make_dir.is_dir():
            continue
        for model_dir in make_dir.iterdir():
            if not model_dir.is_dir():
                continue
            for year_dir in model_dir.iterdir():
                if not year_dir.is_dir():
                    continue
                for entry in year_dir.iterdir():
                    if entry.is_dir():
                        for img in entry.iterdir():
                            if img.suffix.lower() in (".jpg", ".jpeg", ".png"):
                                records.append((make_dir.name, model_dir.name, year_dir.name, img))
                    elif entry.suffix.lower() in (".jpg", ".jpeg", ".png"):
                        records.append((make_dir.name, model_dir.name, year_dir.name, entry))  # no Color subfolder
    records.sort(key=lambda r: (r[0], r[1], r[2], r[3].name))

    if args.sample:
        records = records[: args.sample]

    OUT.mkdir(parents=True, exist_ok=True)

    bad_years = set()
    n_images = 0
    slug_counters = {}  # slug -> next output index

    for make, model, year_tok, img in records:
        year = year_tok if YEAR_RE.match(year_tok) else None
        if year is None:
            bad_years.add(year_tok)

        cls = slug(make, model, year)
        out_dir = OUT / cls
        out_dir.mkdir(parents=True, exist_ok=True)

        idx = slug_counters.get(cls, 1)
        out_img = out_dir / f"{cls}_{idx}{img.suffix.lower()}"
        if not out_img.exists():
            out_img.symlink_to(img.resolve())
        slug_counters[cls] = idx + 1
        n_images += 1

    print(f"images processed: {n_images}")
    print(f"unique output class folders: {len(slug_counters)}")
    print(f"example class folders: {sorted(slug_counters)[:5]}")
    print(f"year segments dropped from slug (non 4-digit 19xx/20xx): {sorted(bad_years)}")


if __name__ == "__main__":
    main()
