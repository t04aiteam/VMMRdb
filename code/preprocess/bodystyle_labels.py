"""Extract body-style labels from the two sources whose folder names already
carry them (boxcars116k_vmmr, stanfordcars_vmmr), and build an ImageFolder-
style tree for training a bodystyle classifier.

Folder-name patterns:
  boxcars116k_vmmr:  <make>_<model>_<style>_mk<N>   e.g. alfaromeo_156_sedan_mk1
                      -- style is always the token right before "_mk<N>".
  stanfordcars_vmmr: <make>_<model...>_<style?>_<year>  e.g. acura_rl_sedan_2012
                      -- style is the token right before the trailing year,
                      IF that token is a real body-style word. Many
                      stanfordcars folders have a trim/performance code there
                      instead (type_r, ss, zr1, srt8...) -- those are
                      dropped, not guessed at.

Vocabulary survey (done by hand against the actual folder names, not
assumed): boxcars has {hatchback, sedan, combi, suv, van, mpv, coupe,
pickup, cabriolet, fastback} as real style tokens (plus a few one-off non-
style noise like "offroad", a stray "mkl" typo -- dropped). stanfordcars has
{sedan, suv, convertible, coupe, cab, hatchback, van, wagon, minivan} as real
style tokens (plus trim codes like ss/zr1/z06/xkr/srt8/abarth -- dropped).
"cab" in stanfordcars means truck "Crew Cab"/"Extended Cab", i.e. pickup --
NOT short for cabriolet/convertible.

VOCAB below merges synonyms across both sources into one target palette:
  sedan, suv, coupe, hatchback, convertible, wagon, van, pickup  (8 classes)

Output:
  DATA/bodystyle_vmmr/<style>/<style>_<idx>.jpg  (symlinks)

Usage:
  uv run code/preprocess/bodystyle_labels.py
"""
import re

from naming import DATA

BOXCARS = DATA / "boxcars116k_vmmr"
STANFORD = DATA / "stanfordcars_vmmr"
OUT = DATA / "bodystyle_vmmr"

VOCAB = {
    "sedan": "sedan",
    "suv": "suv",
    "coupe": "coupe",
    "fastback": "coupe",
    "hatchback": "hatchback",
    "convertible": "convertible",
    "cabriolet": "convertible",
    "wagon": "wagon",
    "combi": "wagon",
    "van": "van",
    "mpv": "van",
    "minivan": "van",
    "pickup": "pickup",
    "cab": "pickup",  # stanfordcars: "Crew Cab"/"Extended Cab" truck config, not cabriolet
}

BOXCARS_RE = re.compile(r"^(?P<rest>.+)_(?P<tok>[a-z]+)_mk[0-9a-z]*$")
STANFORD_RE = re.compile(r"^(?P<rest>.+)_(?P<tok>[a-z0-9]+)_(?:19|20)\d{2}$")


def extract_style(class_dir_name: str, pattern: re.Pattern):
    m = pattern.match(class_dir_name)
    if not m:
        return None
    return VOCAB.get(m.group("tok"))


def main():
    if not BOXCARS.is_dir():
        print(f"ERROR: missing {BOXCARS}")
        return
    if not STANFORD.is_dir():
        print(f"ERROR: missing {STANFORD}")
        return

    OUT.mkdir(parents=True, exist_ok=True)

    style_counters = {}
    dropped = 0
    n_images = 0

    for src, pattern in ((BOXCARS, BOXCARS_RE), (STANFORD, STANFORD_RE)):
        class_dirs = sorted(d for d in src.iterdir() if d.is_dir())
        for class_dir in class_dirs:
            style = extract_style(class_dir.name, pattern)
            if style is None:
                dropped += 1
                continue
            imgs = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
            out_dir = OUT / style
            out_dir.mkdir(parents=True, exist_ok=True)
            idx = style_counters.get(style, 1)
            for img in imgs:
                out_img = out_dir / f"{style}_{idx}{img.suffix.lower()}"
                if not out_img.exists():
                    out_img.symlink_to(img.resolve())
                idx += 1
                n_images += 1
            style_counters[style] = idx

    print(f"images processed: {n_images}")
    print(f"style classes: {sorted(style_counters)}")
    print(f"per-style counts: {style_counters}")
    print(f"class folders dropped (no vocab match): {dropped}")


if __name__ == "__main__":
    main()
