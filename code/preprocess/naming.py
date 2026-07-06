"""Shared class-slug convention for all dataset converters.

Matches VMMRdb / model.pt: lowercase, underscore-separated, "<make>_<model>[_<year>]".
Year is omitted (not "_unknown") when the source dataset has no year label.

dataset/ is gitignored and lives only in the main checkout, not per-worktree ->
converters import MAIN_REPO from here and read/write under MAIN_REPO / "dataset".
"""
import re
from pathlib import Path

MAIN_REPO = Path("/mnt/data/nblong-t04/VMMRdb")
DATA = MAIN_REPO / "dataset"

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NONALNUM = re.compile(r"[^a-z0-9]+")


def _norm_token(s):
    s = _CAMEL.sub("_", s)                 # MarutiSuzuki -> Maruti_Suzuki
    s = s.lower()
    s = _NONALNUM.sub("_", s)              # spaces/hyphens/dots -> _
    return s.strip("_")


def slug(make, model, year=None):
    parts = [_norm_token(make), _norm_token(model)]
    if year:
        parts.append(str(year).strip())
    parts = [p for p in parts if p]
    return "_".join(parts)
