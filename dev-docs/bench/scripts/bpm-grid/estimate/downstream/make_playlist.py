"""Build the 145-track labelled playlist (100 straight, 30 lomanka, 15 lomanka_holdout).

Reads (read-only) the manual-label CSVs in the 0.3.6 rhythmic-regularity eval folder.
Writes downstream/labelled_145.txt (one path per line, playlist order) and
downstream/labelled_145.csv (idx,path,genre,expected,split).
"""
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import bpm_grid_paths as P  # noqa: E402

EVAL = P.REGULARITY_EVAL  # paths.json regularity_eval
rows = []
for name in ("straight.csv", "lomanka.csv", "lomanka_holdout.csv"):
    with (EVAL / name).open(encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append((r["path"], r["genre"], r["expected"], name.replace(".csv", "")))
seen, out = set(), []
for r in rows:
    if r[0] not in seen:
        seen.add(r[0])
        out.append(r)
(P.DOWNSTREAM / "labelled_145.txt").write_text("\n".join(r[0] for r in out) + "\n", encoding="utf-8")
with (P.DOWNSTREAM / "labelled_145.csv").open("w", encoding="utf-8", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["idx", "path", "genre", "expected", "source"])
    for i, r in enumerate(out, 1):
        w.writerow([i, *r])
print(len(out), "tracks;", sum(r[2] == "regular" for r in out), "regular")
