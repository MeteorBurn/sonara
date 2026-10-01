"""Per-track workbook: baseline vs combined (1+2+3+4) for failure inspection.
Writes results/per_track.xlsx (read-only on all inputs)."""

from __future__ import annotations

import pickle
import sys

from openpyxl import Workbook
from openpyxl.styles import Font

import est_common as ec
import pipeline as pl


def main() -> int:
    D = ec.load_data()
    R = pickle.loads((ec.RESULTS / "combined_rows.pkl").read_bytes())
    C = pickle.loads((ec.RESULTS / "fix4_cands.pkl").read_bytes())
    base = {r["idx"]: r for r in R["baseline"]}
    comb = {r["idx"]: r for r in R["1+2+3+4"]}
    cols = ["idx", "split", "file", "refs", "ref_bpm (MIK)", "sonara_bpm", "base_cat", "combined_bpm",
            "combined_cat", "fix4_switched", "fix4 candidates (tempo:s:acf)", "inlier_frac",
            "base_F70", "comb_F70", "base_F20", "comb_F20", "base_F10", "comb_F10",
            "base_phase_ms", "comb_phase_ms", "base_DF70", "comb_DF70", "base_downbeat_p",
            "comb_downbeat_p", "mik_grid", "cue_hits base", "cue_hits comb", "cues", "path"]
    wb = Workbook()
    ws = wb.active
    ws.title = "tracks"
    ws.append(cols)
    for c in ws[1]:
        c.font = Font(bold=True)
    for idx, t in sorted(D.items()):
        b, c = base.get(idx, {}), comb.get(idx, {})
        cands = C.get(idx) or []
        bpm_c = pl.run(t, {1, 4}, cands=cands)["bpm"] if t["status"] == "ok" else None
        cs = "; ".join(f"{x['tempo']:.1f}:{x['s']:.2f}:{(x['acf'] or 0):.2f}" for x in cands)
        ws.append([idx, "dev" if ec.is_dev(idx) else "holdout",
                   t["path"].replace("\\", "/").rsplit("/", 1)[-1], t["comp"]["refs"],
                   t["comp"]["ref_bpm"], t.get("bpm"), b.get("cat"), bpm_c, c.get("cat"),
                   bool(c.get("switched")), cs, c.get("inlier_frac"),
                   b.get("F70"), c.get("F70"), b.get("F20"), c.get("F20"), b.get("F10"), c.get("F10"),
                   b.get("phase_ms"), c.get("phase_ms"), b.get("DF70"), c.get("DF70"),
                   b.get("phase_choice"), c.get("phase_choice"), c.get("mik_src"),
                   b.get("cue_hits"), c.get("cue_hits"), c.get("cue_n"), t["path"]])
    ws.freeze_panes = "D2"
    ws.auto_filter.ref = ws.dimensions
    out = ec.RESULTS / "per_track.xlsx"
    wb.save(out)
    print("written", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
