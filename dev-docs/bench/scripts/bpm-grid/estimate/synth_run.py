"""Run the published Sonara 0.3.7 wheel on estimate/synthetic/*.wav (generated audio only).

Run with: dev-docs/builds/0.3.7-meteorburn.1-9327b16-20260923/smoke-venv/Scripts/python.exe
Writes estimate/synthetic/sonara_out.json.
"""

import json
import sys
from pathlib import Path

import sonara

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bpm_grid_paths as P  # noqa: E402

D = P.ESTIMATE / "synthetic"


def main():
    truth = json.loads((D / "truth.json").read_text(encoding="utf-8"))
    out = {}
    for name in truth:
        r = sonara.analyze_file(str(D / name), sr=22050,
                                features=["bpm", "beats", "beatgrid", "onsets", "onset_bands"],
                                bpm_min=79.0, bpm_max=192.0)
        out[name] = {"bpm": r["bpm"], "bpm_raw": r.get("bpm_raw"), "cands": r["bpm_candidates"],
                     "beats": list(r["beats"]), "downbeats": list(r["downbeats"]),
                     "onset_frames": list(r["onset_frames"]),
                     "onset_strength_bands": [list(map(float, b)) for b in r["onset_strength_bands"]],
                     "band_edges": list(r["onset_band_edges_hz"]),
                     "provenance": {k: r["provenance"].get(k) for k in ("sample_rate", "hop_length")},
                     "sonara_version": sonara.__version__}
    (D / "sonara_out.json").write_text(json.dumps(out), encoding="utf-8")
    print(f"analysed {len(out)} files with sonara {sonara.__version__}")


if __name__ == "__main__":
    main()
