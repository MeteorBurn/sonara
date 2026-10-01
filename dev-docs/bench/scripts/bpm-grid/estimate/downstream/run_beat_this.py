"""Run Beat This! final0 on the 145 labelled tracks by importing the parent
build's beat_this_bpm.py unchanged (ffmpeg CLI -> mono 22050 Hz, minimal
post-processing). Output: downstream/data/beat_this/ (one JSON per track)."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILD = HERE.parents[1]
sys.path.insert(0, str(BUILD))
import beat_this_bpm  # noqa: E402
import bpm_grid_paths as P  # noqa: E402

sys.argv = ["beat_this_bpm.py", "--playlist", str(P.DOWNSTREAM / "labelled_145.txt"),
            "--out-dir", str(P.DOWNSTREAM / "data" / "beat_this"), "--model", "final0",
            "--device", "cuda", "--decoders", "2"]
sys.exit(beat_this_bpm.main())
