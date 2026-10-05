"""Real-music rhythm fixtures for sonara/tests/bpm_accuracy.rs, from dumped envelopes.

    python make_fixtures.py SPEC.tsv OUT_DIR

SPEC.tsv (tab-separated, header): name, envelopes (an `.env` file written by the bench
runner's --dump-envelopes), audio (the source file, for its SHA-256), kind, label_bpm,
tolerance, note. Writes OUT_DIR/<name>.env16 and OUT_DIR/manifest.tsv.

A fixture keeps only what the tempo and beat stage reads: the broadband `onset` and the
below-3.2 kHz `beat` envelope of the analysis pass, quantized to u16 per envelope
(value = q * scale, scale = max / 65535). No audio: an onset envelope cannot be turned
back into the recording. Format, little-endian: b"SNRFIX01", u32 sample rate, u32 hop,
u32 envelope count k, u32 frames n; per envelope u8 name length, name, f32 scale; then
k x n u16 values. The manifest names the expectation the Rust test checks (kind
`integer`: the tempo folded to label_bpm's octave is exactly round(label_bpm);
`fractional`: it is not an integer and lies within tolerance of label_bpm; `not-integer`:
it is not an integer; `tempo`: unfolded, it lies within tolerance of label_bpm, so the
octave counts too).
"""

from __future__ import annotations

import csv
import hashlib
import struct
import sys
from pathlib import Path

KEEP = ("onset", "beat")


def read_env(path: Path) -> tuple[int, int, dict[str, list[float]]]:
    data = path.read_bytes()
    if data[:8] != b"SNRENV01":
        raise SystemExit(f"{path}: not an envelope dump")
    sr, hop, count, frames = struct.unpack_from("<4I", data, 8)
    at, names = 24, []
    for _ in range(count):
        length = data[at]
        names.append(data[at + 1:at + 1 + length].decode("utf-8"))
        at += 1 + length
    out = {}
    for name in names:
        out[name] = list(struct.unpack_from(f"<{frames}f", data, at))
        at += 4 * frames
    return sr, hop, out


def write_fixture(path: Path, sr: int, hop: int, envelopes: dict[str, list[float]]) -> None:
    frames = len(next(iter(envelopes.values())))
    head = bytearray(b"SNRFIX01") + struct.pack("<4I", sr, hop, len(envelopes), frames)
    body = bytearray()
    for name, values in envelopes.items():
        top = max(values) if values else 0.0
        scale = top / 65535.0 if top > 0 else 1.0
        head += bytes([len(name)]) + name.encode("utf-8") + struct.pack("<f", scale)
        body += struct.pack(f"<{frames}H", *(min(65535, max(0, round(v / scale))) for v in values))
    path.write_bytes(bytes(head + body))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    spec, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    with spec.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    manifest = []
    for row in rows:
        sr, hop, env = read_env(Path(row["envelopes"]))
        write_fixture(out_dir / f"{row['name']}.env16", sr, hop, {k: env[k] for k in KEEP})
        manifest.append({"name": row["name"], "kind": row["kind"], "label_bpm": row["label_bpm"],
                         "tolerance": row["tolerance"], "audio_sha256": sha256(Path(row["audio"])),
                         "note": row["note"]})
        print(f"{row['name']}: {(out_dir / (row['name'] + '.env16')).stat().st_size // 1024} KB")
    with (out_dir / "manifest.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
