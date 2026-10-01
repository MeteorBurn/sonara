"""Per-frame HPCP of the fused Sonara pass (analyze.rs:1640-1925, 2402-2410),
re-computed in numpy for the 145 labelled tracks so beat-synchronous chord
labels can be re-derived for shifted / re-tracked beats.

Audio is decoded read-only by sonara.load(path, sr=22050) (Sonara 0.3.7 wheel,
smoke-venv). STFT: zero-padded centre (n_fft/2), periodic Hann 2048, hop 512.
Peaks: strict local maxima of |X| in 40..5000 Hz, parabolic interpolation, top
50 by magnitude; HPCP: 4 harmonics (1/h), cosine weight within 0.5 semitone of
round(pc) % 12 (Sonara's wrap quirk kept: pc in (11.5, 12) contributes 0),
c_ref 261.6256 Hz; per-frame L1 normalisation.

Writes downstream/cache/hpcp/<idx>.npz (float32 12 x n_frames).
Run with the smoke-venv python (needs sonara + numpy).
"""

from __future__ import annotations

import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import bpm_grid_paths as P  # noqa: E402

OUT = P.DOWNSTREAM / "cache" / "hpcp"
SR, NFFT, HOP = 22050, 2048, 512
C_REF = np.float32(261.6256)
MAX_PEAKS = 50


def _round_half_away(x):
    return np.where(x >= 0, np.floor(x + 0.5), np.ceil(x - 0.5))


def hpcp_frames(y):
    y = np.asarray(y, dtype=np.float32)
    pad = NFFT // 2
    yp = np.zeros(len(y) + 2 * pad, dtype=np.float32)
    yp[pad:pad + len(y)] = y
    n_frames = 1 + (len(yp) - NFFT) // HOP
    n = np.arange(NFFT)
    win = (0.5 - 0.5 * np.cos(2 * np.pi * n / NFFT)).astype(np.float32)
    freqs = (np.arange(NFFT // 2 + 1) * SR / NFFT).astype(np.float32)
    lo_bin = int(np.searchsorted(freqs, 40.0))
    hi_bin = int(np.searchsorted(freqs, 5000.0, side="right")) - 1
    out = np.zeros((12, n_frames), dtype=np.float32)
    hw = np.array([1.0, 1 / 2, 1 / 3, 1 / 4], dtype=np.float32)
    B = 2048
    for f0 in range(0, n_frames, B):
        f1 = min(n_frames, f0 + B)
        idx = (np.arange(f0, f1) * HOP)[:, None] + n[None, :]
        X = np.fft.rfft(yp[idx] * win[None, :], axis=1)
        mag = np.abs(X).astype(np.float32)                         # (frames, bins)
        i = np.arange(max(1, lo_bin), min(mag.shape[1] - 1, hi_bin + 1))
        a, b, c = mag[:, i - 1], mag[:, i], mag[:, i + 1]
        is_pk = (b > a) & (b > c)
        den = a - np.float32(2) * b + c
        ok = np.abs(den) > 1e-10
        p = np.where(ok, np.float32(0.5) * (a - c) / np.where(ok, den, 1), 0).astype(np.float32)
        binf = i[None, :].astype(np.float32) + p
        lo = np.floor(binf).astype(np.int64)
        frac = (binf - lo).astype(np.float32)
        inside = ok & (binf >= 0) & (lo < mag.shape[1] - 1)
        lo_c = np.clip(lo, 0, mag.shape[1] - 2)
        f_int = freqs[lo_c] * (1 - frac) + freqs[lo_c + 1] * frac
        pf = np.where(inside, f_int, freqs[i][None, :]).astype(np.float32)
        pm = np.where(ok, b - np.float32(0.25) * (a - c) * p, b).astype(np.float32)
        key = np.where(is_pk, pm, -np.inf)
        k = min(MAX_PEAKS, key.shape[1])
        top = np.argpartition(-key, k - 1, axis=1)[:, :k]
        tm = np.take_along_axis(key, top, axis=1)
        tf = np.take_along_axis(pf, top, axis=1)
        valid = np.isfinite(tm)
        tm = np.where(valid, tm, 0).astype(np.float32)
        col = np.zeros((f1 - f0, 12), dtype=np.float32)
        rows = np.repeat(np.arange(f1 - f0)[:, None], k, axis=1)
        for h in range(4):
            fr = tf / np.float32(h + 1)
            use = valid & (fr >= 20.0)
            semis = np.float32(12.0) * np.log2(np.where(use, fr, 1.0) / C_REF).astype(np.float32)
            pc = np.mod(np.mod(semis, 12.0) + 12.0, 12.0).astype(np.float32)
            center = (_round_half_away(pc).astype(np.int64)) % 12
            dist = np.abs(pc - center.astype(np.float32))
            w = np.where(dist < 0.5, np.cos(np.float32(np.pi) * dist), 0).astype(np.float32)
            contrib = np.where(use, hw[h] * tm * tm * w, 0).astype(np.float32)
            np.add.at(col, (rows, center), contrib)
        s = col.sum(axis=1, keepdims=True)
        col = np.where(s > 0, col / np.where(s > 0, s, 1), col)
        out[:, f0:f1] = col.T
    return out


def work(item):
    idx, path = item
    target = OUT / f"{idx:04d}.npz"
    if target.exists():
        return idx, "cached", 0.0
    t0 = time.perf_counter()
    import sonara
    y, sr = sonara.load(path, sr=SR)
    assert sr == SR
    H = hpcp_frames(y)
    np.savez_compressed(target, hpcp=H, n_samples=len(y))
    return idx, "ok", time.perf_counter() - t0


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    items = []
    with (P.DOWNSTREAM / "labelled_145.csv").open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            items.append((int(r["idx"]), r["path"]))
    with ProcessPoolExecutor(max_workers=4) as pool:
        for n, (idx, st, dt) in enumerate(pool.map(work, items), 1):
            print(f"[{n}/{len(items)}] {idx} {st} {dt:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
