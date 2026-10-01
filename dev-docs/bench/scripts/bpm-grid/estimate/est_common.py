"""Shared code for the offline counterfactual estimate (read-only on all inputs).

- data loading (cache/data.pkl from build_cache.py, cache/mik_tags.json);
- the dev/holdout split (dev = 0-based playlist_idx even, holdout = odd);
- metrics (BPM categories, beat/downbeat F-measure at any tolerance, phase);
- a numpy port of Sonara 0.3.7's tempo estimation, DP beat tracker and
  accent downbeat picker (sonara/src/beat.rs, beatgrid.rs at 459bd3c), fed by
  the broadband onset envelope reconstructed exactly from the stored
  onset_strength_bands (the bands partition the same 128 log-mel rows);
- bootstrap helpers.
"""

from __future__ import annotations

import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BUILD = HERE.parent
sys.path.insert(0, str(BUILD))
import bpm_grid_paths as P  # noqa: E402

CACHE = P.ESTIMATE / "cache"
RESULTS = P.ESTIMATE / "results"
SYNTHETIC = P.ESTIMATE / "synthetic"
SR, HOP = 22050, 512
HOP_SEC = HOP / SR
F32 = np.float32


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_data():
    with (CACHE / "data.pkl").open("rb") as f:
        D = pickle.load(f)
    tags = {int(k): v for k, v in json.loads(
        (CACHE / "mik_tags.json").read_text(encoding="utf-8")).items()}
    for idx, t in D.items():
        t["tag"] = tags.get(idx, {})
    return D


def is_dev(idx: int) -> bool:
    """idx is 1-based; dev = 0-based playlist_idx even."""
    return (idx - 1) % 2 == 0


def frames_to_sec(frames) -> np.ndarray:
    return np.asarray(frames, dtype=float) * HOP_SEC


def fold(bpm, lo=79.0, hi=192.0):
    if bpm is None or not np.isfinite(bpm) or bpm <= 0:
        return bpm
    while bpm < lo:
        bpm *= 2.0
    while bpm > hi:
        bpm /= 2.0
    return bpm


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
MULTIPLES = [(2.0, "x2"), (0.5, "x0.5"), (1.5, "x1.5"), (2 / 3, "x2/3"),
             (4 / 3, "x4/3"), (0.75, "x3/4"), (0.8, "x4/5"), (1.25, "x5/4")]
CAT_ORDER = ["exact", "fine", "minor"] + [m[1] for m in MULTIPLES] + ["major", "n/a"]


def category(value, ref):
    """exact<=0.05 | fine<=0.5 | minor<=2 | multiples (+-2% ratio) | major."""
    if value is None or ref is None:
        return "n/a"
    d = abs(value - ref)
    if d <= 0.05:
        return "exact"
    if d <= 0.5:
        return "fine"
    if d <= 2.0:
        return "minor"
    ratio = value / ref
    for m, label in MULTIPLES:
        if abs(ratio / m - 1.0) <= 0.02:
            return label
    return "major"


def is_critical(cat):
    return cat not in ("exact", "fine", "minor", "n/a")


def match(est, ref, tol, skip=5.0):
    """Greedy one-to-one matching as compare_bpm.f_measure (first `skip` s ignored).

    Returns (hits, n_est, n_ref, offsets est-ref in s)."""
    est = np.asarray(est, dtype=float)
    ref = np.asarray(ref, dtype=float)
    est = est[est >= skip]
    ref = ref[ref >= skip]
    if len(est) == 0 or len(ref) == 0:
        return 0, len(est), len(ref), np.zeros(0)
    used = np.zeros(len(est), dtype=bool)
    offs = []
    pos = np.searchsorted(est, ref)
    for r, i in zip(ref, pos):
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(est) and not used[j] and abs(est[j] - r) <= tol:
                if best is None or abs(est[j] - r) < abs(est[best] - r):
                    best = j
        if best is not None:
            used[best] = True
            offs.append(est[best] - r)
    return len(offs), len(est), len(ref), np.asarray(offs)


def f_measure(est, ref, tol=0.07, skip=5.0):
    hits, ne, nr, offs = match(est, ref, tol, skip)
    if ne == 0 or nr == 0:
        return None
    if hits == 0:
        return 0.0
    p, r = hits / ne, hits / nr
    return 2 * p * r / (p + r)


def phase_ms(est, ref, tol=0.07, skip=5.0):
    _, _, _, offs = match(est, ref, tol, skip)
    return float(np.median(offs) * 1000.0) if len(offs) else None


def cue_hits(downbeats_sec, cues, tol=0.07, skip=5.0):
    """(# cues within tol of a downbeat, # cues considered)."""
    d = np.asarray(downbeats_sec, dtype=float)
    c = [x for x in cues if x >= skip]
    if len(d) == 0 or not c:
        return 0, len(c)
    return int(sum(np.min(np.abs(d - x)) <= tol for x in c)), len(c)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
def boot_rate(x, B=4000, seed=0):
    """Mean of a 0/1 (or real) per-track vector with a percentile 95% CI."""
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return {"n": 0, "mean": None, "lo": None, "hi": None}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(B, len(x)))
    m = x[idx].mean(axis=1)
    return {"n": int(len(x)), "mean": float(x.mean()), "lo": float(np.percentile(m, 2.5)),
            "hi": float(np.percentile(m, 97.5))}


def boot_diff(a, b, B=4000, seed=0):
    """Paired difference mean(b) - mean(a) over the same tracks, 95% CI."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(B, len(a)))
    d = b[idx].mean(axis=1) - a[idx].mean(axis=1)
    return {"n": int(len(a)), "diff": float(b.mean() - a.mean()),
            "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5))}


def boot_median(x, B=4000, seed=0):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return {"n": 0, "median": None, "lo": None, "hi": None}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(B, len(x)))
    m = np.median(x[idx], axis=1)
    return {"n": int(len(x)), "median": float(np.median(x)),
            "lo": float(np.percentile(m, 2.5)), "hi": float(np.percentile(m, 97.5))}


# ---------------------------------------------------------------------------
# Sonara port (beat.rs / beatgrid.rs / onset.rs at 459bd3c), float32 where Sonara is
# ---------------------------------------------------------------------------
_MEL_LOG_STEP = F32(0.06875177747337722)


def _hz_to_mel(f):
    f = F32(f)
    if f < 1000.0:
        return F32(f / F32(200.0 / 3.0))
    return F32(F32(15.0) + F32(np.log(F32(f / F32(1000.0)))) / _MEL_LOG_STEP)


def _mel_to_hz(m):
    m = F32(m)
    if m < 15.0:
        return F32(m * F32(200.0 / 3.0))
    return F32(F32(1000.0) * F32(np.exp(F32((m - F32(15.0)) * _MEL_LOG_STEP))))


def mel_centers(sr=SR, n_mels=128):
    nyq = F32(sr / 2.0)
    lo, hi = _hz_to_mel(0.0), _hz_to_mel(nyq)
    pts = [_mel_to_hz(lo + (hi - lo) * F32(i) / F32(n_mels + 1)) for i in range(n_mels + 2)]
    return np.asarray(pts[1:n_mels + 1], dtype=np.float32)


def band_widths(edges, sr=SR, n_mels=128):
    """Number of mel rows per onset band, as onset_strength_bands_from_log_mel."""
    c = mel_centers(sr, n_mels)
    out = []
    for b in range(len(edges) - 1):
        lo, hi = edges[b], edges[b + 1]
        start = next((i for i, f in enumerate(c) if f >= lo), n_mels)
        if b + 2 == len(edges):
            end = next((i for i, f in enumerate(c) if f > hi), n_mels)
        else:
            end = next((i for i, f in enumerate(c) if f >= hi), n_mels)
        out.append(end - start)
    return np.asarray(out)


def broadband(bands, edges, sr=SR):
    """Reconstruct oenv_padded (mean positive log-mel flux over all 128 rows)."""
    w = band_widths(edges, sr).astype(np.float32)
    assert w.sum() == 128, w
    return ((bands.astype(np.float32) * w[:, None]).sum(axis=0) / F32(128.0)).astype(np.float32)


def autocorrelate(y, max_size):
    n = len(y)
    nfft = 1 << int(math.ceil(math.log2(2 * n)))
    X = np.fft.rfft(np.asarray(y, dtype=np.float64), nfft)
    ac = np.fft.irfft(np.abs(X) ** 2, nfft)[: min(max_size, n)] / nfft
    # Sonara's irfft is normalised and it divides by fft_size once more (verified:
    # stored bpm_candidates scores = numpy irfft / fft_size).
    return ac.astype(np.float32)


def acf_full(oenv, sr=SR, hop=HOP):
    fr = F32(sr) / F32(hop)
    max_lag = int(min(F32(4.0) * fr, F32(len(oenv))))
    return autocorrelate(oenv, max_lag)


def tempo_candidates(acf, start_bpm=120.0, sr=SR, hop=HOP):
    """All (lag, bpm, score) as estimate_tempo builds them."""
    fr = F32(sr) / F32(hop)
    min_lag = int(math.ceil(F32(60.0) * fr / F32(300.0)))
    max_lag = min(int(math.floor(F32(60.0) * fr / F32(30.0))), len(acf) - 1)
    out = []
    for lag in range(min_lag, max_lag + 1):
        bpm = F32(F32(60.0) * fr / F32(lag))
        lp = F32(-0.5) * F32((F32(np.log2(bpm)) - F32(np.log2(F32(start_bpm)))) / F32(1.0)) ** 2
        out.append((lag, float(bpm), float(F32(acf[lag] * (F32(1.0) + F32(np.exp(lp)))))))
    return out


def _best_supported(cands, best, lo, hi, mmin, mmax, ratio):
    pool = [c for c in cands if lo <= c[1] <= hi and mmin <= c[1] / best[1] <= mmax
            and c[2] >= best[2] * ratio]
    if not pool:
        return None
    m = max(c[2] for c in pool)
    return [c for c in pool if c[2] == m][-1]   # Rust max_by keeps the last maximum


def select_preferred(cands):
    m = max(c[2] for c in cands)
    best = [c for c in cands if c[2] == m][-1]  # Rust max_by returns the last maximum
    if best[1] < 75.0:
        c = _best_supported(cands, best, 115.0, 150.0, 1.80, 2.20, 0.50)
        if c:
            return c
    elif best[1] < 90.0:
        c = _best_supported(cands, best, 120.0, 145.0, 1.42, 1.62, 0.75)
        if c:
            return c
    elif best[1] < 95.0 and best[2] >= 4.0:
        c = _best_supported(cands, best, 120.0, 145.0, 1.42, 1.62, 0.85)
        if c:
            return c
    return best


def refine_lag(acf, lag):
    """Fractional lag of the ACF peak (parabolic), as refine_tempo_from_acf_peak."""
    if lag <= 0 or lag + 1 >= len(acf):
        return float(lag)
    l, c, r = (float(F32(acf[lag - 1])), float(F32(acf[lag])), float(F32(acf[lag + 1])))
    if c < l or c < r:
        return float(lag)
    den = l - 2 * c + r
    if abs(den) <= 1e-12:
        return float(lag)
    off = min(0.5, max(-0.5, 0.5 * (l - r) / den))
    return lag + off


def lag_to_bpm(lag, sr=SR, hop=HOP):
    return 60.0 * (sr / hop) / lag


def estimate_tempo(oenv, bpm_min=79.0, bpm_max=192.0, start_bpm=120.0):
    acf = acf_full(oenv)
    cands = tempo_candidates(acf, start_bpm)
    lag, _, _ = select_preferred(cands)
    raw = lag_to_bpm(refine_lag(acf, lag))
    raw = min(320.0, max(30.0, raw))
    ranked = sorted(cands, key=lambda c: -c[2])[:5]
    return {"tempo": min(320.0, max(30.0, fold(raw, bpm_min, bpm_max))), "tempo_raw": raw,
            "lag": lag, "acf": acf, "cands": cands, "top5": [(c[1], c[2]) for c in ranked]}


def local_score(oenv, fpb):
    o = oenv.astype(np.float32)
    mean = F32(o.sum(dtype=np.float32) / F32(len(o)))
    std = F32(np.sqrt(F32(((o - mean) ** 2).sum(dtype=np.float32) / F32(len(o)))))
    on = (o - mean) / (std + F32(1e-10)) if std > 0 else o
    half = fpb
    w = np.exp(-0.5 * (((np.arange(2 * half + 1, dtype=np.float32) - F32(half)) * F32(32.0)
                         / F32(fpb)) ** 2)).astype(np.float32)
    return (np.convolve(on, w, mode="same") / w.sum(dtype=np.float32)).astype(np.float32)


def beat_track_dp(ls, fpb, tightness=100.0):
    """Ellis DP exactly as beat_track_dp (vectorised in blocks of fpb//2 + 1 frames)."""
    n = len(ls)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    ls = ls.astype(np.float32)
    half = fpb // 2
    dmin, dmax = half + 1, 2 * fpb
    lnt = np.zeros(2 * fpb + 2, dtype=np.float32)
    lnt[1:] = np.log(np.arange(1, 2 * fpb + 2, dtype=np.float32))
    log_fpb = F32(np.log(F32(fpb)))
    d = np.arange(dmax, dmin - 1, -1)                      # ascending j <=> descending d
    pen = (F32(tightness) * (lnt[d] - log_fpb) ** 2).astype(np.float32)
    cum = np.zeros(n, dtype=np.float32)
    back = np.zeros(n, dtype=np.int64)
    i0 = 0
    while i0 < n:
        i1 = min(n, i0 + half + 1)
        ii = np.arange(i0, i1)
        J = ii[:, None] - d[None, :]                       # (block, nd) in ascending j
        valid = J >= 0
        S = np.where(valid, cum[np.clip(J, 0, None)] - pen[None, :], -np.inf).astype(np.float32)
        has = valid.any(axis=1)
        k = np.argmax(S, axis=1)
        best = S[np.arange(len(ii)), k]
        cum[ii] = np.where(has, ls[ii] + best, ls[ii]).astype(np.float32)
        back[ii] = np.where(has, J[np.arange(len(ii)), k], ii)
        i0 = i1
    tail = max(0, n - 2 * fpb)
    last = tail + int(np.argmax(cum[tail:]))
    beats = [last]
    cur = last
    while back[cur] != cur and back[cur] < cur:
        cur = back[cur]
        beats.append(cur)
    return np.asarray(beats[::-1], dtype=np.int64)


def trim_beats(ls, beats):
    if len(beats) == 0:
        return beats
    thr = F32(0.01) * max(F32(0.0), ls.max())
    ok = np.nonzero(ls[beats] >= thr)[0]
    if len(ok) == 0:
        return beats
    return beats[ok[0]: ok[-1] + 1]


def track_beats(oenv, tempo, tightness=100.0):
    fpb = int(math.floor(float(F32(60.0) * F32(SR / HOP) / F32(tempo)) + 0.5))  # f32::round
    if fpb == 0:
        return np.zeros(0, dtype=np.int64), fpb
    ls = local_score(oenv, fpb)
    return trim_beats(ls, beat_track_dp(ls, fpb, tightness)), fpb


def accent_downbeat_phase(beats, env, bpb=4, win=2):
    """detect_downbeats: phase with the highest mean peak accent (+-win frames)."""
    n = len(env)
    acc = np.array([env[max(0, b - win): min(n, b + win + 1)].max(initial=0.0)
                    if max(0, b - win) < min(n, b + win + 1) else 0.0 for b in beats])
    best, best_s = 0, -np.inf
    for p in range(min(bpb, len(beats))):
        s = acc[p::bpb].mean()
        if s > best_s:
            best, best_s = p, s
    return best


def sonara_pipeline(oenv, bpm_min=79.0, bpm_max=192.0):
    est = estimate_tempo(oenv, bpm_min, bpm_max)
    beats, fpb = track_beats(oenv, est["tempo"])
    p = accent_downbeat_phase(beats, oenv) if len(beats) else 0
    return est, beats, beats[p::4], fpb
