"""Numpy port of sonara/src/rhythmic_regularity.rs (459bd3c) plus the small
rhythm-derived scalars that feed it or consume beats:

- rr_analyze          rhythmic_regularity::analyze (score, label, confidence,
                      components, windows) - f64 internally like the Rust code;
- bpm_confidence      analyze.rs:2239-2252;
- grid_stability      beatgrid.rs::grid_stability (median/MAD of IBIs);
- grid_regularity     similarity.rs::grid_regularity == analyze.rs
                      aggression_grid_regularity (1 - CV of IBIs);
- danceability        perceptual.rs::danceability_heuristic;
- tempo_curve / tempo_variability   beat.rs;
- fold_bpm_embed      similarity.rs::fold_bpm (also aggression.rs::fold_bpm).

Validated against stored Sonara 0.3.7 output by validate_ports.py.
"""

from __future__ import annotations

import math

import numpy as np

F32 = np.float32
TATUMS = 4
WINDOW_BARS = 4


def _round_half_away(x):
    x = np.asarray(x, dtype=np.float64)
    return np.where(x >= 0, np.floor(x + 0.5), np.ceil(x - 0.5))


def metrical_strides(bpb, tatums=TATUMS):
    strides = []
    for g in range(bpb, 1, -1):
        if bpb % g == 0:
            strides.append(tatums * g)
    s = tatums
    while True:
        strides.append(s)
        if s == 1:
            break
        if s % 2 != 0:
            strides.append(1)
            break
        s //= 2
    return strides


def metrical_weights(bpb, tatums=TATUMS):
    slots = bpb * tatums
    st = metrical_strides(bpb, tatums)
    raw = np.array([sum(1 for d in st if slot % d == 0) for slot in range(slots)], dtype=float)
    span = raw.max() - raw.min()
    if span <= 0:
        return np.zeros(slots)
    return (raw - raw.min()) / span


def _window_max_table(bands, kmax):
    """M[k][:, f] = max(bands[:, f:f+k]) for k = 1..kmax (k=0 unused)."""
    nb, n = bands.shape
    tab = [None, bands.astype(np.float64)]
    cur = tab[1]
    for k in range(2, kmax + 1):
        nxt = np.full((nb, n), -np.inf)
        nxt[:, : n - k + 1] = np.maximum(cur[:, : n - k + 1], bands[:, k - 1:].astype(np.float64))
        # windows running past the end are clipped by the caller (end <= n)
        tab.append(nxt)
        cur = nxt
    return tab


def _accents(bands, beats, phase, bpb, bars_total):
    """accent matrix [band, bar, slot] exactly as rhythmic_regularity::analyze."""
    nb, n = bands.shape
    slots = bpb * TATUMS
    beats = np.asarray(beats, dtype=np.float64)
    idx = phase + np.arange(bars_total)[:, None] * bpb + np.arange(bpb)[None, :]   # (bars, bpb)
    start = beats[idx]
    span = beats[idx + 1] - start
    half = span / (2.0 * TATUMS)
    t = np.arange(TATUMS)
    centre = start[..., None] + span[..., None] * t / TATUMS                       # (bars, bpb, 4)
    lo = centre - half[..., None]
    hi = centre + half[..., None]
    first = np.minimum(np.maximum(_round_half_away(lo), 0.0), n - 1).astype(np.int64)
    end = np.maximum(_round_half_away(hi), 0.0)
    end = np.minimum(np.maximum(end, first + 1), n).astype(np.int64)
    width = (end - first).reshape(bars_total, slots)
    first = first.reshape(bars_total, slots)
    kmax = int(width.max())
    tab = _window_max_table(bands, kmax)
    out = np.zeros((nb, bars_total, slots))
    for k in np.unique(width):
        m = width == k
        out[:, m] = tab[k][:, first[m]]
    # cell_peak folds with an initial 0.0
    return np.maximum(out, 0.0)


def _alignment(contrast, mass, w, uniform, lattice):
    span = lattice - uniform
    if mass <= 0 or span <= 0:
        return 0.0
    weighted = float((contrast * w).sum()) / mass
    return min(1.0, max(0.0, (weighted - uniform) / span))


def _periodicity(contrast, sub_bar):
    slots = len(contrast)
    if slots < 2 or sub_bar == 0 or sub_bar >= slots or slots % sub_bar != 0:
        return 0.0
    step = slots // sub_bar
    X = np.fft.fft(contrast)
    e = np.abs(X[1:]) ** 2
    ac = e.sum()
    rec = e[(np.arange(1, slots) % step) == 0].sum()
    if ac <= 0:
        return 0.0
    null = (sub_bar - 1) / (slots - 1)
    if null >= 1:
        return 0.0
    return min(1.0, max(0.0, (rec / ac - null) / (1 - null)))


def _reliability(bars):
    n_bars, slots = bars.shape
    if n_bars < 2 or slots < 2:
        return 0.0
    prof = bars.mean(axis=0)
    between = prof.var()
    within = bars.var(axis=0).mean()
    noise = within / n_bars
    signal = between - noise
    if signal <= 0:
        return 0.0
    return min(1.0, max(0.0, signal / (signal + noise)))


def _concentration(profile, bpb):
    mass = profile.sum()
    if mass <= 0:
        return 0.0
    on = sum(profile[b * TATUMS] for b in range(bpb) if b * TATUMS < len(profile))
    null = 1.0 / TATUMS
    return min(1.0, max(0.0, (on / mass - null) / (1 - null)))


def _coverage(profile, bpb):
    if bpb < 2:
        return 0.0
    on = np.array([profile[b * TATUMS] if b * TATUMS < len(profile) else 0.0 for b in range(bpb)])
    tot = on.sum()
    if tot <= 0:
        return 0.0
    sh = on / tot
    sh = sh[sh > 0]
    eff = math.exp(float(-(sh * np.log(sh)).sum()))
    return min(1.0, max(0.0, (eff - 1) / (bpb - 1)))


def _wmedian(vals, weights):
    order = np.argsort(np.asarray(vals), kind="stable")
    v = np.asarray(vals)[order]
    w = np.asarray(weights)[order]
    tot = w.sum()
    if tot <= 0:
        return float(v[len(v) // 2])
    cum = np.cumsum(w)
    i = int(np.argmax(cum * 2.0 >= tot))
    if not (cum[i] * 2.0 >= tot):
        return float(v[-1])
    return float(v[i])


def _cos(a, b):
    na, nb = np.sqrt((a * a).sum()), np.sqrt((b * b).sum())
    if na <= 0 or nb <= 0:
        return 0.0
    return min(1.0, max(0.0, float((a * b).sum() / (na * nb))))


def _geomean(f):
    if not f or any(x <= 0 for x in f):
        return 0.0
    return min(1.0, max(0.0, math.exp(sum(math.log(x) for x in f) / len(f))))


def _abstain(conf, bpb, wt):
    return {"regularity": None, "label": None, "confidence": float(conf), "concentration": None,
            "coverage": None, "alignment": None, "periodicity": None, "stability": 0.0,
            "windows_total": wt, "windows_valid": 0, "bpb": bpb}


def rr_analyze(bands, beats, downbeats, bpb=4, bpm_conf=0.0, grid_stab=0.0, ts_conf=None):
    bands = np.asarray(bands, dtype=np.float32)
    beats = [int(b) for b in beats]
    meter = max(1, bpb)
    slots = meter * TATUMS
    nb, n = bands.shape if bands.ndim == 2 else (0, 0)
    if nb == 0 or n == 0 or len(beats) < 2:
        return _abstain(0.0, meter, 0)
    phase = 0
    if len(downbeats):
        try:
            phase = beats.index(int(downbeats[0]))
        except ValueError:
            phase = 0
    usable = len(beats) - 1
    if phase >= usable:
        return _abstain(0.0, meter, 0)
    bars_total = (usable - phase) // meter
    if bars_total == 0:
        return _abstain(0.0, meter, 0)
    acc = _accents(bands, beats, phase, meter, bars_total)
    w = metrical_weights(meter)
    uniform = w.mean()
    lattice = np.mean([w[b * TATUMS] for b in range(meter)])
    st = metrical_strides(meter)
    sub_bar = st[1] if len(st) > 1 else 0
    windows_total = -(-bars_total // WINDOW_BARS)
    meas = []
    for s in range(0, bars_total, WINDOW_BARS):
        e = min(s + WINDOW_BARS, bars_total)
        if e - s < 2:
            continue
        tot = al = pe = re = 0.0
        kick = None
        pooled = np.zeros(slots)
        for b in range(nb):
            bars = acc[b, s:e]
            prof = bars.mean(axis=0)
            con = np.maximum(prof - prof.min(), 0.0)
            mass = float(con.sum())
            if mass <= 0:
                continue
            if b == 0:
                kick = con.copy()
            pooled += con
            tot += mass
            al += mass * _alignment(con, mass, w, uniform, lattice)
            pe += mass * _periodicity(con, sub_bar)
            re += mass * _reliability(bars)
        if tot <= 0:
            continue
        rel = re / tot
        if rel <= 0.5:
            continue
        k = kick if kick is not None else pooled.copy()
        a, p = al / tot, pe / tot
        meas.append({"reg": min(1.0, max(0.0, 0.5 * (a + p))), "conc": _concentration(k, meter),
                     "cov": _coverage(k, meter), "al": a, "pe": p, "rel": rel,
                     "w": tot * rel, "shape": pooled})
    if not meas:
        return _abstain(0.0, meter, windows_total)
    W = np.array([m["w"] for m in meas])
    cons = sum(m["w"] * m["shape"] for m in meas)
    stab = 0.0 if len(meas) < 2 else float(np.mean([_cos(m["shape"], cons) for m in meas]))
    rel = float(np.mean([m["rel"] for m in meas]))

    def agg(key):
        return _wmedian([m[key] for m in meas], W)

    reg = float(F32(min(1.0, max(0.0, agg("reg")))))
    f = [min(1.0, max(0.0, float(bpm_conf))), min(1.0, max(0.0, float(grid_stab))),
         1.0 - 1.0 / math.sqrt(len(meas)), min(1.0, max(0.0, rel)), min(1.0, max(0.0, stab))]
    if ts_conf is not None:
        f.append(min(1.0, max(0.0, float(ts_conf))))
    return {"regularity": reg, "label": "regular" if reg >= 0.5 else "irregular",
            "confidence": float(F32(_geomean(f))), "concentration": agg("conc"), "coverage": agg("cov"),
            "alignment": agg("al"), "periodicity": agg("pe"), "stability": stab,
            "windows_total": windows_total, "windows_valid": len(meas), "bpb": meter,
            "reliability": rel}


# ---------------------------------------------------------------------------
# Small rhythm-derived scalars (f32 like Sonara)
# ---------------------------------------------------------------------------
def bpm_confidence(bpm, n_beats, duration, onset_density, s1):
    bpm, duration = F32(bpm), F32(duration)
    strength = F32(s1) / (F32(s1) + F32(1.2))
    bpm_beats = F32(60.0) * F32(n_beats) / duration if duration > 0 else F32(0.0)
    if bpm > 0 and bpm_beats > 0:
        d = abs(F32(np.log2(F32(bpm / bpm_beats))))
        d = min(d, abs(d - F32(1.0)))
        agree = F32(np.exp(-d / F32(0.10)))
    else:
        agree = F32(0.0)
    dens = F32(min(1.0, max(0.0, F32(onset_density) / F32(4.0))))
    return float(F32(min(1.0, max(0.0, F32(0.5) * strength + F32(0.35) * agree + F32(0.15) * dens))))


def grid_stability(beats):
    b = np.asarray(beats, dtype=np.float32)
    if len(b) < 3:
        return 0.0
    ibi = np.diff(b)
    m = F32(np.median(ibi))
    if m <= 0:
        return 0.0
    mad = F32(np.median(np.abs(ibi - m)))
    return float(F32(min(1.0, max(0.0, F32(1.0) - mad / m))))


def grid_regularity(beats):
    b = np.asarray(beats, dtype=np.int64)
    if len(b) < 3:
        return 0.0
    iv = np.diff(b).astype(np.float32)
    mean = F32(iv.sum(dtype=np.float32) / F32(len(iv)))
    if mean <= 0:
        return 0.0
    var = F32(((iv - mean) ** 2).sum(dtype=np.float32) / F32(len(iv)))
    return float(F32(min(1.0, max(0.0, F32(1.0) - F32(np.sqrt(var)) / mean))))


def danceability(bpm, beats, onset_density):
    beat_reg = grid_regularity(beats) if len(beats) >= 3 else 0.0
    # danceability uses 1 - clamp(cv, 0, 1): identical to grid_regularity's clamp
    tempo = math.exp(-0.5 * ((float(F32(bpm)) - 120.0) / 30.0) ** 2)
    onset = math.exp(-0.5 * ((float(F32(onset_density)) - 4.0) / 2.0) ** 2)
    raw = 0.4 * beat_reg + 0.35 * tempo + 0.25 * onset
    return float(F32(min(1.0, max(0.0, 1.0 / (1.0 + math.exp(-12.3 * (raw - 0.804)))))))


def tempo_curve(beats, sr=22050, hop=512, smooth=5):
    b = np.asarray(beats, dtype=np.float64)
    if len(b) < 2:
        return np.zeros(0)
    dt = np.diff(b) * hop / sr
    bpms = np.where(dt > 0, 60.0 / np.where(dt > 0, dt, 1.0), 0.0)
    if smooth and smooth >= 3 and len(bpms) >= smooth:
        h = smooth // 2
        out = bpms.copy()
        from numpy.lib.stride_tricks import sliding_window_view
        out[h:len(bpms) - h] = np.median(sliding_window_view(bpms, smooth), axis=1)
        bpms = out
    return bpms


def tempo_variability(tc):
    tc = np.asarray(tc, dtype=np.float64)
    if len(tc) == 0 or tc.mean() <= 0:
        return 0.0
    return float(tc.std() / tc.mean())


def fold_bpm_embed(bpm):
    if bpm is None or not np.isfinite(bpm) or bpm <= 0:
        return 0.0
    b = float(bpm)
    while b < 60:
        b *= 2
    while b >= 180:
        b /= 2
    return math.log2(b / 60.0) / math.log2(3.0)
