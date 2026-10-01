"""chords_from_beats / chord_descriptors / chord_events (tonal.rs:199-445,
analyze.rs:3245-3277) over a precomputed per-frame HPCP (hpcp_extract.py)."""
from __future__ import annotations

import numpy as np

NOTE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _templates():
    maj = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32)
    mnr = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32)
    T, names = [], []
    for r in range(12):
        T.append(np.roll(maj, r)); names.append(NOTE[r])
        T.append(np.roll(mnr, r)); names.append(NOTE[r] + "m")
    T = np.asarray(T)
    return T / np.sqrt((T * T).sum(axis=1, keepdims=True)), names


TN, NAMES = _templates()


def boundaries(beats, n_frames):
    b = [int(x) for x in beats]
    out = ([0] if b[0] > 0 else []) + b + ([n_frames] if b[-1] < n_frames else [])
    return np.asarray(out, dtype=np.int64)


def chords_from_beats(H, beats):
    n = H.shape[1]
    if len(beats) == 0 or n == 0:
        return [], np.zeros(0, dtype=np.int64)
    bnd = boundaries(beats, n)
    cs = np.concatenate([np.zeros((12, 1), dtype=np.float64), np.cumsum(H.astype(np.float64), axis=1)], axis=1)
    s = np.minimum(bnd[:-1], n)
    e = np.minimum(bnd[1:], n)
    labels = []
    for a, b in zip(s, e):
        if a >= b:
            labels.append("N")
            continue
        avg = (cs[:, b] - cs[:, a]) / (b - a)
        nrm = np.sqrt((avg * avg).sum())
        if nrm < 1e-10:
            labels.append("C")
            continue
        corr = TN @ (avg / nrm)
        labels.append(NAMES[int(np.argmax(corr))])
    return labels, bnd


def change_rate(labels, duration):
    if not labels or duration <= 0:
        return 0.0
    return sum(1 for a, b in zip(labels[:-1], labels[1:]) if a != b) / duration


def events(labels, bnd, duration, sr=22050, hop=512):
    ev = []
    for i, lab in enumerate(labels):
        st = 0.0 if i == 0 else bnd[i] * hop / sr
        en = bnd[i + 1] * hop / sr
        if ev and ev[-1][0] == lab:
            ev[-1][2] = en
        else:
            ev.append([lab, st, en])
    if ev:
        ev[-1][2] = duration
    return ev
