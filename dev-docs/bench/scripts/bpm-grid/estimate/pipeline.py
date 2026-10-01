"""The simulated fixes, shared by the real-data and synthetic evaluations.

Parameters are fixed here exactly as chosen on the dev split (fix 1, 3, 4) or
from synthetic ground truth (fix 2); see fix1_bpm.py, fix3_downbeat.py,
fix4_tempo.py, synth_latency.py and results/*.json.

A track dict needs: bpm, cands (stored bpm_candidates), beats (frames),
downbeats (frames), onset_frames, onset_strength_bands, band_edges.
"""

from __future__ import annotations

import json

import numpy as np

import est_common as ec
from fix1_bpm import consensus, unwrap
from fix3_downbeat import choose as choose_phase, features as phase_features
from fix4_tempo import candidates_for, pick

LATENCY_S = json.loads((ec.RESULTS / "latency.json").read_text(encoding="utf-8"))["beat_latency_ms"] / 1000.0
FIX1 = dict(thr=ec.HOP_SEC, sep=128, stride=2, refits=2)     # chosen on dev
FIX3 = "two_stage_low"                                        # chosen on dev
FIX4 = dict(rule="s", delta=0.2, rho=0.9)                     # chosen on dev


def fix1_bpm(beat_frames, fallback):
    t = ec.frames_to_sec(beat_frames)
    if len(t) < 16:
        return fallback, None, None
    k = unwrap(t).astype(float)
    b, a, frac = consensus(k, t, **FIX1)
    return ec.fold(60.0 / b), (a, b, k), frac


def fix3_phase(t, beat_frames):
    tt = dict(t)
    tt["beats"] = np.asarray(beat_frames)
    if len(beat_frames) < 8:
        return 0
    _, f = phase_features(tt)
    return choose_phase(FIX3, f)


def fix4_retrack(t, cands=None):
    """Returns (tempo, beat_frames, switched)."""
    if cands is None:
        oenv = ec.broadband(t["onset_strength_bands"], t["band_edges"])
        est = ec.estimate_tempo(oenv)
        cands = candidates_for(t, oenv, est)
    ch = pick(cands, FIX4["rule"], FIX4["delta"], FIX4["rho"])
    return ch["tempo"], ch["beats"], ch is not cands[0]


def accent_phase(t, beat_frames):
    oenv = ec.broadband(t["onset_strength_bands"], t["band_edges"])
    return ec.accent_downbeat_phase(np.asarray(beat_frames), oenv) if len(beat_frames) else 0


def run(t, fixes, cands=None, snap=False):
    """Apply a set of fixes {1,2,3,4} to one track; returns seconds-domain output."""
    beats = np.asarray(t["beats"], dtype=np.int64)
    bpm = t["bpm"]
    switched = False
    if 4 in fixes:
        tempo, nb, switched = fix4_retrack(t, cands)
        if switched:
            beats, bpm = np.asarray(nb, dtype=np.int64), tempo
    if 3 in fixes:
        p = fix3_phase(t, beats)
    elif switched:
        p = accent_phase(t, beats)                   # Sonara's own rule on the new beats
    else:
        stored = np.asarray(t["downbeats"])
        p = int(np.searchsorted(beats, stored[0])) if len(stored) else 0
    frac = None
    line = None
    if 1 in fixes:
        bpm, line, frac = fix1_bpm(beats, bpm)
    beats_s = ec.frames_to_sec(beats)
    if snap:                                          # extra: snap inlier beats to the fitted line
        if line is None:
            _, line, frac = fix1_bpm(beats, bpm)
        if line is not None:
            a, b, k = line
            fitted = a + b * k
            m = np.abs(beats_s - fitted) <= ec.HOP_SEC
            beats_s = np.where(m, fitted, beats_s)
    downs_s = beats_s[p::4]
    onsets_s = ec.frames_to_sec(t.get("onset_frames", []))
    if 2 in fixes:
        beats_s = beats_s - LATENCY_S
        downs_s = downs_s - LATENCY_S
        onsets_s = onsets_s - LATENCY_S
    return {"bpm": bpm, "beats": beats_s, "downbeats": downs_s, "onsets": onsets_s,
            "switched": switched, "inlier_frac": frac, "phase": p}
