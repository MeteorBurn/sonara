"""Frame-domain beats / downbeats / BPM for every fix variant, both sets.

Fix parameters are exactly those chosen on the broken-1000 dev split by the
previous estimate (../pipeline.py: FIX1, FIX3, FIX4); nothing is re-tuned here.

Variants (all in the variant's own frame domain; `shift` = frames dropped from
the front of every onset envelope / band, i.e. the onset pad 3 -> 2):
  base      stored Sonara 0.3.7 beats, downbeats, bpm
  f1        bpm = consensus line fit over the stored beats (folded 79-192)
  f2        pad 3 -> 2 modelled as a pure one-frame shift: beats-1, downbeats-1,
            bands[:, 1:] (onset_frames-1). Downstream measures that sample the
            bands at beat positions are exactly invariant to this.
  f2x       pad 3 -> 2 simulated exactly: the numpy port re-runs tempo, DP
            beat tracker and accent downbeats on bands[:, 1:] (edge effects)
  f3        downbeat phase from the low band (two-stage kick-pair rule)
  f4        tempo candidate re-ranking + re-tracking (Sonara's accent rule for
            the bar phase on switched tracks)
  f123      1 + 2 + 3
  f1234     1 + 2 + 3 + 4
Writes cache/variants_{B,L}.pkl and cache/fix4_cands_L.pkl.
"""

from __future__ import annotations

import pickle
import sys
import time

import numpy as np

import ds_data as dd
import est_common as ec
import pipeline as pl
from fix4_tempo import candidates_for

NAMES = ["base", "f1", "f2", "f2x", "f3", "f4", "f123", "f1234"]


def _s1(t):
    return t["cands"][0][1] if t.get("cands") else 0.0


def build_track(t, cands):
    beats0 = np.asarray(t["beats"], dtype=np.int64)
    downs0 = np.asarray(t["downbeats"], dtype=np.int64)
    bpm0 = float(t["bpm"])
    on0 = np.asarray(t.get("onset_frames", []), dtype=np.int64)
    V = {}

    def rec(bpm, beats, downs, shift=0, switched=False, frac=None, s1=None, onsets=None):
        return {"bpm": float(bpm), "beats": np.asarray(beats, dtype=np.int64),
                "downs": np.asarray(downs, dtype=np.int64), "shift": shift, "switched": bool(switched),
                "fit_frac": frac, "s1": _s1(t) if s1 is None else s1,
                "onsets": (on0 - shift) if onsets is None else onsets}

    V["base"] = rec(bpm0, beats0, downs0)
    b1, _, frac = pl.fix1_bpm(beats0, bpm0)
    V["f1"] = rec(b1, beats0, downs0, frac=frac)
    V["f2"] = rec(bpm0, beats0 - 1, downs0 - 1, shift=1)
    # exact pad-2 re-run on the shifted envelope
    oenv2 = ec.broadband(t["onset_strength_bands"][:, 1:], t["band_edges"])
    est2 = ec.estimate_tempo(oenv2)
    bx, _ = ec.track_beats(oenv2, est2["tempo"])
    px = ec.accent_downbeat_phase(bx, oenv2) if len(bx) else 0
    V["f2x"] = rec(est2["tempo"], bx, bx[px::4], shift=1, s1=est2["top5"][0][1])
    p3 = pl.fix3_phase(t, beats0)
    V["f3"] = rec(bpm0, beats0, beats0[p3::4])
    tempo4, nb4, sw = pl.fix4_retrack(t, cands)
    if sw:
        nb4 = np.asarray(nb4, dtype=np.int64)
        p4 = pl.accent_phase(t, nb4)
        V["f4"] = rec(tempo4, nb4, nb4[p4::4], switched=True)
        beats4 = nb4
        bpm4 = tempo4
    else:
        V["f4"] = rec(bpm0, beats0, downs0)
        beats4, bpm4 = beats0, bpm0
    # combined: fix 1 bpm on the (re-tracked) beats, fix 3 phase, then the pad shift
    b123, _, f123 = pl.fix1_bpm(beats0, bpm0)
    V["f123"] = rec(b123, beats0 - 1, beats0[p3::4] - 1, shift=1, frac=f123)
    b1234, _, f1234 = pl.fix1_bpm(beats4, bpm4)
    p34 = pl.fix3_phase(t, beats4)
    V["f1234"] = rec(b1234, beats4 - 1, beats4[p34::4] - 1, shift=1, switched=sw, frac=f1234)
    return V


def run(which):
    if which == "B":
        D = dd.load_broken()
        C = pickle.loads((ec.RESULTS / "fix4_cands.pkl").read_bytes())
    else:
        D = dd.load_labelled()
        cf = dd.CACHE / "fix4_cands_L.pkl"
        if cf.exists():
            C = pickle.loads(cf.read_bytes())
        else:
            C = {}
            for idx, t in sorted(D.items()):
                if t["status"] != "ok":
                    continue
                oenv = ec.broadband(t["onset_strength_bands"], t["band_edges"])
                est = ec.estimate_tempo(oenv)
                # port fidelity on this set: the selected candidate must be Sonara's bpm
                assert abs(est["tempo"] - t["bpm"]) < 1e-3, (idx, est["tempo"], t["bpm"])
                C[idx] = candidates_for(t, oenv, est)
            cf.write_bytes(pickle.dumps(C))
    out, t0 = {}, time.perf_counter()
    for n, (idx, t) in enumerate(sorted(D.items()), 1):
        if t["status"] != "ok" or "onset_strength_bands" not in t:
            continue
        out[idx] = build_track(t, C.get(idx))
        if n % 200 == 0:
            print(f"  {which} {n} {time.perf_counter() - t0:.0f}s", flush=True)
    (dd.CACHE / f"variants_{which}.pkl").write_bytes(pickle.dumps(out, protocol=pickle.HIGHEST_PROTOCOL))
    sw = sum(v["f4"]["switched"] for v in out.values())
    same2 = sum(len(v["f2x"]["beats"]) == len(v["f2"]["beats"]) and np.array_equal(v["f2x"]["beats"], v["f2"]["beats"])
                for v in out.values())
    print(f"{which}: tracks {len(out)}  fix4 switched {sw}  exact pad-2 beats == shifted beats {same2}")


if __name__ == "__main__":
    for w in (sys.argv[1:] or ["L", "B"]):
        run(w)
