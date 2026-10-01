"""Part B - per-track downstream outputs for every fix variant (both sets).

For each track and variant (variants.py) recompute, from the variant's beats /
downbeats / BPM / band shift, everything Sonara derives from them:
  rhythmic_regularity (+ components, confidence) - exact port (rr_port.py);
  bpm_confidence, grid_stability, tempo_curve / tempo_variability;
  danceability, valence, mood_happy/relaxed/sad (aggressive has no tempo term):
    exact formulas (perceptual.rs, mood.rs) - valence and mood as deltas on the
    stored values, since their non-rhythm terms do not change;
  embedding dims 35 (bpm), 37 (danceability), 38 (grid regularity),
    42 (chord change rate; labelled set only) and 47 (valence);
  vocalness P(vocal) from the bundled MLP on the modified embedding;
  aggression rhythm inputs (features 13, 15, 16, 27, aggression_rhythm) and the
    bounded output drift (aggr_model.py);
  labelled set only: beat-synchronous chord labels from the re-computed HPCP,
    chord change rate, chord-event boundaries;
  structure energy curve (fix 2 only - the onsets move one frame).
Plus an EXPLORATORY variant "oracle_phase" (not one of the four fixes): the
1+2+3+4 beats moved by half a beat on tracks whose beats sit on the offbeat
according to Beat This! - an upper bound for a beat-phase fix, uses the
reference, not deployable.
Writes cache/part_b_tracks.pkl.
"""

from __future__ import annotations

import json
import pickle
import sqlite3
import sys
import time
import zlib

import numpy as np

import aggr_model as am
import chords_port as cp
import ds_data as dd
import est_common as ec
import pipeline as pl
import rr_port as rr

VARS = ["base", "f1", "f2", "f2x", "f3", "f4", "f123", "f1234", "oracle_phase"]
SR, HOP = 22050, 512
FPS = SR / HOP
VOCAL = am.Vocal()
AGG = am.Aggression()


def tempo_term(bpm):
    return min(1.0, max(0.0, (bpm - 60.0) / 120.0))


def energy_curves(which):
    db = (dd.P.DATA / "broken_1000.sqlite") if which == "B" else (dd.P.DOWNSTREAM / "data" / "labelled_145.sqlite")
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    out = {}
    for path, pidx in con.execute("SELECT path, playlist_idx FROM tracks"):
        r = con.execute("SELECT dtype, shape, data FROM arrays WHERE path = ? AND name = 'energy_curve'",
                        (path,)).fetchone()
        if r:
            out[pidx + 1] = np.frombuffer(zlib.decompress(r[2]), dtype=r[0]).reshape(json.loads(r[1])).copy()
    con.close()
    return out


def structure_delta(onsets, energy, duration, shift=1):
    """First-order change of the structure energy curve when every onset moves
    `shift` frames earlier (structure.rs: 1 s windows, 0.5 s hop)."""
    n_frames = 1 + int(round(duration * SR)) // HOP
    win = max(1, int(round(1.0 * FPS)))
    hop = max(1, int(round(0.5 * FPS)))
    starts = np.arange(0, max(n_frames, 1), hop)
    if len(starts) != len(energy):
        starts = starts[: len(energy)]
    ends = np.minimum(starts + win, n_frames)
    nf = np.maximum(ends - starts, 1)
    o0 = np.sort(np.asarray(onsets))
    o1 = o0 - shift
    c0 = np.searchsorted(o0, ends, "left") - np.searchsorted(o0, starts, "left")
    c1 = np.searchsorted(o1, ends, "left") - np.searchsorted(o1, starts, "left")
    secs = nf / FPS
    n0 = np.clip(c0 / secs / 10.0, 0, 1)
    n1 = np.clip(c1 / secs / 10.0, 0, 1)
    e = np.asarray(energy[: len(starts)], dtype=float)
    de = 4 * e * (1 - e) * 0.25 * (n1 - n0)
    lvl0 = 1 + np.round(9 * np.clip((e.mean() - 0.25) / 0.35, 0, 1))
    lvl1 = 1 + np.round(9 * np.clip(((e + de).mean() - 0.25) / 0.35, 0, 1))
    return {"windows": int(len(e)), "windows_changed": int((c0 != c1).sum()), "max_abs_de": float(np.abs(de).max(initial=0)),
            "mean_abs_de": float(np.abs(de).mean()) if len(de) else 0.0, "level_changed": bool(lvl0 != lvl1)}


def bt_local_tempo_at(bt_beats, times):
    ibi = np.diff(bt_beats)
    if len(ibi) < 5:
        return np.full(len(times), np.nan)
    from numpy.lib.stride_tricks import sliding_window_view
    sm = ibi.copy()
    sm[2:-2] = np.median(sliding_window_view(ibi, 5), axis=1)
    i = np.clip(np.searchsorted(bt_beats, times) - 1, 0, len(ibi) - 1)
    out = 60.0 / sm[i]
    out[(times < bt_beats[0]) | (times > bt_beats[-1])] = np.nan
    return out


def oracle_variant(t, Vt, prow):
    v = dict(Vt["f1234"])
    if prow is None or prow.get("phase_class") != "half-beat" or len(v["beats"]) < 8:
        v["oracle_moved"] = False
        return v
    b = v["beats"]
    mid = np.round((b[:-1] + b[1:]) / 2.0).astype(np.int64)
    p = pl.fix3_phase(t, mid + 1)          # fix-3 rule on the unshifted bands
    v = dict(v, beats=mid, downs=mid[p::4], oracle_moved=True)
    return v


def compute_track(t, Vt, H, energy, prow, ref_bpm):
    S = t["S"]
    od, dur = S["onset_density"], S["duration_sec"]
    bands = t["onset_strength_bands"]
    emb0 = np.asarray(S["embedding"], dtype=np.float32)
    harsh = S.get("aggression_harshness")
    out = {}
    for name in VARS:
        v = oracle_variant(t, Vt, prow) if name == "oracle_phase" else Vt[name]
        beats, downs, bpm, sh = v["beats"], v["downs"], v["bpm"], v["shift"]
        bc = rr.bpm_confidence(bpm, len(beats), dur, od, v["s1"])
        gs = rr.grid_stability(beats)
        reg = rr.rr_analyze(bands[:, sh:], beats, downs, 4, bc, gs, None)
        tc = rr.tempo_curve(beats)
        dance = rr.danceability(bpm, beats, od)
        greg = rr.grid_regularity(beats)
        dT = tempo_term(bpm) - tempo_term(S["bpm"])
        dd0 = dance - S["danceability"]
        val = S["valence"] + 0.30 * dT
        happy = S["mood_happy"] + 0.25 * dT + 0.20 * dd0
        relaxed = S["mood_relaxed"] - 0.25 * dT
        sad = S["mood_sad"] - 0.25 * dT
        emb = emb0.copy()
        emb[35] = rr.fold_bpm_embed(bpm)
        emb[37] = min(1.0, max(0.0, dance))
        emb[38] = greg
        emb[47] = min(1.0, max(0.0, val))
        rec = {"bpm": bpm, "n_beats": int(len(beats)), "switched": v.get("switched", False),
               "oracle_moved": v.get("oracle_moved"), "bpm_conf": bc, "grid_stab": gs, "rr": reg,
               "tempo_var": rr.tempo_variability(tc), "dance": dance, "greg": greg, "valence": val,
               "happy": happy, "relaxed": relaxed, "sad": sad,
               "grid_offset": float(beats[0] * HOP / SR) if len(beats) else None}
        # tempo curve level vs Beat This! local tempo
        bb = t["bt"]["beats"]
        if len(tc) and len(bb) >= 16:
            mids = (beats[:-1] + beats[1:]) / 2.0 * HOP / SR
            loc = bt_local_tempo_at(bb, mids)
            ok = np.isfinite(loc)
            rec["tc_bt_agree"] = float(np.mean(np.abs(tc[ok] / loc[ok] - 1) <= 0.04)) if ok.any() else None
        # chords (labelled set only)
        if H is not None:
            labels, bnd = cp.chords_from_beats(H, beats)
            rec["chords"] = labels
            rec["chord_bnd"] = bnd
            rec["ccr"] = cp.change_rate(labels, dur)
            emb[42] = min(1.0, max(0.0, rec["ccr"] / 4.0))
        rec["emb"] = emb
        rec["vocal"] = float(VOCAL.predict(emb)[0])
        # aggression rhythm inputs
        f = {13: rr.fold_bpm_embed(bpm), 15: min(1.0, max(0.0, dance)), 16: greg}
        f[27] = f[15] * f[16] * (1.0 - harsh) if harsh is not None else None
        rec["aggr_f"] = f
        rec["aggr_rhythm"] = (min(1.0, max(0.0, od / 12.0)) + f[15] + f[16]) / 3.0
        out[name] = rec
    # aggression drift bounds vs base
    f0 = out["base"]["aggr_f"]
    for name in VARS[1:]:
        f1 = out[name]["aggr_f"]
        if f0[27] is None:
            out[name]["aggr_bound"] = None
            continue
        dz, lb, unch, tb = AGG.drift_bound(f0, f1)
        out[name]["aggr_bound"] = {"dlogit": dz, "lin_bound": lb, "tree_unchanged": unch, "tree_bound": tb}
    # reference-tempo counterfactual for plausibility (tempo terms at the reference BPM)
    if ref_bpm:
        dT = tempo_term(ref_bpm) - tempo_term(S["bpm"])
        out["_ref"] = {"bpm": ref_bpm, "valence": S["valence"] + 0.30 * dT,
                       "dance_tempo_score": float(np.exp(-0.5 * ((ref_bpm - 120) / 30) ** 2)),
                       "fold": rr.fold_bpm_embed(ref_bpm)}
    # structure energy drift (fix 2: onsets one frame earlier)
    if energy is not None and len(t.get("onset_frames", [])):
        out["_structure_f2"] = structure_delta(t["onset_frames"], energy, dur)
    # BT-window chord labels (labelled set): beat-accurate segmentation reference
    if H is not None and len(t["bt"]["beats"]) >= 16:
        btf = np.unique(np.clip(np.round(t["bt"]["beats"] * SR / HOP).astype(np.int64), 0, H.shape[1] - 1))
        lab, bnd = cp.chords_from_beats(H, btf)
        out["_bt_chords"] = (lab, bnd)
    out["_base_stored"] = {"reg": S.get("rhythmic_regularity"), "dance": S["danceability"], "valence": S["valence"],
                           "vocal": S.get("vocalness"), "emb": emb0, "bpm_conf": S["bpm_confidence"],
                           "chords": S.get("chord_sequence"), "ccr": S.get("chord_change_rate"),
                           "tempo_var": S.get("tempo_variability"), "aggr_rhythm": S.get("aggression_rhythm"),
                           "aggr": S.get("aggression_score")}
    return out


def ref_bpm_for(t, prow):
    c = t.get("comp", {})
    if c.get("refs") == "agree" and c.get("ref_bpm"):
        return float(c["ref_bpm"])
    if prow and prow.get("bt_fit"):
        return float(prow["bt_fit"])
    return None


def main() -> int:
    prows = {(r["set"], r["idx"]): r for r in json.loads((dd.RESULTS / "part_a_tracks.json").read_text())}
    res = {}
    for which in ("L", "B"):
        D = dd.load_labelled() if which == "L" else dd.load_broken()
        V = pickle.loads((dd.CACHE / f"variants_{which}.pkl").read_bytes())
        E = energy_curves(which)
        t0 = time.perf_counter()
        for n, idx in enumerate(sorted(V), 1):
            t = D[idx]
            H = None
            if which == "L":
                H = np.load(dd.CACHE / "hpcp" / f"{idx:04d}.npz")["hpcp"]
            prow = prows.get((which, idx))
            res[(which, idx)] = compute_track(t, V[idx], H, E.get(idx), prow, ref_bpm_for(t, prow))
            if n % 100 == 0:
                print(f"  {which} {n}/{len(V)} {time.perf_counter() - t0:.0f}s", flush=True)
    (dd.CACHE / "part_b_tracks.pkl").write_bytes(pickle.dumps(res, protocol=pickle.HIGHEST_PROTOCOL))
    print("tracks", len(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
