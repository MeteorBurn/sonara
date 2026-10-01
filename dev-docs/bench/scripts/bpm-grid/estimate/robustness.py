"""Robustness checks on the combined result (read-only):
1. artist overlap between the even/odd splits and an artist-disjoint holdout;
2. fix 3 phase vs the independent MIK cue-point phase;
3. composition of the failures that remain after 1+2+3+4.
Writes results/robustness.json."""

from __future__ import annotations

import collections
import json
import pickle
import re
import sys

import numpy as np

import est_common as ec


def artist(t):
    tg = t.get("tags") or {}
    a = (tg.get("artist") or "").strip().lower()
    if not a:
        stem = t["path"].replace("\\", "/").rsplit("/", 1)[-1]
        a = stem.split(" - ")[0].strip().lower()
    return re.split(r",| feat\.| ft\.| & | vs\.? | x ", a)[0].strip()


def main() -> int:
    D = ec.load_data()
    R = pickle.loads((ec.RESULTS / "combined_rows.pkl").read_bytes())
    out = {}
    # ---- 1. artist overlap ----
    A = {i: artist(t) for i, t in D.items() if t["status"] == "ok"}
    dev_art = {A[i] for i in A if ec.is_dev(i)}
    hold = [i for i in A if not ec.is_dev(i)]
    disj = {i for i in hold if A[i] not in dev_art}
    adj = sum(1 for i in A if i + 1 in A and A[i] == A[i + 1])
    out["artists"] = {"holdout_tracks": len(hold), "holdout_artist_in_dev": len(hold) - len(disj),
                      "artist_disjoint_holdout": len(disj), "adjacent_same_artist_pairs": adj,
                      "distinct_artists": len(set(A.values()))}
    out["artist_disjoint_holdout"] = {}
    for name in ("baseline", "fix1", "fix2", "fix3", "fix4", "1+2+3+4"):
        rs = [r for r in R[name] if r["idx"] in disj and r["agree"]]
        out["artist_disjoint_holdout"][name] = {
            "n": len(rs), "exact": float(np.mean([r["cat"] == "exact" for r in rs])),
            "critical": float(np.mean([r["critical"] for r in rs])),
            "F20": float(np.mean([r["F20"] for r in rs if r.get("F20") is not None])),
            "DF70": float(np.mean([r["DF70"] for r in rs if r.get("DF70") is not None]))}
    # ---- 2. fix 3 vs MIK cue phase (independent of Beat This!) ----
    res = collections.Counter()
    for idx, t in D.items():
        if t["status"] != "ok" or t["comp"]["refs"] != "agree":
            continue
        cues = [c for c in (t["tag"].get("cues") or []) if c >= 5.0]
        if len(cues) < 2:
            continue
        b = ec.frames_to_sec(t["beats"])
        hits = [ec.cue_hits(b[p::4], cues)[0] for p in range(4)]
        if max(hits) * 2 <= len(cues):           # cues not on one phase of Sonara's beats
            res["cue_phase_undefined"] += 1
            continue
        p_mik = int(np.argmax(hits))
        rb = {r["idx"]: r for r in R["baseline"]}[idx]
        r3 = {r["idx"]: r for r in R["fix3"]}[idx]
        res["tracks"] += 1
        res["baseline_matches_mik"] += rb["phase_choice"] == p_mik
        res["fix3_matches_mik"] += r3["phase_choice"] == p_mik
        res["fix3_same_pair_as_mik"] += (r3["phase_choice"] - p_mik) % 2 == 0
    out["fix3_vs_mik_cue_phase"] = dict(res)
    # ---- 3. what remains after 1+2+3+4 (agree tracks) ----
    rows = {r["idx"]: r for r in R["1+2+3+4"]}
    base = {r["idx"]: r for r in R["baseline"]}
    rem = collections.Counter()
    for i, r in rows.items():
        if not r["agree"]:
            continue
        if r["critical"]:
            rem["critical:" + r["cat"]] += 1
        elif r["cat"] != "exact":
            rem["noncritical_not_exact"] += 1
    out["remaining_bpm"] = dict(rem)
    # beat phase classes after 1+2+3+4 on non-critical tracks
    ph = collections.Counter()
    for i, r in rows.items():
        t = D.get(i)
        if not r["agree"] or r["critical"] or t["status"] != "ok":
            continue
        f = r.get("F70")
        if f is None:
            continue
        ph["F70>=0.8"] += f >= 0.8
        ph["F70<0.5"] += f < 0.5
        ph["n"] += 1
    out["grid_after_combined_noncritical"] = dict(ph)
    (ec.RESULTS / "robustness.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
