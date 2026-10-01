"""Fix 4: re-rank tempo candidates with Sonara-internal evidence only.

Per track, candidate tempi = the selected tempo plus the stored top-5 ACF
candidates (raw integer-lag BPM), each refined at its own ACF peak and folded to
79-192, de-duplicated by DP period (frames per beat). For every candidate the
exact port of Sonara's DP re-tracks the beats at that tempo, so beats/downbeats
after a switch are real re-tracking, not a bound. Evidence per candidate:
  q    consensus inlier fraction of the re-tracked beats (+-1 frame line band)
  s    mean normalised onset strength (local score) at the re-tracked beats
  acf  ACF score of the candidate lag (as Sonara ranks them)
  met  mean normalised ACF at 1, 2 and 4 candidate periods (metrical support)
Rules are compared on dev (0-based playlist_idx even); holdout reported.
Writes results/fix4_cands.pkl (per-track candidates) and results/fix4.json.
"""

from __future__ import annotations

import json
import pickle
import sys
import time

import numpy as np

import est_common as ec
from fix1_bpm import consensus, unwrap

FR = ec.SR / ec.HOP


def acf_at(acf, lag):
    """Linear interpolation of the ACF at a fractional lag (0 beyond range)."""
    if lag >= len(acf) - 1:
        return 0.0
    i = int(lag)
    f = lag - i
    return float((1 - f) * acf[i] + f * acf[i + 1])


def candidates_for(t, oenv, est):
    acf_long = ec.autocorrelate(oenv, int(16 * FR))     # long ACF for 4-period support
    acf0 = float(acf_long[0]) if acf_long[0] > 0 else 1.0
    pool = [("selected", est["lag"])]
    for bpm, _score in t["cands"]:
        lag = int(round(60.0 * FR / bpm))
        pool.append(("top5", lag))
    seen, out = {}, []
    for src, lag in pool:
        frac = ec.refine_lag(est["acf"], lag)
        tempo = ec.fold(ec.lag_to_bpm(frac))
        fpb = int(np.floor(60.0 * FR / tempo + 0.5))
        if fpb in seen:
            continue
        seen[fpb] = True
        beats, _ = ec.track_beats(oenv, tempo)
        ls = ec.local_score(oenv, fpb)
        if len(beats) >= 16:
            tb = ec.frames_to_sec(beats)
            k = unwrap(tb).astype(float)
            slope, _, q = consensus(k, tb, ec.HOP_SEC, sep=128)
            fit = ec.fold(60.0 / slope)
        else:
            q, fit = 0.0, tempo
        per = 60.0 * FR / tempo                              # folded period in frames
        met = np.mean([acf_at(acf_long, m * per) / acf0 for m in (1, 2, 4)])
        score = next((c[2] for c in est["cands"] if c[0] == lag), None)
        out.append({"src": src, "lag": lag, "tempo": tempo, "fpb": fpb, "q": q, "fit_bpm": fit,
                    "s": float(ls[beats].mean()) if len(beats) else -9.0,
                    "acf": score, "met": float(met), "beats": beats})
    return out


def build(D):
    res, t0 = {}, time.perf_counter()
    for n, (idx, t) in enumerate(sorted(D.items()), 1):
        if t["status"] != "ok":
            continue
        oenv = ec.broadband(t["onset_strength_bands"], t["band_edges"])
        est = ec.estimate_tempo(oenv)
        res[idx] = candidates_for(t, oenv, est)
        if n % 100 == 0:
            print(f"  {n} tracks {time.perf_counter() - t0:.0f}s", flush=True)
    return res


def pick(cands, rule, delta=0.0, rho=0.0):
    base = cands[0]
    best_acf = max((c["acf"] or 0.0) for c in cands)
    pool = [c for c in cands if c is base or (c["acf"] or 0.0) >= rho * best_acf]
    if rule == "q":
        key = lambda c: c["q"]
    elif rule == "q_s":
        key = lambda c: c["q"] + 0.1 * c["s"]
    elif rule == "s":
        key = lambda c: c["s"]
    elif rule == "met":
        key = lambda c: c["met"]
    elif rule == "q_met":
        key = lambda c: c["q"] + c["met"]
    else:
        raise KeyError(rule)
    top = max(pool, key=key)
    if top is base or key(top) - key(base) < delta:
        return base
    return top


def evaluate(D, C, rule, delta, rho, split):
    fixed = regress = stay_bad = stay_good = 0
    switched = 0
    for idx, cands in C.items():
        t = D[idx]
        c = t["comp"]
        if c["refs"] != "agree" or (split != "all" and ec.is_dev(idx) != (split == "dev")):
            continue
        ch = pick(cands, rule, delta, rho)
        switched += ch is not cands[0]
        before = ec.is_critical(ec.category(cands[0]["tempo"], c["ref_bpm"]))
        after = ec.is_critical(ec.category(ch["tempo"], c["ref_bpm"]))
        fixed += before and not after
        regress += (not before) and after
        stay_bad += before and after
        stay_good += (not before) and (not after)
    return {"fixed": fixed, "regressed": regress, "net": fixed - regress, "still_critical": stay_bad,
            "ok": stay_good, "switched": switched}


def main() -> int:
    D = ec.load_data()
    cache = ec.RESULTS / "fix4_cands.pkl"
    if cache.exists():
        C = pickle.loads(cache.read_bytes())
    else:
        C = build(D)
        cache.write_bytes(pickle.dumps(C))
    # ---- oracle ceilings ----
    orc = {}
    for split in ("dev", "holdout", "all"):
        n_crit = in_top5 = in_any = 0
        for idx, cands in C.items():
            t = D[idx]
            c = t["comp"]
            if c["refs"] != "agree" or (split != "all" and ec.is_dev(idx) != (split == "dev")):
                continue
            if not ec.is_critical(ec.category(cands[0]["tempo"], c["ref_bpm"])):
                continue
            n_crit += 1
            top5 = [ec.fold(b) for b, _ in t["cands"]]
            in_top5 += any(abs(x / c["ref_bpm"] - 1) <= 0.02 for x in top5)
            in_any += any(not ec.is_critical(ec.category(x["tempo"], c["ref_bpm"])) for x in cands)
        orc[split] = {"critical": n_crit, "correct_in_top5": in_top5, "correct_among_tracked": in_any}
    print("oracle:", json.dumps(orc))
    # ---- rule search on dev ----
    grid = []
    for rule in ("q", "q_s", "s", "met", "q_met"):
        for delta in (0.0, 0.05, 0.1, 0.2, 0.3, 0.4):
            for rho in (0.0, 0.5, 0.75, 0.9):
                d = evaluate(D, C, rule, delta, rho, "dev")
                grid.append({"rule": rule, "delta": delta, "rho": rho, **d})
    grid.sort(key=lambda g: (-g["net"], g["regressed"]))
    for g in grid[:12]:
        print("dev", g)
    best = grid[0]
    hold = evaluate(D, C, best["rule"], best["delta"], best["rho"], "holdout")
    allr = evaluate(D, C, best["rule"], best["delta"], best["rho"], "all")
    print("chosen:", {k: best[k] for k in ("rule", "delta", "rho")}, "holdout:", hold, "all:", allr)
    # ---- how the simple rules do on holdout (context, not selection) ----
    ctx = {f"{r}_d{d}_r{p}": evaluate(D, C, r, d, p, "holdout")
           for r, d, p in (("q", 0.0, 0.0), ("q", 0.1, 0.0), ("met", 0.0, 0.0), ("s", 0.0, 0.0))}
    (ec.RESULTS / "fix4.json").write_text(json.dumps(
        {"oracle": orc, "grid_dev_top": grid[:25], "chosen": {k: best[k] for k in ("rule", "delta", "rho")},
         "dev": {k: best[k] for k in ("fixed", "regressed", "net", "still_critical", "ok", "switched")},
         "holdout": hold, "all": allr, "holdout_context": ctx}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
