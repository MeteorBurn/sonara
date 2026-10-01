"""Fix 1: final BPM from Sonara's own beat frames with a robust estimator.

Estimators (all on Sonara beat times t_i = frame * 512 / 22050):
  reported   stored ACF-peak BPM (baseline)
  ls_index   least squares t_i ~ a + b*i (naive)
  ls_unwrap  LS on unwrapped beat numbers k_i (gaps counted by round(ibi/median))
  theil_sen  median of pairwise slopes over pairs >= 16 beats apart (on k_i)
  huber      IRLS Huber fit on k_i (c = 1 frame)
  lts        Theil-Sen seed, then LS on inliers |resid| <= 1.5 frames, iterated
  ransac     best 2-point line by inlier count (+-1 frame), then LS on inliers
  coherence  argmax_T |sum exp(2*pi*i*t_i/T)| near the ls_unwrap period
  seg_median median of LS slopes over 32-beat windows
Selection on dev (0-based playlist_idx even) by exact rate (|err|<=0.05 BPM vs
MIK on agree tracks); holdout reported. Then residual-failure characterisation.
Writes results/fix1.json and results/fix1_tracks.json.
"""

from __future__ import annotations

import json
import sys

import numpy as np

import est_common as ec

FRAME = ec.HOP_SEC
MIN_BEATS = 16


def unwrap(t):
    ibi = np.diff(t)
    med = np.median(ibi)
    steps = np.maximum(1, np.round(ibi / med)).astype(int)
    return np.concatenate([[0], np.cumsum(steps)])


def ls(k, t):
    b, a = np.polyfit(k, t, 1)
    return b, a


def theil_sen(k, t, min_sep=16, max_pts=1200):
    if len(k) > max_pts:                      # subsample evenly for very long lists
        sel = np.linspace(0, len(k) - 1, max_pts).astype(int)
        k, t = k[sel], t[sel]
    i, j = np.triu_indices(len(k), 1)
    m = (k[j] - k[i]) >= min_sep
    if m.sum() < 10:
        m = (k[j] - k[i]) >= 1
    s = (t[j][m] - t[i][m]) / (k[j][m] - k[i][m])
    b = np.median(s)
    a = np.median(t - b * k)
    return b, a


def huber(k, t, c=FRAME, iters=30):
    b, a = ls(k, t)
    for _ in range(iters):
        r = t - (a + b * k)
        w = np.where(np.abs(r) <= c, 1.0, c / np.maximum(np.abs(r), 1e-12))
        W = np.sqrt(w)
        A = np.vstack([k * W, W]).T
        sol, *_ = np.linalg.lstsq(A, t * W, rcond=None)
        if abs(sol[0] - b) < 1e-12:
            b, a = sol
            break
        b, a = sol
    return b, a


def lts(k, t, thr=1.5 * FRAME, iters=5):
    b, a = theil_sen(k, t)
    for _ in range(iters):
        r = t - (a + b * k)
        m = np.abs(r) <= thr
        if m.sum() < max(8, 0.3 * len(k)):
            break
        b2, a2 = ls(k[m], t[m])
        if abs(b2 - b) < 1e-12:
            b, a = b2, a2
            break
        b, a = b2, a2
    return b, a


def ransac(k, t, thr=FRAME, n_iter=300, seed=0):
    rng = np.random.default_rng(seed)
    best_n, best = -1, None
    n = len(k)
    for _ in range(n_iter):
        i, j = rng.choice(n, 2, replace=False)
        if abs(k[j] - k[i]) < 16:
            continue
        b = (t[j] - t[i]) / (k[j] - k[i])
        a = t[i] - b * k[i]
        cnt = int((np.abs(t - (a + b * k)) <= thr).sum())
        if cnt > best_n:
            best_n, best = cnt, (b, a)
    if best is None:
        return ls(k, t)
    b, a = best
    m = np.abs(t - (a + b * k)) <= thr
    return ls(k[m], t[m]) if m.sum() >= 8 else best


def consensus(k, t, thr=FRAME, sep=32, stride=2, refits=2):
    """Deterministic RANSAC: candidate lines through beat pairs (i, i+sep) for every
    `stride`-th i; keep the line with most inliers (|resid| <= thr); LS on inliers,
    re-count and re-fit `refits` times. Returns (slope, intercept, inlier_fraction)."""
    n = len(k)
    sep = min(sep, max(1, n // 2))
    i = np.arange(0, n - sep, stride)
    j = i + sep
    dk = k[j] - k[i]
    ok = dk > 0
    i, j, dk = i[ok], j[ok], dk[ok]
    if len(i) == 0:
        b, a = ls(k, t)
        return b, a, 1.0
    B = (t[j] - t[i]) / dk
    A = t[i] - B * k[i]
    cnt = (np.abs(t[None, :] - (A[:, None] + B[:, None] * k[None, :])) <= thr).sum(axis=1)
    m_best = int(np.argmax(cnt))                  # first max: deterministic
    b, a = B[m_best], A[m_best]
    for _ in range(refits + 1):
        m = np.abs(t - (a + b * k)) <= thr
        if m.sum() < 8:
            break
        b, a = ls(k[m], t[m])
    m = np.abs(t - (a + b * k)) <= thr
    return b, a, float(m.mean())


def coherence(t, T0, span=0.004, n=801):
    Ts = T0 * (1 + np.linspace(-span, span, n))
    ph = np.exp(2j * np.pi * t[None, :] / Ts[:, None])
    R = np.abs(ph.sum(axis=1))
    i = int(np.argmax(R))
    if 0 < i < n - 1:                          # parabolic refinement
        l, c, r = R[i - 1], R[i], R[i + 1]
        den = l - 2 * c + r
        off = 0.5 * (l - r) / den if den != 0 else 0.0
        return T0 * (1 + np.interp(i + off, np.arange(n), np.linspace(-span, span, n)))
    return Ts[i]


def seg_median(k, t, win=32):
    s = []
    for st in range(0, len(k) - win + 1, win // 2):
        kk, tt = k[st:st + win], t[st:st + win]
        if kk[-1] - kk[0] >= win // 2:
            s.append(ls(kk, tt)[0])
    return float(np.median(s)) if s else ls(k, t)[0]


def estimators(frames, reported):
    t = ec.frames_to_sec(frames)
    out = {"reported": reported}
    if len(t) < MIN_BEATS:
        for name in ("ls_index", "ls_unwrap", "theil_sen", "huber", "lts", "ransac",
                     "coherence", "seg_median", "cons_1f", "cons_0.5f", "cons_1.5f",
                     "cons_1f_sep64", "cons_1f_noref", "cons_1f_sep128", "cons_1f_sephalf"):
            out[name] = reported
        out["inlier_frac"] = None
        return out
    idx = np.arange(len(t), dtype=float)
    k = unwrap(t).astype(float)
    per = {
        "ls_index": ls(idx, t)[0],
        "ls_unwrap": ls(k, t)[0],
        "theil_sen": theil_sen(k, t)[0],
        "huber": huber(k, t)[0],
        "lts": lts(k, t)[0],
        "ransac": ransac(k, t)[0],
        "seg_median": seg_median(k, t),
    }
    per["coherence"] = coherence(t, per["ls_unwrap"])
    b, _, frac = consensus(k, t, FRAME)
    per["cons_1f"] = b
    per["cons_0.5f"] = consensus(k, t, 0.5 * FRAME)[0]
    per["cons_1.5f"] = consensus(k, t, 1.5 * FRAME)[0]
    per["cons_1f_sep64"] = consensus(k, t, FRAME, sep=64)[0]
    per["cons_1f_noref"] = consensus(k, t, FRAME, refits=0)[0]
    per["cons_1f_sep128"] = consensus(k, t, FRAME, sep=128)[0]
    per["cons_1f_sephalf"] = consensus(k, t, FRAME, sep=max(16, len(t) // 2), stride=1)[0]
    for name, p in per.items():
        out[name] = ec.fold(60.0 / p) if p > 0 else reported
    out["inlier_frac"] = frac
    return out


def summarize(rows, name, split):
    rs = [r for r in rows if r["agree"] and (split == "all" or r["dev"] == (split == "dev"))]
    e = np.array([abs(r["est"][name] - r["ref"]) for r in rs])
    cats = [ec.category(r["est"][name], r["ref"]) for r in rs]
    nc = np.array([not ec.is_critical(c) for c in cats])
    return {"n": len(rs), "exact": int((e <= 0.05).sum()), "exact_rate": float((e <= 0.05).mean()),
            "le_0.5": int(sum(c in ("exact", "fine") for c in cats)),
            "critical": int((~nc).sum()),
            "p50_noncrit": float(np.median(e[nc])), "p90_noncrit": float(np.percentile(e[nc], 90))}


def bt_fit(t):
    """Consensus tempo of Beat This! beats (its 50 fps grid: inliers +-20 ms):
    an independent second reference."""
    b = t["bt"]["beats"]
    if len(b) < MIN_BEATS:
        return None
    k = unwrap(b).astype(float)
    return ec.fold(60.0 / consensus(k, b, 0.02)[0])


def local_tempi(frames, win=64):
    t = ec.frames_to_sec(frames)
    k = unwrap(t).astype(float)
    vals = []
    for st in range(0, len(t) - win + 1, win // 2):
        vals.append(60.0 / ls(k[st:st + win], t[st:st + win])[0])
    return np.asarray(vals)


def main() -> int:
    D = ec.load_data()
    rows = []
    for idx, tr in sorted(D.items()):
        if tr["status"] != "ok":
            continue
        c = tr["comp"]
        rows.append({"idx": idx, "dev": ec.is_dev(idx), "agree": c["refs"] == "agree",
                     "ref": c["ref_bpm"], "est": estimators(tr["beats"], tr["bpm"])})
    names = [n for n in rows[0]["est"].keys() if n != "inlier_frac"]
    table = {n: {s: summarize(rows, n, s) for s in ("dev", "holdout", "all")} for n in names}
    for n in names:
        d, h = table[n]["dev"], table[n]["holdout"]
        print(f"{n:11s} dev exact {d['exact']:3d}/{d['n']} ({d['exact_rate']:.1%}) p50 {d['p50_noncrit']:.3f} "
              f"p90 {d['p90_noncrit']:.3f} | holdout exact {h['exact']:3d}/{h['n']} ({h['exact_rate']:.1%}) "
              f"p50 {h['p50_noncrit']:.3f} p90 {h['p90_noncrit']:.3f}")
    # Selection restricted a priori to deterministic estimators (implementable in
    # Rust without an RNG); random RANSAC is reported for reference only.
    cands = [n for n in names if n not in ("reported", "ransac")]
    chosen = max(cands, key=lambda n: (table[n]["dev"]["exact_rate"], -table[n]["dev"]["p90_noncrit"]))
    print("chosen on dev:", chosen)

    # --- residual failures of the chosen estimator (non-critical agree tracks) ---
    fails = []
    for r in rows:
        if not r["agree"]:
            continue
        v = r["est"][chosen]
        if ec.is_critical(ec.category(v, r["ref"])):
            continue
        tr = D[r["idx"]]
        btf = bt_fit(tr)
        lt = local_tempi(tr["beats"])
        rec = {"idx": r["idx"], "dev": r["dev"], "ref": r["ref"], "est": v, "err": v - r["ref"],
               "bt_fit": btf, "bt_minus_ref": None if btf is None else btf - r["ref"],
               "est_minus_bt": None if btf is None else v - btf,
               "local_tempo_range": float(np.ptp(lt)) if len(lt) > 1 else None,
               "n_beats": int(len(tr["beats"])), "duration": tr["duration"],
               "grid_stability": tr["grid_stability"]}
        # Beat This! local tempo range (for 'tempo varies in the music' evidence)
        bb = tr["bt"]["beats"]
        if len(bb) >= 64:
            kb = unwrap(bb).astype(float)
            lb = [60.0 / ls(kb[s:s + 64], bb[s:s + 64])[0] for s in range(0, len(bb) - 63, 32)]
            rec["bt_local_tempo_range"] = float(np.ptp(lb)) if len(lb) > 1 else None
        rec["ok"] = abs(rec["err"]) <= 0.05
        fails.append(rec)
    bad = [f for f in fails if not f["ok"]]
    good = [f for f in fails if f["ok"]]

    def cls(f):
        if f["bt_fit"] is not None and abs(f["est_minus_bt"]) <= 0.05 and abs(f["bt_minus_ref"]) > 0.05:
            return "reference_doubt (Sonara fit = Beat This! fit != MIK)"
        if (f.get("bt_local_tempo_range") or 0) > 0.5:
            return "tempo_varies (Beat This! local tempo range > 0.5 BPM)"
        if f["bt_fit"] is not None and abs(f["est_minus_bt"]) > 0.05 and abs(f["bt_minus_ref"]) <= 0.05:
            return "sonara_beats (Beat This! fit = MIK, Sonara fit off)"
        return "unclear (all three differ)"
    for f in bad:
        f["class"] = cls(f)
    classes = {}
    for f in bad:
        classes[f["class"]] = classes.get(f["class"], 0) + 1
    print(f"non-critical agree tracks {len(fails)}: exact {len(good)}, not exact {len(bad)}")
    print("failure classes:", json.dumps(classes, indent=1))
    agree_bt = [f for f in fails if f["bt_fit"] is not None]
    print("Beat This! LTS fit vs MIK exact (<=0.05):",
          sum(abs(f["bt_minus_ref"]) <= 0.05 for f in agree_bt), "/", len(agree_bt))
    print("Sonara fit vs Beat This! fit within 0.05:",
          sum(abs(f["est_minus_bt"]) <= 0.05 for f in agree_bt), "/", len(agree_bt))
    def _py(o):
        if isinstance(o, dict):
            return {k: _py(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_py(v) for v in o]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return o
    bad = _py(bad)
    out = {"table": table, "chosen": chosen, "failure_classes": classes,
           "n_noncritical": len(fails), "n_exact": len(good),
           "bt_fit_exact_vs_mik": sum(abs(f["bt_minus_ref"]) <= 0.05 for f in agree_bt),
           "sonara_fit_within_0.05_of_bt_fit": sum(abs(f["est_minus_bt"]) <= 0.05 for f in agree_bt),
           "n_with_bt_fit": len(agree_bt)}
    (ec.RESULTS / "fix1.json").write_text(json.dumps(_py(out), indent=1), encoding="utf-8")
    (ec.RESULTS / "fix1_tracks.json").write_text(json.dumps(
        {"rows": [{"idx": r["idx"], **{k: float(v) if v is not None else None for k, v in r["est"].items()}}
                  for r in rows], "failures": bad}, indent=0, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
