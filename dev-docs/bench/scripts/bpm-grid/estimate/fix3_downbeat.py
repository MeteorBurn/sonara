"""Fix 3: choose the bar phase from Sonara-internal band evidence.

For every track the bar phase p in {0..3} indexes Sonara's own beat list
(downbeats = beats[p::4]); Sonara 0.3.7 picks p by the broadband accent.
Reference phase p* (evaluation only): the p whose downbeats best match Beat
This! downbeats (F +-70 ms), defined on agree tracks where that F >= 0.5.
Variants are compared on dev (0-based playlist_idx even) by phase accuracy;
the chosen one is reported on holdout. Writes results/fix3.json.
"""

from __future__ import annotations

import json
import sys

import numpy as np

import est_common as ec

W = 2  # accent half-window in frames, as beatgrid.rs ACCENT_WINDOW


def peak_at(env, frames, w=W):
    n = len(env)
    out = np.zeros(len(frames))
    for i, f in enumerate(frames):
        lo, hi = max(0, f - w), min(n, f + w + 1)
        out[i] = env[lo:hi].max() if lo < hi else 0.0
    return out


def mean_at(env, frames, w=W):
    n = len(env)
    out = np.zeros(len(frames))
    for i, f in enumerate(frames):
        lo, hi = max(0, f - w), min(n, f + w + 1)
        out[i] = env[lo:hi].mean() if lo < hi else 0.0
    return out


def phase_means(x, n_ph=4):
    return np.array([x[p::n_ph].mean() if len(x[p::n_ph]) else -np.inf for p in range(n_ph)])


def novelty_votes(bands, beats, half=8, top_frac=0.1):
    """Section-change evidence: per-beat log band energy, contrast of the next
    `half` beats vs the previous `half`; vote phase (index mod 4) of the strongest
    local maxima. Returns 4 vote weights."""
    nb = len(beats)
    if nb < 4 * half:
        return np.zeros(4)
    E = np.zeros((nb, bands.shape[0]))
    edges = np.append(beats, beats[-1] + int(np.median(np.diff(beats))))
    for i in range(nb):
        E[i] = bands[:, edges[i]:max(edges[i] + 1, edges[i + 1])].sum(axis=1)
    L = np.log(E + 1e-6)
    nov = np.zeros(nb)
    for i in range(half, nb - half):
        nov[i] = np.linalg.norm(L[i:i + half].mean(0) - L[i - half:i].mean(0))
    loc = [i for i in range(1, nb - 1) if nov[i] > 0 and nov[i] >= nov[i - 1] and nov[i] >= nov[i + 1]]
    if not loc:
        return np.zeros(4)
    loc = sorted(loc, key=lambda i: -nov[i])[:max(1, int(top_frac * nb / 4))]
    v = np.zeros(4)
    for i in loc:
        v[i % 4] += nov[i]
    return v / max(v.sum(), 1e-9)


def features(t):
    beats = np.asarray(t["beats"], dtype=int)
    bands = t["onset_strength_bands"].astype(float)
    broad = ec.broadband(t["onset_strength_bands"], t["band_edges"]).astype(float)
    norm = bands / np.maximum(bands.mean(axis=1, keepdims=True), 1e-9)
    f = {"broad_peak": peak_at(broad, beats)}
    for b in range(bands.shape[0]):
        f[f"b{b}_peak"] = peak_at(norm[b], beats)
        f[f"b{b}_mean"] = mean_at(norm[b], beats)
        f[f"b{b}_peak_w1"] = peak_at(norm[b], beats, 1)
    f["nov"] = novelty_votes(bands, beats)
    return beats, f


def z(v):
    s = v.std()
    return (v - v.mean()) / s if s > 0 else v * 0


def choose(name, f):
    pm = {k: phase_means(v) for k, v in f.items() if k != "nov"}
    if name == "base_broad_peak":
        s = pm["broad_peak"]
    elif name == "low_peak":
        s = pm["b0_peak"]
    elif name == "low_mean":
        s = pm["b0_mean"]
    elif name == "low_peak_w1":
        s = pm["b0_peak_w1"]
    elif name == "low_minus_mid":
        s = pm["b0_peak"] - pm["b1_peak"]
    elif name == "low_minus_mid12":
        s = pm["b0_peak"] - 0.5 * (pm["b1_peak"] + pm["b2_peak"])
    elif name == "low_ratio":
        s = pm["b0_peak"] / np.maximum(pm["b0_peak"] + pm["b1_peak"] + pm["b2_peak"], 1e-9)
    elif name == "low_plus_high":
        s = z(pm["b0_peak"]) + z(pm["b4_peak"] + pm["b3_peak"])
    elif name.startswith("two_stage"):
        # stage 1: the kick pair {p, p+2} by low-minus-mid contrast
        pair = pm["b0_peak"] - pm["b1_peak"]
        k = int(np.argmax(pair[:2] + pair[2:]))          # pair (0,2) vs (1,3)
        cand = [k, k + 2]
        if name == "two_stage_high":
            sec = pm["b3_peak"] + pm["b4_peak"]
        elif name == "two_stage_low":
            sec = pm["b0_peak"]
        elif name == "two_stage_nov":
            sec = f["nov"]
        elif name == "two_stage_nov_low":
            sec = f["nov"] + 0.25 * z(pm["b0_peak"])
        else:
            raise KeyError(name)
        p = cand[int(np.argmax([sec[c] for c in cand]))]
        return p
    elif name == "nov_only":
        s = f["nov"]
    else:
        raise KeyError(name)
    return int(np.argmax(s))


VARIANTS = ["base_broad_peak", "low_peak", "low_mean", "low_peak_w1", "low_minus_mid",
            "low_minus_mid12", "low_ratio", "low_plus_high", "two_stage_high", "two_stage_low",
            "two_stage_nov", "two_stage_nov_low", "nov_only"]


def ref_phase(t, beats_sec):
    bd = t["bt"]["downbeats"]
    if len(bd) < 4:
        return None, None
    fs = [ec.f_measure(beats_sec[p::4], bd, 0.07) or 0.0 for p in range(4)]
    p = int(np.argmax(fs))
    return (p if fs[p] >= 0.5 else None), fs


def main() -> int:
    D = ec.load_data()
    recs = []
    for idx, t in sorted(D.items()):
        if t["status"] != "ok" or t["comp"]["refs"] != "agree" or len(t["beats"]) < 8:
            continue
        beats, f = features(t)
        bs = ec.frames_to_sec(beats)
        p_star, fs = ref_phase(t, bs)
        ch = {v: choose(v, f) for v in VARIANTS}
        # sanity: base reproduces stored downbeats
        stored_p = int(np.searchsorted(beats, t["downbeats"][0])) if len(t["downbeats"]) else 0
        cues = t["tag"].get("cues") or []
        cue = {}
        for v in VARIANTS:
            h, n = ec.cue_hits(bs[ch[v]::4], cues)
            cue[v] = (h, n)
        recs.append({"idx": idx, "dev": ec.is_dev(idx), "p_star": p_star, "fs": fs, "choice": ch,
                     "stored_p": stored_p, "cue": cue})
    assert all(r["choice"]["base_broad_peak"] == r["stored_p"] for r in recs), "base != stored"
    res = {}
    for v in VARIANTS:
        res[v] = {}
        for split in ("dev", "holdout", "all"):
            rs = [r for r in recs if split == "all" or r["dev"] == (split == "dev")]
            ok = [r for r in rs if r["p_star"] is not None]
            acc = np.mean([r["choice"][v] == r["p_star"] for r in ok])
            pair = np.mean([(r["choice"][v] - r["p_star"]) % 2 == 0 for r in ok])
            rel = {}
            for r in ok:
                d = (r["choice"][v] - r["p_star"]) % 4
                rel[d] = rel.get(d, 0) + 1
            df = np.mean([r["fs"][r["choice"][v]] for r in rs if r["fs"] is not None])
            ch_ = sum(r["cue"][v][0] for r in rs)
            cn_ = sum(r["cue"][v][1] for r in rs)
            res[v][split] = {"n_phase": len(ok), "phase_acc": float(acc), "pair_acc": float(pair),
                             "choice_minus_ref": {int(k): int(c) for k, c in sorted(rel.items())},
                             "DF70_mean": float(df), "cue_hits": int(ch_), "cues": int(cn_)}
    for v in VARIANTS:
        d, h = res[v]["dev"], res[v]["holdout"]
        print(f"{v:18s} dev acc {d['phase_acc']:.3f} pair {d['pair_acc']:.3f} DF {d['DF70_mean']:.3f} "
              f"cues {d['cue_hits']}/{d['cues']} {d['choice_minus_ref']} | holdout acc {h['phase_acc']:.3f} "
              f"pair {h['pair_acc']:.3f} DF {h['DF70_mean']:.3f} cues {h['cue_hits']}/{h['cues']}")
    chosen = max((v for v in VARIANTS if v != "base_broad_peak"),
                 key=lambda v: res[v]["dev"]["phase_acc"])
    print("chosen on dev:", chosen)
    (ec.RESULTS / "fix3.json").write_text(json.dumps({"variants": res, "chosen": chosen}, indent=1),
                                          encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
