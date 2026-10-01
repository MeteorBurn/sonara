"""Read-only parsers for Sonara's bundled models (459bd3c):

- vocalness_v2.json   48 -> 32 relu -> 2 softmax MLP (vocal_model.rs / genre.rs);
- aggression rank v3  aggression_model.bin (decode_rank_model in aggression.rs)
                      + aggression_linear.ferricml (FerricML logistic: 39 coefs,
                      no intercept, centred on the artifact's `center`).

Only the rhythm-fed aggression features can be recomputed offline (13 fold_bpm,
15 danceability, 16 grid regularity, 27 dance*regularity*(1-harshness)); the
other 35 need per-frame audio evidence Sonara does not expose. So the
aggression output drift is bounded, not recomputed: the linear-part change is
exact in logit space, the tree part is exactly unchanged unless a split on a
changed feature lies between the old and new value.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import bpm_grid_paths as P  # noqa: E402

SRC = P.REPO / "sonara" / "src"
VOCAL = P.REPO / "python" / "sonara" / "models" / "vocalness_v2.json"


class Vocal:
    def __init__(self, path=VOCAL):
        d = json.loads(path.read_text(encoding="utf-8"))
        self.id = d["id"]
        self.layers = [(np.asarray(L["weights"], dtype=np.float32), np.asarray(L["bias"], dtype=np.float32),
                        L["activation"]) for L in d["layers"]]
        self.vocal_idx = [s.lower() for s in d["labels"]].index("vocal")

    def predict(self, X):
        h = np.atleast_2d(np.asarray(X, dtype=np.float32))
        for W, b, act in self.layers:
            h = h @ W.T + b
            if act == "relu":
                h = np.maximum(h, 0)
            elif act == "softmax":
                h = h - h.max(axis=1, keepdims=True)
                e = np.exp(h)
                h = e / e.sum(axis=1, keepdims=True)
        return h[:, self.vocal_idx]


class Aggression:
    def __init__(self):
        raw = (SRC / "aggression_model.bin").read_bytes()
        body = raw[:-32]
        pos = 0

        def take(fmt):
            nonlocal pos
            v = struct.unpack_from("<" + fmt, body, pos)
            pos += struct.calcsize("<" + fmt)
            return v
        assert body[:8] == b"SNRAGGR2"
        pos = 8
        ver, nfeat, ntrees, mver = take("IIII")
        assert (ver, nfeat, mver) == (2, 39, 3)
        pos += 32
        (self.baseline, self.lw, self.tw, self.slope, self.cint, self.tie, self.hc) = take("7f")
        self.center = np.asarray(take("39f"), dtype=np.float32)
        self.trees = []
        for _ in range(ntrees):
            (nn,) = take("H")
            nodes = []
            for _ in range(nn):
                feat, leaf, left, right, thr, val = take("BBHHff")
                nodes.append((feat, leaf, left, right, thr, val))
            self.trees.append(nodes)
        assert pos == len(body)
        lin = (SRC / "aggression_linear.ferricml").read_bytes()
        assert lin[:8] == b"FERRICML"
        nf = struct.unpack_from("<I", lin, 0x2C)[0]
        assert nf == 39
        self.intercept = struct.unpack_from("<f", lin, 0x44)[0]
        ncoef = struct.unpack_from("<I", lin, 0x48)[0]
        assert ncoef == 39
        self.coef = np.asarray(struct.unpack_from("<39f", lin, 0x4C), dtype=np.float32)
        self.split_features = sorted({n[0] for tr in self.trees for n in tr if not n[1]})

    def tree_thresholds(self, feature):
        return [n[4] for tr in self.trees for n in tr if not n[1] and n[0] == feature]

    def tree_leaf_range(self, tree):
        v = [n[5] for n in tree if n[1]]
        return max(v) - min(v)

    def describe(self):
        return {"trees": len(self.trees), "baseline": self.baseline, "linear_weight": self.lw,
                "tree_weight": self.tw, "calibration_slope": self.slope, "calibration_intercept": self.cint,
                "harshness_correction": self.hc, "intercept": self.intercept,
                "coef_rhythm": {i: float(self.coef[i]) for i in (13, 14, 15, 16, 27)},
                "center_rhythm": {i: float(self.center[i]) for i in (13, 14, 15, 16, 27)},
                "split_features": self.split_features,
                "n_splits_rhythm": {i: len(self.tree_thresholds(i)) for i in (13, 15, 16, 27)}}

    def drift_bound(self, old, new):
        """old/new: dict feature_index -> value for the changed features.
        Returns (delta_logit_linear, max |delta linear output|, tree_unchanged, tree_bound)."""
        dz = sum(float(self.coef[i]) * (new[i] - old[i]) for i in old)
        lin_bound = self.lw * self.slope * abs(dz) / 4.0
        bound, unchanged = 0.0, True
        for tr in self.trees:
            hit = False
            for feat, leaf, _l, _r, thr, _v in tr:
                if leaf or feat not in old:
                    continue
                a, b = old[feat], new[feat]
                if (a <= thr) != (b <= thr):
                    hit = True
                    break
            if hit:
                unchanged = False
                bound += self.tree_leaf_range(tr)
        return dz, lin_bound, unchanged, self.tw * bound
