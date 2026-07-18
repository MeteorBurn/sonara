#!/usr/bin/env python3
"""Test the opt-in similarity embedding vector and the sonara.similarity helper.

The embedding is OPT-IN: it is only produced when analysis is requested with
features=["embedding"], never by a bare mode (compact/playlist/full).

Run as a plain script (like test_api.py): `python tests/python/test_similarity.py`.
"""

import sys

import numpy as np

import sonara

passed = 0
failed = 0
errors = []


def test(name, fn):
    global passed, failed
    try:
        fn()
        passed += 1
        print(f"  PASS  {name}")
    except Exception as e:  # noqa: BLE001
        failed += 1
        errors.append((name, str(e)))
        print(f"  FAIL  {name}: {e}")


# ------------------------------------------------------------
# Test signals (no audio files needed)
# ------------------------------------------------------------
sr = 22050
rng = np.random.default_rng(42)


def kick(bpm, dur=6.0):
    n = int(sr * dur)
    y = np.zeros(n, dtype=np.float32)
    interval = int(60.0 / bpm * sr)
    tail = sr // 20
    env_t = np.arange(tail) / sr
    thump = (np.sin(2 * np.pi * 60.0 * env_t) * np.exp(-30.0 * env_t)).astype(np.float32)
    for pos in range(0, n - tail, interval):
        y[pos:pos + tail] = thump
    return y


y_kick120 = kick(120.0)
y_noise = rng.standard_normal(6 * sr).astype(np.float32)

EMB_FEATURES = ["embedding"]

print("=" * 70)
print("  Testing sonara similarity embedding")
print("=" * 70)

r_emb = sonara.analyze_signal(y_kick120, sr=sr, features=EMB_FEATURES)
r_emb2 = sonara.analyze_signal(y_kick120, sr=sr, features=EMB_FEATURES)

EXPECTED_DIM = sonara.EMBEDDING_DIM


# ------------------------------------------------------------
# Opt-in presence / absence
# ------------------------------------------------------------
def t_present_when_requested():
    assert "embedding" in r_emb, "embedding must be present with features=['embedding']"
    assert "embedding_version" in r_emb, "embedding_version must be present"
    assert r_emb["embedding_version"] == sonara.SIMILARITY_VERSION


def t_right_length():
    assert len(r_emb["embedding"]) == EXPECTED_DIM, (
        f"embedding must be {EXPECTED_DIM} dims, got {len(r_emb['embedding'])}"
    )


def t_absent_in_compact():
    r = sonara.analyze_signal(y_kick120, sr=sr, mode="compact")
    assert "embedding" not in r, "embedding must be absent in compact mode"
    assert "embedding_version" not in r


def t_absent_in_playlist_by_default():
    # Opt-in: even playlist mode must NOT produce the embedding by default.
    r = sonara.analyze_signal(y_kick120, sr=sr, mode="playlist")
    assert "embedding" not in r, "embedding must be opt-in, absent in playlist mode"


def t_values_bounded():
    for i, v in enumerate(r_emb["embedding"]):
        assert np.isfinite(v), f"dim {i} not finite"
        assert 0.0 <= v <= 1.0, f"dim {i} out of [0,1]: {v}"


# ------------------------------------------------------------
# similarity() helper
# ------------------------------------------------------------
def t_self_similarity_is_one():
    s = sonara.similarity(r_emb, r_emb)
    assert abs(s - 1.0) < 1e-6, f"self-similarity must be 1.0, got {s}"


def t_symmetric():
    s_ab = sonara.similarity(r_emb, r_emb2)
    s_ba = sonara.similarity(r_emb2, r_emb)
    assert abs(s_ab - s_ba) < 1e-6, f"similarity must be symmetric: {s_ab} vs {s_ba}"


def t_raw_lists_work():
    a = list(r_emb["embedding"])
    b = list(r_emb["embedding"])
    s = sonara.similarity(a, b)
    assert abs(s - 1.0) < 1e-6, f"raw-list self-similarity must be 1.0, got {s}"


def t_numpy_arrays_work():
    a = np.asarray(r_emb["embedding"], dtype=np.float32)
    s = sonara.similarity(a, a)
    assert abs(s - 1.0) < 1e-6, f"numpy self-similarity must be 1.0, got {s}"


def t_in_unit_interval():
    s = sonara.similarity(r_emb, sonara.analyze_signal(y_noise, sr=sr, features=EMB_FEATURES))
    assert 0.0 <= s <= 1.0, f"similarity must be in [0,1], got {s}"


def t_version_mismatch_raises():
    a = dict(r_emb)
    b = dict(r_emb)
    b["embedding_version"] = a["embedding_version"] + 1
    try:
        sonara.similarity(a, b)
    except ValueError:
        return
    raise AssertionError("expected ValueError on embedding_version mismatch")


def t_missing_embedding_raises():
    r = sonara.analyze_signal(y_kick120, sr=sr, mode="compact")
    try:
        sonara.similarity(r, r)
    except ValueError:
        return
    raise AssertionError("expected ValueError when embedding absent")


test("embedding present when features=['embedding']", t_present_when_requested)
test("embedding has correct length", t_right_length)
test("embedding absent in compact mode", t_absent_in_compact)
test("embedding absent in playlist mode (opt-in)", t_absent_in_playlist_by_default)
test("embedding values finite & in [0,1]", t_values_bounded)
test("similarity(r, r) == 1.0", t_self_similarity_is_one)
test("similarity is symmetric", t_symmetric)
test("similarity works on raw lists", t_raw_lists_work)
test("similarity works on numpy arrays", t_numpy_arrays_work)
test("similarity is in [0,1]", t_in_unit_interval)
test("version mismatch raises ValueError", t_version_mismatch_raises)
test("missing embedding raises ValueError", t_missing_embedding_raises)

print("=" * 70)
print(f"  {passed} passed, {failed} failed")
print("=" * 70)
if failed:
    for name, err in errors:
        print(f"  FAIL {name}: {err}")
    sys.exit(1)
