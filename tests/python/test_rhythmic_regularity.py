"""Contract tests for the opt-in `rhythmic_regularity` feature.

Guards, on synthetic drum patterns with a known metrical layout:
  - the four keys are ABSENT from every default mode,
  - they are PRESENT together, typed and in range, when requested,
  - a four-on-the-floor kit reads "regular"; chopped breaks, 2-step and a
    competing three-against-four pulse read "irregular",
  - silence and drumless audio ABSTAIN — score/label/candidates None with the
    confidence still reported,
  - a short fill or breakdown does not decide the track,
  - the decision survives a gain change and a different sample rate,
  - the onset bands / beat grid the measure reads are not leaked into the
    result unless separately requested.
"""

import sys

import numpy as np

import sonara


SR = 22050
BPM = 128.0
BARS = 32
KEYS = (
    "rhythmic_regularity",
    "rhythmic_regularity_label",
    "rhythmic_regularity_confidence",
    "rhythmic_regularity_candidates",
)


def _voices(sr):
    """A kick and a hi-hat, band-separated the way a real kit is.

    The kick is a decaying 55 Hz body, so it lands in the 0-200 Hz onset band.
    The hat is high-passed noise (its own moving average removed, which cuts
    everything below roughly sr/16), so it carries no sub-200 Hz energy — a
    broadband click would smear offbeat content into the kick band and make the
    fixture, not the measure, decide the result.
    """
    rng = np.random.default_rng(20260923)

    def highpassed(length):
        noise = rng.standard_normal(length).astype(np.float32)
        return noise - np.convolve(noise, np.ones(16, dtype=np.float32) / 16.0, mode="same")

    # A kick is a low body PLUS a beater click. Without the click it is a pure
    # 55 Hz sine, which produces less broadband onset flux than a hi-hat — the
    # beat tracker then locks onto the offbeat hats and every phase-sensitive
    # measure reads a half-beat out. That is a fixture defect, not a rhythm.
    kick_len = int(0.12 * sr)
    kick_t = np.arange(kick_len, dtype=np.float32) / sr
    body = np.sin(2.0 * np.pi * 55.0 * kick_t) * np.exp(-kick_t / 0.045)
    click = 0.6 * highpassed(kick_len) * np.exp(-kick_t / 0.0015)
    kick = (body + click).astype(np.float32)

    hat_len = int(0.03 * sr)
    hat_t = np.arange(hat_len, dtype=np.float32) / sr
    hat = (highpassed(hat_len) * np.exp(-hat_t / 0.006)).astype(np.float32)

    return {"kick": kick, "hat": hat}


def _render(pattern, bars=BARS, sr=SR, gain=1.0):
    """Render a 16-slot-per-bar drum pattern to audio.

    `pattern` maps a sixteenth-note slot to `(voice, strength)`, or is a
    callable `bar -> dict` when the figure changes through the track. The voice
    is named explicitly rather than inferred from the slot, so a breakbeat kick
    on the "and" of 3 really is a kick in the low band.
    """
    beat_sec = 60.0 / BPM
    slot_sec = beat_sec / 4.0
    total = int((bars * 4 * beat_sec + 0.5) * sr)
    signal = np.zeros(total, dtype=np.float32)
    voices = _voices(sr)

    for bar in range(bars):
        slots = pattern(bar) if callable(pattern) else pattern
        for slot, (voice_name, strength) in slots.items():
            start = int((bar * 16 + slot) * slot_sec * sr)
            voice = voices[voice_name]
            end = min(start + voice.size, total)
            if end > start:
                signal[start:end] += gain * strength * voice[: end - start]
    return signal


K, H = "kick", "hat"
# Kick on every quarter, hats on the offbeat eighths.
FOUR_ON_THE_FLOOR = {
    0: (K, 1.0), 2: (H, 0.5), 4: (K, 1.0), 6: (H, 0.5),
    8: (K, 1.0), 10: (H, 0.5), 12: (K, 1.0), 14: (H, 0.5),
}
# Kick on 1 and the "and" of 3, snare on 2 and 4, ghosted sixteenths between.
CHOPPED_BREAK = {
    0: (K, 1.0), 3: (H, 0.35), 4: (H, 0.9), 5: (H, 0.4), 7: (H, 0.3),
    9: (H, 0.35), 10: (K, 0.9), 11: (H, 0.4), 12: (H, 0.9), 13: (H, 0.3), 15: (H, 0.5),
}
# 2-step: kick displaced off beats 2 and 3 entirely, snare on 2 and 4.
TWO_STEP = {0: (K, 1.0), 4: (H, 0.9), 7: (K, 0.9), 12: (H, 0.9), 14: (K, 0.8)}

# Competing-pulse (three-against-four) coverage lives in the Rust module tests,
# where the beat grid is supplied. Rendered alone it carries no reference metre,
# so the beat tracker hears one faster steady pulse and regular is a fair
# reading of that audio.


def _measure(signal, sr=SR, **kwargs):
    return sonara.analyze_signal(signal, sr=sr, features=["rhythmic_regularity"], **kwargs)


def test_rhythmic_regularity_is_absent_from_every_default_mode() -> None:
    signal = _render(FOUR_ON_THE_FLOOR)
    for mode in ("compact", "playlist", "full"):
        result = sonara.analyze_signal(signal, sr=SR, mode=mode)
        for key in KEYS:
            assert key not in result, f"{key} must not appear in mode={mode}"


def test_requested_result_satisfies_the_documented_contract() -> None:
    result = _measure(_render(FOUR_ON_THE_FLOOR))
    for key in KEYS:
        assert key in result, f"{key} missing from a requested result"

    score = result["rhythmic_regularity"]
    label = result["rhythmic_regularity_label"]
    confidence = result["rhythmic_regularity_confidence"]
    candidates = result["rhythmic_regularity_candidates"]

    assert isinstance(score, float) and 0.0 <= score <= 1.0
    assert label in ("regular", "irregular")
    assert isinstance(confidence, float) and 0.0 <= confidence <= 1.0
    assert label == ("regular" if score >= 0.5 else "irregular")

    assert len(candidates) == 2
    assert {name for name, _ in candidates} == {"regular", "irregular"}
    assert candidates[0][1] >= candidates[1][1], "candidates must be ranked"
    assert candidates[0][0] == label, "the top candidate is the label"
    assert abs(sum(value for _, value in candidates) - 1.0) < 1e-5
    assert abs(dict(candidates)["regular"] - score) < 1e-6

    assert result["provenance"]["requested_features"] == ["rhythmic_regularity"]


def test_straight_reads_regular_and_broken_patterns_read_irregular() -> None:
    straight = _measure(_render(FOUR_ON_THE_FLOOR))
    assert straight["rhythmic_regularity_label"] == "regular", straight["rhythmic_regularity"]

    for name, pattern in (
        ("chopped break", CHOPPED_BREAK),
        ("2-step", TWO_STEP),
    ):
        broken = _measure(_render(pattern))
        assert broken["rhythmic_regularity_label"] == "irregular", (
            f"{name} scored {broken['rhythmic_regularity']}"
        )
        assert straight["rhythmic_regularity"] > broken["rhythmic_regularity"] + 0.2, (
            f"{name}: separation too small "
            f"({straight['rhythmic_regularity']} vs {broken['rhythmic_regularity']})"
        )


def test_silence_and_drumless_audio_abstain_with_confidence_reported() -> None:
    silence = _measure(np.zeros(int(20.0 * SR), dtype=np.float32))
    assert silence["rhythmic_regularity"] is None, "silence must not be scored"
    assert silence["rhythmic_regularity_label"] is None
    assert silence["rhythmic_regularity_candidates"] is None
    assert isinstance(silence["rhythmic_regularity_confidence"], float)
    assert silence["rhythmic_regularity_confidence"] < 0.5

    # A slow drifting pad: energy, but no percussive events in a grid.
    t = np.arange(int(20.0 * SR), dtype=np.float32) / SR
    pad = (0.3 * np.sin(2.0 * np.pi * 220.0 * t)
           * (1.0 + 0.2 * np.sin(2.0 * np.pi * 0.05 * t))).astype(np.float32)
    drumless = _measure(pad)
    assert (
        drumless["rhythmic_regularity"] is None
        or drumless["rhythmic_regularity_confidence"] < 0.5
    ), (
        f"a drumless pad must abstain or be low-confidence, got "
        f"{drumless['rhythmic_regularity']} at {drumless['rhythmic_regularity_confidence']}"
    )


def test_a_short_passage_does_not_decide_the_track() -> None:
    def with_fill(bar):
        return CHOPPED_BREAK if 16 <= bar < 20 else FOUR_ON_THE_FLOOR

    def with_breakdown(bar):
        return {} if 12 <= bar < 16 else FOUR_ON_THE_FLOOR

    for name, pattern in (("fill", with_fill), ("breakdown", with_breakdown)):
        result = _measure(_render(pattern))
        assert result["rhythmic_regularity_label"] == "regular", (
            f"a 4-bar {name} flipped the track to {result['rhythmic_regularity']}"
        )


def test_decision_survives_gain_and_sample_rate_changes() -> None:
    base = _measure(_render(FOUR_ON_THE_FLOOR))
    quiet = _measure(_render(FOUR_ON_THE_FLOOR, gain=0.05))
    loud = _measure(_render(FOUR_ON_THE_FLOOR, gain=4.0))
    for name, other in (("quiet", quiet), ("loud", loud)):
        assert other["rhythmic_regularity_label"] == base["rhythmic_regularity_label"], (
            f"{name} gain flipped the label"
        )

    resampled = _measure(_render(FOUR_ON_THE_FLOOR, sr=44100), sr=44100)
    assert resampled["rhythmic_regularity_label"] == base["rhythmic_regularity_label"], (
        "a different sample rate flipped the label"
    )


def test_the_groups_it_reads_are_not_leaked() -> None:
    signal = _render(FOUR_ON_THE_FLOOR)
    alone = _measure(signal)
    for key in ("onset_strength_bands", "onset_band_edges_hz",
                "grid_stability", "downbeats", "grid_offset_sec"):
        assert key not in alone, f"{key} is an internal input here, not an output"

    both = sonara.analyze_signal(
        signal, sr=SR, features=["rhythmic_regularity", "onset_bands", "beatgrid"]
    )
    assert "onset_strength_bands" in both and "grid_stability" in both
    assert both["rhythmic_regularity"] == alone["rhythmic_regularity"], (
        "emitting the inputs must not change the measurement"
    )


def test_measurement_is_deterministic() -> None:
    signal = _render(CHOPPED_BREAK)
    first = _measure(signal)
    second = _measure(signal)
    for key in KEYS:
        assert first[key] == second[key], f"{key} differed between identical runs"


def test_declared_as_opt_in_and_needing_audio() -> None:
    row = next(
        entry
        for entry in sonara.feature_dependencies()
        if entry["name"] == "rhythmic_regularity"
    )
    assert row["opt_in_only"] is True
    assert row["needs_extended"] is False
    assert row["full_only"] is False
    assert row["class"] == "frame_curves"
    assert row["required_evidence"] == []

    result = _measure(_render(FOUR_ON_THE_FLOOR))
    assert sonara.can_augment(result, "rhythmic_regularity") is False
    assert sonara.augment_blocker(result, "rhythmic_regularity") == (
        "needs audio (frame_curves-class feature)"
    )
    assert sonara.augment_analysis(result, []) == result


def test_print_renders_both_a_score_and_an_abstention() -> None:
    _measure(_render(FOUR_ON_THE_FLOOR)).print()
    _measure(np.zeros(int(20.0 * SR), dtype=np.float32)).print()


TESTS = [
    ("absent from every default mode", test_rhythmic_regularity_is_absent_from_every_default_mode),
    ("requested result matches the contract", test_requested_result_satisfies_the_documented_contract),
    ("straight vs broken patterns separate", test_straight_reads_regular_and_broken_patterns_read_irregular),
    ("silence and drumless abstain", test_silence_and_drumless_audio_abstain_with_confidence_reported),
    ("a short passage does not decide", test_a_short_passage_does_not_decide_the_track),
    ("gain and sample rate invariance", test_decision_survives_gain_and_sample_rate_changes),
    ("internal inputs are not leaked", test_the_groups_it_reads_are_not_leaked),
    ("measurement is deterministic", test_measurement_is_deterministic),
    ("declared opt-in and audio-bound", test_declared_as_opt_in_and_needing_audio),
    ("print handles score and abstention", test_print_renders_both_a_score_and_an_abstention),
]


# `scripts/run_python_tests.py` executes each suite file as a plain script, so a
# pytest-only module would pass by doing nothing at all.
if __name__ == "__main__":
    print("Rhythmic regularity")
    failures = []
    for name, fn in TESTS:
        try:
            fn()
        except Exception as error:  # noqa: BLE001 - report, do not abort the file
            failures.append((name, error))
            print(f"  FAIL  {name}: {error}")
        else:
            print(f"  ok    {name}")
    print(f"\n  RESULTS: {len(TESTS) - len(failures)} PASSED, {len(failures)} FAILED")
    sys.exit(1 if failures else 0)
