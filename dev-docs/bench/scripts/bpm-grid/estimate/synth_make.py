"""Synthetic ground-truth audio for the estimate, written only under estimate/synthetic/.

breaks     the exact pattern of ../synthetic_truth.py (kick 1&3, snare 2&4, 8th hats,
           crash on bar 1 every 8 bars), same seed; 5 tempi x 3 sample rates
four       straight house: kick every beat, clap 2&4, open hat on every offbeat,
           soft 16th closed hats, crash every 8 bars; 10 tempi
four_bass  straight techno with an offbeat sub-bass (low-band energy between kicks),
           offbeat hats, clap 2&4; 4 tempi
dnb        two-step at full DnB tempo: kick on 1 and 3-and, snare 2&4, 16th hats; 4 tempi
Truth: every quarter note is a beat; beat 1 of each bar is a downbeat.
"""

from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bpm_grid_paths as P  # noqa: E402

OUT = P.ESTIMATE / "synthetic"
DURATION, FIRST_BEAT = 90.0, 0.5


def kick(sr):
    t = np.arange(int(0.25 * sr)) / sr
    f = 50 + 70 * np.exp(-t * 30)
    return 0.9 * np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-t * 9)


def snare(sr, rng):
    t = np.arange(int(0.18 * sr)) / sr
    return 0.4 * np.sin(2 * np.pi * 190 * t) * np.exp(-t * 25) + \
        rng.standard_normal(len(t)) * np.exp(-t * 22) * 0.5


def hat(sr, rng, amp=0.15, decay=90, dur=0.05):
    t = np.arange(int(dur * sr)) / sr
    n = np.diff(rng.standard_normal(len(t)), prepend=0)
    return amp * n * np.exp(-t * decay)


def crash(sr, rng):
    t = np.arange(int(1.2 * sr)) / sr
    return 0.25 * np.diff(rng.standard_normal(len(t)), prepend=0) * np.exp(-t * 3)


def clap(sr, rng):
    t = np.arange(int(0.15 * sr)) / sr
    n = rng.standard_normal(len(t))
    n = n - np.convolve(n, np.ones(8) / 8, mode="same")   # crude band emphasis
    env = np.exp(-t * 30) * (1 + 0.6 * np.sin(2 * np.pi * 90 * t) ** 2)
    return 0.45 * n * env


def sub(sr):
    t = np.arange(int(0.2 * sr)) / sr
    return 0.6 * np.sin(2 * np.pi * 55 * t) * np.exp(-t * 12) * np.minimum(1, t * 400)


def render(pattern, bpm, sr):
    rng = np.random.default_rng(7)
    y = np.zeros(int(DURATION * sr) + sr)
    period = 60.0 / bpm
    beats, downbeats = [], []

    def put(ev, start):
        i = int(round(start * sr))
        if i < len(y):
            y[i:i + len(ev)] += ev[: len(y) - i]

    k = 0
    while True:
        t = FIRST_BEAT + k * period
        if t > DURATION - 1:
            break
        beats.append(t)
        pos, bar = k % 4, k // 4
        if pos == 0:
            downbeats.append(t)
        if pattern == "breaks":                      # identical to ../synthetic_truth.py
            events = [kick(sr) if pos in (0, 2) else snare(sr, rng)]
            if pos == 0 and bar % 8 == 0:
                events.append(crash(sr, rng))
            for ev, st in [(e, t) for e in events] + [(hat(sr, rng), t), (hat(sr, rng), t + period / 2)]:
                put(ev, st)
        elif pattern == "four":
            put(kick(sr), t)
            if pos in (1, 3):
                put(clap(sr, rng), t)
            if pos == 0 and bar % 8 == 0:
                put(crash(sr, rng), t)
            put(hat(sr, rng, 0.22, 25, 0.15), t + period / 2)            # open hat offbeat
            for q in (0.25, 0.75):
                put(hat(sr, rng, 0.06), t + q * period)                  # soft 16ths
        elif pattern == "four_bass":
            put(kick(sr), t)
            put(sub(sr), t + period / 2)                                   # offbeat bass
            if pos in (1, 3):
                put(clap(sr, rng), t)
            put(hat(sr, rng, 0.2, 40, 0.1), t + period / 2)
        elif pattern == "dnb":
            if pos == 0:
                put(kick(sr), t)
            if pos == 2:
                put(kick(sr), t + period / 2)                              # kick on 3-and
            if pos in (1, 3):
                put(snare(sr, rng), t)
            for q in (0.0, 0.25, 0.5, 0.75):
                put(hat(sr, rng, 0.08), t + q * period)
        k += 1
    y = y[: int(DURATION * sr)]
    return y / np.max(np.abs(y)) * 0.8, beats, downbeats


SETS = [("breaks", t, sr) for t in (120.0, 125.0, 128.0, 131.5, 140.0) for sr in (22050, 44100, 48000)] + \
       [("four", t, 22050) for t in (118.0, 120.0, 122.0, 124.0, 125.0, 126.0, 128.0, 130.0, 132.0, 135.0)] + \
       [("four", t, 44100) for t in (124.0, 128.0)] + \
       [("four_bass", t, 22050) for t in (124.0, 128.0, 132.0, 136.0)] + \
       [("dnb", t, 22050) for t in (170.0, 172.0, 174.0, 176.0)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    truth = {}
    for pattern, bpm, sr in SETS:
        y, beats, downbeats = render(pattern, bpm, sr)
        name = f"{pattern}_{bpm:g}bpm_{sr}.wav"
        with wave.open(str(OUT / name), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes((y * 32767).astype("<i2").tobytes())
        truth[name] = {"pattern": pattern, "bpm": bpm, "sr": sr, "beats": beats, "downbeats": downbeats}
    (OUT / "truth.json").write_text(json.dumps(truth), encoding="utf-8")
    print(f"wrote {len(truth)} files to {OUT}")


if __name__ == "__main__":
    main()
