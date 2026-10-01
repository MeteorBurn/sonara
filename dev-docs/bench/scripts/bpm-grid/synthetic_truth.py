"""Ground-truth synthetic tracks with exactly known beats and downbeats.

Writes 16-bit WAVs to <dataset>/data/synthetic/ and truth.json. Pattern per 4/4 bar:
kick on beats 1 and 3, snare on 2 and 4, closed hat on every eighth; a crash
on beat 1 of every 8th bar. First downbeat at FIRST_BEAT seconds. Every event
starts exactly at its nominal time (sample-accurate at each sample rate).
"""

import json
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bpm_grid_paths as P  # noqa: E402

OUT = P.DATA / "synthetic"
TEMPOS = [120.0, 125.0, 128.0, 131.5, 140.0]
RATES = [22050, 44100, 48000]
DURATION = 90.0
FIRST_BEAT = 0.5


def kick(sr):
    t = np.arange(int(0.25 * sr)) / sr
    f = 50 + 70 * np.exp(-t * 30)
    return 0.9 * np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-t * 9)


def snare(sr, rng):
    t = np.arange(int(0.18 * sr)) / sr
    body = 0.4 * np.sin(2 * np.pi * 190 * t) * np.exp(-t * 25)
    noise = rng.standard_normal(len(t)) * np.exp(-t * 22) * 0.5
    return body + noise


def hat(sr, rng):
    t = np.arange(int(0.05 * sr)) / sr
    n = rng.standard_normal(len(t))
    n = np.diff(n, prepend=0)                         # crude high-pass
    return 0.15 * n * np.exp(-t * 90)


def crash(sr, rng):
    t = np.arange(int(1.2 * sr)) / sr
    n = np.diff(rng.standard_normal(len(t)), prepend=0)
    return 0.25 * n * np.exp(-t * 3)


def render(bpm, sr):
    rng = np.random.default_rng(7)
    y = np.zeros(int(DURATION * sr) + sr)
    period = 60.0 / bpm
    beats, downbeats = [], []
    k = 0
    while True:
        t = FIRST_BEAT + k * period
        if t > DURATION - 1:
            break
        beats.append(t)
        pos = k % 4
        bar = k // 4
        events = [kick(sr) if pos in (0, 2) else snare(sr, rng)]
        if pos == 0:
            downbeats.append(t)
            if bar % 8 == 0:
                events.append(crash(sr, rng))
        for ev, start in [(e, t) for e in events] + [(hat(sr, rng), t), (hat(sr, rng), t + period / 2)]:
            i = int(round(start * sr))
            y[i:i + len(ev)] += ev[: len(y) - i]
        k += 1
    y = y[: int(DURATION * sr)]
    y = y / np.max(np.abs(y)) * 0.8
    return y, beats, downbeats


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    truth = {}
    for bpm in TEMPOS:
        for sr in RATES:
            y, beats, downbeats = render(bpm, sr)
            name = f"breaks_{bpm:g}bpm_{sr}.wav"
            with wave.open(str(OUT / name), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sr)
                w.writeframes((y * 32767).astype("<i2").tobytes())
            truth[name] = {"bpm": bpm, "sr": sr, "beats": beats, "downbeats": downbeats}
    (OUT / "truth.json").write_text(json.dumps(truth), encoding="utf-8")
    print(f"wrote {len(truth)} files to {OUT}")


if __name__ == "__main__":
    main()
