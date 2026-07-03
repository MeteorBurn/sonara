"""TrackAnalysis: dict subclass returned by analyze_file / analyze_signal / analyze_batch."""

from __future__ import annotations

from collections.abc import Callable


Row = tuple[str, str]


def _fmt_duration(sec: float) -> str:
    total = int(round(sec))
    m, s = divmod(total, 60)
    if m >= 60:
        h, m = divmod(m, 60)
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _fmt_confidence(value: object, confidence: object | None) -> str:
    if confidence is None:
        return str(value)
    return f"{value}  (conf {confidence:.2f})"


def _append_if_present(
    rows: list[Row],
    data: dict,
    key: str,
    label: str,
    formatter: Callable[[object], str] = str,
) -> None:
    if key in data:
        rows.append((label, formatter(data[key])))


def _build_rhythm_rows(data: dict) -> list[Row]:
    rows: list[Row] = []
    _append_if_present(rows, data, "bpm", "BPM", lambda v: f"{v:.1f}")
    _append_if_present(rows, data, "n_beats", "Beats", str)
    _append_if_present(rows, data, "onset_density", "Onset density", lambda v: f"{v:.2f}/sec")
    _append_if_present(rows, data, "tempo_variability", "Tempo variability", lambda v: f"{v:.3f}")
    if "time_signature" in data:
        rows.append((
            "Time signature",
            _fmt_confidence(data["time_signature"], data.get("time_signature_confidence")),
        ))
    return rows


def _build_tonal_rows(data: dict) -> list[Row]:
    rows: list[Row] = []
    if "key" in data:
        rows.append(("Key", _fmt_confidence(data["key"], data.get("key_confidence"))))
    _append_if_present(rows, data, "predominant_chord", "Predominant chord", str)
    _append_if_present(rows, data, "chord_change_rate", "Chord changes", lambda v: f"{v:.2f}/sec")
    _append_if_present(rows, data, "dissonance", "Dissonance", lambda v: f"{v:.3f}")
    return rows


def _build_perceptual_rows(data: dict) -> list[Row]:
    rows: list[Row] = []
    for key, label in (
        ("energy", "Energy"),
        ("danceability", "Danceability"),
        ("valence", "Valence"),
        ("acousticness", "Acousticness"),
        ("instrumentalness", "Instrumentalness"),
    ):
        _append_if_present(rows, data, key, label, lambda v: f"{v:.2f}")
    _append_if_present(rows, data, "loudness_lufs", "Loudness", lambda v: f"{v:.1f} LUFS")
    _append_if_present(rows, data, "dynamic_range_db", "Dynamic range", lambda v: f"{v:.1f} dB")
    return rows


def _build_spectral_rows(data: dict) -> list[Row]:
    rows: list[Row] = []
    _append_if_present(rows, data, "spectral_centroid_mean", "Centroid", lambda v: f"{v:.0f} Hz")
    _append_if_present(rows, data, "spectral_bandwidth_mean", "Bandwidth", lambda v: f"{v:.0f} Hz")
    _append_if_present(rows, data, "spectral_rolloff_mean", "Rolloff", lambda v: f"{v:.0f} Hz")
    _append_if_present(rows, data, "spectral_flatness_mean", "Flatness", lambda v: f"{v:.3f}")
    _append_if_present(rows, data, "zero_crossing_rate", "ZCR", lambda v: f"{v:.3f}")
    return rows


def _append_section(lines: list[str], name: str, rows: list[Row]) -> None:
    if not rows:
        return

    lines.append("")
    lines.append(f"  {name}")
    width = max(len(label) for label, _ in rows)
    for label, value in rows:
        lines.append(f"    {label:<{width}}  {value}")


class TrackAnalysis(dict):
    """Result of `sonara.analyze_*`. Behaves as a dict; adds `.print()` for a human-readable summary."""

    def __repr__(self) -> str:
        parts = []
        if "bpm" in self:
            parts.append(f"{self['bpm']:.0f} BPM")
        if "key" in self:
            parts.append(str(self["key"]))
        if "energy" in self:
            parts.append(f"energy {self['energy']:.2f}")
        if "duration_sec" in self:
            parts.append(_fmt_duration(self["duration_sec"]))
        return f"<TrackAnalysis {' | '.join(parts)}>" if parts else "<TrackAnalysis>"

    def print(self) -> None:
        """Print a mode-aware summary, including only fields that were computed."""
        lines: list[str] = []
        if "duration_sec" in self:
            lines.append(f"TrackAnalysis  ({_fmt_duration(self['duration_sec'])})")
        else:
            lines.append("TrackAnalysis")

        for name, rows in (
            ("Rhythm", _build_rhythm_rows(self)),
            ("Tonal", _build_tonal_rows(self)),
            ("Perceptual", _build_perceptual_rows(self)),
            ("Spectral", _build_spectral_rows(self)),
        ):
            _append_section(lines, name, rows)

        print("\n".join(lines))
