"""Sonara: High-performance audio analysis library for music information retrieval."""

from sonara._sonara import *  # noqa: F401, F403
from sonara._sonara import __version__  # noqa: F401 — sourced from Cargo.toml
from sonara._sonara import (
    analyze_file as _analyze_file,
    analyze_signal as _analyze_signal,
    analyze_batch as _analyze_batch,
)
from sonara._result import TrackAnalysis
from sonara import display  # noqa: F401


def analyze_file(path, *, sr=22050, mode="compact", features=None, bpm_min=None, bpm_max=None):
    """Analyze an audio file and return a `TrackAnalysis` (dict subclass with `.print()`)."""
    return TrackAnalysis(_analyze_file(
        path, sr=sr, mode=mode, features=features, bpm_min=bpm_min, bpm_max=bpm_max,
    ))


def analyze_signal(y, *, sr=22050, mode="compact", features=None, bpm_min=None, bpm_max=None):
    """Analyze a signal array and return a `TrackAnalysis` (dict subclass with `.print()`)."""
    return TrackAnalysis(_analyze_signal(
        y, sr=sr, mode=mode, features=features, bpm_min=bpm_min, bpm_max=bpm_max,
    ))


def analyze_batch(paths, *, sr=22050, mode="compact", features=None, bpm_min=None, bpm_max=None):
    """Analyze a list of audio files in parallel; returns a `list[TrackAnalysis]`.

    Errors are isolated per file: the returned list has exactly one entry per
    input path, in the same order as ``paths``. A file that fails to decode does
    not abort the batch — instead its entry is a failure ``TrackAnalysis`` with
    ``path``, ``error`` (human-readable, includes container/codec and cause) and
    ``error_kind`` (a short stable category such as ``"decode"``, ``"io"`` or
    ``"unsupported_format"``). Use ``result.failed`` to distinguish them.
    """
    return [
        TrackAnalysis(r)
        for r in _analyze_batch(
            paths, sr=sr, mode=mode, features=features, bpm_min=bpm_min, bpm_max=bpm_max,
        )
    ]
