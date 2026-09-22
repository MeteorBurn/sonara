import ast
from pathlib import Path

import numpy as np

import sonara


SR = 22050
HOP_LENGTH = 512


def tone_bursts() -> np.ndarray:
    signal = np.zeros(SR * 2, dtype=np.float32)
    for start_sec, duration_sec, frequency in (
        (0.25, 0.12, 120.0),
        (1.25, 0.12, 5000.0),
    ):
        start = int(start_sec * SR)
        length = int(duration_sec * SR)
        time = np.arange(length, dtype=np.float32) / SR
        window = np.sin(np.pi * np.arange(length, dtype=np.float32) / length) ** 2
        signal[start : start + length] += np.sin(2.0 * np.pi * frequency * time) * window
    return signal


def test_standalone_and_fused_analysis_are_frame_aligned() -> None:
    signal = tone_bursts()
    standalone, edges = sonara.onset_strength_bands(signal, sr=SR, hop_length=HOP_LENGTH)
    result = sonara.analyze_signal(signal, sr=SR, features=["onset_bands"])

    assert edges == [0.0, 200.0, 800.0, 3200.0, 8000.0, 11025.0]
    np.testing.assert_allclose(result["onset_band_edges_hz"], edges, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(result["onset_strength_bands"], standalone, rtol=1e-6, atol=1e-6)
    assert standalone.shape[0] == len(edges) - 1
    assert np.all(np.isfinite(standalone))
    assert np.all(standalone >= 0.0)


def test_onset_bands_are_opt_in_and_round_trip_through_augment() -> None:
    signal = tone_bursts()
    for mode in ("compact", "playlist", "full"):
        result = sonara.analyze_signal(signal, sr=SR, mode=mode)
        assert "onset_band_edges_hz" not in result
        assert "onset_strength_bands" not in result

    requested = sonara.analyze_signal(signal, sr=SR, features=["onset_bands"])
    assert sonara.augment_analysis(requested, []) == requested
    assert sonara.can_augment(requested, "onset_bands") is False
    assert sonara.augment_blocker(requested, "onset_bands") == (
        "needs audio (frame_curves-class feature)"
    )


def test_custom_edges_validate_mel_band_coverage() -> None:
    signal = tone_bursts()
    envelopes, edges = sonara.onset_strength_bands(
        signal,
        sr=SR,
        hop_length=HOP_LENGTH,
        band_edges_hz=[0.0, 500.0, 4000.0, 11025.0],
    )
    assert edges == [0.0, 500.0, 4000.0, 11025.0]
    assert envelopes.shape[0] == 3

    try:
        sonara.onset_strength_bands(
            signal,
            sr=SR,
            hop_length=HOP_LENGTH,
            band_edges_hz=[0.0, 1.0, 11025.0],
        )
    except ValueError as error:
        assert "band_edges_hz" in str(error)
    else:
        raise AssertionError("empty mel-band coverage must be rejected")


def test_stub_declares_float32_onset_band_arrays() -> None:
    package_file = sonara.__file__
    assert package_file is not None
    stub_path = Path(package_file).with_name("__init__.pyi")
    tree = ast.parse(stub_path.read_text(encoding="utf-8"), filename=str(stub_path))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "onset_strength_bands"
    )

    input_annotation_node = function.args.args[0].annotation
    return_annotation_node = function.returns
    assert input_annotation_node is not None
    assert return_annotation_node is not None
    input_annotation = ast.unparse(input_annotation_node)
    return_annotation = ast.unparse(return_annotation_node)
    assert input_annotation == "NDArray[np.float32]"
    assert return_annotation == "Tuple[NDArray[np.float32], List[float]]"
