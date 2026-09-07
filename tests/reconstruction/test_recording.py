"""Deterministic admission, persistence, conversion and bounded-writer regressions."""

from dataclasses import replace
import math
from pathlib import Path
import sys
import threading
import time

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "piper_ros_foxy/src/piper_mobile_manipulation"))
sys.path.insert(0, str(ROOT / "tools/bundlesdf"))

from piper_mobile_manipulation.visibility import (
    angle_degrees,
    inside_cone,
    validate_transform,
    visibility_tolerance,
)  # noqa: E402
from piper_mobile_manipulation.reconstruction_recording.core import (
    Frame,
    RecorderConfig,
    RecordingSession,
    mask_metrics,
)  # noqa: E402
from dataset import convert, validate  # noqa: E402
from distortion import diagnose  # noqa: E402


def example(stamp=1_000_000_000, source="first_capture"):
    checker = (np.indices((120, 160)).sum(axis=0) % 2 * 255).astype(np.uint8)
    rgb = np.repeat(checker[:, :, None], 3, axis=2)
    mask = np.zeros((120, 160), np.uint8)
    mask[45:75, 65:95] = 255
    return Frame(
        rgb,
        np.full(mask.shape, 500, np.uint16),
        mask,
        np.array([[200.0, 0, 80], [0, 200.0, 60], [0, 0, 1.0]]),
        np.eye(4),
        stamp,
        stamp,
        stamp,
        0.0,
        "16UC1",
        "base_link",
        "camera_color_optical_frame",
        "session",
        stamp,
        source=source,
        accepted_view=1 if source == "first_capture" else 0,
    )


def session(tmp_path, **kwargs):
    return RecordingSession(tmp_path, "session", [0, 0, 0.5], RecorderConfig(**kwargs))


def wait_written(recorder, count):
    end = time.monotonic() + 3
    while (
        len(recorder.frames) < count and not recorder.error and time.monotonic() < end
    ):
        time.sleep(0.005)
    assert not recorder.error
    assert len(recorder.frames) == count


@pytest.mark.parametrize("limit", [5.0, 10.0, 20.0])
@pytest.mark.parametrize("offset,expected", [(-1e-4, True), (0.0, True), (1e-4, False)])
def test_angle_boundaries(limit, offset, expected):
    radians = math.radians(limit + offset)
    measured = angle_degrees([0, 0, 1], [math.sin(radians), 0, math.cos(radians)])
    assert inside_cone(measured, limit) == expected


@pytest.mark.parametrize(
    "vector", [[0, 0, 0], [0, 0, -1], [1, 0, 0], [np.nan, 0, 1], [np.inf, 1, 1], [1, 2]]
)
def test_invalid_vectors(vector):
    with pytest.raises(ValueError):
        angle_degrees([0, 0, 1], vector)


@pytest.mark.parametrize("angle", [0, -1, 90, 91, np.nan, np.inf])
def test_invalid_parameters(angle):
    with pytest.raises(ValueError):
        visibility_tolerance(angle)
    with pytest.raises(ValueError):
        RecorderConfig(target_visibility_angle_tolerance_deg=angle)


@pytest.mark.parametrize(
    "change",
    [
        dict(sample_rate_hz=4),
        dict(sample_rate_hz=11),
        dict(writer_queue_size=0),
        dict(edge_margin_px=-1),
        dict(minimum_valid_depth_ratio=0),
        dict(recorder_visibility_angle_tolerance_deg=21),
    ],
)
def test_invalid_config(change):
    with pytest.raises(ValueError):
        RecorderConfig(**change)


def test_optical_convention_and_transform():
    assert angle_degrees([0, 0, 1], [0, 0, 2]) == 0
    assert angle_degrees([1, 0, 0], [2, 0, 0]) == 0
    matrix = np.eye(4)
    matrix[:3, 3] = [1, 2, 3]
    assert np.array_equal(validate_transform(matrix) @ [0, 0, 1, 1], [1, 2, 4, 1])
    for mutation in [
        np.diag([1, 1, -1, 1]),
        np.zeros((4, 4)),
        np.eye(3),
        np.full((4, 4), np.nan),
    ]:
        with pytest.raises(ValueError):
            validate_transform(mutation)


@pytest.mark.parametrize(
    "reason,change",
    [
        ("rgb_depth_unsynchronised", dict(depth_stamp_ns=900_000_000)),
        ("rgb_depth_unsynchronised", dict(depth_stamp_ns=1_011_000_000)),
        ("stale_mask", dict(mask_stamp_ns=900_000_000)),
        ("stale_mask", dict(mask_received_age_sec=1.1)),
        ("missing_mask", dict(mask=None)),
        ("wrong_target_or_mask_source", dict(target_id="another-target")),
        ("wrong_target_or_mask_source", dict(mask_source="rectangle_roi")),
        ("missing_tf", dict(tf_stamp_ns=0)),
        ("invalid_intrinsics", dict(K=np.zeros((3, 3)))),
        ("resolution_mismatch", dict(depth=np.zeros((3, 3), np.uint16))),
        ("invalid_depth_encoding", dict(depth_encoding="8UC1")),
        ("insufficient_depth", dict(depth=np.zeros((120, 160), np.uint16))),
        (
            "nonfinite_depth",
            dict(depth=np.full((120, 160), np.nan, np.float32), depth_encoding="32FC1"),
        ),
        ("empty_or_tiny_mask", dict(mask=np.zeros((120, 160), np.uint8))),
        ("excessive_blur", dict(rgb=np.zeros((120, 160, 3), np.uint8))),
    ],
)
def test_rejection_counters(tmp_path, reason, change):
    recorder = session(tmp_path)
    assert not recorder.submit(replace(example(), **change))
    assert recorder.rejected[reason] == 1
    recorder.close()
    assert not recorder.manifest()["complete"]


@pytest.mark.parametrize("edge", ["left", "right", "top", "bottom"])
def test_each_edge(tmp_path, edge):
    frame = example()
    if edge == "left":
        frame.mask[60, 40] = 255
    if edge == "right":
        frame.mask[60, 119] = 255
    if edge == "top":
        frame.mask[40, 80] = 255
    if edge == "bottom":
        frame.mask[79, 80] = 255
    recorder = session(tmp_path)
    assert not recorder.submit(frame)
    assert recorder.rejected["target_clipped"] == 1
    recorder.close()


def test_blur_metric_is_target_interior():
    frame = example()
    _, _, sharp = mask_metrics(frame.mask, frame.rgb)
    blurred = cv2.GaussianBlur(frame.rgb, (7, 7), 2)
    _, _, soft = mask_metrics(frame.mask, blurred)
    assert sharp > soft * 10
    assert sharp > 20


def test_sequence_convert_and_validate(tmp_path):
    recorder = session(tmp_path)
    for i in range(6):
        assert recorder.submit(
            example(
                1_000_000_000 + i * 200_000_000, "first_capture" if i == 0 else "moving"
            )
        )
        wait_written(recorder, i + 1)
    recorder.mission_capture_count = 1
    assert recorder.close("complete")
    assert recorder.final_path.is_dir()
    assert not recorder.path.exists()
    report, manifest = validate(recorder.final_path)
    assert report["frames"] == 6
    assert manifest["mission_capture_count"] == 1
    out = tmp_path / "converted"
    converted = convert(recorder.final_path, out)
    assert converted["converted_frames"] == 6
    assert np.all(cv2.imread(str(out / "depth/00000000.png"), -1) == 500)
    assert np.array_equal(np.loadtxt(out / "cam_K.txt"), example().K)
    with pytest.raises(FileExistsError):
        convert(recorder.final_path, out)
    (recorder.final_path / "00000001/mask.png").write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        validate(recorder.final_path)


def test_distorted_raw_session_rectifies_only_offline(tmp_path):
    recorder = session(tmp_path)
    frame = replace(example(), distortion_model="plumb_bob",
                    distortion_coefficients=(0.169, -0.514, -0.0021, 0.0013, 0.475))
    # Distinct depth values make interpolated/non-nearest resampling detectable.
    frame.depth[:, 80:] = 700
    assert recorder.submit(frame)
    assert recorder.close("complete")
    raw_depth = recorder.final_path / "00000000/depth.npy"
    before = raw_depth.read_bytes()
    output = tmp_path / "rectified"
    convert(recorder.final_path, output, minimum_frames=1)
    assert raw_depth.read_bytes() == before
    assert np.array_equal(np.load(raw_depth), frame.depth)
    K = frame.K
    maps = cv2.initUndistortRectifyMap(
        K, np.asarray(frame.distortion_coefficients), np.eye(3), K,
        (160, 120), cv2.CV_32FC1)
    depth = cv2.imread(str(output / "depth/00000000.png"), -1)
    mask = cv2.imread(str(output / "masks/00000000.png"), -1)
    assert np.array_equal(depth, cv2.remap(frame.depth, *maps, cv2.INTER_NEAREST))
    assert np.array_equal(mask, cv2.remap(frame.mask, *maps, cv2.INTER_NEAREST))
    assert set(np.unique(depth)) <= {0, 500, 700}
    assert set(np.unique(mask)) <= {0, 255}
    assert np.array_equal(np.loadtxt(output / "cam_K.txt"), K)
    _, manifest = validate(recorder.final_path, minimum_frames=1)
    assert manifest['frames'][0]['distortion_model'] == 'plumb_bob'
    report = diagnose(recorder.final_path, tmp_path / 'diagnostic', plane_roi=(5, 5, 50, 50))
    assert report['correction_applied']
    assert report['maximum_displacement_px'] > 0
    assert report['plane_with_calibration']['rms_mm'] < 1e-8
    assert (tmp_path / 'diagnostic/comparison.png').is_file()
    assert raw_depth.read_bytes() == before
    with pytest.raises(FileExistsError):
        diagnose(recorder.final_path, tmp_path / 'diagnostic')
    with pytest.raises(ValueError, match='ROI'):
        diagnose(recorder.final_path, tmp_path / 'bad-roi', plane_roi=(-1, 0, 10, 10))


@pytest.mark.parametrize('model,coefficients', [
    ('equidistant', (0.,) * 4), ('plumb_bob', (0.,) * 4),
    ('plumb_bob', (float('nan'),) * 5), ('', (1.,) * 5),
])
def test_unsupported_distortion_rejects(tmp_path, model, coefficients):
    recorder = session(tmp_path)
    assert not recorder.submit(replace(example(), distortion_model=model,
                                       distortion_coefficients=coefficients))
    assert recorder.rejected['invalid_intrinsics'] == 1
    recorder.close()


def test_distortion_change_rejects(tmp_path):
    recorder = session(tmp_path)
    first = replace(example(), distortion_model='plumb_bob',
                    distortion_coefficients=(0.,) * 5)
    assert recorder.submit(first)
    later = replace(first, rgb_stamp_ns=2_000_000_000, depth_stamp_ns=2_000_000_000,
                    mask_stamp_ns=2_000_000_000, tf_stamp_ns=2_000_000_000,
                    source='moving', distortion_coefficients=(0.1,) * 5)
    assert not recorder.submit(later)
    assert recorder.rejected['intrinsics_or_resolution_changed'] == 1
    recorder.close()


def test_float_depth_precision_and_partial(tmp_path):
    recorder = session(tmp_path)
    frame = example()
    frame.depth = np.full((120, 160), 0.50049, np.float32)
    frame.depth_encoding = "32FC1"
    assert recorder.submit(frame)
    recorder.close("cancelled")
    with pytest.raises(ValueError):
        validate(recorder.final_path, minimum_frames=1)
    validate(recorder.final_path, minimum_frames=1, allow_incomplete=True)
    assert np.array_equal(
        np.load(recorder.final_path / "00000000/depth.npy"), frame.depth
    )
    convert(
        recorder.final_path,
        tmp_path / "partial",
        allow_incomplete=True,
        minimum_frames=1,
    )
    assert np.all(cv2.imread(str(tmp_path / "partial/depth/00000000.png"), -1) == 500)


def test_sampling_duplicates_and_calibration_change(tmp_path):
    recorder = session(tmp_path)
    assert recorder.submit(example())
    assert not recorder.submit(example())
    assert not recorder.submit(example(1_010_000_000, "moving"))
    frame = example(1_200_000_000, "moving")
    frame.K[0, 0] += 1
    assert not recorder.submit(frame)
    assert dict(recorder.rejected) == dict(
        duplicate_or_out_of_order=1,
        sample_rate_throttled=1,
        intrinsics_or_resolution_changed=1,
    )
    recorder.close()


def test_bounded_queue_drop_does_not_wait(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = RecordingSession._persist_frame

    def slow(self, item):
        entered.set()
        assert release.wait(3)
        original(self, item)

    monkeypatch.setattr(RecordingSession, "_persist_frame", slow)
    recorder = session(tmp_path, writer_queue_size=1)
    assert recorder.submit(example())
    assert entered.wait(1)
    assert recorder.submit(example(1_200_000_000, "moving"))
    before = time.monotonic()
    assert not recorder.submit(example(1_400_000_000, "moving"))
    assert time.monotonic() - before < 0.1
    assert recorder.rejected["writer_queue_full"] == 1
    release.set()
    recorder.close("complete")
    assert recorder.manifest()["dropped"] == 1


def test_disk_full_and_unwritable(tmp_path, monkeypatch):
    def fail(*args):
        raise OSError(28, "simulated no space left on device")

    monkeypatch.setattr(RecordingSession, "_persist_frame", fail)
    recorder = session(tmp_path)
    recorder.submit(example())
    recorder.close()
    assert "no space" in recorder.error
    assert not recorder.manifest()["complete"]
    blocked = tmp_path / "file"
    blocked.write_text("not a directory")
    other = session(blocked)
    other.close()
    assert other.error


def test_mask_expansion_and_multi_mission_reset(tmp_path):
    first = session(tmp_path, edge_margin_px=0, blur_gate_enabled=False)
    assert first.submit(example())
    frame = example(1_200_000_000, "moving")
    frame.mask[10:110, 10:150] = 255
    assert not first.submit(frame)
    assert first.rejected["mask_expansion"] == 1
    first.close()
    second = session(tmp_path)
    assert second.submit(example())
    second.close()
    assert first.final_path != second.final_path
    assert not second.rejected


def test_rgb_video_is_not_a_dataset(tmp_path):
    with pytest.raises(ValueError, match="RGB video alone"):
        validate(tmp_path / "scan.mp4")
