import numpy as np
import pytest

from piper_tesseract_foxy.contract import ContractError
from piper_tesseract_foxy.worker import pass_through_blend_geometry


def point(values):
    return {
        'time_from_start_s': 0.0,
        'positions_rad': list(values),
        'velocities_rad_s': [0.0] * 6,
        'accelerations_rad_s2': [0.0] * 6,
    }


def test_pass_through_blend_rounds_corner_and_preserves_endpoints():
    source = [
        point([0.0] * 6),
        point([0.2, 0.0, 0.0, 0.0, 0.0, 0.0]),
        point([0.2, 0.2, 0.0, 0.0, 0.0, 0.0]),
    ]
    blended, evidence = pass_through_blend_geometry(source, 0.05)
    assert evidence['applied'] is True
    assert evidence['blended_corners'] == 1
    assert np.allclose(blended[0]['positions_rad'], source[0]['positions_rad'])
    assert np.allclose(blended[-1]['positions_rad'], source[-1]['positions_rad'])
    positions = np.asarray([item['positions_rad'] for item in blended])
    assert np.any((positions[:, 0] > 0.15) & (positions[:, 1] > 0.0))


def test_pass_through_blend_keeps_complete_reversal_as_cusp():
    source = [
        point([0.0] * 6),
        point([0.2, 0.0, 0.0, 0.0, 0.0, 0.0]),
        point([0.0] * 6),
    ]
    blended, evidence = pass_through_blend_geometry(source, 0.05)
    assert evidence['applied'] is False
    assert len(blended) == len(source)


def test_pass_through_blend_rejects_unbounded_fraction():
    with pytest.raises(ContractError):
        pass_through_blend_geometry(
            [point([0.0] * 6), point([0.1] * 6)], 0.05,
            blend_fraction=0.5)
