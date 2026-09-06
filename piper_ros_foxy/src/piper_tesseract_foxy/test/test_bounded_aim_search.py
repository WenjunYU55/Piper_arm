"""Request-owned cone directions preserve the existing capture aim gate."""
import math

import numpy as np
import pytest

from piper_tesseract_foxy.candidate_selection import (
    effective_final_aim_tolerance, exact_target_aim_candidates,
)


@pytest.mark.parametrize('tolerance', [1.0, 2.0, 5.0, 90.0])
@pytest.mark.parametrize('accepted', [0, 1])
def test_cone_is_bounded_and_first_capture_keeps_five_degree_ceiling(tolerance, accepted):
    limit = effective_final_aim_tolerance(tolerance, accepted)
    candidate = {'camera_position_m': [0, 0, 0], 'candidate_geometry': 'target_ray'}
    result = exact_target_aim_candidates(
        [candidate], [1, 0, 0], [1, 0, 0], limit,
        bounded_cone_search=True)[0]
    directions = result['fallback_look_directions']
    assert len(directions) == (0 if limit == 1 else 4)
    for direction in directions:
        assert np.linalg.norm(direction) == pytest.approx(1)
        angle = math.degrees(math.acos(np.clip(direction[0], -1, 1)))
        assert angle <= min(4, limit - 1) + 1e-9
    assert candidate == {'camera_position_m': [0, 0, 0], 'candidate_geometry': 'target_ray'}


def test_default_tesseract_and_non_ray_candidates_are_unchanged():
    candidate = {'camera_position_m': [0, 0, 0]}
    baseline = exact_target_aim_candidates([candidate], [1, 0, 0], [0, 1, 0], 5)
    assert baseline == exact_target_aim_candidates(
        [candidate], [1, 0, 0], [0, 1, 0], 5, bounded_cone_search=True)
    assert len(baseline[0]['fallback_look_directions']) == 1


def test_existing_fallback_is_preserved_and_cone_never_exceeds_five():
    candidate = {'camera_position_m': [0, 0, 0], 'candidate_geometry': 'target_ray'}
    old = exact_target_aim_candidates([candidate], [1, 0, 0], [1, 1, 1], 5)[0]
    new = exact_target_aim_candidates([candidate], [1, 0, 0], [1, 1, 1], 5,
                                      bounded_cone_search=True)[0]
    assert new['fallback_look_directions'][0] == old['fallback_look_directions'][0]
    assert len(new['fallback_look_directions']) == 5
