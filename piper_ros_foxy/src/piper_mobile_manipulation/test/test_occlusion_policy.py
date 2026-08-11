from piper_mobile_manipulation.occlusion_policy import (
    OccluderEvidence,
    evidence_rejection,
    placement_rejection,
    select_action,
)


def evidence(**changes):
    values = dict(
        track_id='prop-1', object_id=1, label='marker', observation_count=2,
        confirmed_in_probe=True, target_overlap_ratio=0.20,
        closer_depth_ratio=0.15, predicted_surface_gain=0.12,
        predicted_unlocked_viewpoints=2, confidence=0.8, valid=True,
        uncertainty_m=0.01, size_xyz_m=(0.04, 0.02, 0.15),
    )
    values.update(changes)
    return OccluderEvidence(**values)


def test_contact_requires_two_view_depth_ordered_benefit():
    assert evidence_rejection(evidence()) == ''
    assert 'initial and probe' in evidence_rejection(
        evidence(observation_count=1, confirmed_in_probe=False))
    assert 'below 5 percent' in evidence_rejection(
        evidence(closer_depth_ratio=0.04))
    assert 'benefit' in evidence_rejection(evidence(
        predicted_surface_gain=0.09, predicted_unlocked_viewpoints=1))


def test_hand_is_terminal_and_unknown_semantics_never_authorize_contact():
    assert 'terminal' in evidence_rejection(evidence(label='hand'))
    assert 'qualified rigid' in evidence_rejection(evidence(label='obstacle'))


def test_pick_is_preferred_then_push_is_bounded():
    assert select_action(evidence(), True, True, True)['action'] == 'pick_and_place'
    pushed = select_action(evidence(size_xyz_m=(0.10, 0.02, 0.15)), False, True, False, 0)
    assert pushed['action'] == 'push'
    assert pushed['push_distance_m'] == 0.010
    assert select_action(evidence(), False, True, False, 2)['push_distance_m'] == 0.030
    assert not select_action(evidence(), False, True, False, 3)['valid']


def test_placement_respects_target_object_edge_and_swept_footprints():
    assert placement_rejection(
        (0.50, 0.20, 0.03), (0.25, -0.25, 0.03), [],
        (0.10, 0.70, -0.40, 0.40)) == ''
    assert '120mm' in placement_rejection(
        (0.30, -0.25, 0.03), (0.25, -0.25, 0.03), [],
        (0.10, 0.70, -0.40, 0.40))
    assert 'footprint' in placement_rejection(
        (0.50, 0.20, 0.03), (0.25, -0.25, 0.03), [],
        (0.10, 0.70, -0.40, 0.40), [(0.45, 0.55, 0.15, 0.25)])
