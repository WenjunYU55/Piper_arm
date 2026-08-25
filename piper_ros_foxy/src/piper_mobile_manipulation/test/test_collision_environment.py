"""Tests for the invariant platform and next-mission floor selection."""

from pathlib import Path

import pytest

from piper_mobile_manipulation.collision_environment import (
    GROUND_FLOOR,
    TABLETOP_FLOOR,
    read_collision_environment,
    write_collision_environment,
)


def _selection(tmp_path, profile=TABLETOP_FLOOR):
    path = Path(tmp_path) / 'collision_environment.yaml'
    path.write_text(
        'schema_version: 1\nfloor_profile: "%s"  # retained\n' % profile,
        encoding='utf-8')
    return path


def test_gui_floor_selection_changes_only_existing_value(tmp_path):
    path = _selection(tmp_path)
    before = path.read_text(encoding='utf-8')

    selected = write_collision_environment(path, GROUND_FLOOR)

    assert selected.floor_profile == GROUND_FLOOR
    assert selected.floor_z_m == pytest.approx(-0.466)
    assert read_collision_environment(path) == selected
    assert path.read_text(encoding='utf-8') == before.replace(
        '"tabletop"', '"ground"')


def test_floor_selection_is_idempotent(tmp_path):
    path = _selection(tmp_path, GROUND_FLOOR)
    before = path.read_bytes()
    write_collision_environment(path, GROUND_FLOOR)
    assert path.read_bytes() == before


@pytest.mark.parametrize('profile', ['bunker', 'saved', 'unknown', ''])
def test_floor_selection_rejects_non_floor_profiles(tmp_path, profile):
    path = _selection(tmp_path)
    before = path.read_bytes()
    with pytest.raises(ValueError, match='tabletop or ground'):
        write_collision_environment(path, profile)
    assert path.read_bytes() == before


def test_floor_selection_fails_closed_on_duplicate_entry(tmp_path):
    path = _selection(tmp_path)
    path.write_text(
        'floor_profile: "tabletop"\nfloor_profile: "ground"\n',
        encoding='utf-8')
    with pytest.raises(ValueError, match='exactly one'):
        read_collision_environment(path)
