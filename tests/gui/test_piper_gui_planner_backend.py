"""GUI next-mission planner persistence and freeze tests."""

from dataclasses import FrozenInstanceError

import pytest

from piper_gui.planner_backend import (
    read_planner_backend,
    write_planner_backend,
)
from piper_gui.view_model import MissionViewModel, validate_mission_request


def test_gui_persists_each_supported_next_mission_backend(tmp_path):
    path = tmp_path / 'planner_backend.yaml'
    path.write_text('planner_backend: "tesseract"\n', encoding='utf-8')
    assert write_planner_backend(path, 'tesseract') == 'tesseract'
    assert read_planner_backend(path) == 'tesseract'
    assert write_planner_backend(path, 'curobo') == 'curobo'
    assert read_planner_backend(path) == 'curobo'


def test_invalid_persisted_backend_fails_closed(tmp_path):
    path = tmp_path / 'planner_backend.yaml'
    path.write_text('planner_backend: "automatic"\n', encoding='utf-8')
    with pytest.raises(ValueError, match='unsupported planner backend'):
        read_planner_backend(path)


def test_active_mission_request_backend_is_frozen():
    view_model = MissionViewModel()
    request = view_model.begin_submission(
        (0.4, 0.0, 0.12), 'cube', 'curobo')
    assert request.planner_backend == 'curobo'
    with pytest.raises(FrozenInstanceError):
        request.planner_backend = 'tesseract'


def test_unknown_gui_backend_is_rejected():
    with pytest.raises(ValueError, match='unsupported planner backend'):
        validate_mission_request((0.4, 0.0, 0.12), 'cube', 'fallback')
