"""Tests for the backend-neutral planner domain boundary."""

from dataclasses import FrozenInstanceError

import pytest

from piper_mobile_manipulation.planning.backend import (
    MotionPlanRequest,
    PlannerBackend,
    PlannerSelection,
    parse_planner_backend,
)


def test_supported_planner_backends_parse_canonically():
    assert parse_planner_backend('tesseract') is PlannerBackend.TESSERACT
    assert parse_planner_backend(' CUROBO ') is PlannerBackend.CUROBO


def test_unknown_planner_backend_fails_closed():
    with pytest.raises(ValueError, match='exactly tesseract or curobo'):
        parse_planner_backend('automatic')


def test_planner_selection_and_request_are_frozen():
    selection = PlannerSelection(PlannerBackend.TESSERACT)
    request = MotionPlanRequest('mission-1', 'MULTIVIEW_SCAN')
    with pytest.raises(FrozenInstanceError):
        selection.backend = PlannerBackend.CUROBO
    with pytest.raises(FrozenInstanceError):
        request.plan_kind = 'RETURN_HOME'
