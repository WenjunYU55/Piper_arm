"""Shared repository setup for relocated development tests."""

from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(autouse=True)
def repository_working_directory(monkeypatch):
    """Preserve the root working directory assumed by the original tests."""
    monkeypatch.chdir(PROJECT_ROOT)
