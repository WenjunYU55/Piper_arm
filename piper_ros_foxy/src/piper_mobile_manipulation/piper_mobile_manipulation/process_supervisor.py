"""Compatibility facade for infrastructure process supervision."""

from piper_mobile_manipulation.infrastructure.process_supervisor import (
    ProcessHandle,
    ProcessSpec,
    ProcessSupervisor,
    ShutdownReport,
)

__all__ = [
    'ProcessHandle',
    'ProcessSpec',
    'ProcessSupervisor',
    'ShutdownReport',
]
