"""
Application support for the native PiPER operator GUI.

The package deliberately contains no autonomous robot workflow.  Production
mission sequencing belongs to the ``RunTargetScan`` action server.
"""

from .view_model import MissionViewModel

__all__ = ["MissionViewModel"]
