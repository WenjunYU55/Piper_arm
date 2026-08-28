#!/usr/bin/env python3
"""Compatibility launcher for the native GUI implementation.

The implementation lives in :mod:`piper_gui.native_app`.  Keep this facade
while existing operator commands and downstream imports migrate.
"""

from piper_gui.native_app import *  # noqa: F401,F403
from piper_gui.native_app import main


if __name__ == "__main__":
    main()
