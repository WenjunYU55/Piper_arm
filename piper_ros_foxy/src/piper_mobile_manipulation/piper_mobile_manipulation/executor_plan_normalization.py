"""Pure normalization of validated Tesseract paths for the PiPER SDK."""

import numpy as np


def sdk_command_path(path, velocities, accelerations, times, execution_mode,
                     direct_home=False):
    """Collapse a fully validated straight chord to one PiPER MoveJ goal."""
    command_path = [np.asarray(item, dtype=float).copy() for item in path[1:]]
    command_velocities = [
        np.asarray(item, dtype=float).copy() for item in velocities[1:]]
    command_accelerations = [
        np.asarray(item, dtype=float).copy() for item in accelerations[1:]]
    command_times = [float(item) for item in times[1:]]
    mode = str(execution_mode).strip().upper()
    if mode == 'DIRECT_MOVEJ' and not bool(direct_home):
        return (
            [np.asarray(path[-1], dtype=float).copy()],
            [np.zeros(6, dtype=float)],
            [np.zeros(6, dtype=float)],
            [float(times[-1])],
            False,
        )
    return (
        command_path, command_velocities, command_accelerations,
        command_times, bool(not direct_home and mode == 'TESSERACT_STREAM'))
