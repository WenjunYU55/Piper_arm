"""Shared constants and errors for the private schema-v5 contract."""

import re


SCHEMA_VERSION = 5
MAX_FINAL_AIM_OFFSET_DEG = 5.0
JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
TIMING_POLICY = 'tesseract_stream_v3'
COMMAND_RATE_HZ = 20.0
MOVEJ_NOMINAL_VELOCITY_RAD_S = (5.0, 5.0, 5.0, 5.0, 5.0, 3.0)
MAX_PROTOCOL_VELOCITY_RAD_S = 3.0
MAX_PROTOCOL_ACCELERATION_RAD_S2 = 5.0
MAX_BOOTSTRAP_START_LIMIT_TOLERANCE_RAD = 0.04
MAX_CONFIGURED_HOME_START_LIMIT_TOLERANCE_RAD = 0.3
PLAN_KINDS = ('MULTIVIEW_SCAN', 'ROUGH_ACQUISITION', 'RETURN_HOME')
PROVENANCE_SOURCES = ('tracked_target', 'rough_coordinate', 'configured_home')
SCENE_OBSERVATION_MODES = ('perception_snapshot', 'bootstrap_static')
SAFE_ID = re.compile(r'^[a-f0-9]{16,64}$')
SOURCE_REQUEST_ID = re.compile(r'^[A-Za-z0-9_.:-]{8,128}$')
MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_CANDIDATE_VIEWS = 100
MAX_OBSTACLES = 256
MAX_CAPTURE_VIEWPOINTS = 13
MAX_SEGMENTS = MAX_CAPTURE_VIEWPOINTS + 1
MAX_POINTS_PER_SEGMENT = 60000
QUEUE_NAMES = ('requests', 'processing', 'responses', 'failed')
HEALTH_FILENAME = 'worker_health.json'
MAX_HEALTH_BYTES = 16 * 1024


class ContractError(ValueError):
    """Raised when untrusted planning data violates the spool contract."""
