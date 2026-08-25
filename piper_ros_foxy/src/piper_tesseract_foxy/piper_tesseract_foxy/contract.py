"""Compatibility façade for the versioned Tesseract planning contract."""

from piper_tesseract_foxy.contract_core import ContractError  # noqa: F401
from piper_tesseract_foxy.contract_hashing import (  # noqa: F401
    attach_digest,
    canonical_bytes,
    sha256_file,
    sha256_value,
    verify_digest,
)
from piper_tesseract_foxy.contract_request import validate_request  # noqa: F401
from piper_tesseract_foxy.contract_response import validate_response  # noqa: F401
from piper_tesseract_foxy.contract_spool import Spool  # noqa: F401
from piper_tesseract_foxy.contract_validation import (  # noqa: F401
    angular_separation_deg,
    COMMAND_RATE_HZ,
    finite_vector,
    HEALTH_FILENAME,
    JOINT_NAMES,
    MAX_BOOTSTRAP_START_LIMIT_TOLERANCE_RAD,
    MAX_CANDIDATE_VIEWS,
    MAX_CAPTURE_VIEWPOINTS,
    MAX_CONFIGURED_HOME_START_LIMIT_TOLERANCE_RAD,
    MAX_FILE_BYTES,
    MAX_FINAL_AIM_OFFSET_DEG,
    MAX_HEALTH_BYTES,
    MAX_OBSTACLES,
    MAX_POINTS_PER_SEGMENT,
    MAX_PROTOCOL_ACCELERATION_RAD_S2,
    MAX_PROTOCOL_VELOCITY_RAD_S,
    MAX_SEGMENTS,
    motion_limits_digest,
    MOVEJ_NOMINAL_VELOCITY_RAD_S,
    PLAN_KINDS,
    PROVENANCE_SOURCES,
    QUEUE_NAMES,
    require_sha256,
    SAFE_ID,
    SCENE_OBSERVATION_MODES,
    SCHEMA_VERSION,
    SOURCE_REQUEST_ID,
    target_ray_position_matches,
    TIMING_POLICY,
    trajectory_digest,
    validate_motion_limits,
    validate_plan_identity,
)
