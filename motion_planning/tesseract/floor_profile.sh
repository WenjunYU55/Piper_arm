#!/usr/bin/env bash

# The PiPER/L515/Bunker geometry is invariant. The profile chooses only the
# support-plane height encoded in the selected, hash-bound manifest.
FLOOR_PROFILE="${PIPER_FLOOR_PROFILE:-tabletop}"
case "$FLOOR_PROFILE" in
  tabletop)
    COLLISION_MANIFEST_NAME="collision_model.yaml"
    COLLISION_URDF_NAME="piper_planning.urdf"
    ;;
  ground)
    COLLISION_MANIFEST_NAME="collision_model_ground.yaml"
    COLLISION_URDF_NAME="piper_planning_ground.urdf"
    ;;
  *)
    echo "PIPER_FLOOR_PROFILE must be exactly tabletop or ground." >&2
    exit 2
    ;;
esac
COLLISION_SRDF_NAME="piper_bunker.srdf"
