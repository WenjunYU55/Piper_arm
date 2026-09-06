#!/usr/bin/env bash
# Caller must source the freshly built Foxy workspace and choose an unused domain.
set -euo pipefail
if [ "$#" -ne 2 ]; then
  echo 'Usage: replay_bag.sh BAG_DIRECTORY NEW_OUTPUT_ROOT' >&2
  exit 2
fi
BUNDLESDF_BAG="$(realpath "$1")"
BUNDLESDF_OUTPUT="$(realpath -m "$2")"
test ! -e "$BUNDLESDF_OUTPUT" || { echo 'Refusing existing output root' >&2; exit 2; }
: "${ROS_DOMAIN_ID:?Set ROS_DOMAIN_ID to an unused isolated replay domain first}"
if [ "$ROS_DOMAIN_ID" = 42 ] || [ "$ROS_DOMAIN_ID" = 0 ]; then
  echo 'Refusing production/default ROS domain; select an unused replay domain.' >&2
  exit 2
fi
python3 - "$BUNDLESDF_BAG" <<'PY'
import pathlib, sys, yaml
root = pathlib.Path(sys.argv[1])
metadata = yaml.safe_load((root / 'metadata.yaml').read_text())['rosbag2_bagfile_information']
names = {record['topic_metadata']['name'] for record in metadata['topics_with_message_count']}
required = {'/camera/color/image_raw', '/camera/aligned_depth_to_color/image_raw', '/camera/color/camera_info',
            '/piper/sam2_target_mask', '/tf', '/piper/reconstruction_keyframe', '/piper/scan_session_history',
            '/piper/scan_execution_status', '/piper/supervised_workflow_status'}
missing = required - names
if missing:
    raise SystemExit('Incompatible bag; missing raw evidence or accepted-capture identity topics: ' + ', '.join(sorted(missing)))
PY
mkdir -p "$BUNDLESDF_OUTPUT"
BUNDLESDF_DEFAULT_ANGLE="$(python3 -c 'from piper_mobile_manipulation.visibility import DEFAULT_VISIBILITY_ANGLE_DEG; print(DEFAULT_VISIBILITY_ANGLE_DEG)')"
# Task identity must match the bag's recorded ReconstructionKeyframe.task_id.
python3 "$(ros2 pkg prefix piper_mobile_manipulation)/lib/piper_mobile_manipulation/reconstruction_recorder_node.py" --ros-args \
  -p output_root:="$BUNDLESDF_OUTPUT" \
  -p task_id:="${PIPER_REPLAY_TASK_ID:-}" \
  -p target_visibility_angle_tolerance_deg:="${PIPER_REPLAY_VISIBILITY_DEG:-$BUNDLESDF_DEFAULT_ANGLE}" \
  > "$BUNDLESDF_OUTPUT/recorder.log" 2>&1 &
BUNDLESDF_RECORDER_PID=$!
finish_recorder() {
  kill -INT "$BUNDLESDF_RECORDER_PID" 2>/dev/null || true
  wait "$BUNDLESDF_RECORDER_PID" || true
}
trap finish_recorder EXIT INT TERM
sleep 3
# Foxy has no --clock option. Recorder admission and TF use original message
# stamps; replay scheduling and shutdown use monotonic time. Playback launches
# no driver, mission engine, planner or command publisher.
ros2 bag play "$BUNDLESDF_BAG" --rate 1.0 --topics \
  /camera/color/image_raw /camera/aligned_depth_to_color/image_raw \
  /camera/color/camera_info /piper/sam2_target_mask /tf /tf_static \
  /piper/reconstruction_keyframe /piper/scan_session_history \
  /piper/scan_execution_status /piper/supervised_workflow_status
sleep 4
