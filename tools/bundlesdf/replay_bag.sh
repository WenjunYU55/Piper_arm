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
BUNDLESDF_ARCHIVED_ANGLE="$(python3 - "$BUNDLESDF_BAG" "$BUNDLESDF_DEFAULT_ANGLE" <<'PYANGLE'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1]).parent / 'recording.json'
print(json.loads(path.read_text()).get('target_visibility_angle_tolerance_deg', sys.argv[2])
      if path.is_file() else sys.argv[2])
PYANGLE
)"
BUNDLESDF_ARCHIVED_TASK="$(python3 - "$BUNDLESDF_BAG" <<'PYTASK'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1]).parent / 'recording.json'
print(json.loads(path.read_text()).get('task_id', '') if path.is_file() else '')
PYTASK
)"
BUNDLESDF_TASK_ARGS=()
BUNDLESDF_REPLAY_TASK="${PIPER_REPLAY_TASK_ID:-$BUNDLESDF_ARCHIVED_TASK}"
if [ -n "$BUNDLESDF_REPLAY_TASK" ]; then
  BUNDLESDF_TASK_ARGS=(-p "task_id:=$BUNDLESDF_REPLAY_TASK")
fi
BUNDLESDF_CONFIG_ARGS=()
for BUNDLESDF_CONFIG_NAME in scan_quality_params.yaml reconstruction_recording_params.yaml; do
  BUNDLESDF_CONFIG_PATH="$(dirname "$BUNDLESDF_BAG")/config/$BUNDLESDF_CONFIG_NAME"
  if [ -f "$BUNDLESDF_CONFIG_PATH" ]; then
    BUNDLESDF_CONFIG_ARGS+=(--params-file "$BUNDLESDF_CONFIG_PATH")
  fi
done
# Task identity must match the bag's recorded ReconstructionKeyframe.task_id.
python3 "$(ros2 pkg prefix piper_mobile_manipulation)/lib/piper_mobile_manipulation/reconstruction_recorder_node.py" --ros-args \
  -p output_root:="$BUNDLESDF_OUTPUT" \
  "${BUNDLESDF_TASK_ARGS[@]}" "${BUNDLESDF_CONFIG_ARGS[@]}" \
  -p target_visibility_angle_tolerance_deg:="${PIPER_REPLAY_VISIBILITY_DEG:-$BUNDLESDF_ARCHIVED_ANGLE}" \
  > "$BUNDLESDF_OUTPUT/recorder.log" 2>&1 &
BUNDLESDF_RECORDER_PID=$!
finish_recorder() {
  kill -INT "$BUNDLESDF_RECORDER_PID" 2>/dev/null || true
  wait "$BUNDLESDF_RECORDER_PID" || true
}
trap finish_recorder EXIT INT TERM
sleep 3
if ! kill -0 "$BUNDLESDF_RECORDER_PID" 2>/dev/null; then
  cat "$BUNDLESDF_OUTPUT/recorder.log" >&2
  exit 2
fi
# Foxy has no --clock option. Recorder admission and TF use original message
# stamps; replay scheduling and shutdown use monotonic time. Playback launches
# no driver, mission engine, planner or command publisher.
ros2 bag play "$BUNDLESDF_BAG" --rate 1.0 --topics \
  /camera/color/image_raw /camera/aligned_depth_to_color/image_raw \
  /camera/color/camera_info /piper/sam2_target_mask /tf /tf_static \
  /piper/reconstruction_keyframe /piper/scan_session_history \
  /piper/scan_execution_status /piper/supervised_workflow_status
sleep 4
if ! kill -0 "$BUNDLESDF_RECORDER_PID" 2>/dev/null; then
  cat "$BUNDLESDF_OUTPUT/recorder.log" >&2
  exit 2
fi
finish_recorder
trap - EXIT
python3 - "$BUNDLESDF_OUTPUT" <<'PYRESULT'
import json, pathlib, sys
manifests = [json.loads(path.read_text())
             for path in pathlib.Path(sys.argv[1]).glob('*/manifest.json')]
accepted = sum(item.get('accepted', 0) for item in manifests)
if not accepted:
    raise SystemExit('Replay produced no accepted frames; inspect recorder.log and session rejection counters')
print(json.dumps(dict(sessions=len(manifests), accepted_frames=accepted,
                      completed_sessions=sum(bool(item.get('complete')) for item in manifests))))
PYRESULT
