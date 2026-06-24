#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
failures=0

pass() { printf 'PASS  %s\n' "$1"; }
fail() { printf 'FAIL  %s\n' "$1" >&2; failures=$((failures + 1)); }

if [ -r /etc/os-release ] && . /etc/os-release && [ "${ID:-}" = ubuntu ] && [ "${VERSION_ID:-}" = 20.04 ]; then
  pass "Ubuntu 20.04"
else
  fail "Ubuntu 20.04 is required"
fi

if [ -f /opt/ros/foxy/setup.bash ]; then
  # shellcheck disable=SC1091
  set +u
  source /opt/ros/foxy/setup.bash
  set -u
  pass "ROS 2 Foxy base environment"
else
  fail "/opt/ros/foxy/setup.bash"
fi

if [ -f "$ROOT/piper_ros_foxy/install/setup.bash" ]; then
  # shellcheck disable=SC1091
  set +u
  source "$ROOT/piper_ros_foxy/install/setup.bash"
  set -u
  pass "PiPER workspace overlay"
else
  fail "PiPER workspace has not been built"
fi

if [ -f "$ROOT/L515_camera/realsense_ws/install/setup.bash" ]; then
  # shellcheck disable=SC1091
  set +u
  source "$ROOT/L515_camera/realsense_ws/install/setup.bash"
  set -u
  pass "RealSense workspace overlay"
else
  fail "RealSense workspace has not been built"
fi

for command_name in colcon git ip ros2; do
  if command -v "$command_name" >/dev/null 2>&1; then
    pass "command: $command_name"
  else
    fail "command: $command_name"
  fi
done

python3 - <<'PY'
import sys
from importlib import import_module
from importlib.metadata import version

required = ('can', 'cv2', 'numpy', 'piper_sdk', 'rclpy', 'scipy', 'tkinter', 'yaml')
missing = []
for name in required:
    try:
        import_module(name)
    except Exception as exc:
        missing.append(f'{name}: {exc}')
if missing:
    print('FAIL  Python imports: ' + '; '.join(missing), file=sys.stderr)
    raise SystemExit(1)
print('PASS  Python imports')
expected = {'piper-sdk': '0.6.1', 'python-can': '4.5.0'}
wrong = [f'{name}={version(name)} (expected {wanted})' for name, wanted in expected.items() if version(name) != wanted]
if wrong:
    print('FAIL  Python versions: ' + '; '.join(wrong), file=sys.stderr)
    raise SystemExit(1)
print('PASS  piper-sdk==0.6.1 and python-can==4.5.0')
PY
if [ "$?" -ne 0 ]; then failures=$((failures + 1)); fi

for package_name in piper piper_description piper_mobile_manipulation piper_msgs realsense2_camera; do
  if ros2 pkg prefix "$package_name" >/dev/null 2>&1; then
    pass "ROS package: $package_name"
  else
    fail "ROS package: $package_name"
  fi
done

if [ "$failures" -eq 0 ]; then
  echo "Installation verification passed. Hardware connectivity is not tested."
  exit 0
fi

echo "Installation verification failed with $failures problem(s)." >&2
exit 1
