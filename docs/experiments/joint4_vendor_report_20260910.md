# Draft vendor report — PiPER near-zero Joint 4 feedback discontinuity

Not sent. Prepared from read-only local measurements for manufacturer review.

Hardware query: H-V1.2-1, arm software S-V1.5-8 with build string 250221; SDK 0.6.1; SocketCAN at 1 Mbit/s. Attached camera L515 f1120648 remained mounted. No load, motor-zero, firmware, gains or controller configuration changed during diagnostics.

## Reproduction without driving the robot
With the ROS arm driver stopped, open a passive SocketCAN receiver. Observe ID 0x2A6 (joint 3/4 feedback) and motor/status IDs. This does not require constructing an SDK interface or transmitting commands. Record on an already stationary, disabled arm; do not induce motion or reset as part of this reproduction.

Raw 0x2A6 payload alternates between `000003d3fffffe89` and `000003d300000000`: J3 stays +0.979 degrees, J4 alternates -0.375 degrees / 0.000 degrees. The installed vendor decoder agrees exactly with independent signed big-endian decoding across all 3999 packets from a 20-second capture. All six low-speed motor status bytes remain zero (disabled/no reported faults); J4 status 0x264 payload is `00e600221c000000`. No arm command IDs are present in passive captures. The controller Cartesian pose feedback also switches between two payload sets.

A 30-second simultaneous capture produced 192 J4 transitions but wrist-mounted independent IMU angular-speed norm never exceeded 0.01062 rad/s. Reported 0.375-degree jumps over 5 ms imply approximately 1.31 rad/s if interpreted as physical rotation. IMU caveats: gyro 100 Hz, accelerometer 200 Hz, receipt-time overlap rather than common hardware clock; RealSense reports unavailable IMU calibration and uses defaults. This strongly contradicts repeated full-size physical oscillation, without excluding tiny physical movement or encoder noise.

All 35 retained recording folders contain 179849 Joint 4 samples in total. Near zero: 22795 exact zeros, 12365 nonzero samples with magnitude less than 1 degree, no nonzero sample below 0.302 degree. This suggests an undocumented zero-suppression band around +/-0.3 degree; it is not asserted as a proven firmware specification. Similar isolated pulses appear in 11 folders, earliest retained on 7 September 2026 at 14:27 UK. Several align with disabling; other occurrences mean it is not demonstrated to be exclusive to disabled mode.

The host startup check required two seconds within 0.005 rad (~0.286 degrees) of an anchor, so repeated controller reports across that apparent cutoff prevented startup. A narrow disabled-only host mitigation has been tested; original feedback and powered checks remain unchanged. We are seeking the underlying cause rather than treating that mitigation as a firmware repair.

## Checks already completed
- Firmware queried on 0x4AF without changing settings.
- Gripper parameter response on 0x47E is `6446010000000000`: assistance/friction is already 1, excluding the 255 setting discussed in issue 33 as the observed configuration.
- SDK/ROS stopped during raw reproduction, so host concurrent SDK sends are not required to reproduce.
- No bus-off or CAN error counters were recorded in the inspected host state; historical RX drops existed. Received alternating payloads are real frames, not filled-in missing values.
- 0x254 motor-position field stays zero but is documented as integer radians and cannot independently resolve these small angles.
- No resetting, teaching-role changes, motor zero changes, firmware upgrade, enable or movement command was performed to investigate this issue.

## Questions for AgileX
1. Does S-V1.5-8 suppress joint readings around +/-0.3 degrees, or substitute commanded zero for measured position in any state? What exact signal and threshold are involved?
2. Does 0x2A6 use the same sensor/estimate when motors are enabled, disabled, and transitioning? Is that estimate related to encoder offsets or motor-side quantization?
3. Is the alternating pattern a known defect, and which released firmware version specifically corrects it?
4. Is there a documented read-only higher-resolution position signal to distinguish encoder fluctuation from zero-suppression without recalibrating or resetting references?
5. Would a proposed firmware update preserve joint-zero references, mode-0 kinematics and existing command/feedback contracts? What requalification is required?

Related upstream issues reviewed: https://github.com/agilexrobotics/piper_sdk/issues/33 ; https://github.com/agilexrobotics/piper_sdk/issues/67 . The 1.7-6+ MIT twitch correction is a different powered-command symptom, not evidence of a fix for this case.
