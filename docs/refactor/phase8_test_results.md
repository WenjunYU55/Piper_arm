# Phase 8 test and build results

Date: 2026-08-14

No robot, camera, perception, planning or production mission process was
started. The GUI launch smoke used isolated `ROS_DOMAIN_ID=224`, created the Tk
application for 0.5 seconds, then closed it through the normal application,
executor, node and rclpy cleanup path.

| Check | Result |
|---|---|
| Phase 8 plus legacy GUI/transport tests | 44 passed |
| Root GUI/reconstruction/calibration tests | 69 passed, 1 intentional skip |
| Complete `piper_mobile_manipulation/test` | 565 passed |
| New Phase 8 module flake8/pep257 | pass, no findings |
| AI YAML parse | 11 documents loaded |
| Isolated-domain real Tk launch/close | pass |
| Five-package `colcon build --symlink-install` | pass |
| Aggregate five-package colcon functional tests | pass |
| Aggregate lint baseline | unchanged known debt: 93 flake8 findings and 21 pep257 findings, represented by 116 xUnit failures; 933 tests, 0 errors, 1 skip overall |

The aggregate lint counts match the Phase 7 baseline. Phase 8 does not modify
the ROS package files that own those pre-existing findings. The new
`piper_gui/` modules and Phase 8 test are independently lint-clean.
