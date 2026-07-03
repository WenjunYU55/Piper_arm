# Supervised cube workflow verification handoff

Status as of 2026-07-03. All verification so far was dry-run; the PiPER arm was not enabled or
commanded.

## Verified

- GPU perception runs on CUDA at approximately 7–13 FPS and tracks the labelled green cube.
- The target landmark is normally `LOCKED` and valid, with approximately 1–4 mm measurement error
  and 2–3 px projection error. An occasional single-frame insufficient-depth rejection was seen.
- The pen is detected as a blocking obstacle with valid depth and a stable `base_link` transform.
- The observed pen centroid was approximately `(0.755, 0.158, 0.259)` m in `base_link`.
- The workflow package tests and build passed before live verification.
- The coordinator is dry-run only and has no arm-command publisher.

## Current stopping point

The pen centroid is outside the configured workspace because `x` is approximately 0.755 m while
`workspace_x_max` is 0.70 m. The planner now explicitly rejects an obstacle whose starting centroid
is outside the workspace. Physical obstacle removal and multi-view scanning remain unverified.

The coordinator process started, but the first `/supervised_cube_workflow/start` call was made before
the process existed. A later call appeared to wait and was not diagnosed before stopping.

## Resume remotely (no arm motion)

1. Start the perception and coordinator terminals with `ROS_DOMAIN_ID=42` and
   `ROS_LOCALHOST_ONLY=1`. Never run `enable_piper.sh` for this test.
2. Start monitors for `/piper/supervised_workflow_status` and `/piper/removal_plan` before calling
   `/supervised_cube_workflow/start` because they carry event messages.
3. Confirm the service exists with:

   ```bash
   ros2 node list
   ros2 service list | grep supervised_cube_workflow
   ros2 service type /supervised_cube_workflow/start
   ```

4. Call it with a bounded wait:

   ```bash
   timeout 10s ros2 service call /supervised_cube_workflow/start std_srvs/srv/Trigger '{}'
   echo "exit code: $?"
   ```

5. Expected result with the current scene: an invalid dry-run plan whose reason says the obstacle
   center is outside the configured workspace. The arm must remain disabled.

If the service times out with exit code 124, save the three diagnostic outputs above and the
coordinator terminal log.

## Resume in the lab

1. Keep the cube fixed. With the arm disabled, move only the pen toward the robot base until its
   stable `base_centroid.x` lies between 0.10 and 0.70 m.
2. Verify `valid: true` and `scene_blocked: true` on `/piper/obstacle_instances_3d`.
3. Repeat the coordinator dry-run and inspect its proposed plan and RViz markers.
4. Approve only after manually checking workspace clearance. For initial validation, move the pen
   by hand and use `confirm_action_complete`; do not use arm motion.
5. At `SCAN_READY`, capture 5–8 stationary viewpoints and verify `/piper/target_model` convergence.
6. Real arm removal remains a separate, supervised safety-validation phase.
