# Base-frame dry-run perception and scanning

This workflow keeps `base_link` as the global frame and never publishes an arm motion command.
The operator moves the camera manually and acknowledges survey/removal/scan steps through services.

## Start order

1. Start the PiPER driver without enabling the arm: `./start_piper.sh`.
2. Publish the accepted hand-eye transform: `./L515_camera/run_hand_eye_tf.sh`.
3. Start the complete GPU perception pipeline with mandatory base-frame TF:

   ```bash
   PIPER_CLOUD_FRAME=base_link PIPER_CLOUD_REQUIRE_TF=true \
     ./L515_camera/run_gpu_vision_pipeline.sh
   ```

4. Start typed active-scan planning and capture:

   ```bash
   source L515_camera/source_l515_environment.sh
   ros2 launch piper_mobile_manipulation active_scan_capture_debug.launch.py
   ```

5. Start the supervised removal/scan coordinator:

   ```bash
   ./L515_camera/run_supervised_cube_workflow.sh
   ```

## Key interfaces

- Raw camera estimate: `/piper/target/raw_camera`
- Raw base estimate: `/piper/target/raw_base`
- Filtered base estimate: `/piper/target/filtered_base`
- 50 ms prediction: `/piper/target/predicted_base`
- Typed candidates: `/piper/scan_viewpoints`
- IK-filtered candidates: `/piper/reachable_scan_viewpoints`
- Conservative scene map: `/piper/scene_objects`
- Typed removal plan: `/piper/removal_plan_typed`
- Typed workflow state: `/piper/scan_status`
- Dry-run command service: `/supervised_cube_workflow/command`

Start the operator-guided survey and capture at least three manually separated views:

```bash
ros2 service call /supervised_cube_workflow/command \
  piper_mobile_manipulation/srv/ScanCommand "{command: 0, viewpoint_index: 0, reason: ''}"

ros2 service call /supervised_cube_workflow/command \
  piper_mobile_manipulation/srv/ScanCommand "{command: 1, viewpoint_index: 0, reason: ''}"
```

Repeat command `1` after each manual viewpoint. A removal plan is valid only when the obstacle is
whitelisted, the destination and clearances pass the conservative geometry checks, and every
approach/action/destination/retreat waypoint has a bounded IK solution. `ground` is semantically
safe, but the configured workspace floor remains non-penetrable.

Approve and confirm a manually completed removal with commands `2` and `3`. The coordinator then
requires a fresh survey before clearing another blocker or allowing scanning.

## Reconstruction outputs

The target cloud node transforms masked depth into `base_link`, applies confidence-gated 6-DoF ICP,
and rejects registrations below 0.5 fitness or above 10 mm RMSE. Full-resolution captures and their
registration metadata are saved under `datasets/target_clouds/views`; final PLY and YAML manifests
are saved under `datasets/target_clouds`.

IK success is not collision certification. Unknown, invalid, stale, or non-whitelisted objects remain
protected, unseen space is considered occupied, and this workflow contains no real-arm executor.
