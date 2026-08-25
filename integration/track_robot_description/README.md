# Tracked-robot description drop-in

This bundle makes the tracked robot the authoritative TF root while retaining
the current PiPER, gripper, holder and L515 description.

The generated tree is:

```text
robot_bottom
└── base_link                  Bunker chassis / odometry base
    ├── sensor_station_link
    └── arm_base_link          PiPER model root
        ├── piper_base_link    gateway coordinate frame (identity alias)
        └── link1 ... camera frames
```

`piper_base_link` is deliberately an empty identity frame. It gives the
tracked-robot-facing gateway an unambiguous arm coordinate frame without
renaming the existing `arm_base_link` or introducing duplicate geometry.

## Install into Track-robot-workspace

From this PiPER repository:

```bash
TRACK_REPO=/absolute/path/to/Track-robot-workspace

cp integration/track_robot_description/drop_in/src/piper_description/urdf/piper_description.xacro \
  "$TRACK_REPO/src/piper_description/urdf/piper_description.xacro"
cp integration/track_robot_description/drop_in/src/bunker_pro2/urdf/bunker_pro2.urdf \
  "$TRACK_REPO/src/bunker_pro2/urdf/bunker_pro2.urdf"

cd "$TRACK_REPO"
colcon build --packages-select piper_description bunker_pro2
colcon test --packages-select bunker_pro2
colcon test-result --all --verbose
```

No mesh replacement is required for the source repository pinned in
`drop_in/description_bundle_manifest.json`. The generated arm-only Xacro uses
the PiPER mesh assets already present in that repository.

The two files must be replaced together. Replacing only the arm Xacro leaves
the master tracked URDF stale; replacing only the master URDF makes the
repository's source-equivalence test compare against an older arm definition.

## Target-coordinate contract

The tracked robot publishes its normal localization transform:

```text
odom -> base_link
```

It submits `RunTargetScan` with the rough target's original timestamp and
`rough_target.header.frame_id = odom`. The existing PiPER gateway snapshots:

```text
odom -> piper_base_link
```

at that timestamp, rotates the position covariance, and writes the transformed
point to the private arm mission as local `base_link` coordinates. The tracked
base must be stationary before physical arm dispatch; later base movement does
not reinterpret the admitted target.

The fixed tracked transform is:

```text
base_link -> arm_base_link
xyz = [0.39, 0.0, 0.016] m
rpy = [0.0, 0.0, 0.0] rad
```

The physical ground is 0.45 m below tracked `base_link`, so it is 0.466 m
below the arm planning origin. This matches the PiPER `ground` floor profile.

## Regeneration

The ready-to-copy files are generated deterministically from the current local
PiPER description and the hash-pinned tracked-robot description:

```bash
python3 tools/build_track_robot_description_bundle.py
```

Use `--tracked-urdf` to build from an explicitly reviewed local upstream file.
The generator excludes the PiPER repository's arm-centric embedded Bunker
visual links because the tracked master already owns that geometry.
