# Bunker Pro 2 collision-model sources

These two binary STL files were imported without modification from
`ZZY5825/Track-robot-workspace` at commit
`c8bf4db35c7a196aa26c0add0f2549fa1c973980` on 2026-08-24.

- `bunker_pro2_base_link.STL`: upstream Bunker Pro 2 chassis mesh,
  SHA-256 `aa9f81c3d75c28e6720fda247bc01d052d17d9343b1e050d13ea2f7a5e60ce9d`.
- `bunker_pro2_FullCase.STL`: sensor-station assembly mesh,
  SHA-256 `62148eb78b3e3294210e01cab448c4b0f099aacecfbe38d4409f8c2bf8bdd8d5`.

Repository URL: <https://github.com/ZZY5825/Track-robot-workspace>

Only these platform assets are imported. The PiPER arm, joint limits, L515,
holder, calibration, and planning-frame semantics remain owned by this
repository. Generated collision meshes place the platform into this project's
arm-base `base_link` frame and retain every source triangle.
