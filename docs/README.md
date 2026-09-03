# Documentation

Use this page to find the authoritative document for a task. Operator-facing
instructions are kept separate from architecture history and machine-oriented
maintenance records.

## Get started

| Task | Start here |
|---|---|
| Understand the project | [`README.md`](../README.md) |
| Install a clean workstation | [`CLEAN_INSTALL.md`](../CLEAN_INSTALL.md) |
| Verify hardware and software versions | [`reference/validated-environments.md`](reference/validated-environments.md) |
| Run, cancel, recover, or shut down | [`OPERATOR_COMMANDS.md`](../OPERATOR_COMMANDS.md) |
| Diagnose the L515 | [`L515_camera/README.md`](../L515_camera/README.md) |

## Understand the system

| Topic | Document |
|---|---|
| End-to-end runtime and process boundaries | [`architecture/system-overview.md`](architecture/system-overview.md) |
| Package responsibilities and dependency direction | [`ARCHITECTURE.md`](../ARCHITECTURE.md) |
| Tesseract/cuRobo selection and generic plan contract | [`architecture/motion_planner_backends.md`](architecture/motion_planner_backends.md) |
| Canonical and generated asset policy | [`architecture/asset_policy.md`](architecture/asset_policy.md) |
| Current architecture and limitations | [`ARCHITECTURE.md`](../ARCHITECTURE.md) and [`reference/validated-environments.md`](reference/validated-environments.md) |
| Dated August 2026 handoff snapshot | [`historical/system_handoff_2026_08_11.md`](historical/system_handoff_2026_08_11.md) |
| Tracked-robot integration contract | [`integration/track_robot_description/README.md`](../integration/track_robot_description/README.md) |

## Develop and test

| Area | Document or command |
|---|---|
| Repository-wide test ownership | [`tests/README.md`](../tests/README.md) |
| Perception environment and tests | [`AI_perception_tests/README.md`](../AI_perception_tests/README.md) |
| Reconstruction setup | [`CLEAN_INSTALL.md` section 9](../CLEAN_INSTALL.md#9-optional-offline-tsdf-reconstruction) |
| Refactor baseline and equivalence evidence | [`refactor/`](refactor/) |
| Full Tesseract/cuRobo test commands | [`architecture/motion_planner_backends.md`](architecture/motion_planner_backends.md#running-and-testing) |
| Record paired scan results and build Excel/figures | [`experiments/results_campaign.md`](experiments/results_campaign.md) |

## Research and maintenance records

- `historical/` contains old experiment and session notes. It is not the
  current operator procedure.
- `refactor/` contains architecture-audit, characterization, test, and
  equivalence evidence.
- `ai/` is the machine-readable architecture and maintenance map. Its required
  first-pass order is `00-index.yaml`, `05-admin.yaml`, `10-system-map.yaml`,
  and `30-contracts.yaml`.

## Document authority

When documents disagree, use this order:

1. Current code, configuration, and generated interface definitions.
2. `docs/ai/` architecture contracts and guardrails.
3. `OPERATOR_COMMANDS.md` for supported operation and physical safety.
4. `CLEAN_INSTALL.md` for workstation reproduction.
5. Current architecture documents.
6. Historical/refactor records only as evidence of earlier decisions.

Do not use an old handoff, session note, or copied command as authority for
physical motion.
