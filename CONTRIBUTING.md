# Contributing

Contributions should preserve the system's external ROS, operator, dataset, and
safety behavior unless a change is explicitly proposed and reviewed as a
behavioral change.

## Before editing

1. Read `docs/ai/00-index.yaml`, `docs/ai/05-admin.yaml`,
   `docs/ai/10-system-map.yaml`, and `docs/ai/30-contracts.yaml`.
2. Use `docs/ai/20-modules.yaml` to identify the owner and allowed edit paths.
3. Use `docs/ai/40-flows.yaml` and `docs/ai/50-guardrails.yaml` to identify the
   required regression and smoke checks.
4. Start from a clean feature branch. Do not mix datasets, generated models,
   build outputs, virtual environments, or unrelated refactors into the change.

## Architecture rules

- Keep ROS/UI/CLI components as adapters around application and domain logic.
- Keep mission progression in the mission engine; do not create a second
  mission for a planner, perception method, or deployment platform.
- Keep NBV independent from the motion-planner backend.
- Keep planner workers command-free. They must not import or publish through the
  PiPER driver path.
- Route every executable plan through common normalization, validation,
  `PlanAuthorizer`, runtime gates, and `TrajectoryRunner`.
- Preserve fail-closed defaults and do not silently relax numeric limits.
- Prefer cohesive modules with clear ownership over small helper-file sprawl.

## Tests

Run the tests named by the affected flow and guardrail first. For the common
cross-package suite:

```bash
cd ~/Piper_arm
source ./source_piper_foxy_environment.sh
python3 -m pytest -q tests
```

Run package-local suites for changed ROS packages. GPU tests must remain
separate, opt-in, and command-free. A successful import is not planner or
hardware qualification.

## Documentation

Update all stale `docs/ai/*.yaml` records after changing ownership, contracts,
topics, services, parameters, process order, failure behavior, required tests,
or paths. Update operator and installation documentation when a supported
command changes.

## Pull requests

Keep each change focused. State:

- the owned subsystem and problem;
- whether external behavior changes;
- safety boundaries affected;
- tests and command-free qualifications run;
- tests not run and why;
- documentation updated;
- any physical validation still required.

Do not include credentials, device serial numbers, private datasets, model
weights, virtual environments, generated build trees, or large diagnostic
artifacts.
