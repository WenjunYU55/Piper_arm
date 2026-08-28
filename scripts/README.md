# Operator and setup scripts

Scripts are grouped by responsibility:

- `setup/`: host dependencies and one-time CAN service provisioning.
- `robot/`: direct driver/CAN commissioning commands.
- `calibration/`: guarded calibration entry points.

The daily automatic mission and GUI launchers remain at repository root because
they are the supported public entry points. Follow `OPERATOR_COMMANDS.md`; do
not infer physical-motion authority from a script being executable.
