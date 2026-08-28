# Legacy direct-motion utilities

These utilities publish real joint commands when the PiPER driver is enabled.
They are retained for historical diagnostics but are not part of the supported
automatic mission or normal operator workflow.

The two commands are intentionally distinct:

- `reset_arm.py` publishes one historically recorded non-zero joint target.
- `reset_piper.py` publishes an all-zero joint target.

Their old names were ambiguous, so they are quarantined here. Do not use them
as recovery or emergency-stop tools. Use `OPERATOR_COMMANDS.md` for supported
cancellation, staged home, disable, and emergency procedures.
