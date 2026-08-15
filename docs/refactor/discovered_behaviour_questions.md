# Phase 1 discovered behavior questions

These observations were found while executing the no-hardware Phase 1
characterization harness. They describe the checked-out production behavior;
Phase 1 does not change or reinterpret them.

## Cancellation received during terminal shutdown

The action cancel callback accepts cancellation at every phase. Once
`run_pipeline` has completed, however, `execute_cb` calls `safe_shutdown`
without passing the goal handle. A cancel arriving while the completed scan is
returning home, holding, disabling, or terminating child processes is therefore
not consulted by that terminal transaction. If shutdown succeeds, the
production method selects a `SUCCEEDED` result and calls the action handle's
`succeed()` transition even though `is_cancel_requested` became true during
shutdown. The pure characterization double records that call; whether Foxy's
real action state machine accepts this exact late-cancel race remains an
integration question.

This is now protected by:

- `test_cancel_arriving_during_terminal_return_home_does_not_interrupt_home`;
- `test_cancel_arriving_during_process_cleanup_does_not_change_result`.

Safety is not weakened: home, hold, disable, and cleanup still complete. The
open product-contract question is whether a late cancel should continue to mean
"do not interrupt safe shutdown, report the completed scan as success" or
whether the wire result should become `CANCELLED` after the same shutdown.
Changing that classification is explicitly outside Phase 1.

## Exhausted visual-replacement budget failure code

Nine consecutive `VIEW_REJECTED` outcomes exercise the configured eight
replacement replans and then fail with reason `visual replacement budget
exhausted`. The legacy string classifier maps that reason to `MISSION_FAILED`,
not `INSUFFICIENT_CAPTURE_QUALITY`, because the reason contains neither the
word `capture` nor `quality`.

The behavior is protected by
`test_fresh_capture_rejections_consume_eight_replans_then_fail`. The open
compatibility question is whether tracked-robot clients depend on the existing
generic code or whether a later, explicitly versioned change should return the
more specific capture-quality code. Phase 1 preserves `MISSION_FAILED`.

## Confirmed intentional safety behavior

These results initially look asymmetric but agree with the Phase 0 safety
invariants and are not proposed for correction:

- Missing joint feedback before enable performs never-enabled cleanup and can
  report safe shutdown without issuing home or disable commands.
- Invalid/stale joint or arm-status authority after an approved motion blocks
  automatic home and motor disable service calls, leaves the arm in the
  current hold/recovery state, and reports `NEEDS_OPERATOR`.
- A motor-axis dropout forbids hold/home/service-disable commands, waits only
  for driver-owned all-six-disabled feedback, then cleans up child processes.
- A fresh but unhealthy camera-clock report enters the bounded hold/refresh
  path; a missing camera-clock report is also treated as a transient freshness
  gap until its recovery timeout.

These paths remain characterization targets, not refactor opportunities.
