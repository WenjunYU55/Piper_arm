"""
ROS-action lifecycle adapter used by the native GUI.

The adapter maps callbacks and cancellation only.  Mission sequencing and all
safety decisions remain in the production action server.
"""

from dataclasses import dataclass
import threading
import uuid

from .view_model import MissionResultView


@dataclass(frozen=True)
class MissionClientEvent:
    kind: str
    payload: object


class MissionActionClient:
    def __init__(
            self, action_client, goal_builder, outcome_names, event_sink,
            task_id_factory=None, thread_factory=None):
        self._action_client = action_client
        self._goal_builder = goal_builder
        self._outcome_names = dict(outcome_names)
        self._event_sink = event_sink
        self._task_id_factory = task_id_factory or (
            lambda: "gui-sim-" + uuid.uuid4().hex)
        self._thread_factory = thread_factory or self._new_thread
        self._lock = threading.Lock()
        self._task_id = ""
        self._goal_handle = None

    @staticmethod
    def _new_thread(target, args):
        return threading.Thread(target=target, args=args, daemon=True)

    @property
    def active_task_id(self):
        with self._lock:
            return self._task_id

    def submit(self, request):
        with self._lock:
            if self._task_id:
                admitted = False
            else:
                task_id = self._task_id_factory()
                self._task_id = task_id
                admitted = True
        if not admitted:
            self._emit("submission_failed", "a GUI mission is already active")
            return False
        thread = self._thread_factory(self._submit, (task_id, request))
        thread.start()
        return True

    def _submit(self, task_id, request):
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self._fail_submission(
                task_id,
                "autonomous mission action server is unavailable; "
                "start run_target_scan_mission.sh first",
            )
            return
        try:
            goal = self._goal_builder(task_id, request)
            future = self._action_client.send_goal_async(
                goal,
                feedback_callback=lambda message, bound=task_id:
                self._feedback(bound, message.feedback),
            )
        except Exception as exc:
            self._fail_submission(task_id, "mission goal failed: %s" % exc)
            return
        future.add_done_callback(
            lambda completed, bound=task_id:
            self._goal_response(completed, bound))

    def _feedback(self, task_id, feedback):
        if self._is_current(task_id):
            self._emit("feedback", feedback)

    def _goal_response(self, future, task_id):
        if not self._is_current(task_id):
            return
        try:
            handle = future.result()
        except Exception as exc:
            self._fail_submission(task_id, "mission goal failed: %s" % exc)
            return
        if handle is None or not handle.accepted:
            self._fail_submission(task_id, "mission goal was rejected")
            return
        with self._lock:
            if self._task_id != task_id:
                return
            self._goal_handle = handle
        self._emit("accepted", task_id)
        handle.get_result_async().add_done_callback(
            lambda completed, bound=task_id: self._result(completed, bound))

    def _result(self, future, task_id):
        if not self._is_current(task_id):
            return
        try:
            result = future.result().result
            view = MissionResultView(
                task_id=task_id,
                outcome=self._outcome_names.get(result.outcome, "UNKNOWN"),
                reason=str(result.reason),
                failure_code=str(result.failure_code),
                retryable=bool(result.retryable),
                safe_shutdown=bool(result.safe_shutdown),
                capture_count=int(result.capture_count),
                dataset_path=str(result.dataset_path),
                manifest_sha256=str(result.manifest_sha256),
                mesh_job_id=str(result.mesh_job_id),
            )
        except Exception as exc:
            self._clear(task_id)
            self._emit("submission_failed", "mission result failed: %s" % exc)
            return
        self._clear(task_id)
        self._emit("result", view)

    def cancel(self):
        with self._lock:
            handle = self._goal_handle
        if handle is None:
            self._emit("cancel_unavailable", "no accepted GUI mission is active")
            return False
        try:
            handle.cancel_goal_async()
        except Exception as exc:
            self._emit("cancel_unavailable", "mission cancel failed: %s" % exc)
            return False
        self._emit("cancel_requested", self.active_task_id)
        return True

    def _fail_submission(self, task_id, message):
        if self._clear(task_id):
            self._emit("submission_failed", str(message))

    def _clear(self, task_id):
        with self._lock:
            if self._task_id != task_id:
                return False
            self._task_id = ""
            self._goal_handle = None
            return True

    def _is_current(self, task_id):
        with self._lock:
            return self._task_id == task_id

    def _emit(self, kind, payload):
        self._event_sink(MissionClientEvent(kind, payload))
