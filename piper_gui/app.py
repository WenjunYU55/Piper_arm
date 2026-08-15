"""Lifecycle bootstrap shared by the native GUI entry point and smoke tests."""

import queue
import threading


def run_gui(
        ros_runtime, node_factory, app_factory, root_factory,
        executor_factory):
    events = queue.Queue()
    ros_runtime.init()
    node = node_factory(events)
    executor = executor_factory()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    root = root_factory()
    app = app_factory(root, node, events)
    try:
        root.mainloop()
    finally:
        app.shutdown()
        executor.shutdown()
        node.destroy_node()
        ros_runtime.shutdown()
        spin_thread.join(timeout=2.0)
