"""Characterize dependency boundaries before moving production modules."""

import ast
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / 'piper_mobile_manipulation'
TESSERACT_ROOT = (
    Path(__file__).resolve().parents[2]
    / 'piper_tesseract_foxy'
    / 'piper_tesseract_foxy'
)

ROS_IMPORT_ROOTS = frozenset({
    'builtin_interfaces',
    'geometry_msgs',
    'message_filters',
    'piper_msgs',
    'rclpy',
    'sensor_msgs',
    'std_msgs',
    'std_srvs',
    'tf2_geometry_msgs',
    'tf2_ros',
    'trajectory_msgs',
    'visualization_msgs',
})

PURE_MOBILE_MODULES = frozenset({
    'camera_timestamp_health.py',
    'capability_map.py',
    'capture_coordinator.py',
    'collision_environment.py',
    'configuration.py',
    'executor_recovery.py',
    'failure_model.py',
    'heavy_refresh_contract.py',
    'home_pose.py',
    'mission_core.py',
    'mission_engine.py',
    'mission_spool.py',
    'motion_limit_stability.py',
    'nbv_coverage.py',
    'obstacle_geometry.py',
    'occlusion_policy.py',
    'plan_authorizer.py',
    'process_supervisor.py',
    'ray_hard_culls.py',
    'ray_mission_diagnostics.py',
    'reconstruction_jobs.py',
    'safety_evaluator.py',
    'scan_capture.py',
    'scan_execution_modes.py',
    'scan_motion.py',
    'scan_session_memory.py',
    'scan_trajectory.py',
    'startup_gates.py',
    'supervised_workflow.py',
    'surface_coverage.py',
    'target_acquisition.py',
    'target_envelope.py',
    'target_landmark_geometry.py',
    'telemetry_store.py',
    'trajectory_runner.py',
    'view_generation.py',
    'viewpoint_rays.py',
})


def imported_modules(path, current_module=None, is_package=False):
    """Return the explicit import targets in one Python source file."""
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported = node.module or ''
            if node.level and current_module:
                current_parts = current_module.split('.')
                package_parts = (
                    current_parts if is_package else current_parts[:-1])
                keep = len(package_parts) - (node.level - 1)
                imported = '.'.join(
                    package_parts[:max(0, keep)]
                    + ([node.module] if node.module else []))
            if imported and node.module:
                imports.append(imported)
            if node.level and not node.module:
                imports.extend(
                    '.'.join((imported, alias.name)).strip('.')
                    for alias in node.names
                    if alias.name != '*')
    return imports


def module_name(path, package_root, package_name):
    """Return an import name for a source path beneath a package root."""
    relative = path.relative_to(package_root).with_suffix('')
    parts = list(relative.parts)
    if parts[-1] == '__init__':
        parts.pop()
    return '.'.join([package_name] + parts)


def internal_dependency_graph(package_root, package_name):
    """Build the explicit internal-import graph for one Python package."""
    paths = sorted(package_root.rglob('*.py'))
    modules = {
        module_name(path, package_root, package_name): path
        for path in paths
        if '__pycache__' not in path.parts
    }
    graph = {name: set() for name in modules}
    for name, path in modules.items():
        for imported in imported_modules(
                path, current_module=name, is_package=path.name == '__init__.py'):
            matches = [
                candidate for candidate in modules
                if imported == candidate
                or imported.startswith(candidate + '.')
            ]
            if matches:
                graph[name].add(max(matches, key=len))
    return graph


def dependency_cycles(graph):
    """Return all cycles found by a depth-first graph traversal."""
    cycles = []
    active = []
    active_set = set()
    complete = set()

    def visit(node):
        if node in complete:
            return
        if node in active_set:
            start = active.index(node)
            cycles.append(tuple(active[start:] + [node]))
            return
        active.append(node)
        active_set.add(node)
        for dependency in sorted(graph[node]):
            visit(dependency)
        active.pop()
        active_set.remove(node)
        complete.add(node)

    for node in sorted(graph):
        visit(node)
    return cycles


def test_named_domain_and_application_modules_remain_ros_independent():
    """Keep established pure owners usable without starting ROS."""
    violations = []
    for filename in sorted(PURE_MOBILE_MODULES):
        path = PACKAGE_ROOT / filename
        assert path.is_file(), filename
        for imported in imported_modules(path):
            root = imported.split('.', 1)[0]
            if root in ROS_IMPORT_ROOTS or imported.endswith('_node'):
                violations.append((filename, imported))
    assert violations == []


def test_tesseract_worker_and_contract_remain_ros_independent():
    """Keep the isolated planner process free of ROS imports."""
    violations = []
    for filename in ('contract.py', 'worker.py'):
        for imported in imported_modules(TESSERACT_ROOT / filename):
            if imported.split('.', 1)[0] in ROS_IMPORT_ROOTS:
                violations.append((filename, imported))
    assert violations == []


def test_production_python_packages_have_no_internal_import_cycles():
    """Prevent structural moves from introducing bidirectional imports."""
    mobile = internal_dependency_graph(
        PACKAGE_ROOT, 'piper_mobile_manipulation')
    tesseract = internal_dependency_graph(
        TESSERACT_ROOT, 'piper_tesseract_foxy')
    assert dependency_cycles(mobile) == []
    assert dependency_cycles(tesseract) == []
