import json
import math
import os
import re

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    RegisterEventHandler,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def cfg(name):
    return os.path.join(
        get_package_share_directory('piper_mobile_manipulation'), 'config', name)


def final_aim_tolerance(path):
    """Read the frozen later-view aim value; first lock stays capped at 5°."""
    with open(path, encoding='utf-8') as stream:
        text = stream.read()
    matches = re.findall(
        r'^\s*final_capture_aim_tolerance_deg\s*:\s*'
        r'([-+]?(?:\d+(?:\.\d*)?|\.\d+))',
        text, flags=re.MULTILINE)
    if len(matches) != 1:
        raise RuntimeError(
            'scan_execution_params.yaml must contain exactly one '
            'final_capture_aim_tolerance_deg')
    value = float(matches[0])
    if not 1.0 <= value <= 90.0:
        raise RuntimeError(
            'final_capture_aim_tolerance_deg must be within 1.0-90.0')
    return value


def home_override():
    result = {}
    for environment_name, parameter_name in (
            ('PIPER_RETURN_HOME_POSITIONS_RAD', 'return_home_positions_rad'),
            ('PIPER_PRE_HOME_POSITIONS_RAD', 'pre_home_positions_rad')):
        raw = os.environ.get(environment_name, '').strip()
        if not raw:
            continue
        try:
            values = [float(value) for value in json.loads(raw)]
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError('invalid %s: %s' % (environment_name, exc))
        if len(values) != 6 or not all(math.isfinite(value) for value in values):
            raise RuntimeError(
                '%s must contain six finite values' % environment_name)
        result[parameter_name] = values
    return result


def generate_launch_description():
    root = os.environ.get('PIPER_ARM_ROOT', '/home/prl/Piper_arm')
    planner_backend = os.environ.get(
        'PIPER_PLANNER_BACKEND', 'tesseract').strip().lower()
    if planner_backend not in ('tesseract', 'curobo'):
        raise RuntimeError(
            'PIPER_PLANNER_BACKEND must be tesseract or curobo')
    planner_spool = (
        os.environ.get('PIPER_TESSERACT_SPOOL', os.path.join(
            os.environ.get('XDG_RUNTIME_DIR', '/tmp'),
            'piper_tesseract_plans'))
        if planner_backend == 'tesseract'
        else os.environ.get('PIPER_CUROBO_SPOOL', os.path.join(
            os.environ.get('XDG_RUNTIME_DIR', '/tmp'),
            'piper_curobo_plans')))
    enable_motion = DeclareLaunchArgument(
        'enable_real_arm_motion',
        default_value='false',
        description='Opt in to approved real joint commands; never enables motors.',
    )
    speed = DeclareLaunchArgument(
        'speed_percent',
        default_value='5.0',
        description='SDK speed percentage from 1 through 100.',
    )
    max_views = DeclareLaunchArgument(
        'max_execution_viewpoints',
        default_value='13',
        description='Maximum automatically executed viewpoints for this run.',
    )
    min_views = DeclareLaunchArgument(
        'min_execution_viewpoints',
        default_value='13',
        description='Minimum validated viewpoints required to offer a proposal.',
    )
    auto_capture = DeclareLaunchArgument(
        'auto_capture',
        default_value='true',
        description='Capture after settling; disable only for supervised motion commissioning.',
    )
    mission_policy = DeclareLaunchArgument(
        'allow_mission_policy',
        default_value='false',
        description='Allow task/hash/deadline-bound autonomous approvals.',
    )
    closed_loop_one_view = DeclareLaunchArgument(
        'closed_loop_one_view',
        default_value='false',
        description='Plan one observation/capture before measured-state replanning.',
    )
    floor_profile = DeclareLaunchArgument(
        'floor_profile',
        default_value=os.environ.get('PIPER_FLOOR_PROFILE', 'tabletop'),
        choices=['tabletop', 'ground'],
        description=(
            'Startup-only support plane; combined platform geometry is invariant.'),
    )
    profile_is_ground = [
        "'", LaunchConfiguration('floor_profile'), "' == 'ground'",
    ]
    manifest_name = PythonExpression([
        "'collision_model_ground.yaml' if ", *profile_is_ground,
        " else 'collision_model.yaml'",
    ])
    profile_floor = PythonExpression([
        "'-0.466' if ", *profile_is_ground, " else '0.005'",
    ])
    tesseract_model_dir = os.path.join(
        get_package_share_directory('piper_tesseract_foxy'), 'model')
    scan_params = cfg('scan_planning_params.yaml')
    quality_params = cfg('scan_quality_params.yaml')
    capture_params = cfg('scan_capture_params.yaml')
    workflow_params = cfg('supervised_cube_workflow_params.yaml')
    execution_params = cfg('scan_execution_params.yaml')
    frozen_final_aim_tolerance_deg = final_aim_tolerance(execution_params)
    selected_home = home_override()
    bridge = Node(
        package='piper_tesseract_foxy',
        executable='motion_planner_bridge',
        name='motion_planner',
        output='screen',
        parameters=[
            os.path.join(
                get_package_share_directory('piper_tesseract_foxy'),
                'config', 'tesseract_bridge_params.yaml'),
            {
                'spool_root': planner_spool,
                'planner_backend': planner_backend,
                'hand_eye_calibration_path': os.environ.get(
                    'PIPER_HAND_EYE_CALIBRATION', os.path.join(
                        root,
                        'L515_camera/calibration/hand_eye/'
                        'session_20260808_straight_mount/'
                        'calibration_result.yaml')),
                'joint_bounds_path': os.path.join(root, 'piper_joint_bounds.json'),
                'robot_xacro_path': os.path.join(
                    root,
                    'piper_ros_foxy/src/piper_description/urdf/piper_description.xacro'),
                'srdf_path': PathJoinSubstitution([
                    tesseract_model_dir, 'piper_bunker.srdf']),
                'collision_manifest_path': PathJoinSubstitution([
                    tesseract_model_dir, manifest_name]),
                'speed_percent': ParameterValue(
                    LaunchConfiguration('speed_percent'), value_type=float),
                'max_execution_viewpoints': ParameterValue(
                    LaunchConfiguration('max_execution_viewpoints'),
                    value_type=int),
                'closed_loop_one_view': ParameterValue(
                    LaunchConfiguration('closed_loop_one_view'),
                    value_type=bool),
                'final_capture_aim_tolerance_deg':
                    frozen_final_aim_tolerance_deg,
                **selected_home,
            },
        ],
    )
    reachability = Node(
        package='piper_mobile_manipulation',
        executable='viewpoint_reachability_filter_node.py',
        name='viewpoint_reachability_filter',
        output='screen',
        parameters=[scan_params, {
            'floor_z_m': ParameterValue(profile_floor, value_type=float),
        }],
    )
    workflow = Node(
        package='piper_mobile_manipulation',
        executable='supervised_cube_workflow_node.py',
        name='supervised_cube_workflow',
        output='screen',
        parameters=[workflow_params, {
            'min_views': ParameterValue(
                LaunchConfiguration('min_execution_viewpoints'),
                value_type=int),
            'max_views': ParameterValue(
                LaunchConfiguration('max_execution_viewpoints'),
                value_type=int),
        }],
    )
    executor = Node(
        package='piper_mobile_manipulation',
        executable='scan_viewpoint_executor_node.py',
        name='scan_viewpoint_executor',
        output='screen',
        parameters=[execution_params, {
            'enable_real_arm_motion': ParameterValue(
                LaunchConfiguration('enable_real_arm_motion'), value_type=bool),
            'speed_percent': ParameterValue(
                LaunchConfiguration('speed_percent'), value_type=float),
            'max_execution_viewpoints': ParameterValue(
                LaunchConfiguration('max_execution_viewpoints'), value_type=int),
            'min_execution_viewpoints': ParameterValue(
                LaunchConfiguration('min_execution_viewpoints'), value_type=int),
            'auto_capture': ParameterValue(
                LaunchConfiguration('auto_capture'), value_type=bool),
            'allow_mission_policy': ParameterValue(
                LaunchConfiguration('allow_mission_policy'), value_type=bool),
            'closed_loop_one_view': ParameterValue(
                LaunchConfiguration('closed_loop_one_view'), value_type=bool),
            'hand_eye_calibration_path': os.environ.get(
                'PIPER_HAND_EYE_CALIBRATION', os.path.join(
                    root,
                    'L515_camera/calibration/hand_eye/'
                    'session_20260808_straight_mount/'
                    'calibration_result.yaml')),
            'joint_bounds_path': os.path.join(root, 'piper_joint_bounds.json'),
            'floor_z_m': ParameterValue(profile_floor, value_type=float),
            **selected_home,
        }],
    )
    planner = Node(
        package='piper_mobile_manipulation',
        executable='scan_viewpoint_planner_node.py',
        name='scan_viewpoint_planner',
        output='screen',
        parameters=[scan_params, {
            'session_max_views': ParameterValue(
                LaunchConfiguration('max_execution_viewpoints'),
                value_type=int),
        }],
    )
    acquisition = Node(
        package='piper_mobile_manipulation',
        executable='scan_target_acquisition_node.py',
        name='scan_target_acquisition',
        output='screen',
        parameters=[scan_params],
    )
    capture = Node(
        package='piper_mobile_manipulation',
        executable='scan_capture_node.py',
        name='scan_capture',
        output='screen',
        parameters=[
            # Reuse capture_timeout_sec from the executor configuration.  Put
            # capture-specific settings last so unrelated overlapping values
            # can never override the recorder's command-free safety profile.
            execution_params,
            scan_params,
            capture_params,
            {
                'capture_mode': 'service',
                'task_id': os.environ.get('PIPER_MISSION_TASK_ID', ''),
                'mission_sha256': os.environ.get(
                    'PIPER_MISSION_SHA256', ''),
                'target_label': os.environ.get(
                    'PIPER_TARGET_LABEL', 'green cube'),
                'target_profile': os.environ.get(
                    'PIPER_TARGET_PROFILE', 'green_cube'),
                'target_prompt': os.environ.get(
                    'PIPER_TARGET_PROMPT', 'green cube .'),
                'calibration_sha256': os.environ.get(
                    'PIPER_CALIBRATION_SHA256', ''),
                'max_frames_per_scan': ParameterValue(
                    LaunchConfiguration('max_execution_viewpoints'),
                    value_type=int),
            },
        ],
    )
    critical_nodes = (
        bridge, reachability, workflow, executor, planner, acquisition, capture)
    shutdown_handlers = [
        RegisterEventHandler(OnProcessExit(
            target_action=node,
            on_exit=[EmitEvent(event=Shutdown(
                reason='critical supervised scan component exited'))],
        ))
        for node in critical_nodes
    ]
    return LaunchDescription([
        enable_motion,
        speed,
        max_views,
        min_views,
        auto_capture,
        mission_policy,
        closed_loop_one_view,
        floor_profile,
        *shutdown_handlers,
        TimerAction(period=2.5, actions=[bridge]),
        # Foxy/Fast DDS can expose endpoints without delivering callbacks when
        # all six Python participants start simultaneously.  A command-free,
        # visual-only node starts immediately to complete discovery; safety
        # consumers then start in order before the high-rate candidate producer.
        # This sequence is part of the runtime load-order contract.
        TimerAction(period=0.5, actions=[reachability]),
        TimerAction(period=1.0, actions=[workflow]),
        TimerAction(period=1.5, actions=[executor]),
        # Start the visual-only participant immediately.  On this Foxy/Fast DDS
        # host the first Python participant can miss already-running high-rate
        # publishers during discovery; no safety decision depends on this node.
        Node(
            package='piper_mobile_manipulation',
            executable='active_scan_debug_overlay_node.py',
            name='active_scan_debug_overlay',
            output='screen',
            parameters=[scan_params, quality_params],
        ),
        TimerAction(period=2.0, actions=[planner]),
        TimerAction(period=2.25, actions=[acquisition]),
        # Raw capture is deliberately started after the command/safety
        # participants. It is read-only and saves only on the executor's
        # accepted, settled MULTIVIEW_SCAN capture service call.
        TimerAction(period=3.0, actions=[capture]),
    ])
