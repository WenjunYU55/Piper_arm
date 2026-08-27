"""Golden public-interface and numerical policy characterization."""

import hashlib
from pathlib import Path

from piper_mobile_manipulation.mission_core import (
    DEFAULT_DEADLINE_SEC,
    HEARTBEAT_TIMEOUT_SEC,
    MAX_DEADLINE_SEC,
    MAX_FEATURE_CAPTURES,
    MAX_OCCLUSION_ACTIONS,
    MAX_PENDING_MISSIONS,
    MISSION_QUEUE_COALESCE_SEC,
    REQUIRED_CAPTURES,
)
from piper_mobile_manipulation.scan_viewpoint_executor_node import (
    MAX_RGBD_CAPTURE_READINESS_RETRIES,
)
from piper_mobile_manipulation.target_scan_mission_node import (
    ACQUISITION_SERVICE_TIMEOUT_SEC,
    MAX_SCAN_QUALITY_REPLANS,
    MAX_SCAN_TARGET_DRIFT_REPLANS,
    PLAN_APPROVAL_TRANSIENT_TIMEOUT_SEC,
    PLAN_REQUEST_QUEUE_TIMEOUT_SEC,
    PLAN_RESULT_TIMEOUT_SEC,
    SCAN_VISUAL_REACQUISITION_TIMEOUT_SEC,
    WORKFLOW_ASSESSMENT_TIMEOUT_SEC,
)


INTERFACE_HASHES = {
    'piper_mobile_manipulation/action/RunTargetScan.action':
        'c54319336ada0442c789fc52e99124d52eeee3fae758dfad571efa197f67c565',
    'piper_mobile_manipulation/msg/CameraTimestampHealth.msg':
        '7fe0ef05befe75084473cc6b33aca12571cdf0000d7f7cf1f82d95cbcd7362e2',
    'piper_mobile_manipulation/msg/Detection2D.msg':
        '0daeb656e82e047f5b3fe2420b2368e556d243d5bb544974c71adee84cb2d728',
    'piper_mobile_manipulation/msg/HandoffTarget.msg':
        '33046d72ddb94868020649611fa4511b85b2b2f095dd4af68cac354091fbe00a',
    'piper_mobile_manipulation/msg/ManipulationCommand.msg':
        '7e9a132d59bbf2c321e7919bc2865893897a2460fc1eeb8e9d9c94a92f0889cb',
    'piper_mobile_manipulation/msg/ManipulationState.msg':
        'ba48588e4d2767eff12f1491d85de9b0a151a968d1171c29c0f0fabdb03869cf',
    'piper_mobile_manipulation/msg/MeshJobStatus.msg':
        'de8c524f064ff6d004b1822b2dc02bd3f5b9552addd233d910f2f6b7588eb502',
    'piper_mobile_manipulation/msg/MotionPlan.msg':
        '09ddcf657abda42c1067afb414f6f7ba29278b5dc25748b2ffb1e7c097ce058d',
    'piper_mobile_manipulation/msg/MotionPlanStatus.msg':
        'f2bed83e293ccc3f8f3a94ceb4e1b7f281956d1da1af37dc8cb819a317dd4559',
    'piper_mobile_manipulation/msg/ObstacleInstance3D.msg':
        'e52aeea80788f434511ce60217a88726b03ae884b343d70da7807ff411093f5c',
    'piper_mobile_manipulation/msg/ObstacleInstance3DArray.msg':
        'b23ec64535088309b88301865feb6b88b5d46c321bba5e69dfa572784edd4b6f',
    'piper_mobile_manipulation/msg/OccluderAction.msg':
        '79f2676c522d0230222110a465481de8e4f7e3002c38d8991b9fadd7a5639f75',
    'piper_mobile_manipulation/msg/PlannerReadiness.msg':
        '5303cc1e1628d643901520579a8ebdd75e739a6099b5b5b896134e0f228d90b7',
    'piper_mobile_manipulation/msg/ScanExecutionPlan.msg':
        '968ce6790c642ffb0f2b43c5ccd81dae94cc60810d13d80c806575a63b067c9b',
    'piper_mobile_manipulation/msg/ScanExecutionStatus.msg':
        '42c1ba9840d9a563af4814875a69b75efb8c5dafca30920f569254f7f04fd491',
    'piper_mobile_manipulation/msg/ServoCommand.msg':
        '238daea5cb43bc9ffbd7a636e3a11b199d9a5e4596b8f08c0437fdac02872248',
    'piper_mobile_manipulation/msg/Target3D.msg':
        '4f71fa5b87ce61536349ba7af0efda73909a3c8b5ac9edb851658907f7700bf9',
    'piper_mobile_manipulation/msg/TargetError.msg':
        'da1c1e966be5bb3839f081a25987d885a9f478613eaf327ec3999a589074c278',
    'piper_mobile_manipulation/msg/TesseractPlan.msg':
        'c5f35beabf869e90132fe1c3ff3536b2963a09544835afc8c1502820fb34c1b8',
    'piper_mobile_manipulation/msg/TesseractPlanStatus.msg':
        '9cc9d6b1918493e91b8450c4cbe1749892893ff8807cee73fba6fdec1e1c9fd8',
    'piper_mobile_manipulation/msg/TesseractReadiness.msg':
        'eeb75c671a44f54b20487ef9d964be183fde2d9746d99187c72644ed497fac89',
    'piper_mobile_manipulation/msg/TrackedTarget.msg':
        'b1bd4559f6226e17681ba466aa96d19b273d8031460be9d538736410391e7d30',
    'piper_mobile_manipulation/msg/TrackingHealth.msg':
        '74b8dfcbc5f9c7d8cde7cdb2775741062a4fa8a787c0b551c8c8b48c5446c31b',
    'piper_mobile_manipulation/srv/ApproveScanExecution.srv':
        'd3c973619eb5d621a0192d6698889aa6b3dd263cbce35b3601172c64e2cb06b1',
    'piper_mobile_manipulation/srv/AuthorizeMission.srv':
        '8ce2eeefc0d59a23bcbd74162dc650652ac22f7061399451c71238f772ec542e',
    'piper_mobile_manipulation/srv/ExecuteHomeStage.srv':
        '452d6569fd0f240d729b9cb1147aa37a8407b1b842dc1e2b0b0106cc0e85ac83',
    'piper_mobile_manipulation/srv/GetMeshJobResult.srv':
        '87710d941bb0d82d62e764204801cea4cb8b104a66230acc23d42e7af3741f2f',
    'piper_mobile_manipulation/srv/GetTargetScanResult.srv':
        'f04674deb158d275638d839ba811b8a3ef0f35aa153a7e6064c7197ec3383a5e',
    'piper_mobile_manipulation/srv/PrepareAcquisition.srv':
        'f4ca21ad6af0c6ba8e796776e7c2b2918dabee42edf291392383457f5e8e6e82',
    'piper_mobile_manipulation/srv/ReportTrackedRobotHomed.srv':
        '20bbd39d2d8f442cd891b318c0a0a1cd4691a9c5378ed66d19e31bb71149ad84',
    'piper_mobile_manipulation/srv/RequestMotionPlan.srv':
        '886b8a2299d02745c4aa39e101614ab56dcc12684365f754fc5acfa2cf116906',
    'piper_mobile_manipulation/srv/RequestTesseractPlan.srv':
        'ab62ebcb0038e7b6acd753ffbf5be169e54516a36b70b928c7b5139ec04218d2',
    'piper_msgs/msg/PiperMotionLimits.msg':
        '8091870c4cee8beb0ef200ade2dbbdfba41d88cb90485371215b0972b1432293',
    'piper_msgs/msg/PiperStatusMsg.msg':
        'c83298a91ff856230b76530c708dfa8e0dc7fdfc6ef6884a981b05d771dc541b',
    'piper_msgs/msg/PosCmd.msg':
        '9856176d34660f1652eae0b397a4db5849003c9e834ef3a33fe7f84af18982af',
    'piper_msgs/srv/Enable.srv':
        '6dac7964189ce9edbb219d009b8190b1da6848b2ee8f7791b56aec3fa92767e0',
}


def test_all_public_ros_interface_files_match_phase_zero_wire_contract():
    source_root = Path(__file__).resolve().parents[2]
    discovered = {}
    for package in ('piper_mobile_manipulation', 'piper_msgs'):
        for kind in ('action', 'msg', 'srv'):
            directory = source_root / package / kind
            if not directory.is_dir():
                continue
            for path in sorted(directory.iterdir()):
                if path.suffix not in ('.action', '.msg', '.srv'):
                    continue
                key = str(path.relative_to(source_root))
                discovered[key] = hashlib.sha256(path.read_bytes()).hexdigest()

    assert discovered == INTERFACE_HASHES


def test_mission_retry_deadline_and_capture_limits_match_phase_zero():
    assert DEFAULT_DEADLINE_SEC == 1200.0
    assert MAX_DEADLINE_SEC == 1200.0
    assert HEARTBEAT_TIMEOUT_SEC == 5.0
    assert MAX_PENDING_MISSIONS == 8
    assert MISSION_QUEUE_COALESCE_SEC == 1.0
    assert REQUIRED_CAPTURES == 8
    assert MAX_FEATURE_CAPTURES == 24
    assert MAX_OCCLUSION_ACTIONS == 6
    assert MAX_SCAN_QUALITY_REPLANS == 8
    assert MAX_SCAN_TARGET_DRIFT_REPLANS == 8
    assert MAX_RGBD_CAPTURE_READINESS_RETRIES == 10


def test_mission_service_and_reacquisition_timeouts_match_phase_zero():
    assert ACQUISITION_SERVICE_TIMEOUT_SEC == 8.0
    assert WORKFLOW_ASSESSMENT_TIMEOUT_SEC == 75.0
    assert PLAN_REQUEST_QUEUE_TIMEOUT_SEC == 12.0
    assert PLAN_RESULT_TIMEOUT_SEC == 185.0
    assert PLAN_APPROVAL_TRANSIENT_TIMEOUT_SEC == 5.0
    assert SCAN_VISUAL_REACQUISITION_TIMEOUT_SEC == 30.0


def test_primary_public_ros_names_remain_present_at_their_owners():
    package_root = Path(__file__).resolve().parents[1]
    mission = (
        package_root / 'piper_mobile_manipulation'
        / 'target_scan_mission_node.py').read_text(encoding='utf-8')
    executor = (
        package_root / 'piper_mobile_manipulation'
        / 'scan_viewpoint_executor_node.py').read_text(encoding='utf-8')
    configuration = (
        package_root / 'piper_mobile_manipulation'
        / 'configuration.py').read_text(encoding='utf-8')
    executor_contract = executor + configuration

    for name in (
            '/piper/run_target_scan',
            '/piper/get_target_scan_result',
            '/piper/scan_execution_plan',
            '/piper/scan_execution_status',
            '/piper/scan_capture_status',
            '/piper/scan_session_history',
            '/piper/planner_readiness',
            '/piper/camera_timestamp_health',
            '/joint_states_single',
            '/arm_status'):
        assert name in mission
    assert "super().__init__('scan_viewpoint_executor')" in executor
    for name in (
            "'~/approve'",
            "'~/authorize_mission'",
            "'~/cancel'",
            "'~/hold'",
            "'~/refresh_plan'",
            "'~/diagnostic_state'",
            '/scan_capture/capture_view',
            '/piper/tracking_health',
            '/joint_ctrl_single'):
        assert name in executor_contract
