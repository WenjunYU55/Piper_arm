from pathlib import Path
import xml.etree.ElementTree as ElementTree


ROOT = Path(__file__).resolve().parent
NAMESPACE = {"f": "http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles"}


def test_gui_transport_profile_is_loopback_udp_only():
    document = ElementTree.parse(ROOT / "fastdds_gui_udp_only.xml")
    profiles = document.find("f:profiles", NAMESPACE)
    assert profiles is not None

    descriptors = profiles.findall(
        "f:transport_descriptors/f:transport_descriptor", NAMESPACE)
    assert len(descriptors) == 1
    assert descriptors[0].findtext("f:type", namespaces=NAMESPACE) == "UDPv4"
    assert descriptors[0].findtext(
        "f:interfaceWhiteList/f:address", namespaces=NAMESPACE) == "127.0.0.1"

    participant = profiles.find("f:participant", NAMESPACE)
    assert participant is not None
    assert participant.get("is_default_profile") == "true"
    assert participant.findtext(
        "f:rtps/f:useBuiltinTransports", namespaces=NAMESPACE) == "false"
    assert participant.findtext(
        "f:rtps/f:userTransports/f:transport_id",
        namespaces=NAMESPACE,
    ) == "piper_gui_udp_v4"

    # Foxy's rmw_fastrtps owns endpoint QoS. Loading XML endpoint profiles on
    # this install switches services to fixed-size histories and drops the
    # larger PrepareAcquisition response.
    assert profiles.findall("f:publisher", NAMESPACE) == []
    assert profiles.findall("f:subscriber", NAMESPACE) == []


def test_gui_and_scan_launchers_enforce_foxy_transport_contract():
    for relative_path in (
            "start_gui.sh",
            "L515_camera/run_supervised_viewpoint_execution.sh"):
        launcher = (ROOT / relative_path).read_text(encoding="utf-8")
        assert "fastdds_gui_udp_only.xml" in launcher
        assert "RMW_FASTRTPS_USE_QOS_FROM_XML=0" in launcher
        assert "ROS_LOCALHOST_ONLY=0" in launcher


def test_l515_parameter_transactions_are_process_bounded():
    for relative_path in (
            "L515_camera/start_l515_camera.sh",
            "L515_camera/start_l515_camera_low_bandwidth.sh"):
        launcher = (ROOT / relative_path).read_text(encoding="utf-8")
        assert "camera_param()" in launcher
        assert "timeout --signal=TERM --kill-after=1s 2s" in launcher
        # Keep the raw CLI invocation inside the bounded helper. Every actual
        # preset/global-time transaction must call that helper instead.
        assert launcher.count("ros2 param") == 1
        assert launcher.count("camera_param set --no-daemon --spin-time 0.5") == 3
        assert launcher.count("camera_param get --no-daemon --spin-time 0.5") == 3


def test_supervised_scan_launch_shuts_down_on_critical_child_exit():
    launch_source = (
        ROOT
        / "piper_ros_foxy/src/piper_mobile_manipulation/launch/"
        "supervised_viewpoint_execution.launch.py"
    ).read_text(encoding="utf-8")
    assert "OnProcessExit" in launch_source
    assert "critical_nodes" in launch_source
    assert "Shutdown(" in launch_source
