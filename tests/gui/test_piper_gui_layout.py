"""Native GUI layout regressions."""

from piper_gui_native import fitted_gui_geometry, primary_monitor_geometry


def test_gui_fills_a_normal_display_without_crossing_screen_edges():
    assert fitted_gui_geometry(1920, 1080) == (1872, 984, 24, 48)


def test_gui_shrinks_to_fit_a_small_display():
    width, height, x_position, y_position = fitted_gui_geometry(1024, 600)

    assert (width, height) == (976, 504)
    assert x_position >= 0 and y_position >= 0
    assert x_position + width <= 1024
    assert y_position + height <= 600


def test_gui_caps_itself_on_a_large_display_and_stays_centered():
    assert fitted_gui_geometry(3840, 2160) == (1920, 1080, 960, 540)


def test_primary_monitor_is_selected_from_a_multi_monitor_desktop():
    output = """Monitors: 2
 0: +*DP-0 2560/597x1440/336+0+0 DP-0
 1: +DP-2 1920/598x1080/336+2560+360 DP-2
"""

    assert primary_monitor_geometry(output) == (0, 0, 2560, 1440)


def test_monitor_parser_falls_back_to_first_monitor_without_primary_marker():
    output = " 0: +DP-2 1920/598x1080/336+2560+360 DP-2\n"

    assert primary_monitor_geometry(output) == (2560, 360, 1920, 1080)
