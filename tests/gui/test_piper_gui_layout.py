"""Native GUI layout regressions."""

from pathlib import Path
import tkinter as tk
from tkinter import ttk

import pytest

from piper_gui_native import fitted_gui_geometry, primary_monitor_geometry


PROJECT_ROOT = Path(__file__).resolve().parents[2]


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


def test_automatic_scan_tab_has_a_width_fitted_vertical_scroll_area():
    source = (
        PROJECT_ROOT / "piper_gui" / "native_app.py"
    ).read_text(encoding="utf-8")

    helper = source.split("def build_vertical_scroll_area", 1)[1].split(
        "def clamp", 1)[0]
    build = source.split("def _build(self)", 1)[1].split(
        "def _build_results_campaign", 1)[0]
    assert "tk.Canvas" in helper
    assert "ttk.Scrollbar" in helper
    assert "yscrollcommand=scrollbar.set" in helper
    assert 'scrollregion=canvas.bbox("all")' in helper
    assert "width=event.width" in helper
    assert "build_vertical_scroll_area(" in build
    assert "self._build_automatic_scan(automatic_content)" in build


@pytest.fixture
def scroll_widgets():
    from piper_gui.native_app import build_vertical_scroll_area
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip('Tk display required for widget event test')
    root.geometry('400x180')
    parent = ttk.Frame(root)
    parent.pack(fill='both', expand=True)
    content, canvas, scrollbar = build_vertical_scroll_area(parent)
    combo = ttk.Combobox(content, values=('20 Hz', '100 Hz', '200 Hz'), state='readonly')
    combo.set('20 Hz')
    combo.pack()
    text = tk.Text(content, height=2)
    text.insert('end', '\n'.join(str(i) for i in range(50)))
    text.pack()
    ttk.Frame(content, height=700).pack()
    root.update()
    try:
        yield root, canvas, combo, text
    finally:
        root.destroy()


@pytest.mark.parametrize('sequence,options', [
    ('<Button-5>', {}), ('<MouseWheel>', {'delta': -120}),
    ('<MouseWheel>', {'delta': -1}),
])
def test_wheel_scrolls_page_without_changing_combo(scroll_widgets, sequence, options):
    root, canvas, combo, _text = scroll_widgets
    before = canvas.yview()
    combo.event_generate(sequence, **options)
    root.update()
    assert canvas.yview()[0] > before[0]
    assert combo.get() == '20 Hz'
    combo.event_generate('<Button-4>')
    root.update()
    assert canvas.yview()[0] == pytest.approx(before[0])


def test_nested_text_and_other_widgets_do_not_scroll_page(scroll_widgets):
    root, canvas, _combo, text = scroll_widgets
    before = canvas.yview()
    text.event_generate('<Button-5>')
    root.update()
    assert canvas.yview() == before
    assert text.yview()[0] > 0
    outside = ttk.Button(root, text='Other tab')
    outside.pack()
    root.update()
    before = canvas.yview()
    outside.event_generate('<Button-5>')
    root.update()
    assert canvas.yview() == before


def test_destroy_removes_private_wheel_bindings(scroll_widgets):
    root, canvas, combo, _text = scroll_widgets
    tag = combo.bindtags()[0]
    assert tag.startswith('PiPERScroll:')
    assert root.bind_class(tag, '<MouseWheel>')
    canvas.destroy()
    assert not root.bind_class(tag, '<MouseWheel>')
