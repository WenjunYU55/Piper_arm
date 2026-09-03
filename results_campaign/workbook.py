"""Dependency-free XLSX writer for campaign result tables."""

from __future__ import annotations

import math
from pathlib import Path
import re
from typing import Any, Iterable, List, Mapping
from xml.sax.saxutils import escape
import zipfile


_INVALID_SHEET = re.compile(r"[\\/*?:\[\]]")


def _column_name(index: int) -> str:
    result = ''
    value = index + 1
    while value:
        value, remainder = divmod(value - 1, 26)
        result = chr(65 + remainder) + result
    return result


def _sheet_name(value: str, used: set) -> str:
    base = _INVALID_SHEET.sub('_', str(value))[:31] or 'Sheet'
    name = base
    counter = 2
    while name in used:
        suffix = '_%d' % counter
        name = base[:31 - len(suffix)] + suffix
        counter += 1
    used.add(name)
    return name


def _columns(rows: List[Mapping[str, Any]]) -> List[str]:
    output = []
    for row in rows:
        for key in row:
            if key not in output:
                output.append(str(key))
    return output or ['note']


def _cell(reference: str, value: Any, style: int = 0) -> str:
    style_text = ' s="%d"' % style if style else ''
    if isinstance(value, bool):
        return '<c r="%s" t="b"%s><v>%d</v></c>' % (reference, style_text, int(value))
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(float(value)):
            return '<c r="%s"%s><v>%.15g</v></c>' % (reference, style_text, float(value))
        value = ''
    if value is None:
        value = ''
    if isinstance(value, (dict, list, tuple)):
        import json
        value = json.dumps(value, sort_keys=True)
    text = escape(str(value))
    return '<c r="%s" t="inlineStr"%s><is><t xml:space="preserve">%s</t></is></c>' % (reference, style_text, text)


def _worksheet(rows: List[Mapping[str, Any]]) -> str:
    if not rows:
        rows = [{'note': 'No data recorded yet.'}]
    columns = _columns(rows)
    xml_rows = []
    header = ''.join(_cell('%s1' % _column_name(index), key, 1) for index, key in enumerate(columns))
    xml_rows.append('<row r="1">%s</row>' % header)
    for row_index, row in enumerate(rows, 2):
        cells = ''.join(_cell('%s%d' % (_column_name(index), row_index), row.get(key)) for index, key in enumerate(columns))
        xml_rows.append('<row r="%d">%s</row>' % (row_index, cells))
    widths = ''.join('<col min="%d" max="%d" width="%g" customWidth="1"/>' % (
        index + 1, index + 1, min(48.0, max(12.0, len(key) + 2.0))) for index, key in enumerate(columns))
    end = '%s%d' % (_column_name(len(columns) - 1), len(rows) + 1)
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<dimension ref="A1:%s"/><sheetViews><sheetView workbookViewId="0">'
            '<pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>'
            '</sheetView></sheetViews><cols>%s</cols><sheetData>%s</sheetData>'
            '<autoFilter ref="A1:%s"/></worksheet>') % (end, widths, ''.join(xml_rows), end)


def write_xlsx(path: Path, sheets: Mapping[str, Iterable[Mapping[str, Any]]]) -> Path:
    """Write a valid, compact XLSX workbook without pandas/openpyxl."""
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    used = set()
    normalized = [(_sheet_name(name, used), list(rows)) for name, rows in sheets.items()]
    content_types = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
    ]
    for index in range(len(normalized)):
        content_types.append('<Override PartName="/xl/worksheets/sheet%d.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' % (index + 1))
    content_types.append('</Types>')
    workbook_sheets = ''.join('<sheet name="%s" sheetId="%d" r:id="rId%d"/>' % (
        escape(name, {'"': '&quot;'}), index + 1, index + 1) for index, (name, _) in enumerate(normalized))
    workbook = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
                'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
                '<sheets>%s</sheets></workbook>') % workbook_sheets
    rels = ''.join('<Relationship Id="rId%d" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet%d.xml"/>' % (index + 1, index + 1) for index in range(len(normalized)))
    rels += '<Relationship Id="rId%d" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>' % (len(normalized) + 1)
    styles = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
              '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
              '<fonts count="2"><font><sz val="11"/><name val="Calibri"/></font>'
              '<font><b/><color rgb="FFFFFFFF"/><sz val="11"/><name val="Calibri"/></font></fonts>'
              '<fills count="3"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill>'
              '<fill><patternFill patternType="solid"><fgColor rgb="FF1F4E78"/><bgColor indexed="64"/></patternFill></fill></fills>'
              '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
              '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
              '<cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
              '<xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1"/></cellXfs>'
              '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>')
    with zipfile.ZipFile(str(destination), 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('[Content_Types].xml', ''.join(content_types))
        archive.writestr('_rels/.rels', '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        archive.writestr('xl/workbook.xml', workbook)
        archive.writestr('xl/_rels/workbook.xml.rels', '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">%s</Relationships>' % rels)
        archive.writestr('xl/styles.xml', styles)
        for index, (_, rows) in enumerate(normalized, 1):
            archive.writestr('xl/worksheets/sheet%d.xml' % index, _worksheet(rows))
    return destination
