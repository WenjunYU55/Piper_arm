"""Fast structural checks for the generated architecture diagrams."""

from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[3]
ASSETS = ROOT / "docs" / "assets" / "readme" / "architecture"
FOCUSED = {
    "capture-reconstruction-pipeline.svg",
    "execution-safety-pipeline.svg",
    "hardware-topology.svg",
    "perception-pipeline.svg",
    "planner-backend-pipeline.svg",
    "viewpoint-planning-pipeline.svg",
}


def main():
    files = sorted(ASSETS.glob("*.svg"))
    assert len(files) == 7, f"expected 7 SVGs, found {len(files)}"

    dimensions = {}
    for path in files:
        root = ET.parse(path).getroot()
        dimensions[path.name] = root.attrib["viewBox"].split()[2:]

    assert dimensions["system-overview.svg"] == ["1320", "5050"]
    assert all(dimensions[name][0] == "1280" for name in FOCUSED)

    master = (ASSETS / "system-overview.svg").read_text(encoding="utf-8")
    assert 'transform="scale(' not in master, "master diagram must use native coordinates"

    for document in (ROOT / "README.md", ROOT / "docs" / "architecture" / "system-diagrams.md"):
        content = document.read_text(encoding="utf-8")
        for name in dimensions:
            assert name in content, f"{name} is not embedded in {document.name}"
        assert content.count('width="1000"') >= 7, f"diagram embeds are too small in {document.name}"

    print("PASS: 7 SVGs parse, canvases match, and README embeds use full width")


if __name__ == "__main__":
    main()

