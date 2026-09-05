#!/usr/bin/env python3
"""Extract a self-contained Revo2 hand subtree from a combined G1 URDF."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g1-urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--revo2-package-root",
        type=Path,
        required=True,
        help="directory containing meshes/revo2_right_hand",
    )
    parser.add_argument("--base-link", default="right_base_link")
    return parser.parse_args()


def extract_hand_subtree(
    source: Path,
    output: Path,
    package_root: Path,
    *,
    base_link: str = "right_base_link",
) -> Path:
    source_root = ET.parse(source).getroot()
    links = {element.attrib["name"]: element for element in source_root.findall("link")}
    joints = list(source_root.findall("joint"))
    if base_link not in links:
        raise ValueError(f"base link {base_link!r} does not exist in {source}")

    children: dict[str, list[ET.Element]] = {}
    for joint in joints:
        parent = joint.find("parent").attrib["link"]
        children.setdefault(parent, []).append(joint)
    selected_links = {base_link}
    selected_joints: set[str] = set()
    pending = [base_link]
    while pending:
        parent = pending.pop()
        for joint in children.get(parent, []):
            child = joint.find("child").attrib["link"]
            selected_joints.add(joint.attrib["name"])
            if child not in selected_links:
                selected_links.add(child)
                pending.append(child)

    result = ET.Element("robot", name=f"{source_root.get('name', 'g1')}_revo2_hand")
    for element in source_root:
        if element.tag == "link" and element.attrib["name"] in selected_links:
            result.append(deepcopy(element))
        elif element.tag == "joint" and element.attrib["name"] in selected_joints:
            result.append(deepcopy(element))

    package_prefix = "package://revo2_description/"
    for mesh in result.findall(".//mesh"):
        filename = mesh.get("filename")
        if filename and filename.startswith(package_prefix):
            relative = filename.removeprefix(package_prefix)
            resolved = (package_root / relative).resolve()
            if not resolved.is_file():
                raise FileNotFoundError(f"Revo2 mesh does not exist: {resolved}")
            mesh.set("filename", str(resolved))
        elif filename and not Path(filename).is_absolute():
            resolved = (source.parent / filename).resolve()
            if not resolved.is_file():
                raise FileNotFoundError(f"relative mesh does not exist: {resolved}")
            mesh.set("filename", str(resolved))

    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(result, space="  ")
    ET.ElementTree(result).write(output, encoding="utf-8", xml_declaration=True)
    return output


def main() -> None:
    args = _arguments()
    output = extract_hand_subtree(
        args.g1_urdf,
        args.output,
        args.revo2_package_root,
        base_link=args.base_link,
    )
    print(output)


if __name__ == "__main__":
    main()
