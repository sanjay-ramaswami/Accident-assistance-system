"""Export the public EDEN-RICAPPS code-to-chapter rows without the identifier column."""

from __future__ import annotations

import argparse
import csv
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data/triage_dataset/eden_ricapps_zenodo_19606710/classification_manchester_ed_presenting_complaints.xlsx"
DEFAULT_OUTPUT = ROOT / "data/triage_dataset/eden_ricapps_zenodo_19606710/complaint_code_chapter.csv"
DEFAULT_EXCLUDED = ROOT / "data/triage_dataset/eden_ricapps_zenodo_19606710/excluded_ambiguous_codes.csv"
NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pkg": "http://schemas.openxmlformats.org/package/2006/relationships",
}
EXPECTED_LABELS = {
    "1": "Cardiorespiratory",
    "2": "Acute neurological",
    "3": "Sepsis / potential infection",
    "4": "Medical-surgical abdominal",
    "5": "Major trauma",
    "6": "Debility and fatigue",
    "7": "Initial low vital risk",
    "8": "Other complaints",
}
SOURCE_LABELS = {
    "1": "Cardiorrespiratorio",
    "2": "Neurológico agudo",
    "3": "Sepsis / infección potencial",
    "4": "Abdominal médico-quirúrgico",
    "5": "Trauma mayor",
    "6": "Debilidad y fatiga",
    "7": "Bajo riesgo vital inicial",
    "8": "Otros motivos",
}


def _column_index(cell_ref: str) -> int:
    letters = re.match(r"[A-Z]+", cell_ref)
    if not letters:
        raise ValueError(f"Invalid spreadsheet cell reference: {cell_ref}")
    index = 0
    for char in letters.group():
        index = index * 26 + ord(char) - ord("A") + 1
    return index - 1


def read_rows(source: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Read the workbook's first worksheet with the Python standard library."""
    with zipfile.ZipFile(source) as workbook:
        book = ET.fromstring(workbook.read("xl/workbook.xml"))
        relationships = ET.fromstring(workbook.read("xl/_rels/workbook.xml.rels"))
        targets = {node.attrib["Id"]: node.attrib["Target"] for node in relationships}
        sheets = book.findall(".//main:sheet", NS)
        if len(sheets) != 1:
            raise ValueError("Expected exactly one worksheet in the source workbook")
        target = targets[sheets[0].attrib[f"{{{NS['rel']}}}id"]].lstrip("/")
        if not target.startswith("xl/"):
            target = f"xl/{target}"

        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in workbook.namelist():
            strings = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
            shared_strings = [
                "".join(text.text or "" for text in item.findall(".//main:t", NS))
                for item in strings.findall("main:si", NS)
            ]

        sheet = ET.fromstring(workbook.read(target))
        rows: list[list[str]] = []
        for row_node in sheet.findall(".//main:sheetData/main:row", NS):
            values: dict[int, str] = {}
            for cell in row_node.findall("main:c", NS):
                value_node = cell.find("main:v", NS)
                value = value_node.text if value_node is not None else ""
                if cell.attrib.get("t") == "s" and value:
                    value = shared_strings[int(value)]
                elif cell.attrib.get("t") == "inlineStr":
                    value = "".join(text.text or "" for text in cell.findall(".//main:t", NS))
                values[_column_index(cell.attrib["r"])] = value
            rows.append([values.get(index, "") for index in range(4)])

    if not rows:
        raise ValueError("Source workbook has no rows")
    header = rows[0]
    required = ("MotivoConsultaMATSET_EDIT", "cat_manchester_def", "Categoria_Manchester")
    try:
        indices = [header.index(name) for name in required]
    except ValueError as error:
        raise ValueError("Source workbook columns do not match the inspected data dictionary") from error

    by_code: dict[str, set[tuple[str, str]]] = {}
    for row in rows[1:]:
        if not any(row):
            continue
        raw_code, label_id, label = (row[index].strip() for index in indices)
        code = raw_code.upper()
        if not code or label_id not in EXPECTED_LABELS:
            raise ValueError("Source row has a missing code or an unknown chapter label")
        if SOURCE_LABELS[label_id].casefold() != label.casefold():
            raise ValueError(f"Chapter label does not match the documented mapping for id {label_id}")
        english_label = EXPECTED_LABELS[label_id]
        by_code.setdefault(code, set()).add((label_id, english_label))

    clean = []
    excluded = []
    for code, targets in sorted(by_code.items()):
        if len(targets) != 1:
            excluded.extend(
                {"normalized_code": code, "chapter_id": label_id, "chapter_label": label}
                for label_id, label in sorted(targets, key=lambda item: int(item[0]))
            )
            continue
        label_id, label = next(iter(targets))
        clean.append({"complaint_code": code, "chapter_id": label_id, "chapter_label": label})
    return clean, excluded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--excluded-output", type=Path, default=DEFAULT_EXCLUDED)
    parser.add_argument("--force", action="store_true", help="Allow replacing the derived CSV")
    args = parser.parse_args()
    if not args.force and (args.output.exists() or args.excluded_output.exists()):
        raise SystemExit("Derived output already exists; pass --force to replace it")
    records, excluded = read_rows(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("complaint_code", "chapter_id", "chapter_label"))
        writer.writeheader()
        writer.writerows(records)
    with args.excluded_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("normalized_code", "chapter_id", "chapter_label"))
        writer.writeheader()
        writer.writerows(excluded)
    print(f"Wrote {len(records)} unique, de-identified mappings; excluded {len(excluded)} conflicting code/label rows")


if __name__ == "__main__":
    main()
