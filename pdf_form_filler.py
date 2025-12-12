"""
Utility functions for exporting and importing PDF form data via CSV files.

Features
--------
* Export a PDF's form fields to CSV with columns "Field Title" and "Value".
* Fill a PDF from a single CSV mapping of field names to values.
* Fill a PDF multiple times from several CSV files and bundle the results into a zip archive.

The module lazily imports the optional ``pypdf`` dependency so that the rest of the
repository can be used without it. Install ``pypdf`` before using the functions
here:

    pip install pypdf

The command line interface exposes three subcommands:

    python pdf_form_filler.py export --pdf input.pdf --csv fields.csv
    python pdf_form_filler.py fill --pdf input.pdf --csv values.csv --out filled.pdf
    python pdf_form_filler.py fill-multiple --pdf input.pdf --csvs a.csv b.csv --zip output.zip
"""
from __future__ import annotations

import argparse
import csv
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List


def _require_pypdf():
    """Import and return the :mod:`pypdf` package, raising a helpful error if missing."""
    try:
        import pypdf
    except ImportError as exc:  # pragma: no cover - exercised in runtime usage
        raise ImportError(
            "The 'pypdf' package is required for PDF form operations. Install it with 'pip install pypdf'."
        ) from exc
    return pypdf


def _read_csv_mapping(csv_path: Path) -> Dict[str, str]:
    """Return a mapping of field names to values from a two-column CSV file."""
    mapping: Dict[str, str] = {}
    with csv_path.open(newline="", encoding="utf-8") as fp:
        reader = csv.reader(fp)
        header = next(reader, None)
        if header and len(header) >= 2:
            # If the first row looks like a header, skip it unless values also make sense as data.
            if header[0].lower().startswith("field") and header[1].lower().startswith("value"):
                pass
            else:
                mapping[header[0]] = header[1]
        for row in reader:
            if len(row) >= 2:
                mapping[row[0]] = row[1]
    return mapping


def _extract_field_value(field_data: Dict) -> str:
    """Extract a user-facing value from a field dictionary returned by pypdf."""
    for key in ("/V", "value", "/DV", "/AS"):
        value = field_data.get(key)
        if value is not None:
            return str(value)
    return ""


def export_fields_to_csv(pdf_path: Path, output_csv_path: Path) -> None:
    """Export form fields from *pdf_path* to a CSV at *output_csv_path*.

    The CSV contains two columns: "Field Title" and "Value". Existing files are overwritten.
    """
    pypdf = _require_pypdf()
    reader = pypdf.PdfReader(str(pdf_path))
    fields = reader.get_fields() or {}

    with output_csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["Field Title", "Value"])
        for field_name, data in fields.items():
            writer.writerow([field_name, _extract_field_value(data)])


def fill_pdf_form(pdf_path: Path, field_values: Dict[str, str], output_pdf_path: Path) -> None:
    """Fill *pdf_path* with *field_values* and write the result to *output_pdf_path*."""
    pypdf = _require_pypdf()
    reader = pypdf.PdfReader(str(pdf_path))
    writer = pypdf.PdfWriter()
    writer.clone_document_from_reader(reader)

    for page in writer.pages:
        writer.update_page_form_field_values(page, field_values)

    writer._root_object.update({pypdf.generic.NameObject("/NeedAppearances"): pypdf.generic.BooleanObject(True)})

    with output_pdf_path.open("wb") as out_file:
        writer.write(out_file)


def fill_pdf_from_csv(pdf_path: Path, csv_path: Path, output_pdf_path: Path) -> None:
    """Fill a PDF form using values from a two-column CSV file."""
    mapping = _read_csv_mapping(csv_path)
    fill_pdf_form(pdf_path, mapping, output_pdf_path)


def fill_multiple_from_csv(
    pdf_path: Path, csv_paths: Iterable[Path], output_zip_path: Path, suffix: str = "filled"
) -> List[Path]:
    """Fill a PDF form for each CSV and bundle the outputs into a zip archive.

    Returns the list of generated PDF paths (inside a temporary directory) that were zipped.
    """
    generated_paths: List[Path] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for index, csv_path in enumerate(csv_paths, start=1):
            stem = csv_path.stem or f"submission_{index}"
            output_pdf = Path(tmpdir) / f"{stem}_{suffix}.pdf"
            fill_pdf_from_csv(pdf_path, csv_path, output_pdf)
            generated_paths.append(output_pdf)

        with zipfile.ZipFile(output_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for pdf_file in generated_paths:
                archive.write(pdf_file, arcname=pdf_file.name)

    return generated_paths


# -----------------
# Command Line Tool
# -----------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export and import PDF form data via CSV")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="Export form fields to CSV")
    export_parser.add_argument("--pdf", required=True, type=Path, help="Path to source PDF form")
    export_parser.add_argument("--csv", required=True, type=Path, help="Destination CSV path")

    fill_parser = subparsers.add_parser("fill", help="Fill a PDF form from a CSV mapping")
    fill_parser.add_argument("--pdf", required=True, type=Path, help="Path to source PDF form")
    fill_parser.add_argument("--csv", required=True, type=Path, help="CSV file containing field values")
    fill_parser.add_argument("--out", required=True, type=Path, help="Path for filled PDF output")

    multi_parser = subparsers.add_parser("fill-multiple", help="Fill multiple PDFs and zip the results")
    multi_parser.add_argument("--pdf", required=True, type=Path, help="Path to source PDF form")
    multi_parser.add_argument("--csvs", required=True, nargs="+", type=Path, help="CSV files to apply to the form")
    multi_parser.add_argument("--zip", dest="zip_path", required=True, type=Path, help="Destination zip path")

    return parser.parse_args()


def _cmd_export(args: argparse.Namespace) -> None:
    export_fields_to_csv(args.pdf, args.csv)
    print(f"Exported form fields from {args.pdf} to {args.csv}")


def _cmd_fill(args: argparse.Namespace) -> None:
    fill_pdf_from_csv(args.pdf, args.csv, args.out)
    print(f"Filled {args.pdf} using {args.csv} -> {args.out}")


def _cmd_fill_multiple(args: argparse.Namespace) -> None:
    csv_paths = [Path(p) for p in args.csvs]
    fill_multiple_from_csv(args.pdf, csv_paths, args.zip_path)
    print(f"Filled {len(csv_paths)} PDFs and saved archive to {args.zip_path}")


def main() -> None:
    args = _parse_args()
    if args.command == "export":
        _cmd_export(args)
    elif args.command == "fill":
        _cmd_fill(args)
    elif args.command == "fill-multiple":
        _cmd_fill_multiple(args)
    else:  # pragma: no cover - guarded by argparse
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":  # pragma: no cover - manual usage
    main()
