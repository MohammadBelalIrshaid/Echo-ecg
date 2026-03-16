import csv
import os
import re
from collections import Counter
from pathlib import Path

DATA_ROOT = Path(r"E:\Hourani\Echo-Ecg\Raw_Datasets")
DATASETS = ["CAMUS_public", "CardiacNet", "echo-eg", "HMC-QU", "AbnormCardiacEchoVideos"]

ECHO_EXTENSIONS = {
    ".dcm",
    ".dicom",
    ".nii",
    ".nii.gz",
    ".avi",
    ".mp4",
    ".mov",
    ".mkv",
    ".mpg",
    ".mpeg",
    ".mhd",
    ".mha",
    ".nrrd",
    ".npy",
    ".npz",
}

VIEW_ORDER = ["A2C", "A4C", "PLAX", "PSAX", "UNKNOWN"]
FORMAT_FAMILIES = [
    ("DICOM", [".dcm", ".dicom"]),
    ("NIfTI", [".nii", ".nii.gz"]),
    ("AVI", [".avi"]),
    ("MP4", [".mp4"]),
    ("MOV", [".mov"]),
    ("MKV", [".mkv"]),
    ("MPG", [".mpg"]),
    ("MPEG", [".mpeg"]),
    ("MHD", [".mhd"]),
    ("MHA", [".mha"]),
    ("NRRD", [".nrrd"]),
    ("NumPy", [".npy", ".npz"]),
]

VIEW_PATTERNS = {
    "A4C": [
        re.compile(r"(?<![A-Z0-9])A4C(?![A-Z0-9])"),
        re.compile(r"(?<![A-Z0-9])4CH(?![A-Z0-9])"),
        re.compile(r"(?<![A-Z0-9])AP4(?![A-Z0-9])"),
        re.compile(r"APICAL[\s_\-]*4(?:CH)?"),
    ],
    "A2C": [
        re.compile(r"(?<![A-Z0-9])A2C(?![A-Z0-9])"),
        re.compile(r"(?<![A-Z0-9])2CH(?![A-Z0-9])"),
        re.compile(r"(?<![A-Z0-9])AP2(?![A-Z0-9])"),
        re.compile(r"APICAL[\s_\-]*2(?:CH)?"),
    ],
    "PLAX": [
        re.compile(r"(?<![A-Z0-9])PLAX(?![A-Z0-9])"),
        re.compile(r"PARASTERNAL[\s_\-]*LONG"),
        re.compile(r"LONG[\s_\-]*AXIS"),
        re.compile(r"LONGAXIS"),
    ],
    "PSAX": [
        re.compile(r"(?<![A-Z0-9])PSAX(?![A-Z0-9])"),
        re.compile(r"(?<![A-Z0-9])SAX(?![A-Z0-9])"),
        re.compile(r"SHORT[\s_\-]*AXIS"),
        re.compile(r"SHORTAXIS"),
    ],
}


def normalize_extension(filename: str) -> str:
    lower = filename.lower()
    if lower.endswith(".nii.gz"):
        return ".nii.gz"
    return Path(lower).suffix


def is_probable_echo_array(path_str: str) -> bool:
    s = path_str.lower()
    positive_tokens = [
        "echo",
        "cine",
        "video",
        "view",
        "a4c",
        "a2c",
        "4ch",
        "2ch",
        "plax",
        "psax",
        "apical",
        "parasternal",
        "shortaxis",
        "longaxis",
    ]
    negative_tokens = [
        "label",
        "labels",
        "mask",
        "seg",
        "annotation",
        "annot",
        "groundtruth",
        "metadata",
        "meta",
        "split",
        "index",
    ]
    has_positive = any(tok in s for tok in positive_tokens)
    has_negative = any(tok in s for tok in negative_tokens)
    return has_positive and not has_negative


def is_echo_view_file(file_path: Path, ext: str) -> bool:
    if ext not in ECHO_EXTENSIONS:
        return False
    if ext in {".npy", ".npz"}:
        return is_probable_echo_array(str(file_path))
    return True


def detect_view_label(path_str: str) -> str:
    s = path_str.upper()
    for label in ["A4C", "A2C", "PLAX", "PSAX"]:
        for pattern in VIEW_PATTERNS[label]:
            if pattern.search(s):
                return label
    return "UNKNOWN"


def ordered_labels(view_counter: Counter) -> list:
    return [label for label in VIEW_ORDER if view_counter.get(label, 0) > 0]


def format_view_label_list(view_counter: Counter) -> str:
    labels = ordered_labels(view_counter)
    if not labels:
        return "-"
    if labels == ["UNKNOWN"]:
        return "UNKNOWN"
    return ", ".join(labels)


def format_n_per_view(view_counter: Counter) -> str:
    labels = ordered_labels(view_counter)
    if len(labels) <= 1:
        return "-"
    return ", ".join(f"{label}: {view_counter[label]}" for label in labels)


def format_file_formats(ext_counter: Counter) -> str:
    if not ext_counter:
        return "-"
    parts = []
    for family_name, family_exts in FORMAT_FAMILIES:
        present = [ext for ext in family_exts if ext_counter.get(ext, 0) > 0]
        if present:
            ext_part = "/".join(present)
            parts.append(f"{family_name} ({ext_part})")
    return ", ".join(parts) if parts else "-"


def markdown_escape(text: str) -> str:
    return text.replace("|", r"\|")


def print_markdown_table(rows: list) -> None:
    headers = [
        "Standardized Dataset Name",
        "View label",
        "N sample (Belal)",
        "N sample per view (Belal)",
        "File format (Belal)",
    ]

    str_rows = []
    for row in rows:
        str_rows.append(
            [
                str(row["Standardized Dataset Name"]),
                str(row["View label"]),
                str(row["N sample (Belal)"]),
                str(row["N sample per view (Belal)"]),
                str(row["File format (Belal)"]),
            ]
        )

    widths = [len(h) for h in headers]
    for r in str_rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))

    def fmt_line(cells):
        escaped = [markdown_escape(c) for c in cells]
        return "| " + " | ".join(escaped[i].ljust(widths[i]) for i in range(len(cells))) + " |"

    print(fmt_line(headers))
    print("| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |")
    for r in str_rows:
        print(fmt_line(r))


def write_csv(rows: list, output_path: Path) -> None:
    fieldnames = [
        "Standardized Dataset Name",
        "View label",
        "N sample (Belal)",
        "N sample per view (Belal)",
        "File format (Belal)",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def analyze_dataset(dataset_path: Path) -> dict:
    scanned_files = 0
    matched_files = 0
    ext_counter = Counter()
    view_counter = Counter()

    for root, _, files in os.walk(dataset_path):
        scanned_files += len(files)
        for name in files:
            file_path = Path(root) / name
            ext = normalize_extension(name)
            if not is_echo_view_file(file_path, ext):
                continue
            matched_files += 1
            ext_counter[ext] += 1
            view = detect_view_label(str(file_path))
            view_counter[view] += 1

    total_samples = matched_files
    view_label = format_view_label_list(view_counter)
    n_per_view = format_n_per_view(view_counter)
    file_formats = format_file_formats(ext_counter)

    return {
        "total_samples": total_samples,
        "view_label": view_label,
        "n_per_view": n_per_view,
        "file_formats": file_formats,
        "scanned_files": scanned_files,
        "matched_files": matched_files,
        "ext_counter": ext_counter,
    }


def main() -> None:
    warnings = []
    rows = []
    diagnostics = {}

    for dataset in DATASETS:
        dataset_path = DATA_ROOT / dataset
        if not dataset_path.exists() or not dataset_path.is_dir():
            warnings.append(f"WARNING: Dataset folder not found: {dataset_path}")
            rows.append(
                {
                    "Standardized Dataset Name": dataset,
                    "View label": "-",
                    "N sample (Belal)": 0,
                    "N sample per view (Belal)": "-",
                    "File format (Belal)": "-",
                }
            )
            diagnostics[dataset] = {
                "scanned_files": 0,
                "matched_files": 0,
                "ext_counter": Counter(),
            }
            continue

        result = analyze_dataset(dataset_path)
        rows.append(
            {
                "Standardized Dataset Name": dataset,
                "View label": result["view_label"],
                "N sample (Belal)": result["total_samples"],
                "N sample per view (Belal)": result["n_per_view"],
                "File format (Belal)": result["file_formats"],
            }
        )
        diagnostics[dataset] = {
            "scanned_files": result["scanned_files"],
            "matched_files": result["matched_files"],
            "ext_counter": result["ext_counter"],
        }

    for w in warnings:
        print(w)

    print_markdown_table(rows)

    script_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    csv_path = script_dir / "echo_dataset_summary.csv"
    write_csv(rows, csv_path)
    print(f"\nSaved CSV: {csv_path}")

    print("\nDiagnostics:")
    for dataset in DATASETS:
        info = diagnostics[dataset]
        ext_counter = info["ext_counter"]
        top_exts = ext_counter.most_common(5)
        if top_exts:
            top_exts_str = ", ".join(f"{ext} ({count})" for ext, count in top_exts)
        else:
            top_exts_str = "-"
        print(
            f"- {dataset}: scanned={info['scanned_files']}, "
            f"matched_echo_view_files={info['matched_files']}, "
            f"top5_matched_extensions={top_exts_str}"
        )


if __name__ == "__main__":
    main()