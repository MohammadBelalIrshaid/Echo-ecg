from __future__ import annotations

import csv
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import nibabel as nib
import numpy as np
import pandas as pd


ROOT = Path(r"E:\Hourani\Echo-Ecg")
RAW_ROOT = ROOT / "Raw_Datasets"
MODELS_ROOT = ROOT / "External_Val_Models"

ECHO_VIEW_REPO = MODELS_ROOT / "echo-view-classifier-master"
ECHO_QC_REPO = MODELS_ROOT / "EchocardiographyQC-master"
ECHO_PRIME_REPO = MODELS_ROOT / "EchoPrime-main"

ECHO_VIEW_OUT = ECHO_VIEW_REPO / "processed_datasets"
ECHO_QC_OUT = ECHO_QC_REPO / "processed_datasets" / "SourceImage"
ECHO_PRIME_OUT = ECHO_PRIME_REPO / "processed_datasets"


@dataclass
class Sample:
    dataset: str
    sample_id: str
    source_path: Path
    source_type: str  # "nii" or "mp4"
    raw_label: str
    patient_id: str


def slug(text: str) -> str:
    out = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        else:
            out.append("_")
    s = "".join(out)
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def normalize_raw_label(raw: str) -> str:
    s = (raw or "").strip()
    if not s:
        return ""
    s = s.replace("-", "_").replace(" ", "_")
    s = s.replace("(", "").replace(")", "")
    while "__" in s:
        s = s.replace("__", "_")
    return s.upper().strip("_")


def canonical_view(raw_label: str) -> Optional[str]:
    s = normalize_raw_label(raw_label)
    if s in {"", "?", "NAN", "VIEW", "1"}:
        return None

    if "SUBCOSTAL" in s or s == "IVC":
        return "SUBCOSTAL"
    if "SSN" in s or "SUPRASTERNAL" in s:
        return "SSN"

    if "A2C" in s or "2CH" in s or "APEX_2CH" in s:
        return "A2C"
    if "A3C" in s or "3CH" in s or "APEX_3CH" in s:
        return "A3C"
    if "A4C" in s or "4CH" in s or "APEX_4CH" in s:
        return "A4C"
    if "A5C" in s or "5CH" in s or "APEX_5CH" in s:
        return "A5C"

    if "PLAX" in s:
        return "PLAX"

    if "PSAX" in s or ("SAX" in s and "PLAX" not in s):
        if any(k in s for k in ("AORT", "GREAT_VESSEL", "AV", "VALVE")):
            return "PSAX_AV"
        if any(k in s for k in ("PAPILLARY", "MV")):
            return "PSAX_MV"
        if any(k in s for k in ("APEX", "APICAL")):
            return "PSAX_AP"
        return "PSAX"

    if s == "AORTA":
        return "SSN"

    return None


def echo_view_label(canonical: Optional[str]) -> Optional[str]:
    m = {
        "A2C": "a2c",
        "A3C": "a3c",
        "A4C": "a4c",
        "A5C": "a5c",
        "PLAX": "plax",
        "PSAX_AV": "psax-av",
        "PSAX_MV": "psax-mv",
        "PSAX_AP": "psax-ap",
    }
    return m.get(canonical)


def echo_qc_label(canonical: Optional[str]) -> Optional[str]:
    m = {
        "A2C": "A2C",
        "A3C": "A3C",
        "A4C": "A4C",
        "A5C": "A5C",
        "PLAX": "PLAX",
        "PSAX_AV": "PSAX AV",
        "PSAX_MV": "PSAX MV",
        "PSAX_AP": "PSAX AP",
        "SSN": "SSN",
    }
    return m.get(canonical)


def echo_prime_label(raw_label: str, canonical: Optional[str]) -> Optional[str]:
    s = normalize_raw_label(raw_label)
    if s.startswith("DOPPLER_A"):
        return "Apical_Doppler"
    if s.startswith("DOPPLER_PLAX"):
        return "Doppler_Parasternal_Long"
    if s.startswith("DOPPLER_PSAX"):
        return "Doppler_Parasternal_Short"

    m = {
        "A2C": "A2C",
        "A3C": "A3C",
        "A4C": "A4C",
        "A5C": "A5C",
        "PLAX": "Parasternal_Long",
        "PSAX": "Parasternal_Short",
        "PSAX_AV": "Parasternal_Short",
        "PSAX_MV": "Parasternal_Short",
        "PSAX_AP": "Parasternal_Short",
        "SSN": "SSN",
        "SUBCOSTAL": "Subcostal",
    }
    return m.get(canonical)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_frames(frames: np.ndarray) -> np.ndarray:
    arr0 = np.asarray(frames)
    if arr0.dtype == np.uint8:
        arr = arr0
    else:
        arr = arr0.astype(np.float32, copy=False)
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    if arr.ndim != 3:
        raise ValueError(f"Expected frames with ndim 2 or 3, got {arr.ndim}")

    if arr.dtype == np.uint8:
        return arr

    mn = float(np.nanmin(arr))
    mx = float(np.nanmax(arr))
    if not np.isfinite(mn) or not np.isfinite(mx) or math.isclose(mx, mn):
        out = np.zeros_like(arr, dtype=np.uint8)
    else:
        out = ((arr - mn) / (mx - mn) * 255.0).clip(0, 255).astype(np.uint8)
    return out


def load_nifti_frames(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    # Keep on-disk dtype (often uint8) to avoid expensive float64 expansion.
    arr = np.asarray(img.dataobj)
    arr = np.squeeze(arr)

    if arr.ndim == 2:
        return normalize_frames(arr)

    if arr.ndim == 3:
        dims = arr.shape
        time_axis = int(np.argmin(dims))
        if dims[time_axis] > 80 and dims[-1] <= 80:
            time_axis = 2
        arr = np.moveaxis(arr, time_axis, 0)
        return normalize_frames(arr)

    if arr.ndim >= 4:
        # Take first channel-like axis if present, then recurse.
        dims = list(arr.shape)
        ch_axis = next((i for i, d in enumerate(dims) if d <= 4), 0)
        arr = np.take(arr, 0, axis=ch_axis)
        arr = np.squeeze(arr)
        if arr.ndim == 2:
            return normalize_frames(arr)
        if arr.ndim == 3:
            dims = arr.shape
            time_axis = int(np.argmin(dims))
            if dims[time_axis] > 80 and dims[-1] <= 80:
                time_axis = 2
            arr = np.moveaxis(arr, time_axis, 0)
            return normalize_frames(arr)

    raise ValueError(f"Unsupported NIfTI shape for {path}: {arr.shape}")


def load_mp4_frames(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    frames: List[np.ndarray] = []
    ok, frame = cap.read()
    while ok:
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        frames.append(gray)
        ok, frame = cap.read()
    cap.release()
    if not frames:
        raise ValueError(f"No frames decoded from {path}")
    arr = np.stack(frames, axis=0)
    return normalize_frames(arr)


def write_mp4_from_frames(frames: np.ndarray, out_path: Path, fps: float = 30.0) -> None:
    ensure_dir(out_path.parent)
    h, w = frames.shape[1], frames.shape[2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video writer for {out_path}")
    for i in range(frames.shape[0]):
        bgr = cv2.cvtColor(frames[i], cv2.COLOR_GRAY2BGR)
        writer.write(bgr)
    writer.release()


def write_middle_jpg(frames: np.ndarray, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    mid = frames[frames.shape[0] // 2]
    if not cv2.imwrite(str(out_path), mid):
        raise RuntimeError(f"Failed to write image {out_path}")


def write_qc_series(frames: np.ndarray, series_dir: Path, view_label: str, sample: Sample) -> None:
    ensure_dir(series_dir)
    width = max(3, len(str(frames.shape[0] - 1)))
    for i in range(frames.shape[0]):
        frame_path = series_dir / f"{i:0{width}d}.jpg"
        if not frame_path.exists():
            cv2.imwrite(str(frame_path), frames[i])

    metadata_path = series_dir / "metadata.txt"
    if not metadata_path.exists():
        uid = slug(sample.sample_id)
        metadata = [
            f"UID,{uid}",
            "Date,",
            "Machine,Unknown",
            f"Length,{frames.shape[0]}",
            f"View,{view_label}",
            "Scale,1.0",
            f"Folder,{sample.dataset}",
            f"Filename,{sample.sample_id}",
        ]
        metadata_path.write_text("\n".join(metadata) + "\n", encoding="utf-8")


def hardlink_or_copy(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def collect_camus_samples() -> List[Sample]:
    out: List[Sample] = []
    base = RAW_ROOT / "CAMUS_public" / "database_nifti"
    for f in base.rglob("*_half_sequence.nii.gz"):
        name = f.name
        if "_gt" in name:
            continue
        raw = "A2C" if "_2CH_" in name else "A4C" if "_4CH_" in name else ""
        if not raw:
            continue
        patient = f.parent.name
        sample_id = f.stem.replace(".nii", "")
        out.append(Sample("CAMUS_public", sample_id, f, "nii", raw, patient))
    return out


def collect_cardiacnet_samples() -> List[Sample]:
    out: List[Sample] = []
    base = RAW_ROOT / "CardiacNet"
    # CardiacNet includes regular *_image.nii files and a subset where files are
    # nested under directories named like "111_image.nii/PHI01.nii".
    # We treat every non-label .nii file as an image sample.
    for f in base.rglob("*.nii"):
        if not f.is_file():
            continue
        if f.name.endswith("_label.nii"):
            continue
        sample_id = f.stem
        out.append(Sample("CardiacNet", sample_id, f, "nii", "A4C", "CardiacNet"))
    return out


def collect_echo_eg_p10_samples() -> List[Sample]:
    out: List[Sample] = []
    xlsx = RAW_ROOT / "echo-eg" / "ECHO_Outputs.xlsx"
    p10_dir = RAW_ROOT / "echo-eg" / "P10_MP4"
    df = pd.read_excel(xlsx, header=None)
    for _, row in df.iterrows():
        file_id = str(row.iloc[0]).strip()
        raw_label = str(row.iloc[1]).strip()
        if not file_id or file_id.lower() == "nan":
            continue
        if normalize_raw_label(raw_label) in {"", "?", "NAN", "VIEW", "1"}:
            continue
        mp4 = p10_dir / f"{file_id}.mp4"
        if not mp4.exists():
            continue
        out.append(Sample("echo-eg_P10", file_id, mp4, "mp4", raw_label, "P10_MP4"))
    return out


def collect_echo_eg_mimicqa_samples() -> List[Sample]:
    out: List[Sample] = []
    base = RAW_ROOT / "echo-eg" / "mimic-iv-echo-ext" / "mimic-iv-echo-ext-mimicechoqa-a-benchmark-dataset-for-echocardiogram-based-visual-question-answering-1.0.0"
    jpath = base / "MIMICEchoQA.json"
    data = json.loads(jpath.read_text(encoding="utf-8"))

    rel_to_label: Dict[str, str] = {}
    for row in data:
        videos = row.get("videos", [])
        if not videos:
            continue
        rel = str(videos[0]).replace("\\", "/").strip()
        label = str(row.get("view", "")).strip()
        if not rel or not label:
            continue
        if rel not in rel_to_label:
            rel_to_label[rel] = label

    for rel, raw_label in rel_to_label.items():
        # JSON uses "mimic-iv-echo/..." while files are under "MIMICEchoQA/..."
        fixed_rel = rel
        if fixed_rel.startswith("mimic-iv-echo/"):
            fixed_rel = "MIMICEchoQA/" + fixed_rel[len("mimic-iv-echo/") :]
        src = base / fixed_rel
        if not src.exists():
            continue
        sample_id = src.stem
        patient = src.parent.name
        out.append(Sample("echo-eg_MIMICEchoQA", sample_id, src, "mp4", raw_label, patient))
    return out


def collect_all_samples() -> List[Sample]:
    samples: List[Sample] = []
    samples.extend(collect_camus_samples())
    samples.extend(collect_cardiacnet_samples())
    samples.extend(collect_echo_eg_p10_samples())
    samples.extend(collect_echo_eg_mimicqa_samples())
    return samples


def main() -> None:
    ensure_dir(ECHO_VIEW_OUT)
    ensure_dir(ECHO_QC_OUT)
    ensure_dir(ECHO_PRIME_OUT)

    samples = collect_all_samples()
    print(f"Collected samples: {len(samples)}")

    ev_rows: List[Dict[str, str]] = []
    qc_rows: List[Dict[str, str]] = []
    ep_rows: List[Dict[str, str]] = []
    skipped_rows: List[Dict[str, str]] = []

    for i, s in enumerate(samples, start=1):
        if i % 100 == 0:
            print(f"Processing {i}/{len(samples)} ...")

        canonical = canonical_view(s.raw_label)
        ev_label = echo_view_label(canonical)
        qc_label = echo_qc_label(canonical)
        ep_label = echo_prime_label(s.raw_label, canonical)

        # echo-view-classifier is not trained on SSN/Subcostal.
        if s.dataset.startswith("echo-eg") and canonical in {"SSN", "SUBCOSTAL"}:
            ev_label = None

        if ev_label is None and qc_label is None and ep_label is None:
            skipped_rows.append(
                {
                    "dataset": s.dataset,
                    "sample_id": s.sample_id,
                    "source_path": str(s.source_path),
                    "raw_label": s.raw_label,
                    "canonical": str(canonical),
                    "reason": "no_model_mapping",
                }
            )
            continue

        need_frames = False
        ev_out_path: Optional[Path] = None
        qc_series_dir: Optional[Path] = None
        ep_out_path: Optional[Path] = None

        if ev_label:
            ev_out_path = ECHO_VIEW_OUT / s.dataset / ev_label / f"{slug(s.sample_id)}.jpg"
            need_frames = need_frames or not ev_out_path.exists()
        if qc_label:
            qc_series_dir = ECHO_QC_OUT / s.dataset / s.patient_id / slug(s.sample_id)
            need_frames = need_frames or not (qc_series_dir / "metadata.txt").exists()
        if ep_label:
            ep_out_path = ECHO_PRIME_OUT / s.dataset / ep_label / f"{slug(s.sample_id)}.mp4"

        frames: Optional[np.ndarray] = None

        # EchoPrime can keep MP4 as-is (hard link/copy), no decoding needed.
        ep_done_by_link = False
        if ep_label and s.source_type == "mp4" and ep_out_path is not None:
            if not ep_out_path.exists():
                hardlink_or_copy(s.source_path, ep_out_path)
            ep_done_by_link = True

        if need_frames or (ep_label and s.source_type == "nii"):
            try:
                if s.source_type == "nii":
                    frames = load_nifti_frames(s.source_path)
                else:
                    frames = load_mp4_frames(s.source_path)
            except Exception as e:
                skipped_rows.append(
                    {
                        "dataset": s.dataset,
                        "sample_id": s.sample_id,
                        "source_path": str(s.source_path),
                        "raw_label": s.raw_label,
                        "canonical": str(canonical),
                        "reason": f"decode_failed:{e}",
                    }
                )
                continue

        if ev_label and ev_out_path is not None:
            if not ev_out_path.exists():
                if frames is None:
                    frames = load_nifti_frames(s.source_path) if s.source_type == "nii" else load_mp4_frames(s.source_path)
                write_middle_jpg(frames, ev_out_path)
            ev_rows.append(
                {
                    "dataset": s.dataset,
                    "sample_id": s.sample_id,
                    "source_path": str(s.source_path),
                    "raw_label": s.raw_label,
                    "canonical": str(canonical),
                    "model_label": ev_label,
                    "output_path": str(ev_out_path),
                }
            )

        if qc_label and qc_series_dir is not None:
            if not (qc_series_dir / "metadata.txt").exists():
                if frames is None:
                    frames = load_nifti_frames(s.source_path) if s.source_type == "nii" else load_mp4_frames(s.source_path)
                write_qc_series(frames, qc_series_dir, qc_label, s)
            qc_rows.append(
                {
                    "dataset": s.dataset,
                    "sample_id": s.sample_id,
                    "source_path": str(s.source_path),
                    "raw_label": s.raw_label,
                    "canonical": str(canonical),
                    "model_label": qc_label,
                    "output_path": str(qc_series_dir),
                }
            )

        if ep_label and ep_out_path is not None:
            if not ep_out_path.exists():
                if ep_done_by_link:
                    pass
                else:
                    if frames is None:
                        frames = load_nifti_frames(s.source_path) if s.source_type == "nii" else load_mp4_frames(s.source_path)
                    write_mp4_from_frames(frames, ep_out_path)
            ep_rows.append(
                {
                    "dataset": s.dataset,
                    "sample_id": s.sample_id,
                    "source_path": str(s.source_path),
                    "raw_label": s.raw_label,
                    "canonical": str(canonical),
                    "model_label": ep_label,
                    "output_path": str(ep_out_path),
                }
            )

    def write_manifest(path: Path, rows: List[Dict[str, str]]) -> None:
        ensure_dir(path.parent)
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        fields = list(rows[0].keys())
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    write_manifest(ECHO_VIEW_OUT / "manifest.csv", ev_rows)
    write_manifest(ECHO_QC_REPO / "processed_datasets" / "manifest.csv", qc_rows)
    write_manifest(ECHO_PRIME_OUT / "manifest.csv", ep_rows)
    write_manifest(ROOT / "processing_skipped_samples.csv", skipped_rows)

    print("Done.")
    print(f"echo-view-classifier samples: {len(ev_rows)}")
    print(f"EchocardiographyQC samples: {len(qc_rows)}")
    print(f"EchoPrime samples: {len(ep_rows)}")
    print(f"Skipped samples: {len(skipped_rows)}")


if __name__ == "__main__":
    main()
