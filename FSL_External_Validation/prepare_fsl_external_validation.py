from __future__ import annotations

import csv
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import cv2
import pandas as pd


ROOT = Path(r"E:\Hourani\Echo-Ecg")
FSL_ROOT = ROOT / "FSL_External_Validation"
MODEL_DIR = FSL_ROOT / "model"
TRIALS_DIR = FSL_ROOT / "trials"
SOURCE_MANIFEST = ROOT / "External_Val_Models" / "EchoPrime-main" / "processed_datasets" / "manifest.csv"
P10_QUALITY_XLSX = ROOT / "Raw_Datasets" / "echo-eg" / "ECHO_Outputs.xlsx"

QUALITY_RANK = {
    "Good": 0,
    "Intermediate": 1,
    "Bad": 2,
    "Very Bad": 3,
    "Terrible": 4,
}
QUALITY_ORDER = list(QUALITY_RANK.keys())
FRAME_TRIALS = (1, 5, 10)
FULL_VIDEO_TRIAL_COUNT = 1


@dataclass
class SelectedSample:
    label: str
    sample_id: str
    source_video: Path
    quality: str
    quality_rank: int
    selection_index: int


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def hardlink_or_copy(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_middle_frame(src_video: Path, dst_image: Path) -> None:
    ensure_dir(dst_image.parent)
    if dst_image.exists():
        return

    cap = cv2.VideoCapture(str(src_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {src_video}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mid_idx = max(0, frame_count // 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not decode middle frame from: {src_video}")
    if not cv2.imwrite(str(dst_image), frame):
        raise RuntimeError(f"Could not write image: {dst_image}")


def load_quality_table() -> pd.DataFrame:
    df = pd.read_excel(P10_QUALITY_XLSX, header=None, usecols=[0, 4])
    df.columns = ["sample_id", "quality"]
    df["sample_id"] = df["sample_id"].astype(str).str.strip()
    df["quality"] = df["quality"].astype(str).str.strip()
    df = df[
        df["sample_id"].ne("nan")
        & df["quality"].ne("nan")
        & df["quality"].ne("Quality")
    ].drop_duplicates(subset=["sample_id"], keep="first")
    df["quality_rank"] = df["quality"].map(QUALITY_RANK).fillna(999).astype(int)
    return df


def load_ranked_p10_samples() -> pd.DataFrame:
    manifest_df = pd.read_csv(SOURCE_MANIFEST)
    manifest_df = manifest_df[manifest_df["dataset"] == "echo-eg_P10"].copy()
    quality_df = load_quality_table()
    merged = manifest_df.merge(quality_df, on="sample_id", how="left")
    merged["quality"] = merged["quality"].fillna("Unknown")
    merged["quality_rank"] = merged["quality_rank"].fillna(999).astype(int)
    merged["source_video"] = merged["output_path"].map(Path)
    merged = merged[merged["source_video"].map(Path.exists)].copy()
    merged = merged.sort_values(
        by=["model_label", "quality_rank", "sample_id"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    return merged


def select_top_samples_per_label(df: pd.DataFrame, count: int) -> List[SelectedSample]:
    selections: List[SelectedSample] = []
    for label in sorted(df["model_label"].unique()):
        label_df = df[df["model_label"] == label].copy().head(count)
        for idx, row in enumerate(label_df.itertuples(index=False), start=1):
            selections.append(
                SelectedSample(
                    label=label,
                    sample_id=row.sample_id,
                    source_video=row.source_video,
                    quality=row.quality,
                    quality_rank=int(row.quality_rank),
                    selection_index=idx,
                )
            )
    return selections


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_frame_trial(trial_name: str, selections: List[SelectedSample]) -> List[Dict[str, object]]:
    trial_dir = TRIALS_DIR / trial_name
    frames_dir = trial_dir / "frames"
    rows: List[Dict[str, object]] = []
    for item in selections:
        label_dir = frames_dir / item.label
        dst_image = label_dir / f"{item.selection_index:02d}_{item.sample_id}.jpg"
        write_middle_frame(item.source_video, dst_image)
        rows.append(
            {
                "trial": trial_name,
                "modality": "frame",
                "label": item.label,
                "selection_index": item.selection_index,
                "sample_id": item.sample_id,
                "quality": item.quality,
                "quality_rank": item.quality_rank,
                "source_video": str(item.source_video),
                "output_asset": str(dst_image),
            }
        )
    write_csv(trial_dir / "manifest.csv", rows)
    return rows


def build_video_trial(trial_name: str, selections: List[SelectedSample]) -> List[Dict[str, object]]:
    trial_dir = TRIALS_DIR / trial_name
    videos_dir = trial_dir / "videos"
    rows: List[Dict[str, object]] = []
    for item in selections:
        label_dir = videos_dir / item.label
        dst_video = label_dir / f"{item.selection_index:02d}_{item.sample_id}.mp4"
        hardlink_or_copy(item.source_video, dst_video)
        rows.append(
            {
                "trial": trial_name,
                "modality": "video",
                "label": item.label,
                "selection_index": item.selection_index,
                "sample_id": item.sample_id,
                "quality": item.quality,
                "quality_rank": item.quality_rank,
                "source_video": str(item.source_video),
                "output_asset": str(dst_video),
            }
        )
    write_csv(trial_dir / "manifest.csv", rows)
    return rows


def write_summary_files(ranked_df: pd.DataFrame, trial_rows: Dict[str, List[Dict[str, object]]]) -> None:
    summary_rows: List[Dict[str, object]] = []
    for label in sorted(ranked_df["model_label"].unique()):
        label_df = ranked_df[ranked_df["model_label"] == label]
        quality_counts = label_df["quality"].value_counts().to_dict()
        row: Dict[str, object] = {
            "label": label,
            "available_total": int(len(label_df)),
            "available_good": int(quality_counts.get("Good", 0)),
            "available_intermediate": int(quality_counts.get("Intermediate", 0)),
            "available_bad": int(quality_counts.get("Bad", 0)),
            "available_very_bad": int(quality_counts.get("Very Bad", 0)),
            "available_terrible": int(quality_counts.get("Terrible", 0)),
        }
        for trial_name, rows in trial_rows.items():
            row[f"selected_{trial_name}"] = sum(1 for r in rows if r["label"] == label)
        summary_rows.append(row)
    write_csv(FSL_ROOT / "selection_summary.csv", summary_rows)


def write_readme(ranked_df: pd.DataFrame) -> None:
    labels = sorted(ranked_df["model_label"].unique())
    ssn_count = int((ranked_df["model_label"] == "SSN").sum())
    lines = [
        "# FSL External Validation",
        "",
        "## Model",
        "",
        "Downloaded from `https://huggingface.co/unsloth/Qwen3.5-9B-GGUF`.",
        "Files included:",
        "- `Qwen3.5-9B-Q4_K_M.gguf`",
        "- `mmproj-F16.gguf`",
        "- `README.md` from the Hugging Face repo",
        "",
        "Assumption used: `Q4_K_M` plus `mmproj-F16` for a practical local multimodal GGUF setup.",
        "",
        "## Trial Dataset",
        "",
        "Support examples are sourced from `echo-eg_P10` in `External_Val_Models/EchoPrime-main/processed_datasets/manifest.csv`.",
        "Ranking is based on column E in `Raw_Datasets/echo-eg/ECHO_Outputs.xlsx` using:",
        "- `Good`",
        "- `Intermediate`",
        "- `Bad`",
        "- `Very Bad`",
        "- `Terrible`",
        "",
        "For each label, the best available examples were selected in that order.",
        f"Labels included: {', '.join(labels)}.",
        f"`SSN` has only {ssn_count} available P10 examples, so the 10-shot frame trial is capped at 7 for that label.",
        "",
        "Trials created:",
        "- `frames_1_per_label`",
        "- `frames_5_per_label`",
        "- `frames_10_per_label`",
        "- `full_video_1_per_label`",
        "",
        "Each trial folder includes a `manifest.csv`.",
        "A cross-trial availability summary is written to `selection_summary.csv`.",
    ]
    (FSL_ROOT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dir(MODEL_DIR)
    ensure_dir(TRIALS_DIR)

    ranked_df = load_ranked_p10_samples()
    (FSL_ROOT / "ranked_p10_candidates.csv").write_text(
        ranked_df.drop(columns=["source_video"]).to_csv(index=False),
        encoding="utf-8",
    )

    trial_rows: Dict[str, List[Dict[str, object]]] = {}
    for n in FRAME_TRIALS:
        trial_name = f"frames_{n}_per_label"
        selections = select_top_samples_per_label(ranked_df, n)
        trial_rows[trial_name] = build_frame_trial(trial_name, selections)

    video_trial_name = f"full_video_{FULL_VIDEO_TRIAL_COUNT}_per_label"
    video_selections = select_top_samples_per_label(ranked_df, FULL_VIDEO_TRIAL_COUNT)
    trial_rows[video_trial_name] = build_video_trial(video_trial_name, video_selections)

    write_summary_files(ranked_df, trial_rows)
    write_readme(ranked_df)

    print(f"Prepared FSL external validation assets under: {FSL_ROOT}")
    for trial_name, rows in trial_rows.items():
        print(f"{trial_name}: {len(rows)} assets")


if __name__ == "__main__":
    main()
