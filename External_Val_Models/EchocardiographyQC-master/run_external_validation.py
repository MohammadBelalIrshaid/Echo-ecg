from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validation_utils import compute_classification_metrics, save_standard_outputs


CLASS_ORDER = ["PLAX", "PSAX AV", "PSAX MV", "PSAX AP", "A2C", "A3C", "A4C", "A5C", "SSN"]


def parse_args() -> argparse.Namespace:
    repo_dir = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description=(
            "Evaluate EchocardiographyQC predictions from conf.npy files.\n"
            "Run dicom-inferrer first, then point --infer-results to its output root."
        )
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=repo_dir / "processed_datasets" / "manifest.csv",
    )
    p.add_argument(
        "--infer-results",
        type=Path,
        default=repo_dir / "processed_datasets" / "inferResults",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "external_validation_results",
    )
    return p.parse_args()


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def main() -> None:
    args = parse_args()
    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    repo_dir = Path(__file__).resolve().parent
    # User-friendly handling if placeholder string was passed literally.
    if str(args.infer_results).strip().upper() == "PATH_TO_INFER_RESULTS":
        candidates = [
            repo_dir / "processed_datasets" / "inferResults",
            repo_dir / "inferResults",
        ]
        found = next((c for c in candidates if c.exists()), None)
        if found is not None:
            args.infer_results = found

    if not args.infer_results.exists():
        raise FileNotFoundError(
            f"Inference results folder not found: {args.infer_results}\n"
            "Run dicom-inferrer-1.0.0.exe on processed_datasets/SourceImage first.\n"
            "Then re-run with --infer-results <that_output_folder>."
        )

    mdf = pd.read_csv(args.manifest)
    mdf = mdf[mdf["model_label"].isin(CLASS_ORDER)].copy()
    if mdf.empty:
        raise RuntimeError("No valid ground-truth samples in manifest.")

    # UID is the series folder name used in metadata.txt and expected inferResults folder name.
    mdf["uid"] = mdf["output_path"].apply(lambda x: Path(str(x)).name)
    gt_map = {
        row.uid: {
            "true_label": row.model_label,
            "dataset": row.dataset,
            "sample_id": row.sample_id,
            "source_path": row.source_path,
            "raw_label": row.raw_label,
            "canonical": row.canonical,
            "output_path": row.output_path,
        }
        for row in mdf.itertuples(index=False)
    }

    rows = []
    probs_list = []
    infer_uid_set = set()
    for conf_path in args.infer_results.rglob("conf.npy"):
        uid = conf_path.parent.name
        infer_uid_set.add(uid)
        if uid not in gt_map:
            continue
        arr = np.load(conf_path)
        if arr.ndim != 2:
            continue

        if arr.shape[0] == len(CLASS_ORDER):
            frame_logits = arr.T  # (T, C)
        elif arr.shape[1] == len(CLASS_ORDER):
            frame_logits = arr
        else:
            continue

        frame_probs = softmax(frame_logits, axis=1)
        series_probs = frame_probs.mean(axis=0)
        pred_idx = int(np.argmax(series_probs))
        pred_label = CLASS_ORDER[pred_idx]

        base = gt_map[uid]
        row = {
            "uid": uid,
            "dataset": base["dataset"],
            "sample_id": base["sample_id"],
            "source_path": base["source_path"],
            "raw_label": base["raw_label"],
            "canonical": base["canonical"],
            "true_label": base["true_label"],
            "pred_label": pred_label,
            "correct": int(base["true_label"] == pred_label),
            "conf_path": str(conf_path),
        }
        for i, cls in enumerate(CLASS_ORDER):
            row[f"prob_{cls.replace(' ', '_')}"] = float(series_probs[i])
        rows.append(row)
        probs_list.append(series_probs)

    if not rows:
        manifest_uids = set(gt_map.keys())
        overlap = manifest_uids & infer_uid_set
        manifest_examples = sorted(list(manifest_uids))[:8]
        infer_examples = sorted(list(infer_uid_set))[:8]
        raise RuntimeError(
            "No matched predictions found.\n"
            "Ensure --infer-results contains conf.npy folders named with UIDs from processed_datasets/SourceImage.\n"
            f"Manifest UIDs: {len(manifest_uids)} | Infer UIDs: {len(infer_uid_set)} | Overlap: {len(overlap)}\n"
            f"Manifest UID examples: {manifest_examples}\n"
            f"Infer UID examples: {infer_examples}"
        )

    pred_df = pd.DataFrame(rows)
    probs = np.vstack(probs_list)
    summary, per_label, cm = compute_classification_metrics(
        y_true_labels=pred_df["true_label"].tolist(),
        y_pred_labels=pred_df["pred_label"].tolist(),
        class_order=CLASS_ORDER,
        probs=probs,
    )
    save_standard_outputs(
        args.output_dir,
        pred_df,
        summary,
        per_label,
        cm,
        CLASS_ORDER,
        probs=probs,
        model_name="EchocardiographyQC-master",
    )

    print("Done.")
    print(f"Matched series: {len(pred_df)}")
    print(f"Accuracy: {summary['accuracy']:.4f}")
    print(f"Macro-F1: {summary['macro_f1']:.4f}")
    print(f"MCC: {summary['mcc']:.4f}")


if __name__ == "__main__":
    main()
