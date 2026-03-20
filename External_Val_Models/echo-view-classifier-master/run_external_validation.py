from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validation_utils import compute_classification_metrics, save_standard_outputs


CLASS_ORDER = ["plax", "psax-av", "psax-mv", "psax-ap", "a4c", "a5c", "a3c", "a2c"]


def parse_args() -> argparse.Namespace:
    repo_dir = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="External validation for echo-view-classifier.")
    p.add_argument(
        "--manifest",
        type=Path,
        default=repo_dir / "processed_datasets" / "manifest.csv",
    )
    p.add_argument(
        "--model",
        type=Path,
        default=repo_dir / "model" / "mymodel_echocv_500-500-8_adam_16_0.9394.h5",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "external_validation_results",
    )
    p.add_argument("--batch-size", type=int, default=32)
    return p.parse_args()


def load_eval_data(manifest_path: Path) -> pd.DataFrame:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    df = pd.read_csv(manifest_path)
    df = df[df["model_label"].isin(CLASS_ORDER)].copy()
    df = df[df["output_path"].apply(lambda x: Path(str(x)).exists())].copy()
    if df.empty:
        raise RuntimeError("No valid samples found in manifest/output paths.")
    return df.reset_index(drop=True)


def _read_image_robust(path: str) -> np.ndarray | None:
    # Standard OpenCV path loader.
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is not None:
        return img
    # Fallback for occasional Windows/OpenCV path issues.
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        if buf.size == 0:
            return None
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def load_images(paths: list[str]) -> tuple[np.ndarray | None, list[int], list[str]]:
    # Pre-allocate once and fill in-place to avoid extra allocation from np.stack.
    x = np.empty((len(paths), 224, 224, 3), dtype=np.float32)
    n_good = 0
    good_idx: list[int] = []
    bad_paths: list[str] = []
    for i, p in enumerate(paths):
        img = _read_image_robust(p)
        if img is None:
            bad_paths.append(p)
            continue
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x[n_good] = img.astype(np.float32, copy=False)
        n_good += 1
        good_idx.append(i)
    if n_good == 0:
        return None, good_idx, bad_paths
    return x[:n_good], good_idx, bad_paths


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        repo_dir = Path(__file__).resolve().parent
        fallback = repo_dir / "mymodel_echocv_500-500-8_adam_16_0.9394.h5"
        if fallback.exists():
            args.model = fallback
        else:
            raise FileNotFoundError(
                f"Model weights not found: {args.model}\n"
                f"Also checked fallback: {fallback}\n"
                "Expected path from README: ./model/mymodel_echocv_500-500-8_adam_16_0.9394.h5"
            )

    df = load_eval_data(args.manifest)
    # Inference-only load to avoid legacy optimizer config incompatibilities
    # (e.g., old H5 files storing Adam(lr=...) with newer Keras versions).
    model = load_model(args.model, compile=False)

    probs_chunks = []
    kept_chunks = []
    skipped_image_rows = []

    max_bs = max(1, int(args.batch_size))
    cur_bs = max_bs
    idx = 0
    while idx < len(df):
        chunk = df.iloc[idx : idx + cur_bs]
        try:
            x, good_idx, bad_paths = load_images(chunk["output_path"].tolist())
            if bad_paths:
                for bp in bad_paths:
                    skipped_image_rows.append({"output_path": bp, "reason": "cv2_read_failed"})
            if x is not None:
                probs = model.predict(x, verbose=0)
                probs_chunks.append(probs)
                kept_chunks.append(chunk.iloc[good_idx].copy())
            idx += len(chunk)
            if cur_bs < max_bs:
                cur_bs = min(max_bs, cur_bs * 2)
            del x
            gc.collect()
        except MemoryError:
            if cur_bs == 1:
                # Skip this sample if even single-sample allocation fails.
                bad_path = str(chunk.iloc[0]["output_path"])
                skipped_image_rows.append({"output_path": bad_path, "reason": "memory_error"})
                idx += 1
            else:
                cur_bs = max(1, cur_bs // 2)
            gc.collect()

    if not probs_chunks:
        raise RuntimeError("No readable images were found for inference.")

    kept_df = pd.concat(kept_chunks, axis=0, ignore_index=True)
    probs = np.vstack(probs_chunks)
    pred_idx = np.argmax(probs, axis=1)
    pred_labels = [CLASS_ORDER[i] for i in pred_idx]
    true_labels = kept_df["model_label"].tolist()

    predictions_df = kept_df[
        ["dataset", "sample_id", "source_path", "raw_label", "canonical", "model_label", "output_path"]
    ].copy()
    predictions_df = predictions_df.rename(columns={"model_label": "true_label"})
    predictions_df["pred_label"] = pred_labels
    predictions_df["correct"] = (predictions_df["true_label"] == predictions_df["pred_label"]).astype(int)
    for i, cls in enumerate(CLASS_ORDER):
        predictions_df[f"prob_{cls}"] = probs[:, i]

    summary, per_label, cm = compute_classification_metrics(
        y_true_labels=true_labels,
        y_pred_labels=pred_labels,
        class_order=CLASS_ORDER,
        probs=probs,
    )
    save_standard_outputs(
        args.output_dir,
        predictions_df,
        summary,
        per_label,
        cm,
        CLASS_ORDER,
        probs=probs,
        model_name="echo-view-classifier-master",
    )
    if skipped_image_rows:
        pd.DataFrame(skipped_image_rows).to_csv(args.output_dir / "skipped_images.csv", index=False)

    print("Done.")
    print(f"Samples evaluated: {len(predictions_df)}")
    if skipped_image_rows:
        print(f"Samples skipped (read failure): {len(skipped_image_rows)}")
    print(f"Accuracy: {summary['accuracy']:.4f}")
    print(f"Macro-F1 (all labels): {summary['macro_f1_all_labels']:.4f}")
    if summary["macro_f1_present_labels"] is not None:
        print(f"Macro-F1 (present labels only): {summary['macro_f1_present_labels']:.4f}")
    print(f"MCC: {summary['mcc']:.4f}")


if __name__ == "__main__":
    main()
