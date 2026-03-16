from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validation_utils import compute_classification_metrics, save_standard_outputs


PSAX_VARIANTS = {"Parasternal_Short", "Doppler_Parasternal_Short"}
COLLAPSED_PSAX = "PSAX"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="External validation for EchoPrime view classification.")
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parent / "processed_datasets" / "manifest.csv",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "external_validation_results",
    )
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return p.parse_args()


class ViewOnlyRuntime:
    def __init__(self, view_classifier: torch.nn.Module, device: torch.device):
        self.view_classifier = view_classifier
        self.device = device
        self.frames_to_take = 32
        self.frame_stride = 2
        self.video_size = 224
        self.mean = torch.tensor([29.110628, 28.076836, 29.096405]).reshape(3, 1, 1, 1)
        self.std = torch.tensor([47.989223, 46.456997, 47.20083]).reshape(3, 1, 1, 1)


def load_view_only_runtime(repo_dir: Path, device_arg: str, class_count: int) -> ViewOnlyRuntime:
    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this environment.")

    vc_path = repo_dir / "model_data" / "weights" / "view_classifier.pt"
    if not vc_path.exists():
        raise FileNotFoundError(
            f"EchoPrime view-classifier weights not found: {vc_path}\n"
            "Download model_data.zip from EchoPrime release and extract under this repo."
        )

    vc_state_dict = torch.load(str(vc_path), map_location=device)
    view_classifier = torchvision.models.convnext_base()
    view_classifier.classifier[-1] = torch.nn.Linear(view_classifier.classifier[-1].in_features, class_count)
    view_classifier.load_state_dict(vc_state_dict)
    view_classifier.to(device)
    view_classifier.eval()
    for param in view_classifier.parameters():
        param.requires_grad = False
    return ViewOnlyRuntime(view_classifier=view_classifier, device=device)


def collapse_label(label: str) -> str:
    if label in PSAX_VARIANTS:
        return COLLAPSED_PSAX
    return label


def collapse_probabilities(probs: np.ndarray, class_order: list[str]) -> tuple[np.ndarray, list[str]]:
    collapsed_order: list[str] = []
    for cls in class_order:
        ccls = collapse_label(cls)
        if ccls not in collapsed_order:
            collapsed_order.append(ccls)

    collapsed_probs = np.zeros((probs.shape[0], len(collapsed_order)), dtype=np.float64)
    idx_map = {cls: i for i, cls in enumerate(collapsed_order)}
    for j, cls in enumerate(class_order):
        collapsed_probs[:, idx_map[collapse_label(cls)]] += probs[:, j]
    return collapsed_probs, collapsed_order


def load_video_tensor(mp4_path: Path, ep, utils_module) -> torch.Tensor:
    cap = cv2.VideoCapture(str(mp4_path))
    frames = []
    ok, frame = cap.read()
    while ok:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        ok, frame = cap.read()
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded: {mp4_path}")

    x = np.zeros((len(frames), 224, 224, 3), dtype=np.float32)
    for i, fr in enumerate(frames):
        x[i] = utils_module.crop_and_scale(fr)

    t = torch.as_tensor(x, dtype=torch.float32).permute(3, 0, 1, 2)  # (3, T, 224, 224)
    t.sub_(ep.mean).div_(ep.std)

    if t.shape[1] < ep.frames_to_take:
        pad = torch.zeros((3, ep.frames_to_take - t.shape[1], ep.video_size, ep.video_size), dtype=torch.float32)
        t = torch.cat((t, pad), dim=1)
    t = t[:, 0 : (0 + ep.frames_to_take) : ep.frame_stride, :, :]  # -> (3,16,224,224)
    return t


def main() -> None:
    args = parse_args()

    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    # EchoPrime expects repo-relative assets/model_data paths.
    repo_dir = Path(__file__).resolve().parent
    old_cwd = Path.cwd()
    try:
        import os

        os.chdir(str(repo_dir))
        if str(repo_dir) not in sys.path:
            sys.path.insert(0, str(repo_dir))

        try:
            import utils
        except Exception as e:
            raise RuntimeError(
                "Failed importing EchoPrime utils. Ensure repo requirements are installed "
                "(notably torch/torchvision/transformers/opencv/pydicom).\n"
                f"Original error: {e}"
            )

        raw_class_order = list(utils.COARSE_VIEWS)
        ep = load_view_only_runtime(repo_dir, args.device, len(raw_class_order))

        df = pd.read_csv(args.manifest)
        df = df[df["model_label"].isin(raw_class_order)].copy()
        df = df[df["output_path"].apply(lambda x: Path(str(x)).exists())].reset_index(drop=True)
        if df.empty:
            raise RuntimeError("No valid evaluation samples found in manifest/output paths.")

        rows = []
        probs_all = []
        for start in range(0, len(df), args.batch_size):
            chunk = df.iloc[start : start + args.batch_size]
            vids = []
            keep_rows = []
            for r in chunk.itertuples(index=False):
                try:
                    vids.append(load_video_tensor(Path(r.output_path), ep, utils))
                    keep_rows.append(r)
                except Exception as e:
                    print(f"Skipping {r.sample_id}: {e}")

            if not vids:
                continue

            stack = torch.stack(vids, dim=0)  # (N,3,16,224,224)
            first_frames = stack[:, :, 0, :, :].to(ep.device)
            with torch.no_grad():
                logits = ep.view_classifier(first_frames)
                probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            pred_idx = np.argmax(probs, axis=1)
            pred_labels = [raw_class_order[i] for i in pred_idx]

            for i, r in enumerate(keep_rows):
                row = {
                    "dataset": r.dataset,
                    "sample_id": r.sample_id,
                    "source_path": r.source_path,
                    "raw_label": r.raw_label,
                    "canonical": r.canonical,
                    "true_label": r.model_label,
                    "pred_label": pred_labels[i],
                    "correct": int(r.model_label == pred_labels[i]),
                    "output_path": r.output_path,
                }
                rows.append(row)
                probs_all.append(probs[i])

        if not rows:
            raise RuntimeError("No predictions produced.")

        pred_df = pd.DataFrame(rows).reset_index(drop=True)
        probs_arr_raw = np.vstack(probs_all)
        probs_arr, class_order = collapse_probabilities(probs_arr_raw, raw_class_order)
        pred_df["true_label_raw"] = pred_df["true_label"]
        pred_df["pred_label_raw"] = pred_df["pred_label"]
        pred_df["true_label"] = pred_df["true_label_raw"].map(collapse_label)
        pred_df["pred_label"] = pred_df["pred_label_raw"].map(collapse_label)
        pred_df["correct"] = (pred_df["true_label"] == pred_df["pred_label"]).astype(int)
        for j, cls in enumerate(class_order):
            pred_df[f"prob_{cls}"] = probs_arr[:, j]

        summary, per_label, cm = compute_classification_metrics(
            y_true_labels=pred_df["true_label"].tolist(),
            y_pred_labels=pred_df["pred_label"].tolist(),
            class_order=class_order,
            probs=probs_arr,
        )
        save_standard_outputs(
            args.output_dir,
            pred_df,
            summary,
            per_label,
            cm,
            class_order,
            probs=probs_arr,
            model_name="EchoPrime-main",
        )

        print("Done.")
        print(f"Samples: {len(pred_df)}")
        print(f"Accuracy: {summary['accuracy']:.4f}")
        print(f"Macro-F1: {summary['macro_f1']:.4f}")
        print(f"MCC: {summary['mcc']:.4f}")
    finally:
        import os

        os.chdir(str(old_cwd))


if __name__ == "__main__":
    main()
