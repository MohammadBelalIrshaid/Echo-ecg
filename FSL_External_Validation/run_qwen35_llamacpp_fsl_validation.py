from __future__ import annotations

import argparse
import base64
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

import cv2
import numpy as np
import pandas as pd


ROOT = Path(r"E:\Hourani\Echo-Ecg")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from External_Val_Models.validation_utils import compute_classification_metrics, save_standard_outputs


FSL_ROOT = ROOT / "FSL_External_Validation"
DEFAULT_MODEL = FSL_ROOT / "model" / "Qwen3.5-9B-Q8_0.gguf"
DEFAULT_MMPROJ = FSL_ROOT / "model" / "mmproj-F16.gguf"
DEFAULT_TRIALS_DIR = FSL_ROOT / "trials"
DEFAULT_QUERY_MANIFEST = ROOT / "External_Val_Models" / "EchoPrime-main" / "processed_datasets" / "manifest.csv"
CACHE_DIR = FSL_ROOT / "cache"
RESULTS_DIR = FSL_ROOT / "results"


LABEL_LINE_RE = re.compile(r'"label"\s*:\s*"([^"]+)"', re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Few-shot external validation with Qwen 3.5 9B GGUF on llama.cpp."
    )
    p.add_argument(
        "--llama-server",
        type=Path,
        required=True,
        help="Path to llama.cpp's llama-server executable.",
    )
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--mmproj", type=Path, default=DEFAULT_MMPROJ)
    p.add_argument(
        "--support-trial",
        choices=[
            "frames_1_per_label",
            "frames_5_per_label",
            "frames_10_per_label",
            "full_video_1_per_label",
        ],
        default="frames_5_per_label",
    )
    p.add_argument(
        "--trials-dir",
        type=Path,
        default=DEFAULT_TRIALS_DIR,
    )
    p.add_argument(
        "--query-manifest",
        type=Path,
        default=DEFAULT_QUERY_MANIFEST,
    )
    p.add_argument(
        "--query-dataset",
        default="echo-eg_MIMICEchoQA",
        help="Dataset name to evaluate from the query manifest.",
    )
    p.add_argument(
        "--query-render-mode",
        choices=["middle_frame", "contact_sheet"],
        default="contact_sheet",
    )
    p.add_argument(
        "--support-video-render-mode",
        choices=["middle_frame", "contact_sheet"],
        default="contact_sheet",
    )
    p.add_argument("--contact-sheet-frames", type=int, default=9)
    p.add_argument("--contact-sheet-cols", type=int, default=3)
    p.add_argument("--max-queries", type=int, default=0, help="0 means all queries.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--ctx-size", type=int, default=16384)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--gpu-layers", type=int, default=99)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=0.1)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--server-start-timeout", type=int, default=300)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to FSL_External_Validation/results/<support-trial>__<query-dataset>",
    )
    return p.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def post_json(url: str, payload: Dict) -> Dict:
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_json(url: str) -> Dict:
    req = Request(url, headers={"Accept": "application/json"})
    with urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_for_server(base_url: str, timeout_seconds: int) -> None:
    deadline = time.time() + timeout_seconds
    last_error: Optional[str] = None
    while time.time() < deadline:
        try:
            data = get_json(f"{base_url}/v1/models")
            if data.get("data"):
                return
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
        time.sleep(2)
    raise RuntimeError(f"Timed out waiting for llama-server at {base_url}. Last error: {last_error}")


def launch_llama_server(args: argparse.Namespace) -> subprocess.Popen:
    cmd = [
        str(args.llama_server),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--model",
        str(args.model),
        "--mmproj",
        str(args.mmproj),
        "--ctx-size",
        str(args.ctx_size),
        "--threads",
        str(args.threads),
        "--gpu-layers",
        str(args.gpu_layers),
    ]
    return subprocess.Popen(cmd, cwd=str(args.llama_server.parent))


def read_support_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"trial", "modality", "label", "sample_id", "output_asset"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Support manifest missing columns: {sorted(missing)}")
    return df


def read_query_manifest(path: Path, dataset_name: str, max_queries: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["dataset"] == dataset_name].copy()
    df = df[df["output_path"].apply(lambda x: Path(str(x)).exists())].copy()
    if max_queries > 0:
        df = df.head(max_queries).copy()
    if df.empty:
        raise RuntimeError(f"No queries found for dataset '{dataset_name}' in {path}")
    return df.reset_index(drop=True)


def encode_image_base64(path: Path) -> str:
    raw = path.read_bytes()
    return base64.b64encode(raw).decode("ascii")


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def sample_video_frames(video_path: Path, count: int) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        raise RuntimeError(f"No frames reported for video: {video_path}")

    if count == 1:
        indices = [frame_count // 2]
    else:
        indices = sorted({int(round(x)) for x in np.linspace(0, frame_count - 1, num=count)})

    frames: List[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"Could not decode frames from video: {video_path}")
    return frames


def make_contact_sheet(frames: List[np.ndarray], cols: int) -> np.ndarray:
    resized: List[np.ndarray] = []
    target_h, target_w = 224, 224
    for frame in frames:
        resized.append(cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA))

    rows = int(math.ceil(len(resized) / cols))
    blank = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    while len(resized) < rows * cols:
        resized.append(blank.copy())

    row_images = []
    for r in range(rows):
        start = r * cols
        row_images.append(np.hstack(resized[start : start + cols]))
    return np.vstack(row_images)


def render_video_asset(
    src_video: Path,
    cache_path: Path,
    mode: str,
    frame_count: int,
    cols: int,
) -> Path:
    ensure_dir(cache_path.parent)
    if cache_path.exists():
        return cache_path

    frames = sample_video_frames(src_video, 1 if mode == "middle_frame" else frame_count)
    if mode == "middle_frame":
        image = frames[0]
    else:
        image = make_contact_sheet(frames, cols)

    if not cv2.imwrite(str(cache_path), image):
        raise RuntimeError(f"Could not write rendered asset: {cache_path}")
    return cache_path


def support_asset_image_path(row: pd.Series, args: argparse.Namespace) -> Path:
    asset_path = Path(str(row["output_asset"]))
    modality = str(row["modality"])
    if modality == "frame":
        return asset_path
    cache_path = CACHE_DIR / "support" / args.support_trial / row["label"] / f"{safe_name(row['sample_id'])}.jpg"
    return render_video_asset(
        src_video=asset_path,
        cache_path=cache_path,
        mode=args.support_video_render_mode,
        frame_count=args.contact_sheet_frames,
        cols=args.contact_sheet_cols,
    )


def query_asset_image_path(row: pd.Series, args: argparse.Namespace) -> Path:
    src_video = Path(str(row["output_path"]))
    cache_path = CACHE_DIR / "query" / args.query_dataset / f"{safe_name(row['sample_id'])}.jpg"
    return render_video_asset(
        src_video=src_video,
        cache_path=cache_path,
        mode=args.query_render_mode,
        frame_count=args.contact_sheet_frames,
        cols=args.contact_sheet_cols,
    )


def build_messages(
    label_order: List[str],
    support_df: pd.DataFrame,
    query_image: Path,
) -> List[Dict]:
    content: List[Dict] = []
    label_str = ", ".join(label_order)
    content.append(
        {
            "type": "text",
            "text": (
                "Classify the echocardiography view into exactly one label from this list: "
                f"{label_str}. "
                "Use the support examples first, then classify the final query image. "
                'Return JSON only in the form {"label": "<one label>", "confidence": <0 to 1>, "reason": "<short>"}.'  # noqa: E501
            ),
        }
    )

    for row in support_df.itertuples(index=False):
        support_image = Path(str(row.rendered_image))
        content.append(
            {
                "type": "text",
                "text": (
                    f"Support example. Label: {row.label}. "
                    f"Sample ID: {row.sample_id}. "
                    f"Quality: {row.quality}."
                ),
            }
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_image_base64(support_image)}"
                },
            }
        )

    content.append({"type": "text", "text": "Query example. Predict exactly one label from the allowed list."})
    content.append(
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encode_image_base64(query_image)}"
            },
        }
    )

    return [
        {
            "role": "system",
            "content": (
                "You are a precise echocardiography view classifier. "
                "Choose exactly one allowed label and answer in JSON only."
            ),
        },
        {"role": "user", "content": content},
    ]


def extract_prediction_label(text: str, label_order: List[str]) -> str:
    match = LABEL_LINE_RE.search(text)
    if match:
        candidate = match.group(1).strip()
        if candidate in label_order:
            return candidate

    for label in label_order:
        if re.search(rf"\b{re.escape(label)}\b", text):
            return label

    raise ValueError(f"Could not map model response to a known label. Response: {text}")


def chat_completion(base_url: str, messages: List[Dict], args: argparse.Namespace) -> Dict:
    payload = {
        "model": "local-model",
        "messages": messages,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }
    return post_json(f"{base_url}/v1/chat/completions", payload)


def main() -> None:
    args = parse_args()
    if not args.llama_server.exists():
        raise FileNotFoundError(f"llama-server not found: {args.llama_server}")
    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not args.mmproj.exists():
        raise FileNotFoundError(f"mmproj not found: {args.mmproj}")

    support_manifest = args.trials_dir / args.support_trial / "manifest.csv"
    if not support_manifest.exists():
        raise FileNotFoundError(f"Support manifest not found: {support_manifest}")

    output_dir = args.output_dir or (RESULTS_DIR / f"{args.support_trial}__{args.query_dataset}")
    ensure_dir(output_dir)
    ensure_dir(CACHE_DIR)

    support_df = read_support_manifest(support_manifest)
    support_df["rendered_image"] = support_df.apply(lambda row: str(support_asset_image_path(row, args)), axis=1)
    support_df = support_df.sort_values(by=["label", "selection_index", "sample_id"]).reset_index(drop=True)
    label_order = sorted(support_df["label"].unique().tolist())

    query_df = read_query_manifest(args.query_manifest, args.query_dataset, args.max_queries)
    query_df["rendered_image"] = query_df.apply(lambda row: str(query_asset_image_path(row, args)), axis=1)
    query_df = query_df.reset_index(drop=True)

    base_url = f"http://{args.host}:{args.port}"
    server = launch_llama_server(args)
    try:
        wait_for_server(base_url, args.server_start_timeout)

        rows: List[Dict[str, object]] = []
        probs: Optional[np.ndarray] = None

        for idx, row in enumerate(query_df.itertuples(index=False), start=1):
            messages = build_messages(label_order, support_df, Path(str(row.rendered_image)))
            resp = chat_completion(base_url, messages, args)
            text = resp["choices"][0]["message"]["content"]
            pred_label = extract_prediction_label(text, label_order)
            rows.append(
                {
                    "dataset": row.dataset,
                    "sample_id": row.sample_id,
                    "source_path": row.source_path,
                    "raw_label": row.raw_label,
                    "canonical": row.canonical,
                    "true_label": row.model_label,
                    "pred_label": pred_label,
                    "correct": int(row.model_label == pred_label),
                    "query_rendered_image": row.rendered_image,
                    "model_response": text,
                }
            )
            if idx % 10 == 0:
                print(f"Processed {idx}/{len(query_df)} queries")

        pred_df = pd.DataFrame(rows)
        summary, per_label, cm = compute_classification_metrics(
            y_true_labels=pred_df["true_label"].tolist(),
            y_pred_labels=pred_df["pred_label"].tolist(),
            class_order=label_order,
            probs=probs,
        )
        summary["support_trial"] = args.support_trial
        summary["query_dataset"] = args.query_dataset
        summary["query_render_mode"] = args.query_render_mode
        summary["support_video_render_mode"] = args.support_video_render_mode
        summary["llama_server"] = str(args.llama_server)
        summary["model_path"] = str(args.model)
        summary["mmproj_path"] = str(args.mmproj)

        save_standard_outputs(
            output_dir=output_dir,
            predictions_df=pred_df,
            metrics_summary=summary,
            per_label_df=per_label,
            conf_matrix=cm,
            class_order=label_order,
            probs=probs,
            model_name="qwen3.5-9b-q8_0-llama.cpp",
        )
        support_df.to_csv(output_dir / "support_manifest_resolved.csv", index=False)
        query_df.to_csv(output_dir / "query_manifest_resolved.csv", index=False)
        (output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")

        print("Done.")
        print(f"Queries: {len(pred_df)}")
        print(f"Accuracy: {summary['accuracy']:.4f}")
        print(f"Macro-F1 (all labels): {summary['macro_f1_all_labels']:.4f}")
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=15)


if __name__ == "__main__":
    main()
