from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


def normalize_view_label_text(label: str) -> str:
    s = (label or "").strip()
    if not s:
        return ""
    s = s.replace("-", "_").replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s.upper()


def is_plax_variant(label: str) -> bool:
    norm = normalize_view_label_text(label)
    if not norm:
        return False
    if norm in {"PARASTERNAL_LONG", "DOPPLER_PARASTERNAL_LONG"}:
        return True
    if norm == "PLAX":
        return True
    if norm.startswith("PLAX_") or norm.endswith("_PLAX") or "_PLAX_" in norm:
        return True
    return False


def collapse_plax_label(label: str, canonical: str = "PLAX") -> str:
    if is_plax_variant(label):
        return canonical
    return label


def _nan_to_none(v):
    if isinstance(v, float) and np.isnan(v):
        return None
    return v


def compute_classification_metrics(
    y_true_labels: Sequence[str],
    y_pred_labels: Sequence[str],
    class_order: Sequence[str],
    probs: Optional[np.ndarray] = None,
) -> Tuple[Dict, pd.DataFrame, np.ndarray]:
    class_to_idx = {c: i for i, c in enumerate(class_order)}
    y_true = np.array([class_to_idx[x] for x in y_true_labels], dtype=int)
    y_pred = np.array([class_to_idx[x] for x in y_pred_labels], dtype=int)

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_order)))

    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(len(class_order)), zero_division=0
    )
    per_label = pd.DataFrame(
        {
            "label": list(class_order),
            "precision": p,
            "recall": r,
            "f1": f1,
            "support": support,
        }
    )

    present_mask = support > 0
    if np.any(present_mask):
        macro_precision_present = float(np.mean(p[present_mask]))
        macro_recall_present = float(np.mean(r[present_mask]))
        macro_f1_present = float(np.mean(f1[present_mask]))
    else:
        macro_precision_present = None
        macro_recall_present = None
        macro_f1_present = None

    macro_precision_all = float(np.mean(p))
    macro_recall_all = float(np.mean(r))
    macro_f1_all = float(np.mean(f1))

    summary = {
        "n_samples": int(len(y_true)),
        "n_classes": int(len(class_order)),
        "n_present_labels": int(np.sum(present_mask)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        # Backward-compatible keys: strict averaging over full model label set.
        "macro_precision": macro_precision_all,
        "macro_recall": macro_recall_all,
        "macro_f1": macro_f1_all,
        # Explicitly labeled definitions.
        "macro_precision_all_labels": macro_precision_all,
        "macro_recall_all_labels": macro_recall_all,
        "macro_f1_all_labels": macro_f1_all,
        "macro_precision_present_labels": macro_precision_present,
        "macro_recall_present_labels": macro_recall_present,
        "macro_f1_present_labels": macro_f1_present,
        "brier_score": None,
        "auc_roc_macro_ovr": None,
        "auc_pr_macro_ovr": None,
    }

    if probs is not None:
        probs = np.asarray(probs, dtype=np.float64)
        if probs.shape != (len(y_true), len(class_order)):
            raise ValueError(
                f"Probability shape mismatch: got {probs.shape}, expected {(len(y_true), len(class_order))}"
            )
        probs = np.clip(probs, 1e-12, 1.0)
        probs = probs / probs.sum(axis=1, keepdims=True)
        y_true_oh = label_binarize(y_true, classes=np.arange(len(class_order)))
        if y_true_oh.shape[1] == 1:
            # Binary edge case
            y_true_oh = np.hstack([1 - y_true_oh, y_true_oh])
        brier = np.mean(np.sum((probs - y_true_oh) ** 2, axis=1))
        summary["brier_score"] = float(brier)

        roc_vals: List[float] = []
        pr_vals: List[float] = []
        for i in range(len(class_order)):
            yi = y_true_oh[:, i]
            pi = probs[:, i]
            n_pos = int(np.sum(yi))
            n_neg = int(len(yi) - n_pos)
            if n_pos > 0 and n_neg > 0:
                roc_vals.append(float(roc_auc_score(yi, pi)))
            if n_pos > 0:
                pr_vals.append(float(average_precision_score(yi, pi)))

        summary["auc_roc_macro_ovr"] = float(np.mean(roc_vals)) if roc_vals else None
        summary["auc_pr_macro_ovr"] = float(np.mean(pr_vals)) if pr_vals else None

    summary = {k: _nan_to_none(v) for k, v in summary.items()}
    return summary, per_label, cm


def save_standard_outputs(
    output_dir: Path,
    predictions_df: pd.DataFrame,
    metrics_summary: Dict,
    per_label_df: pd.DataFrame,
    conf_matrix: np.ndarray,
    class_order: Sequence[str],
    probs: Optional[np.ndarray] = None,
    model_name: Optional[str] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_df.to_csv(output_dir / "predictions.csv", index=False)
    per_label_df.to_csv(output_dir / "per_label_metrics.csv", index=False)
    per_label_df.to_csv(output_dir / "combined_per_label_metrics.csv", index=False)

    with (output_dir / "metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    cm_df = pd.DataFrame(conf_matrix, index=class_order, columns=class_order)
    cm_df.to_csv(output_dir / "confusion_matrix.csv")

    # Per-dataset breakdown: one-vs-rest per label + macro metrics per dataset.
    if "dataset" in predictions_df.columns:
        ds_macro_rows = []
        ds_label_rows = []
        for ds_name, ds_df in predictions_df.groupby("dataset"):
            ds_idx = ds_df.index.to_numpy()
            ds_probs = probs[ds_idx] if probs is not None else None
            ds_summary, ds_per_label, _ = compute_classification_metrics(
                y_true_labels=ds_df["true_label"].tolist(),
                y_pred_labels=ds_df["pred_label"].tolist(),
                class_order=class_order,
                probs=ds_probs,
            )
            ds_macro_rows.append(
                {
                    "dataset": ds_name,
                    "n_samples": ds_summary["n_samples"],
                    "n_present_labels": ds_summary["n_present_labels"],
                    "accuracy": ds_summary["accuracy"],
                    "mcc": ds_summary["mcc"],
                    "macro_precision": ds_summary["macro_precision"],
                    "macro_recall": ds_summary["macro_recall"],
                    "macro_f1": ds_summary["macro_f1"],
                    "macro_precision_all_labels": ds_summary["macro_precision_all_labels"],
                    "macro_recall_all_labels": ds_summary["macro_recall_all_labels"],
                    "macro_f1_all_labels": ds_summary["macro_f1_all_labels"],
                    "macro_precision_present_labels": ds_summary["macro_precision_present_labels"],
                    "macro_recall_present_labels": ds_summary["macro_recall_present_labels"],
                    "macro_f1_present_labels": ds_summary["macro_f1_present_labels"],
                    "brier_score": ds_summary["brier_score"],
                    "auc_roc_macro_ovr": ds_summary["auc_roc_macro_ovr"],
                    "auc_pr_macro_ovr": ds_summary["auc_pr_macro_ovr"],
                }
            )
            ds_per_label.insert(0, "dataset", ds_name)
            ds_label_rows.append(ds_per_label)

        pd.DataFrame(ds_macro_rows).to_csv(output_dir / "dataset_macro_metrics.csv", index=False)
        pd.concat(ds_label_rows, ignore_index=True).to_csv(
            output_dir / "dataset_per_label_metrics.csv", index=False
        )

    if model_name:
        pd.DataFrame(
            [
                {
                    "model": model_name,
                    "n_samples": metrics_summary["n_samples"],
                    "n_present_labels": metrics_summary["n_present_labels"],
                    "accuracy": metrics_summary["accuracy"],
                    "mcc": metrics_summary["mcc"],
                    "macro_precision": metrics_summary["macro_precision"],
                    "macro_recall": metrics_summary["macro_recall"],
                    "macro_f1": metrics_summary["macro_f1"],
                    "macro_precision_all_labels": metrics_summary["macro_precision_all_labels"],
                    "macro_recall_all_labels": metrics_summary["macro_recall_all_labels"],
                    "macro_f1_all_labels": metrics_summary["macro_f1_all_labels"],
                    "macro_precision_present_labels": metrics_summary["macro_precision_present_labels"],
                    "macro_recall_present_labels": metrics_summary["macro_recall_present_labels"],
                    "macro_f1_present_labels": metrics_summary["macro_f1_present_labels"],
                    "brier_score": metrics_summary["brier_score"],
                    "auc_roc_macro_ovr": metrics_summary["auc_roc_macro_ovr"],
                    "auc_pr_macro_ovr": metrics_summary["auc_pr_macro_ovr"],
                }
            ]
        ).to_csv(output_dir / "model_macro_metrics.csv", index=False)

    plt.figure(figsize=(max(8, len(class_order) * 0.6), max(6, len(class_order) * 0.5)))
    sns.heatmap(cm_df, annot=False, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=200)
    plt.close()

    plt.figure(figsize=(max(8, len(class_order) * 0.6), 5))
    sns.barplot(data=per_label_df, x="label", y="f1")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.title("F1 per Label")
    plt.tight_layout()
    plt.savefig(output_dir / "f1_per_label.png", dpi=200)
    plt.close()

    if probs is not None:
        y_true_idx = np.array([class_order.index(x) for x in predictions_df["true_label"]], dtype=int)
        y_true_oh = label_binarize(y_true_idx, classes=np.arange(len(class_order)))
        if y_true_oh.shape[1] == 1:
            y_true_oh = np.hstack([1 - y_true_oh, y_true_oh])

        roc_plotted = False
        plt.figure(figsize=(8, 6))
        for i, cls in enumerate(class_order):
            n_pos = int(y_true_oh[:, i].sum())
            n_neg = int(len(y_true_oh[:, i]) - n_pos)
            if n_pos == 0 or n_neg == 0:
                continue
            fpr, tpr, _ = roc_curve(y_true_oh[:, i], probs[:, i])
            plt.plot(fpr, tpr, label=cls)
            roc_plotted = True
        if roc_plotted:
            plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
            plt.title("ROC Curves (One-vs-Rest)")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend(fontsize=7, ncol=2)
            plt.tight_layout()
            plt.savefig(output_dir / "roc_curves_ovr.png", dpi=200)
        plt.close()

        pr_plotted = False
        plt.figure(figsize=(8, 6))
        for i, cls in enumerate(class_order):
            n_pos = int(y_true_oh[:, i].sum())
            if n_pos == 0:
                continue
            prec, rec, _ = precision_recall_curve(y_true_oh[:, i], probs[:, i])
            plt.plot(rec, prec, label=cls)
            pr_plotted = True
        if pr_plotted:
            plt.title("PR Curves (One-vs-Rest)")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.legend(fontsize=7, ncol=2)
            plt.tight_layout()
            plt.savefig(output_dir / "pr_curves_ovr.png", dpi=200)
        plt.close()
