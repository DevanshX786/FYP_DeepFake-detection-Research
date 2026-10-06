from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


def calculate_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Calculate comprehensive evaluation metrics for binary classification.

    Args:
        y_true: Ground truth binary labels (0 = Real, 1 = Fake) shape [N].
        y_probs: Predicted probabilities P(fake) in [0, 1] shape [N].
        threshold: Classification decision threshold (default: 0.5).

    Returns:
        Dictionary containing accuracy, balanced_accuracy, precision, recall, f1, roc_auc.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_probs = np.asarray(y_probs, dtype=float)
    y_pred = (y_probs >= threshold).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        if len(np.unique(y_true)) > 1:
            auc = float(roc_auc_score(y_true, y_probs))
        else:
            auc = 0.5
    except ValueError:
        auc = 0.5

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def calculate_per_manipulation_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    methods: List[str],
    threshold: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """Compute performance broken down by manipulation category (e.g. DeepFakes, Face2Face, etc.).

    Args:
        y_true: Ground truth labels.
        y_probs: Predicted probabilities.
        methods: List of manipulation method strings corresponding to each sample.
        threshold: Decision threshold.

    Returns:
        Dictionary mapping manipulation method name to metric dictionary.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_probs = np.asarray(y_probs, dtype=float)
    methods = np.asarray(methods)

    results = {}
    unique_methods = np.unique(methods)

    # Real samples mask
    real_mask = (y_true == 0)

    for method in unique_methods:
        if method in ("real", "original", "youtube"):
            continue

        method_mask = (methods == method)
        # Subset containing real videos + this specific fake method
        subset_mask = real_mask | method_mask

        if np.sum(subset_mask) > 0 and np.sum(method_mask) > 0:
            sub_true = y_true[subset_mask]
            sub_probs = y_probs[subset_mask]
            results[str(method)] = calculate_metrics(sub_true, sub_probs, threshold=threshold)

    return results
