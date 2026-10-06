import os
from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    save_path: str,
    threshold: float = 0.5,
    title: str = "Confusion Matrix",
) -> None:
    """Generate and save confusion matrix visualization."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    y_pred = (np.asarray(y_probs) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.8)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i, s=str(cm[i, j]), va="center", ha="center", size="xx-large", weight="bold")

    plt.title(title, pad=20, fontsize=14, fontweight="bold")
    fig.colorbar(cax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Real (0)", "Fake (1)"], fontsize=11)
    ax.set_yticklabels(["Real (0)", "Fake (1)"], fontsize=11)
    plt.xlabel("Predicted Label", fontsize=12, labelpad=10)
    plt.ylabel("Ground Truth Label", fontsize=12, labelpad=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    save_path: str,
    model_name: str = "CDTC-Net",
) -> None:
    """Generate and save ROC Curve plot with AUC annotation."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#2563EB", lw=2.5, label=f"{model_name} (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="#9CA3AF", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.5000)")

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("False Positive Rate (FPR)", fontsize=12)
    plt.ylabel("True Positive Rate (TPR)", fontsize=12)
    plt.title(f"Receiver Operating Characteristic (ROC) — {model_name}", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    save_path: str,
    model_name: str = "CDTC-Net",
) -> None:
    """Generate and save Precision-Recall curve."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, color="#10B981", lw=2.5, label=f"{model_name} (PR-AUC = {pr_auc:.4f})")

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title(f"Precision-Recall Curve — {model_name}", fontsize=13, fontweight="bold")
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_training_curves(
    csv_log_path: str,
    save_path: str,
) -> None:
    """Plot training vs validation loss and metrics across epochs from CSV log."""
    if not os.path.exists(csv_log_path):
        return

    df = pd.read_csv(csv_log_path)
    if "epoch" not in df.columns:
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Loss plot
    ax1.plot(df["epoch"], df["train_loss"], label="Train Loss", color="#EF4444", lw=2)
    if "val_loss" in df.columns:
        ax1.plot(df["epoch"], df["val_loss"], label="Val Loss", color="#3B82F6", lw=2)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss (BCE)", fontsize=11)
    ax1.set_title("Training & Validation Loss", fontsize=12, fontweight="bold")
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.6)

    # AUC & Accuracy plot
    if "val_auc" in df.columns:
        ax2.plot(df["epoch"], df["val_auc"], label="Val ROC-AUC", color="#10B981", lw=2)
    if "val_acc" in df.columns:
        ax2.plot(df["epoch"], df["val_acc"], label="Val Accuracy", color="#8B5CF6", lw=2)
    if "val_f1" in df.columns:
        ax2.plot(df["epoch"], df["val_f1"], label="Val F1", color="#F59E0B", lw=2)

    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Score", fontsize=11)
    ax2.set_title("Validation Metrics Progression", fontsize=12, fontweight="bold")
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
