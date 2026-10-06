"""Evaluation, metrics, plotting, and cross-dataset testing."""
from src.evaluation.metrics import calculate_metrics, calculate_per_manipulation_metrics
from src.evaluation.plots import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve, plot_training_curves
from src.evaluation.evaluate import evaluate_checkpoint
from src.evaluation.cross_dataset import evaluate_cross_dataset

__all__ = [
    "calculate_metrics",
    "calculate_per_manipulation_metrics",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_precision_recall_curve",
    "plot_training_curves",
    "evaluate_checkpoint",
    "evaluate_cross_dataset",
]
