import os
import torch
from typing import Any, Dict, Optional


class CheckpointManager:
    """Manages model checkpoint saving and loading with metric tracking."""

    def __init__(
        self,
        checkpoint_dir: str,
        monitor: str = "val_auc",
        mode: str = "max",
        save_best_only: bool = True,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        if self.mode == "max":
            self.best_score = float("-inf")
            self.is_better = lambda score, best: score > best
        else:
            self.best_score = float("inf")
            self.is_better = lambda score, best: score < best

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        filename: Optional[str] = None,
    ) -> bool:
        """Save model checkpoint if it achieves a new best metric score or if requested.

        Args:
            model: PyTorch model.
            optimizer: PyTorch optimizer.
            epoch: Current epoch number.
            metrics: Dictionary of evaluated metrics.
            filename: Optional explicit filename.

        Returns:
            True if a new best checkpoint was saved, False otherwise.
        """
        current_score = metrics.get(self.monitor, None)
        is_best = False

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "best_score": self.best_score,
        }

        if current_score is not None and self.is_better(current_score, self.best_score):
            self.best_score = current_score
            is_best = True
            state["best_score"] = self.best_score
            best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            torch.save(state, best_path)

        if not self.save_best_only or filename is not None:
            save_name = filename if filename else f"checkpoint_epoch_{epoch:03d}.pt"
            path = os.path.join(self.checkpoint_dir, save_name)
            torch.save(state, path)

        # Always keep latest model
        latest_path = os.path.join(self.checkpoint_dir, "latest_model.pt")
        torch.save(state, latest_path)

        return is_best

    def load(
        self,
        model: torch.nn.Module,
        checkpoint_path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: torch.device = torch.device("cpu"),
    ) -> Dict[str, Any]:
        """Load model state and optionally optimizer state from checkpoint.

        Args:
            model: Model to load weights into.
            checkpoint_path: Path to checkpoint file.
            optimizer: Optional optimizer to load state into.
            device: Target torch device.

        Returns:
            Loaded state dictionary.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return checkpoint
