from typing import Tuple
import torch
import torch.nn as nn


class TemporalDifferenceModule(nn.Module):
    """Explicit Temporal Difference module computing signed and absolute frame transitions."""

    def __init__(
        self,
        feature_dim: int = 256,
        output_dim: int = 256,
        dropout: float = 0.1,
    ):
        """Initialize TemporalDifferenceModule.

        Args:
            feature_dim: Dimension of each frame representation (default: 256).
            output_dim: Dimension of projected difference token (default: 256).
            dropout: Dropout probability.
        """
        super().__init__()
        in_dim = feature_dim * 2  # [D_t ; A_t] = signed + absolute
        self.projection = nn.Sequential(
            nn.Linear(in_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, frame_features: torch.Tensor) -> torch.Tensor:
        """Compute temporal difference tokens.

        Args:
            frame_features: Tensor of shape [B, N, feature_dim].

        Returns:
            Difference tokens of shape [B, N - 1, output_dim].
        """
        # F_{t+1} - F_t for t = 1 ... N - 1
        d_t = frame_features[:, 1:, :] - frame_features[:, :-1, :]
        a_t = torch.abs(d_t)

        diff_concat = torch.cat([d_t, a_t], dim=-1)  # [B, N - 1, 2 * feature_dim]
        diff_tokens = self.projection(diff_concat)    # [B, N - 1, output_dim]

        return diff_tokens
