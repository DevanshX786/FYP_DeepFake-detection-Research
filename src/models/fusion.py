import torch
import torch.nn as nn


class CrossDomainFusion(nn.Module):
    """Cross-Domain Fusion layer combining RGB spatial features and Frequency spectral features."""

    def __init__(
        self,
        rgb_dim: int = 256,
        freq_dim: int = 256,
        output_dim: int = 256,
        dropout: float = 0.2,
    ):
        """Initialize CrossDomainFusion.

        Args:
            rgb_dim: Dimension of RGB feature tokens.
            freq_dim: Dimension of Frequency feature tokens.
            output_dim: Dimension of fused frame representation.
            dropout: Dropout probability.
        """
        super().__init__()
        in_features = rgb_dim + freq_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(in_features, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, rgb_tokens: torch.Tensor, freq_tokens: torch.Tensor) -> torch.Tensor:
        """Fuse RGB and Frequency token representations.

        Args:
            rgb_tokens: Tensor of shape [B, N, rgb_dim].
            freq_tokens: Tensor of shape [B, N, freq_dim].

        Returns:
            Fused frame tokens of shape [B, N, output_dim].
        """
        concat_tokens = torch.cat([rgb_tokens, freq_tokens], dim=-1)
        fused = self.fusion_mlp(concat_tokens)
        return fused
