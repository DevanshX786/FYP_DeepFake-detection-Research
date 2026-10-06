import torch
import torch.nn as nn

from src.models.frequency_encoder import FrequencyEncoder


class FrequencyOnlyModel(nn.Module):
    """Frequency-only Baseline: 2D FFT/DCT CNN followed by Temporal Average Pooling."""

    def __init__(
        self,
        in_channels: int = 1,
        feature_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.freq_encoder = FrequencyEncoder(
            in_channels=in_channels,
            output_dim=feature_dim,
        )
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, rgb_frames: torch.Tensor = None, freq_maps: torch.Tensor = None) -> torch.Tensor:
        """Forward pass.

        Args:
            rgb_frames: Unused.
            freq_maps: [B, N, 1, 224, 224]

        Returns:
            Logits [B, 1].
        """
        assert freq_maps is not None, "Frequency maps required for FrequencyOnlyModel"
        tokens = self.freq_encoder(freq_maps)  # [B, N, feature_dim]
        video_repr = tokens.mean(dim=1)        # [B, feature_dim]
        logits = self.classifier(video_repr)   # [B, 1]
        return logits
