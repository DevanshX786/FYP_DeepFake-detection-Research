import torch
import torch.nn as nn

from src.models.rgb_encoder import ConvNeXtRGBEncoder


class RGBOnlyModel(nn.Module):
    """Spatial-only Baseline: ConvNeXt-Tiny followed by Temporal Average Pooling."""

    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        feature_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.rgb_encoder = ConvNeXtRGBEncoder(
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            output_dim=feature_dim,
        )
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, rgb_frames: torch.Tensor, freq_maps: torch.Tensor = None) -> torch.Tensor:
        """Forward pass.

        Args:
            rgb_frames: [B, N, 3, 224, 224]
            freq_maps: Unused.

        Returns:
            Logits [B, 1].
        """
        # [B, N, feature_dim]
        tokens = self.rgb_encoder(rgb_frames)
        # Temporal average pooling across frames
        video_repr = tokens.mean(dim=1)  # [B, feature_dim]
        logits = self.classifier(video_repr)
        return logits
