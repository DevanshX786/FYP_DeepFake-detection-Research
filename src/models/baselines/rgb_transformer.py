import torch
import torch.nn as nn

from src.models.rgb_encoder import ConvNeXtRGBEncoder
from src.models.temporal_transformer import TemporalConsistencyTransformer


class RGBTransformerModel(nn.Module):
    """RGB + Temporal Transformer Baseline (Ablation A0)."""

    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        feature_dim: int = 256,
        transformer_heads: int = 8,
        transformer_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.rgb_encoder = ConvNeXtRGBEncoder(
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            output_dim=feature_dim,
        )
        self.transformer = TemporalConsistencyTransformer(
            embed_dim=feature_dim,
            num_heads=transformer_heads,
            num_layers=transformer_layers,
            feedforward_dim=512,
            dropout=0.1,
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
        frame_tokens = self.rgb_encoder(rgb_frames)          # [B, N, feature_dim]
        video_repr = self.transformer(frame_tokens, diff_tokens=None)  # [B, feature_dim]
        logits = self.classifier(video_repr)                 # [B, 1]
        return logits
