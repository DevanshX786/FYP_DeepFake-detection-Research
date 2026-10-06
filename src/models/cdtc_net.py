from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn

from src.models.rgb_encoder import ConvNeXtRGBEncoder
from src.models.frequency_encoder import FrequencyEncoder
from src.models.fusion import CrossDomainFusion
from src.models.temporal_difference import TemporalDifferenceModule
from src.models.temporal_transformer import TemporalConsistencyTransformer


class CDTCNet(nn.Module):
    """Cross-Domain Temporal Consistency Network (CDTC-Net) for Deepfake Video Detection."""

    def __init__(
        self,
        pretrained_backbone: bool = True,
        freeze_rgb_backbone: bool = True,
        feature_dim: int = 256,
        freq_in_channels: int = 1,
        transformer_heads: int = 8,
        transformer_layers: int = 2,
        transformer_ff_dim: int = 512,
        dropout_classifier: float = 0.3,
    ):
        """Initialize CDTC-Net.

        Args:
            pretrained_backbone: Whether to use ImageNet pretrained ConvNeXt-Tiny.
            freeze_rgb_backbone: Whether to freeze initial ConvNeXt weights.
            feature_dim: Common feature dimensionality (256-D).
            freq_in_channels: Number of frequency spectrum input channels (1 for grayscale).
            transformer_heads: Number of attention heads in temporal transformer.
            transformer_layers: Number of transformer encoder layers.
            transformer_ff_dim: Feedforward dimension in transformer layers.
            dropout_classifier: Classifier dropout probability.
        """
        super().__init__()

        # 1. Spatial RGB Branch
        self.rgb_encoder = ConvNeXtRGBEncoder(
            pretrained=pretrained_backbone,
            freeze_backbone=freeze_rgb_backbone,
            output_dim=feature_dim,
        )

        # 2. Frequency Spectral Branch
        self.freq_encoder = FrequencyEncoder(
            in_channels=freq_in_channels,
            output_dim=feature_dim,
        )

        # 3. Cross-Domain Fusion
        self.fusion = CrossDomainFusion(
            rgb_dim=feature_dim,
            freq_dim=feature_dim,
            output_dim=feature_dim,
            dropout=0.2,
        )

        # 4. Explicit Temporal Difference Features
        self.temp_diff = TemporalDifferenceModule(
            feature_dim=feature_dim,
            output_dim=feature_dim,
            dropout=0.1,
        )

        # 5. Temporal Consistency Transformer
        self.transformer = TemporalConsistencyTransformer(
            embed_dim=feature_dim,
            num_heads=transformer_heads,
            num_layers=transformer_layers,
            feedforward_dim=transformer_ff_dim,
            dropout=0.1,
        )

        # 6. Video Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout_classifier),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        rgb_frames: torch.Tensor,
        freq_maps: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through CDTC-Net.

        Args:
            rgb_frames: Sequence of RGB face crops [B, N, 3, 224, 224].
            freq_maps: Sequence of Frequency spectrum maps [B, N, 1, 224, 224].

        Returns:
            Logits of shape [B, 1] for binary deepfake detection.
        """
        # Step 1: Extract spatial and frequency tokens
        rgb_tokens = self.rgb_encoder(rgb_frames)    # [B, N, 256]
        freq_tokens = self.freq_encoder(freq_maps)   # [B, N, 256]

        # Step 2: Cross-domain fusion per frame
        frame_tokens = self.fusion(rgb_tokens, freq_tokens)  # [B, N, 256]

        # Step 3: Compute explicit inter-frame difference tokens
        diff_tokens = self.temp_diff(frame_tokens)           # [B, N - 1, 256]

        # Step 4: Temporal sequence modeling with [CLS] token
        video_repr = self.transformer(frame_tokens, diff_tokens)  # [B, 256]

        # Step 5: Final classification logit
        logits = self.classifier(video_repr)  # [B, 1]

        return logits

    @torch.no_grad()
    def predict_probability(
        self,
        rgb_frames: torch.Tensor,
        freq_maps: torch.Tensor,
    ) -> torch.Tensor:
        """Compute predicted deepfake probability P(fake) in [0, 1].

        Args:
            rgb_frames: [B, N, 3, 224, 224]
            freq_maps: [B, N, 1, 224, 224]

        Returns:
            Probabilities [B, 1].
        """
        logits = self.forward(rgb_frames, freq_maps)
        return torch.sigmoid(logits)
