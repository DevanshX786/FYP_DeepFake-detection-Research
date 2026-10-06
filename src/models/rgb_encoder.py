import torch
import torch.nn as nn
import torchvision.models as models


class ConvNeXtRGBEncoder(nn.Module):
    """Spatial RGB Feature Encoder based on ConvNeXt-Tiny with linear projection to 256-D."""

    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        output_dim: int = 256,
    ):
        """Initialize ConvNeXtRGBEncoder.

        Args:
            pretrained: Whether to load ImageNet pre-trained weights.
            freeze_backbone: If True, freeze backbone parameters to stabilize early training.
            output_dim: Projected output feature dimension (default: 256).
        """
        super().__init__()
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        backbone = models.convnext_tiny(weights=weights)

        # Extract feature extractor and pooling layers, discard classifier head
        self.features = backbone.features
        self.avgpool = backbone.avgpool

        # ConvNeXt-Tiny native feature dimension is 768
        in_features = 768
        self.projection = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(in_features, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

        if freeze_backbone:
            self.set_backbone_freeze(True)

    def set_backbone_freeze(self, freeze: bool = True) -> None:
        """Freeze or unfreeze backbone parameters.

        Args:
            freeze: True to freeze, False to unfreeze for full fine-tuning.
        """
        for param in self.features.parameters():
            param.requires_grad = not freeze

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: RGB tensor of shape [B, N, 3, H, W] or [BatchSize, 3, H, W].

        Returns:
            Projected RGB feature representations [B, N, output_dim] or [BatchSize, output_dim].
        """
        if x.dim() == 5:
            # Sequence format: [B, N, C, H, W]
            B, N, C, H, W = x.shape
            x = x.view(B * N, C, H, W)
            feat = self.features(x)
            pooled = self.avgpool(feat)
            proj = self.projection(pooled)
            return proj.view(B, N, -1)
        else:
            feat = self.features(x)
            pooled = self.avgpool(feat)
            return self.projection(pooled)
