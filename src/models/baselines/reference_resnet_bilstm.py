import torch
import torch.nn as nn
import torchvision.models as models


class ReferenceResNetBiLSTM(nn.Module):
    """Independent implementation of the Reference Project Baseline: ResNet50 + BiLSTM.

    Pipeline:
        Face Crops [B, N, 3, 224, 224]
        -> ResNet50 (2048-D frame features)
        -> 1-Layer BiLSTM (256 units/dir = 512-D output)
        -> Dropout(0.5)
        -> Linear(512, 1) Logit
    """

    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        hidden_dim: int = 256,
        dropout: float = 0.5,
    ):
        """Initialize ReferenceResNetBiLSTM baseline.

        Args:
            pretrained: Whether to load ImageNet weights for ResNet50.
            freeze_backbone: Whether to freeze ResNet50 feature extractor.
            hidden_dim: Hidden dimension per direction in BiLSTM (default: 256).
            dropout: Dropout probability before classification layer (default: 0.5).
        """
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)

        # Remove final FC classification layer
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        in_features = resnet.fc.in_features  # 2048

        # 1-layer Bidirectional LSTM
        self.bilstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)
        # Bidirectional LSTM produces 2 * hidden_dim
        self.fc = nn.Linear(hidden_dim * 2, 1)

        if freeze_backbone:
            self.set_backbone_freeze(True)

    def set_backbone_freeze(self, freeze: bool = True) -> None:
        """Freeze or unfreeze backbone parameters."""
        for param in self.feature_extractor.parameters():
            param.requires_grad = not freeze

    def forward(self, rgb_frames: torch.Tensor, freq_maps: torch.Tensor = None) -> torch.Tensor:
        """Forward pass.

        Args:
            rgb_frames: Tensor of shape [B, N, 3, 224, 224].
            freq_maps: Ignored in spatial-only baseline, accepted for uniform interface.

        Returns:
            Logits of shape [B, 1].
        """
        B, N, C, H, W = rgb_frames.shape

        # Extract ResNet50 frame features
        x = rgb_frames.view(B * N, C, H, W)
        features = self.feature_extractor(x)  # [B * N, 2048, 1, 1]
        features = features.view(B, N, -1)     # [B, N, 2048]

        # Process sequence through BiLSTM
        lstm_out, (hn, cn) = self.bilstm(features)  # [B, N, 512]

        # Pool across sequence (last time step or temporal max/mean)
        # Standard: take final forward and backward states
        seq_repr = lstm_out[:, -1, :]  # [B, 512]
        dropped = self.dropout(seq_repr)
        logits = self.fc(dropped)       # [B, 1]

        return logits

