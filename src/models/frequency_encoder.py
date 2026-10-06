from typing import List
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Convolutional building block for the frequency encoder."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FrequencyEncoder(nn.Module):
    """Dedicated lightweight convolutional encoder for 2D Frequency (FFT/DCT) magnitude spectra."""

    def __init__(
        self,
        in_channels: int = 1,
        channels: List[int] = None,
        output_dim: int = 256,
    ):
        """Initialize FrequencyEncoder.

        Args:
            in_channels: Number of input channels (default: 1 for grayscale magnitude).
            channels: Feature channels per conv stage (default: [32, 64, 128, 256]).
            output_dim: Projected output dimension (default: 256).
        """
        super().__init__()
        if channels is None:
            channels = [32, 64, 128, 256]

        layers = []
        c_in = in_channels
        for c_out in channels:
            layers.append(ConvBlock(c_in, c_out, stride=2))
            c_in = c_out

        self.encoder = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(channels[-1], output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Frequency tensor of shape [B, N, 1, H, W] or [BatchSize, 1, H, W].

        Returns:
            Projected frequency feature representations [B, N, output_dim] or [BatchSize, output_dim].
        """
        if x.dim() == 5:
            B, N, C, H, W = x.shape
            x = x.view(B * N, C, H, W)
            feat = self.encoder(x)
            pooled = self.global_pool(feat)
            proj = self.projection(pooled)
            return proj.view(B, N, -1)
        else:
            feat = self.encoder(x)
            pooled = self.global_pool(feat)
            return self.projection(pooled)
