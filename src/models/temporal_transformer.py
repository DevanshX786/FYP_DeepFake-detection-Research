import math
import torch
import torch.nn as nn


class TemporalConsistencyTransformer(nn.Module):
    """Lightweight Temporal Consistency Transformer Encoder with [CLS] token pooling."""

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        feedforward_dim: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 64,
    ):
        """Initialize TemporalConsistencyTransformer.

        Args:
            embed_dim: Token embedding dimension (default: 256).
            num_heads: Number of attention heads (default: 8).
            num_layers: Number of Transformer encoder layers (default: 2).
            feedforward_dim: Dimension of feedforward hidden layer (default: 512).
            dropout: Dropout rate (default: 0.1).
            max_seq_len: Maximum supported sequence length for positional embeddings.
        """
        super().__init__()
        self.embed_dim = embed_dim

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Learnable positional embeddings
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        self.pos_drop = nn.Dropout(p=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embed_dim),
        )

    def forward(
        self,
        frame_tokens: torch.Tensor,
        diff_tokens: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass through temporal Transformer.

        Args:
            frame_tokens: Tensor of frame tokens [B, N, embed_dim].
            diff_tokens: Optional tensor of difference tokens [B, N - 1, embed_dim].

        Returns:
            Video-level representation tensor [B, embed_dim] extracted from [CLS] token.
        """
        B = frame_tokens.shape[0]

        # Expand CLS token for batch
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]

        if diff_tokens is not None:
            # Concatenate [CLS, frame_1 ... frame_N, diff_1 ... diff_N-1]
            tokens = torch.cat([cls_tokens, frame_tokens, diff_tokens], dim=1)
        else:
            tokens = torch.cat([cls_tokens, frame_tokens], dim=1)

        seq_len = tokens.shape[1]
        tokens = tokens + self.pos_embedding[:, :seq_len, :]
        tokens = self.pos_drop(tokens)

        encoded = self.transformer_encoder(tokens)

        # Return [CLS] token as video embedding
        cls_out = encoded[:, 0, :]  # [B, embed_dim]
        return cls_out
