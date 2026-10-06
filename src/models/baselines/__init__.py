"""Baseline models for experimental comparison and ablation studies."""
from src.models.baselines.reference_resnet_bilstm import ReferenceResNetBiLSTM
from src.models.baselines.rgb_only import RGBOnlyModel
from src.models.baselines.rgb_transformer import RGBTransformerModel
from src.models.baselines.frequency_only import FrequencyOnlyModel

__all__ = [
    "ReferenceResNetBiLSTM",
    "RGBOnlyModel",
    "RGBTransformerModel",
    "FrequencyOnlyModel",
]

