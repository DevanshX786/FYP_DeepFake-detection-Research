"""Baseline models for experimental comparison and ablation studies."""
from src.models.baselines.senior_resnet_bilstm import SeniorResNetBiLSTM
from src.models.baselines.rgb_only import RGBOnlyModel
from src.models.baselines.rgb_transformer import RGBTransformerModel
from src.models.baselines.frequency_only import FrequencyOnlyModel

__all__ = [
    "SeniorResNetBiLSTM",
    "RGBOnlyModel",
    "RGBTransformerModel",
    "FrequencyOnlyModel",
]
