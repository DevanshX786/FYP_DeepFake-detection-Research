import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from src.evaluation.cross_dataset import evaluate_cross_dataset
from src.evaluation.evaluate import evaluate_checkpoint
from src.models.baselines.frequency_only import FrequencyOnlyModel
from src.models.baselines.rgb_only import RGBOnlyModel
from src.models.baselines.rgb_transformer import RGBTransformerModel
from src.models.baselines.reference_resnet_bilstm import ReferenceResNetBiLSTM
from src.models.cdtc_net import CDTCNet


def build_model(model_name: str) -> torch.nn.Module:
    """Build model instance given model identifier."""
    if model_name == "cdtc_net":
        return CDTCNet(pretrained_backbone=False, freeze_rgb_backbone=False)
    elif model_name == "reference_resnet_bilstm":
        return ReferenceResNetBiLSTM(pretrained=False)
    elif model_name == "rgb_only":
        return RGBOnlyModel(pretrained=False, freeze_backbone=False)
    elif model_name == "rgb_transformer":
        return RGBTransformerModel(pretrained=False, freeze_backbone=False)
    elif model_name == "frequency_only":
        return FrequencyOnlyModel()
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate deepfake detection model checkpoint.")
    parser.add_argument("--model_name", type=str, default="cdtc_net", choices=["cdtc_net", "reference_resnet_bilstm", "rgb_only", "rgb_transformer", "frequency_only"])
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint.")
    parser.add_argument("--test_csv", type=str, required=True, help="Path to test split CSV.")
    parser.add_argument("--output_dir", type=str, default="experiments/evaluation", help="Output directory for metrics and plots.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--cross_dataset", action="store_true", help="Flag if evaluating cross-dataset (e.g. Celeb-DF).")
    args = parser.parse_args()

    model = build_model(args.model_name)

    if args.cross_dataset:
        evaluate_cross_dataset(
            model=model,
            celebdf_csv_path=args.test_csv,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            model_name=args.model_name,
        )
    else:
        evaluate_checkpoint(
            model=model,
            test_csv_path=args.test_csv,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            model_name=args.model_name,
        )


if __name__ == "__main__":
    main()
