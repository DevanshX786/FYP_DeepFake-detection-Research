import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from src.data.splits import create_identity_safe_splits


def main():
    parser = argparse.ArgumentParser(description="Create leakage-free train/val/test splits based on source video identity.")
    parser.add_argument("--metadata_csv", type=str, default="data/metadata/ffpp_metadata.csv", help="Path to input metadata CSV.")
    parser.add_argument("--output_dir", type=str, default="data/splits", help="Directory to save train.csv, val.csv, test.csv.")
    parser.add_argument("--train_ratio", type=float, default=0.70, help="Train split proportion.")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation split proportion.")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Test split proportion.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting.")
    args = parser.parse_args()

    if not os.path.exists(args.metadata_csv):
        print(f"Error: metadata file not found at {args.metadata_csv}")
        return

    df = pd.read_csv(args.metadata_csv)
    train_df, val_df, test_df = create_identity_safe_splits(
        metadata_df=df,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    print(f"Splits generated successfully with seed {args.seed}!")
    print(f"  Train samples: {len(train_df)} ({len(train_df['source_video_id'].unique())} unique source IDs)")
    print(f"  Val samples:   {len(val_df)} ({len(val_df['source_video_id'].unique())} unique source IDs)")
    print(f"  Test samples:  {len(test_df)} ({len(test_df['source_video_id'].unique())} unique source IDs)")


if __name__ == "__main__":
    main()
