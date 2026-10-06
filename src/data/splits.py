import os
import random
from typing import Dict, List, Optional, Tuple
import pandas as pd


def create_identity_safe_splits(
    metadata_df: pd.DataFrame,
    output_dir: str = "data/splits",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split metadata into Train/Val/Test subsets strictly by source_video_id / identity.

    Guarantees:
    - Zero data leakage: all manipulated variants derived from a source video
      are strictly assigned to the same split as the source video.
    - No video or frame from the test set appears during training or validation.

    Args:
        metadata_df: DataFrame containing at least ['source_video_id', 'label', 'video_path'].
        output_dir: Directory where split CSVs will be saved.
        train_ratio: Fraction for training set.
        val_ratio: Fraction for validation set.
        test_ratio: Fraction for testing set.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Split ratios must sum to 1.0"
    assert "source_video_id" in metadata_df.columns, "Metadata must contain 'source_video_id' column"

    unique_sources = sorted(metadata_df["source_video_id"].dropna().unique().tolist())
    rng = random.Random(seed)
    rng.shuffle(unique_sources)

    n_total = len(unique_sources)
    n_train = int(round(n_total * train_ratio))
    n_val = int(round(n_total * val_ratio))

    train_sources = set(unique_sources[:n_train])
    val_sources = set(unique_sources[n_train:n_train + n_val])
    test_sources = set(unique_sources[n_train + n_val:])

    # Assign split column
    def assign_split(src_id):
        if src_id in train_sources:
            return "train"
        elif src_id in val_sources:
            return "val"
        elif src_id in test_sources:
            return "test"
        return "unknown"

    df = metadata_df.copy()
    df["split"] = df["source_video_id"].apply(assign_split)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
        val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
        test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    return train_df, val_df, test_df


def load_split_dataframe(split_csv_path: str) -> pd.DataFrame:
    """Load and validate split dataframe from CSV file.

    Args:
        split_csv_path: Path to CSV split file.

    Returns:
        Loaded pandas DataFrame.
    """
    if not os.path.exists(split_csv_path):
        raise FileNotFoundError(f"Split file not found: {split_csv_path}")

    df = pd.read_csv(split_csv_path)
    required_cols = {"video_path", "label", "source_video_id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Split CSV missing required columns: {missing}")

    return df
