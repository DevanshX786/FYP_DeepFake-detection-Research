import argparse
import glob
import os
import re
import sys
import zipfile
from typing import List
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def extract_video_id(filepath: str) -> str:
    """Extract base source video ID from filename or path.

    FaceForensics++ format:
    - Original: '000.mp4', '001.mp4' -> source ID = '000'
    - Manipulated: '000_003.mp4' -> source ID = '000'
    Celeb-DF format:
    - 'id0_0000.mp4' -> source ID = 'id0'
    - 'YouTube-real/00001.mp4' -> source ID = 'yt_00001'
    """
    basename = os.path.basename(filepath)
    name, _ = os.path.splitext(basename)

    # FF++ manipulation: XXX_YYY
    ffpp_match = re.match(r"^(\d+)_\d+$", name)
    if ffpp_match:
        return ffpp_match.group(1)

    # FF++ original: XXX
    ffpp_orig = re.match(r"^(\d+)$", name)
    if ffpp_orig:
        return ffpp_orig.group(1)

    # Celeb-DF: idX_...
    celeb_match = re.match(r"^(id\d+)_.*$", name)
    if celeb_match:
        return celeb_match.group(1)

    return name


def scan_ffpp_dataset(data_path: str, compression: str = "c23") -> pd.DataFrame:
    """Scan FaceForensics++ from either an extracted folder or an immutable ZIP archive."""
    records = []
    supported_exts = {".mp4", ".avi", ".mov"}
    manip_dirs = {"deepfakes": "deepfakes", "face2face": "face2face", "faceswap": "faceswap", "neuraltextures": "neuraltextures"}

    # Check if data_path is a ZIP file or directory containing a ZIP
    zip_candidates = []
    if os.path.isfile(data_path) and data_path.lower().endswith(".zip"):
        zip_candidates = [data_path]
    elif os.path.isdir(data_path):
        for f in os.listdir(data_path):
            if "faceforensics" in f.lower() and f.lower().endswith(".zip"):
                zip_candidates.append(os.path.join(data_path, f))

    if zip_candidates:
        zip_file = zip_candidates[0]
        print(f"Scanning FaceForensics++ directly from immutable ZIP archive: {zip_file}")
        with zipfile.ZipFile(zip_file, "r") as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                ext = os.path.splitext(info.filename)[1].lower()
                if ext not in supported_exts:
                    continue

                parts = info.filename.replace("\\", "/").split("/")
                # Pattern: 'FaceForensics++_C23/Deepfakes/000_003.mp4'
                if len(parts) >= 3:
                    cat = parts[1].lower()
                    fname = parts[2]
                elif len(parts) == 2:
                    cat = parts[0].lower()
                    fname = parts[1]
                else:
                    continue

                if cat in ("original", "real", "youtube"):
                    records.append({
                        "video_path": f"{zip_file}::{info.filename}",
                        "label": 0,
                        "dataset": "faceforensics++",
                        "manipulation_method": "original",
                        "source_video_id": extract_video_id(fname),
                        "compression": compression,
                    })
                elif cat in manip_dirs:
                    records.append({
                        "video_path": f"{zip_file}::{info.filename}",
                        "label": 1,
                        "dataset": "faceforensics++",
                        "manipulation_method": cat,
                        "source_video_id": extract_video_id(fname),
                        "compression": compression,
                    })
        return pd.DataFrame(records)

    # Extracted folder fallback
    print(f"Scanning FaceForensics++ from extracted directory: {data_path}")
    orig_paths = [
        os.path.join(data_path, "original"),
        os.path.join(data_path, "original_sequences", "youtube", compression, "videos"),
        os.path.join(data_path, "original_sequences", "youtube", compression),
    ]
    for p in orig_paths:
        if os.path.exists(p):
            for f in os.listdir(p):
                ext = os.path.splitext(f)[1].lower()
                if ext in supported_exts:
                    records.append({
                        "video_path": os.path.join(p, f).replace("\\", "/"),
                        "label": 0,
                        "dataset": "faceforensics++",
                        "manipulation_method": "original",
                        "source_video_id": extract_video_id(f),
                        "compression": compression,
                    })
            break

    for m in ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]:
        m_paths = [
            os.path.join(data_path, m),
            os.path.join(data_path, "manipulated_sequences", m, compression, "videos"),
            os.path.join(data_path, "manipulated_sequences", m, compression),
        ]
        for p in m_paths:
            if os.path.exists(p):
                for f in os.listdir(p):
                    ext = os.path.splitext(f)[1].lower()
                    if ext in supported_exts:
                        records.append({
                            "video_path": os.path.join(p, f).replace("\\", "/"),
                            "label": 1,
                            "dataset": "faceforensics++",
                            "manipulation_method": m.lower(),
                            "source_video_id": extract_video_id(f),
                            "compression": compression,
                        })
                break

    return pd.DataFrame(records)


def scan_celebdf_dataset(data_path: str) -> pd.DataFrame:
    """Scan Celeb-DF v2 from either an extracted folder or an immutable ZIP archive."""
    records = []
    supported_exts = {".mp4", ".avi"}

    zip_candidates = []
    if os.path.isfile(data_path) and data_path.lower().endswith(".zip"):
        zip_candidates = [data_path]
    elif os.path.isdir(data_path):
        for f in os.listdir(data_path):
            if "celeb" in f.lower() and f.lower().endswith(".zip"):
                zip_candidates.append(os.path.join(data_path, f))

    if zip_candidates:
        zip_file = zip_candidates[0]
        print(f"Scanning Celeb-DF v2 directly from immutable ZIP archive: {zip_file}")
        with zipfile.ZipFile(zip_file, "r") as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                ext = os.path.splitext(info.filename)[1].lower()
                if ext not in supported_exts:
                    continue

                parts = info.filename.replace("\\", "/").split("/")
                if len(parts) >= 2:
                    folder = parts[0]
                    fname = parts[1]
                else:
                    continue

                if folder in ("Celeb-real", "YouTube-real"):
                    records.append({
                        "video_path": f"{zip_file}::{info.filename}",
                        "label": 0,
                        "dataset": "celeb_df",
                        "manipulation_method": "real",
                        "source_video_id": extract_video_id(fname),
                    })
                elif folder == "Celeb-synthesis":
                    records.append({
                        "video_path": f"{zip_file}::{info.filename}",
                        "label": 1,
                        "dataset": "celeb_df",
                        "manipulation_method": "celeb_synthesis",
                        "source_video_id": extract_video_id(fname),
                    })
        return pd.DataFrame(records)

    # Extracted folder
    print(f"Scanning Celeb-DF v2 from extracted directory: {data_path}")
    for folder, label, method in [("Celeb-real", 0, "real"), ("YouTube-real", 0, "real"), ("Celeb-synthesis", 1, "celeb_synthesis")]:
        p = os.path.join(data_path, folder)
        if os.path.exists(p):
            for f in os.listdir(p):
                ext = os.path.splitext(f)[1].lower()
                if ext in supported_exts:
                    records.append({
                        "video_path": os.path.join(p, f).replace("\\", "/"),
                        "label": label,
                        "dataset": "celeb_df",
                        "manipulation_method": method,
                        "source_video_id": extract_video_id(f),
                    })

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Scan dataset folders/archives and build standardized metadata CSV.")
    parser.add_argument("--dataset_type", type=str, required=True, choices=["ffpp", "celebdf"], help="Dataset type: 'ffpp' or 'celebdf'.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to raw dataset folder or ZIP archive.")
    parser.add_argument("--output_csv", type=str, required=True, help="Destination CSV path.")
    parser.add_argument("--compression", type=str, default="c23", help="Compression level for FF++ (default: c23).")
    args = parser.parse_args()

    if args.dataset_type == "ffpp":
        df = scan_ffpp_dataset(args.data_dir, compression=args.compression)
    else:
        df = scan_celebdf_dataset(args.data_dir)

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"\nScanned {len(df)} total video records. Metadata saved to: {args.output_csv}")
    if len(df) > 0:
        print("\nClass distribution:")
        print(df["label"].value_counts())
        print("\nManipulation breakdown:")
        print(df["manipulation_method"].value_counts())


if __name__ == "__main__":
    main()
