import os
import sys
import zipfile
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def extract_faceforensics(
    zip_path: str = "data/raw/FaceForensics++ Dataset (C23).zip",
    output_dir: str = "data/raw/faceforensics",
):
    """Extract FaceForensics++ from immutable ZIP file without modifying the ZIP archive."""
    if not os.path.exists(zip_path):
        print(f"Error: {zip_path} not found.")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"Opening immutable archive: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        target_members = []

        for m in members:
            if m.is_dir():
                continue
            parts = m.filename.split("/")
            # e.g., 'FaceForensics++_C23/Deepfakes/000_003.mp4' -> 'Deepfakes/000_003.mp4'
            if len(parts) >= 2:
                target_members.append((m, os.path.join(*parts[1:])))

        print(f"Extracting {len(target_members)} FaceForensics++ files to {output_dir}...")
        for member, rel_path in tqdm(target_members, desc="Extracting FaceForensics++"):
            dest_path = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            if not os.path.exists(dest_path) or os.path.getsize(dest_path) != member.file_size:
                with zf.open(member) as src, open(dest_path, "wb") as dst:
                    dst.write(src.read())

    print(f"FaceForensics++ extraction complete -> {output_dir}")


def extract_celebdf(
    zip_path: str = "data/raw/Celeb-DF-v2.zip",
    output_dir: str = "data/raw/celebdf",
):
    """Extract Celeb-DF v2 from immutable ZIP file without modifying the ZIP archive."""
    if not os.path.exists(zip_path):
        print(f"Error: {zip_path} not found.")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"Opening immutable archive: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        print(f"Extracting {len(members)} Celeb-DF v2 files to {output_dir}...")
        for member in tqdm(members, desc="Extracting Celeb-DF v2"):
            dest_path = os.path.join(output_dir, member.filename)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            if not os.path.exists(dest_path) or os.path.getsize(dest_path) != member.file_size:
                with zf.open(member) as src, open(dest_path, "wb") as dst:
                    dst.write(src.read())

    print(f"Celeb-DF v2 extraction complete -> {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract immutable dataset archives.")
    parser.add_argument("--dataset", type=str, default="all", choices=["ffpp", "celebdf", "all"])
    args = parser.parse_args()

    if args.dataset in ("ffpp", "all"):
        extract_faceforensics()
    if args.dataset in ("celebdf", "all"):
        extract_celebdf()
