#!/usr/bin/env python3
"""
download_models.py
===================

Downloads fine-tuned model .zip files from the GitHub release and extracts
them into MODELS_DIR (from config.py), matching the folder names
MODEL_CHOICES in models.py expects.

Usage
-----
Download every model in AVAILABLE_MODELS:
    python download_models.py

Download only specific models:
    python download_models.py patb_model QTrain_patb_uthmani_model

Re-download even if the folder already exists:
    python download_models.py --force

List what's available without downloading anything:
    python download_models.py --list
"""

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

import config as cfg

# Update owner/repo/tag if you cut a new release under a different tag.
RELEASE_URL = (
    "https://github.com/CAMeL-Lab/MAQAM-Quranic-Corpus/releases/download/models-v1"
)

# Folder names that actually have a .zip attached to the release above.
# Keep this in sync with what's uploaded -- not every entry in
# models.MODEL_CHOICES necessarily has a release asset.
AVAILABLE_MODELS = [
    "patb_model",
    "QAll_imlaay_uthmani_patb_model",
    "QTrain_imlaay_uthmani_model",
    "QTrain_imlaay_uthmani_patb_model",
    "QTrain_patb_imlaay_model",
    "QTrain_patb_uthmani_model",
]


def download_model(name: str, models_dir: Path, force: bool = False):
    target = models_dir / name
    if target.exists() and not force:
        print(f"  {name}: already exists, skipping (use --force to re-download)")
        return True

    if name not in AVAILABLE_MODELS:
        print(f"  {name}: not a recognized model name. "
              f"Available: {', '.join(AVAILABLE_MODELS)}")
        return False

    models_dir.mkdir(parents=True, exist_ok=True)
    zip_path = models_dir / f"{name}.zip"
    url = f"{RELEASE_URL}/{name}.zip"

    print(f"  {name}: downloading from {url} ...")
    try:
        urllib.request.urlretrieve(url, zip_path)
    except Exception as exc:
        print(f"  {name}: download failed -- {exc}")
        return False

    print(f"  {name}: extracting ...")
    if target.exists() and force:
        shutil.rmtree(target)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(models_dir)
    zip_path.unlink()

    if not target.exists():
        print(f"  {name}: WARNING -- extracted, but {target} not found. "
              f"Check the zip's internal folder name matches {name!r}.")
        return False

    print(f"  {name}: done -> {target}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*",
                         help="Specific model names to download. Omit to download all of AVAILABLE_MODELS.")
    parser.add_argument("--force", action="store_true",
                         help="Re-download and overwrite even if the folder already exists.")
    parser.add_argument("--models-dir", default=None,
                         help="Override config.MODELS_DIR.")
    parser.add_argument("--list", action="store_true",
                         help="List available models and exit.")
    args = parser.parse_args()

    if args.list:
        print("Available models:")
        for name in AVAILABLE_MODELS:
            print(f"  - {name}")
        return

    models_dir = Path(args.models_dir or cfg.MODELS_DIR)
    to_download = args.models or AVAILABLE_MODELS

    print(f"Models directory: {models_dir}\n")
    results = [download_model(name, models_dir, force=args.force) for name in to_download]

    if not all(results):
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())