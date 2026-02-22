#!/usr/bin/env python3
"""
Batch downscale + grayscale all images in a folder using OpenCV.
- Reads images from INPUT_DIR (non-recursive by default)
- Writes to OUTPUT_DIR = INPUT_DIR/processed
- Output filename is the same as input filename
"""

import os
import sys
from pathlib import Path
import cv2


# =========================
# CONFIG
# =========================
INPUT_DIR = "../data/log_cabin_2"  # <-- set this
OUT_SUBDIR = "processed"            # output folder name under INPUT_DIR

TARGET_W = 2856*2
TARGET_H = 2142*2                   # <-- set this

# Interpolation choice:
# - INTER_AREA: best for downscaling
# - INTER_LINEAR: decent general
INTERP = cv2.INTER_AREA

# Process common image extensions
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in EXTS


def process_one(in_path: Path, out_path: Path) -> bool:
    img = cv2.imread(str(in_path), cv2.IMREAD_COLOR)
    if img is None:
        return False

    # grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # resize to exact target (may change aspect ratio)
    resized = cv2.resize(gray, (TARGET_W, TARGET_H), interpolation=INTERP)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save single-channel grayscale (OpenCV will write properly for most formats)
    ok = cv2.imwrite(str(out_path), resized)
    return bool(ok)


def main() -> int:
    in_dir = Path(INPUT_DIR).expanduser().resolve()
    if not in_dir.exists() or not in_dir.is_dir():
        print(f"[ERROR] INPUT_DIR is not a valid directory: {in_dir}")
        return 2

    out_dir = in_dir / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in in_dir.iterdir() if is_image_file(p)])

    if not files:
        print(f"[WARN] No images found in: {in_dir}")
        return 0

    n_ok = 0
    n_fail = 0

    for p in files:
        out_path = out_dir / p.name
        ok = process_one(p, out_path)
        if ok:
            n_ok += 1
        else:
            n_fail += 1
            print(f"[FAIL] {p.name}")

    print(f"[DONE] processed={n_ok} failed={n_fail} output_dir={out_dir}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
