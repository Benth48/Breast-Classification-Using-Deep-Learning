"""
Applies CLAHE to the L channel in LAB colorspace, then the same reproducible
per-class train/val/test split as the full pipeline — no border crop, no
Doppler crop.

Reads from  : data/raw/{normal,benign,malignant}/
Writes to   : data/clahe/{train,val,test}/{normal,benign,malignant}/

Safe to re-run — skips already-processed files.

Usage:
    # All defaults
    python src/clahe.py

    # Custom params
    python src/clahe.py \\
        --raw_dir data/raw \\
        --out_dir data/clahe \\
        --clip_limit 2.0 --tile_size 8 \\
        --val_split 0.15 --test_split 0.15 --seed 42
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
CLASS_NAMES      = ["normal", "benign", "malignant"]
SPLITS           = ["train", "val", "test"]


# ──────────────────────────────────────────────────────────────────────────────
# 1. CLAHE
# ──────────────────────────────────────────────────────────────────────────────

def apply_clahe(
    image_bgr: np.ndarray,
    clip_limit: float = 2.0,
    tile_size:  int   = 8,
) -> np.ndarray:
    """
    Apply CLAHE to the L channel in LAB colorspace.

    Pipeline: BGR → LAB → equalize L → LAB → BGR

    Args:
        image_bgr  : BGR uint8 image (as returned by cv2.imread)
        clip_limit : CLAHE contrast clip limit
        tile_size  : CLAHE tile grid size (tile_size × tile_size)

    Returns:
        BGR uint8 image with equalized luminance.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_size, tile_size),
    )
    l_eq = clahe.apply(l)

    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

# DIRECTORY WALK + SPLIT + SAVE

def collect_images(class_dir: Path) -> list[Path]:
    """Collect all image files, excluding BUSI-style mask files (*_mask.*)."""
    return sorted([
        p for p in class_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
        and "_mask" not in p.stem.lower()
    ])


def split_files(
    files: list[Path],
    val_split: float,
    test_split: float,
    seed: int,
) -> tuple[list[Path], list[Path], list[Path]]:
    """Reproducible stratified split for one class."""
    files = files.copy()
    random.seed(seed)
    random.shuffle(files)

    n       = len(files)
    n_val   =  int(n * val_split)
    n_test  =  int(n * test_split)

    train = files[: n - n_val - n_test]
    val   = files[n - n_val - n_test : n - n_test]
    test  = files[n - n_test :]

    return train, val, test


def process_and_save(
    src_path: Path,
    dst_path: Path,
    clip_limit: float,
    tile_size:  int,
) -> bool:
    """
    Load → CLAHE → save one image.
    Skips if destination already exists (resumable pipeline).

    Returns:
        True if the image was processed, False if skipped.
    """
    if dst_path.exists():
        return False

    image = cv2.imread(str(src_path))
    if image is None:
        print(f"  [WARN] Could not read: {src_path} — skipping.")
        return False

    processed = apply_clahe(image, clip_limit, tile_size)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst_path), processed)

    return True


def run(
    raw_dir: Path,
    out_dir: Path,
    clip_limit: float,
    tile_size:  int,
    val_split:  float,
    test_split: float,
    seed:       int,
) -> None:
    print("=" * 64)
    print("  Preprocessing Pipeline: CLAHE only")
    print("=" * 64)
    print(f"  Source : {raw_dir}")
    print(f"  Output : {out_dir}")
    print(f"  Splits : train={1-val_split-test_split:.0%}"
          f"  val={val_split:.0%}  test={test_split:.0%}")
    print(f"  CLAHE  : clip={clip_limit}  tile={tile_size}×{tile_size}")
    print(f"  Seed   : {seed}")
    print("=" * 64)

    total_processed = 0
    total_skipped   = 0

    stats = {cls: {split: 0 for split in SPLITS} for cls in CLASS_NAMES}

    for class_name in CLASS_NAMES:
        class_dir = raw_dir / class_name

        if not class_dir.exists():
            print(f"\n  [WARN] '{class_dir}' not found — skipping.")
            continue

        all_files = collect_images(class_dir)
        if not all_files:
            print(f"\n  [WARN] No images in '{class_dir}' — skipping.")
            continue

        train_files, val_files, test_files = split_files(
            all_files, val_split, test_split, seed
        )
        split_map = {"train": train_files, "val": val_files, "test": test_files}

        print(f"\n  [{class_name.upper()}]  "
              f"total={len(all_files)}  "
              f"train={len(train_files)}  "
              f"val={len(val_files)}  "
              f"test={len(test_files)}")

        for split_name, split_files_ in split_map.items():
            stats[class_name][split_name] = len(split_files_)
            dst_class_dir = out_dir / split_name / class_name

            for src_path in tqdm(
                split_files_,
                desc=f"    {split_name:>5}",
                unit="img",
                leave=False,
            ):
                dst_path = dst_class_dir / src_path.name

                if process_and_save(src_path, dst_path, clip_limit, tile_size):
                    total_processed += 1
                else:
                    total_skipped += 1

    # Summary 
    print("\n" + "=" * 64)
    print("  CLASS / SPLIT SUMMARY")
    print("=" * 64)

    header = f"  {'Class':<12}" + "".join(f"{s:>8}" for s in SPLITS) + f"{'Total':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for class_name in CLASS_NAMES:
        row_counts = [stats[class_name][s] for s in SPLITS]
        row_total  = sum(row_counts)
        print(f"  {class_name:<12}" + "".join(f"{c:>8}" for c in row_counts) + f"{row_total:>8}")

    col_totals  = [sum(stats[c][s] for c in CLASS_NAMES) for s in SPLITS]
    grand_total = sum(col_totals)
    print("  " + "-" * (len(header) - 2))
    print(f"  {'TOTAL':<12}" + "".join(f"{t:>8}" for t in col_totals) + f"{grand_total:>8}")

    print(f"\n  Processed : {total_processed} images")
    print(f"  Skipped   : {total_skipped} (already existed or unreadable)")
    print(f"  Output    : {out_dir.resolve()}")
    print("=" * 64)


# CLI


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CLAHE-only preprocessing + split")

    # Paths
    p.add_argument("--raw_dir", default="data/raw",   help="Raw data directory")
    p.add_argument("--out_dir", default="data/clahe", help="Output directory")

    # CLAHE
    p.add_argument("--clip_limit", type=float, default=2.0)
    p.add_argument("--tile_size",  type=int,   default=8)

    # Split
    p.add_argument("--val_split",  type=float, default=0.15)
    p.add_argument("--test_split", type=float, default=0.15)
    p.add_argument("--seed",       type=int,   default=42)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.val_split + args.test_split > 1.0:
        raise ValueError("val_split + test_split must be less than 1.0")

    run(
        raw_dir=Path(args.raw_dir),
        out_dir=Path(args.out_dir),
        clip_limit=args.clip_limit,
        tile_size=args.tile_size,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
    )
