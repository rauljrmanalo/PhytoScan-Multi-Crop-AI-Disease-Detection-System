"""
inspect_dataset.py
═══════════════════════════════════════════════════════════════════════
Run this BEFORE training to verify your dataset is correctly structured
and shows a visual sample grid from each class.

Usage:
    python inspect_dataset.py --data_dir ./rice_dataset
    python inspect_dataset.py --data_dir ./rice_dataset --show_samples 4
"""

import os
import sys
import argparse
import json
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# Must match train_rice_vgg16.py exactly
EXPECTED_CLASSES = {
    "bacterial_leaf_blight" : "bacterial_blight",
    "brown_spot"            : "brown_spot",
    "healthy"               : "healthy",
    "leaf_blast"            : "leaf_blast",
    "leaf_scald"            : "leaf_scald",
    "narrow_brown_spot"     : "narrow_brown_spot",
    "neck_blast"            : "neck_blast",
    "rice_hispa"            : "rice_hispa",
    "sheath_blight"         : "sheath_blight",
    "tungro"                : "tungro",
}
IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def count_images(folder: str) -> int:
    return sum(
        1 for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS
    )


def inspect(data_dir: str, show_samples: int, output_dir: str):
    print(f"\n{'='*60}")
    print(f"  PhytoScan — Dataset Inspector")
    print(f"  Dataset path: {data_dir}")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)

    splits = ["train", "test"]
    stats = {}
    all_ok = True

    for split in splits:
        split_path = os.path.join(data_dir, split)
        print(f"\n[{split.upper()}]  {split_path}")

        if not os.path.isdir(split_path):
            print(f"  ❌  Folder not found: {split_path}")
            all_ok = False
            continue

        found_classes = {
            d for d in os.listdir(split_path)
            if os.path.isdir(os.path.join(split_path, d))
        }

        stats[split] = {}
        split_total = 0

        for cls_folder, disease_id in EXPECTED_CLASSES.items():
            cls_path = os.path.join(split_path, cls_folder)
            if cls_folder not in found_classes:
                print(f"  ❌  Missing class folder: '{cls_folder}'")
                all_ok = False
                continue

            n = count_images(cls_path)
            stats[split][cls_folder] = n
            split_total += n
            status = "✅" if n > 0 else "⚠️ "
            print(f"  {status}  {cls_folder:22s} → {n:5d} images  (maps to: '{disease_id}')")

        extra = found_classes - set(EXPECTED_CLASSES.keys())
        if extra:
            print(f"  ⚠️  Unexpected folders (will be ignored): {extra}")

        print(f"  {'─'*48}")
        print(f"  {'TOTAL':22s}   {split_total:5d} images")

    # ── Class balance check ───────────────────────────────────────────
    if "train" in stats and stats["train"]:
        counts = list(stats["train"].values())
        if counts:
            ratio = max(counts) / (min(counts) + 1e-6)
            print(f"\n  Class imbalance ratio (train): {ratio:.1f}x")
            if ratio > 3:
                print(f"  ⚠️  High imbalance — consider class_weight during training")
            else:
                print(f"  ✅  Class balance is acceptable")

    # ── Save stats JSON ───────────────────────────────────────────────
    stats_path = os.path.join(output_dir, "dataset_stats.json")
    with open(stats_path, 'w') as f:
        json.dump({"dataset_path": data_dir, "splits": stats}, f, indent=2)
    print(f"\n  Stats saved → {stats_path}")

    # ── Distribution bar chart ────────────────────────────────────────
    if "train" in stats and stats["train"]:
        fig, axes = plt.subplots(1, len(splits), figsize=(5 * len(splits), 5))
        if len(splits) == 1:
            axes = [axes]

        colors = ['#4a8c35', '#e74c3c', '#f39c12', '#e67e22']

        for ax, split in zip(axes, splits):
            if split not in stats or not stats[split]:
                ax.set_visible(False)
                continue
            cls_names = list(stats[split].keys())
            counts = list(stats[split].values())
            bars = ax.bar(cls_names, counts, color=colors[:len(cls_names)], edgecolor='black', linewidth=0.5)
            for bar, count in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                        str(count), ha='center', va='bottom', fontsize=9, fontweight='bold')
            ax.set_title(f'{split.upper()} split', fontsize=12, fontweight='bold')
            ax.set_ylabel('Number of images')
            ax.set_xticklabels(cls_names, rotation=20, ha='right')
            ax.set_ylim(0, max(counts) * 1.15)
            ax.grid(axis='y', alpha=0.3)

        plt.suptitle('Rice Dataset — Class Distribution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        dist_path = os.path.join(output_dir, "dataset_distribution.png")
        plt.savefig(dist_path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  Distribution chart → {dist_path}")

    # ── Sample image grid ─────────────────────────────────────────────
    if show_samples > 0:
        print(f"\n  Generating sample image grid ({show_samples} per class)...")
        train_path = os.path.join(data_dir, 'train')
        n_classes = len(EXPECTED_CLASSES)

        fig, axes = plt.subplots(n_classes, show_samples,
                                  figsize=(show_samples * 2.5, n_classes * 2.5))
        fig.suptitle('Sample Images per Class (from train/)', fontsize=13, fontweight='bold')

        for row_i, (cls_folder, disease_id) in enumerate(EXPECTED_CLASSES.items()):
            cls_path = os.path.join(train_path, cls_folder)
            if not os.path.isdir(cls_path):
                continue

            imgs = [
                f for f in os.listdir(cls_path)
                if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS
            ]
            np.random.seed(42)
            selected = np.random.choice(imgs, min(show_samples, len(imgs)), replace=False)

            for col_i in range(show_samples):
                ax = axes[row_i][col_i] if n_classes > 1 else axes[col_i]
                ax.axis('off')
                if col_i < len(selected):
                    img_path = os.path.join(cls_path, selected[col_i])
                    try:
                        img = mpimg.imread(img_path)
                        ax.imshow(img)
                    except Exception:
                        ax.text(0.5, 0.5, 'load err', ha='center', va='center',
                                transform=ax.transAxes, fontsize=7)
                if col_i == 0:
                    ax.set_ylabel(f"{cls_folder}\n({disease_id})",
                                  fontsize=7.5, labelpad=4,
                                  rotation=0, ha='right', va='center')

        plt.tight_layout()
        sample_path = os.path.join(output_dir, "dataset_samples.png")
        plt.savefig(sample_path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  Sample grid saved  → {sample_path}")

    # ── Final verdict ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    if all_ok:
        print("  ✅  Dataset looks good — ready for training.")
        print(f"\n  Run training with:")
        print(f"    python train_rice_vgg16.py --data_dir {data_dir}")
    else:
        print("  ❌  Fix the issues above before training.")
    print(f"{'='*60}\n")

    return all_ok


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inspect rice leaf disease dataset')
    parser.add_argument('--data_dir',     default='./rice_dataset',
                        help='Root folder containing train/ and test/')
    parser.add_argument('--show_samples', type=int, default=4,
                        help='Number of sample images to show per class (0 to skip)')
    parser.add_argument('--output_dir',   default='./models/weights',
                        help='Where to save inspection charts')
    args = parser.parse_args()

    ok = inspect(args.data_dir, args.show_samples, args.output_dir)
    sys.exit(0 if ok else 1)
