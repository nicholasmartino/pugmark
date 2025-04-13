#!/usr/bin/env python3
import argparse
import os
import re

import tensorflow as tf

from src.training.CheckpointValidator import (
    inspect_checkpoint_files,
    validate_checkpoint_changes,
)
from src.training.Globals import PATH


def main():
    """Check if checkpoints are actually changing during training"""
    parser = argparse.ArgumentParser(description="Check checkpoint integrity")
    parser.add_argument(
        "--checkpoint_dir",
        default=f"{PATH}/footprints/checkpoint",
        help="Path to checkpoint directory",
    )
    parser.add_argument(
        "--test_dir", default=f"{PATH}/footprints/test", help="Path to test images"
    )
    args = parser.parse_args()

    # Inspect checkpoint file sizes
    print("Checking checkpoint files...")
    results = inspect_checkpoint_files(args.checkpoint_dir, latest_n=5)

    # Report if all data files have the same size
    if "all_same_size" in results:
        print(f"All checkpoint data files have same size: {results['all_same_size']}")
        if results["all_same_size"]:
            print(f"Data file size: {results['data_file_size']} bytes")

    # Load a test image
    test_files = tf.io.gfile.glob(f"{args.test_dir}/*.png")
    test_image = None
    if test_files:
        # Just pick the first image
        img_path = test_files[0]
        print(f"Using sample image: {img_path}")
        img = tf.io.decode_png(tf.io.read_file(img_path), channels=3)
        img = tf.image.resize(img, [256, 256])
        img = tf.cast(img, tf.float32) / 127.5 - 1
        test_image = tf.expand_dims(img, 0)  # Add batch dimension

    # Find checkpoints - extract checkpoint numbers and sort numerically
    # Get all index files
    index_files = tf.io.gfile.glob(f"{args.checkpoint_dir}/*.index")
    checkpoints = []

    if index_files:
        # Extract checkpoints with their numbers
        numbered_checkpoints = []
        other_checkpoints = []

        for idx_file in index_files:
            base_path = idx_file[:-6]  # Remove .index
            # Try to extract checkpoint number
            match = re.search(r"ckpt-(\d+)", idx_file)
            if match:
                number = int(match.group(1))
                numbered_checkpoints.append((number, base_path))
            else:
                other_checkpoints.append(base_path)

        # Sort numbered checkpoints by number (descending)
        numbered_checkpoints.sort(key=lambda x: x[0], reverse=True)

        # Combine sorted checkpoints lists (numbered first, then others)
        checkpoints = [path for _, path in numbered_checkpoints]
        checkpoints.extend(other_checkpoints)

        # Take just the most recent 10
        checkpoints = checkpoints[:10]

        # Log what we found
        print(f"Found {len(checkpoints)} checkpoints:")
        for i, ckpt in enumerate(checkpoints):
            print(f"  {i+1}. {os.path.basename(ckpt)}")
    else:
        print("No checkpoints found in directory")

    # Compare first vs last checkpoint
    if len(checkpoints) >= 2:
        print(f"Comparing first vs last checkpoint:")
        results = validate_checkpoint_changes(
            checkpoints[0], checkpoints[-1], test_image
        )
        if not results.get("weights_differ", True):
            print(
                "WARNING! Checkpoints have identical weights - training may be broken"
            )
        else:
            print(f"Weight difference: {results.get('weight_diff_magnitude', 0)}")


if __name__ == "__main__":
    main()
