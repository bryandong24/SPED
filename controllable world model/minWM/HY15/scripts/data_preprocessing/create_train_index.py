#!/usr/bin/env python
"""
Generate train_index.json from a directory of .pt latent files.

Usage:
    python HY15/scripts/data_preprocessing/create_train_index.py /path/to/latents_dir

This scans the directory for all .pt files and writes a train_index.json
in the same directory with absolute paths. If train_index.json already exists,
it will be overwritten.
"""
import argparse
import glob
import json
import os


def main():
    parser = argparse.ArgumentParser(description="Generate train_index.json from .pt files")
    parser.add_argument("data_dir", help="Directory containing .pt latent files")
    parser.add_argument("--output", "-o", default=None,
                        help="Output path for train_index.json (default: <data_dir>/train_index.json)")
    parser.add_argument("--recursive", "-r", action="store_true",
                        help="Recursively search subdirectories")
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    pattern = os.path.join(data_dir, "**/*.pt") if args.recursive else os.path.join(data_dir, "*.pt")
    pt_files = sorted(glob.glob(pattern, recursive=args.recursive))

    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {data_dir}")

    index = [{"latent_path": os.path.abspath(p)} for p in pt_files]

    output_path = args.output or os.path.join(data_dir, "train_index.json")
    with open(output_path, "w") as f:
        json.dump(index, f, indent=2)

    print(f"Created {output_path} with {len(index)} samples")


if __name__ == "__main__":
    main()
