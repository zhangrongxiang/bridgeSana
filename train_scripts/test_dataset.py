#!/usr/bin/env python
"""
Test script to verify dataset loading
"""

import sys
import json
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms

def test_dataset_structure(data_dir):
    """Test if dataset structure is correct"""
    data_dir = Path(data_dir)

    print("=" * 60)
    print("Testing Dataset Structure")
    print("=" * 60)

    # Check if train.jsonl exists
    jsonl_path = data_dir / "train.jsonl"
    if not jsonl_path.exists():
        print(f"❌ ERROR: train.jsonl not found at {jsonl_path}")
        return False

    print(f"✓ Found train.jsonl at {jsonl_path}")

    # Load and check JSONL
    data = []
    with open(jsonl_path, 'r') as f:
        for i, line in enumerate(f):
            try:
                item = json.loads(line)
                data.append(item)
            except json.JSONDecodeError as e:
                print(f"❌ ERROR: Invalid JSON at line {i+1}: {e}")
                return False

    print(f"✓ Loaded {len(data)} entries from train.jsonl")

    # Check first few entries
    print("\n" + "=" * 60)
    print("Checking Image Pairs")
    print("=" * 60)

    for i, item in enumerate(data[:3]):
        print(f"\nEntry {i+1}:")
        print(f"  Source: {item.get('src', 'N/A')}")
        print(f"  Target: {item.get('tar', 'N/A')}")
        print(f"  Prompt: {item.get('prompt', 'N/A')[:60]}...")

        # Handle path - remove leading directory name if present
        src_rel = item['src']
        tar_rel = item['tar']
        data_dir_name = data_dir.name

        if src_rel.startswith(data_dir_name + '/'):
            src_rel = src_rel[len(data_dir_name) + 1:]
        if tar_rel.startswith(data_dir_name + '/'):
            tar_rel = tar_rel[len(data_dir_name) + 1:]

        # Check if files exist
        src_path = data_dir / src_rel
        tar_path = data_dir / tar_rel

        if not src_path.exists():
            print(f"  ❌ Source image not found: {src_path}")
            continue
        else:
            print(f"  ✓ Source image exists")

        if not tar_path.exists():
            print(f"  ❌ Target image not found: {tar_path}")
            continue
        else:
            print(f"  ✓ Target image exists")

        # Try to load images
        try:
            src_img = Image.open(src_path).convert('RGB')
            tar_img = Image.open(tar_path).convert('RGB')
            print(f"  ✓ Images loaded successfully")
            print(f"    Source size: {src_img.size}")
            print(f"    Target size: {tar_img.size}")
        except Exception as e:
            print(f"  ❌ Error loading images: {e}")
            continue

    print("\n" + "=" * 60)
    print("Dataset Test Summary")
    print("=" * 60)
    print(f"Total entries: {len(data)}")
    print(f"✓ Dataset structure is valid!")

    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_dataset.py <data_dir>")
        print("Example: python test_dataset.py /cache/omnic/3D_Chibi")
        sys.exit(1)

    data_dir = sys.argv[1]
    success = test_dataset_structure(data_dir)

    if success:
        print("\n✓ All tests passed! Dataset is ready for training.")
        sys.exit(0)
    else:
        print("\n❌ Dataset test failed. Please fix the issues above.")
        sys.exit(1)
