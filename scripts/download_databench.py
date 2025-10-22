#!/usr/bin/env python3
"""
Download and prepare DataBench dataset for our Table QA system.

DataBench is from SemEval 2025 Task 8:
- HuggingFace: cardiffnlp/databench
- Baseline: 26-27%
- Top (AILS): 85.63%
- Our Target: 70%
"""

import os
import json
from datasets import load_dataset
import pandas as pd
from pathlib import Path

def download_databench(output_dir: str = "data/databench"):
    """Download DataBench dataset from HuggingFace."""

    print("=" * 60)
    print("Downloading DataBench from HuggingFace...")
    print("=" * 60)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Download dataset
        # DataBench has two configs: 'qa' and 'semeval'
        # We use 'semeval' for SemEval 2025 Task 8
        print("\n[1/4] Loading dataset from cardiffnlp/databench (semeval config)...")
        dataset = load_dataset("cardiffnlp/databench", "semeval")

        print(f"\n[2/4] Dataset loaded successfully!")
        print(f"Available splits: {list(dataset.keys())}")

        # Show dataset info
        for split_name in dataset.keys():
            split_data = dataset[split_name]
            print(f"\n{split_name}: {len(split_data)} samples")
            if len(split_data) > 0:
                print(f"  Columns: {split_data.column_names}")

        # Save to local files
        print(f"\n[3/4] Saving to {output_dir}...")

        for split_name in dataset.keys():
            split_data = dataset[split_name]

            # Save as JSONL
            output_file = output_path / f"{split_name}.jsonl"

            with open(output_file, 'w', encoding='utf-8') as f:
                for sample in split_data:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')

            print(f"  ✓ Saved {split_name}.jsonl ({len(split_data)} samples)")

        # Show sample data
        print("\n[4/4] Sample data from first split:")
        first_split = list(dataset.keys())[0]
        sample = dataset[first_split][0]

        print("\nSample structure:")
        for key, value in sample.items():
            if isinstance(value, str) and len(value) > 100:
                print(f"  {key}: {value[:100]}... (truncated)")
            elif isinstance(value, list) and len(value) > 5:
                print(f"  {key}: [{len(value)} items]")
            else:
                print(f"  {key}: {value}")

        print("\n" + "=" * 60)
        print("✓ DataBench download complete!")
        print(f"✓ Data saved to: {output_dir}")
        print("=" * 60)

        return dataset

    except Exception as e:
        print(f"\n✗ Error downloading DataBench: {e}")
        print("\nTroubleshooting:")
        print("1. Check internet connection")
        print("2. Verify HuggingFace datasets library: pip install datasets")
        print("3. Try manually: https://huggingface.co/datasets/cardiffnlp/databench")
        raise


if __name__ == "__main__":
    # Set data directory
    data_dir = "/media/liuyu/DataDrive/DASFAA-Table/data/databench"

    print("\n🚀 Starting DataBench download...")
    print(f"📁 Output directory: {data_dir}\n")

    dataset = download_databench(data_dir)

    print("\n✅ Next steps:")
    print("1. Analyze data structure: python scripts/analyze_databench.py")
    print("2. Run baseline test: python scripts/evaluate_databench.py --num_samples 50")
    print("3. Compare with AILS baseline (26%)")
