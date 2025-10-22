#!/usr/bin/env python3
"""
Analyze DataBench dataset structure and download table data.

DataBench structure:
- QA pairs: question, answer, type, columns_used, etc.
- Table data: Separate parquet files per dataset
"""

import json
import pandas as pd
from pathlib import Path
from collections import Counter
import sys

def analyze_databench_qa(data_dir: str = "data/databench"):
    """Analyze DataBench QA pairs."""

    print("=" * 70)
    print("DataBench Dataset Analysis")
    print("=" * 70)

    data_path = Path(data_dir)

    # Load all splits
    splits = {}
    for split_file in ["train.jsonl", "dev.jsonl", "test.jsonl"]:
        file_path = data_path / split_file
        if not file_path.exists():
            print(f"✗ {split_file} not found")
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            splits[split_file.replace('.jsonl', '')] = [json.loads(line) for line in f]

    # Statistics
    print("\n📊 Dataset Statistics")
    print("-" * 70)
    for split_name, data in splits.items():
        print(f"{split_name:10s}: {len(data):4d} samples")

    # Analyze question types
    print("\n📈 Question Types Distribution")
    print("-" * 70)
    all_types = []
    for split_data in splits.values():
        all_types.extend([sample['type'] for sample in split_data])

    type_counts = Counter(all_types)
    for qtype, count in type_counts.most_common():
        pct = 100 * count / len(all_types)
        print(f"  {qtype:15s}: {count:4d} ({pct:5.1f}%)")

    # Analyze datasets
    print("\n📁 Source Datasets")
    print("-" * 70)
    all_datasets = []
    for split_data in splits.values():
        all_datasets.extend([sample['dataset'] for sample in split_data])

    dataset_counts = Counter(all_datasets)
    print(f"  Total unique datasets: {len(dataset_counts)}")
    print(f"  Top 10 datasets:")
    for ds_name, count in dataset_counts.most_common(10):
        print(f"    {ds_name:20s}: {count:3d} questions")

    # Show sample questions
    print("\n📝 Sample Questions (first 5 from dev)")
    print("-" * 70)
    dev_data = splits.get('dev', [])
    for i, sample in enumerate(dev_data[:5], 1):
        print(f"\n[{i}] Question: {sample['question']}")
        print(f"    Answer: {sample['answer']}")
        print(f"    Type: {sample['type']}")
        print(f"    Dataset: {sample['dataset']}")
        print(f"    Columns used: {sample['columns_used']}")

    # Important info about table data
    print("\n" + "=" * 70)
    print("⚠️  IMPORTANT: Table Data Location")
    print("=" * 70)
    print("\nDataBench stores table data separately in parquet files.")
    print("To load table data for a question:")
    print("\n```python")
    print("import pandas as pd")
    print("dataset_id = '001_Forbes'  # from sample['dataset']")
    print("df = pd.read_parquet(f'hf://datasets/cardiffnlp/databench/data/{dataset_id}/all.parquet')")
    print("```")
    print("\nOr for 20-row sample:")
    print("```python")
    print("df = pd.read_parquet(f'hf://datasets/cardiffnlp/databench/data/{dataset_id}/sample.parquet')")
    print("```")

    return splits, list(dataset_counts.keys())


def download_sample_tables(dataset_ids, output_dir: str = "data/databench/tables", max_datasets: int = 10):
    """Download sample table data for first N datasets."""

    print("\n" + "=" * 70)
    print(f"Downloading Sample Tables (first {max_datasets} datasets)")
    print("=" * 70)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for i, ds_id in enumerate(dataset_ids[:max_datasets], 1):
        try:
            print(f"\n[{i}/{max_datasets}] Downloading {ds_id}...")

            # Download sample (20 rows) - faster than full table
            df = pd.read_parquet(f"hf://datasets/cardiffnlp/databench/data/{ds_id}/sample.parquet")

            # Save locally
            save_path = output_path / f"{ds_id}_sample.parquet"
            df.to_parquet(save_path)

            print(f"  ✓ Shape: {df.shape} (rows x cols)")
            print(f"  ✓ Columns: {list(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}")
            print(f"  ✓ Saved to: {save_path}")

        except Exception as e:
            print(f"  ✗ Error: {e}")

    print(f"\n✓ Downloaded {min(max_datasets, len(dataset_ids))} sample tables")
    print(f"✓ Saved to: {output_dir}")


if __name__ == "__main__":
    # Set data directory
    data_dir = "/media/liuyu/DataDrive/DASFAA-Table/data/databench"

    # Analyze QA pairs
    splits, dataset_ids = analyze_databench_qa(data_dir)

    # Download sample tables
    download_sample_tables(dataset_ids, output_dir=f"{data_dir}/tables", max_datasets=5)

    print("\n" + "=" * 70)
    print("✅ Analysis Complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Adapt our Table QA system to DataBench format")
    print("2. Create evaluation script for 50 samples")
    print("3. Run baseline test (target: 55-60% vs 26% baseline)")
