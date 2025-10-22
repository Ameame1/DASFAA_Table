#!/usr/bin/env python3
"""
Preprocess real TabFact dataset from raw format to JSONL
"""

import json
import csv
import random
from pathlib import Path
from tqdm import tqdm

def load_table_csv(table_id, csv_dir):
    """Load table from CSV file"""
    csv_path = csv_dir / table_id

    if not csv_path.exists():
        return None

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='#')
        rows = list(reader)

    if not rows:
        return None

    # First row is header
    header = rows[0]
    data_rows = rows[1:]

    return {
        'header': header,
        'rows': data_rows
    }

def preprocess_tabfact(
    json_file: str,
    csv_dir: str,
    output_file: str,
    max_samples: int = None,
    train_ratio: float = 0.8,
    seed: int = 42
):
    """
    Preprocess TabFact data

    Args:
        json_file: Path to r1_training_all.json or r2_training_all.json
        csv_dir: Directory containing CSV files
        output_file: Output JSONL file
        max_samples: Maximum number of samples to process (None = all)
        train_ratio: Ratio for train/dev split
        seed: Random seed
    """

    print(f"Loading TabFact data from {json_file}...")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    csv_dir_path = Path(csv_dir)

    # Collect all samples
    all_samples = []
    skipped_tables = 0

    for table_id, table_data in tqdm(data.items(), desc="Processing tables"):
        # Load table CSV
        table = load_table_csv(table_id, csv_dir_path)

        if table is None:
            skipped_tables += 1
            continue

        # table_data is a list: [statements, labels, ...]
        statements = table_data[0]
        labels = table_data[1]

        # Create samples for each statement
        for statement, label in zip(statements, labels):
            sample = {
                'table_id': table_id,
                'table': table,
                'statement': statement,
                'label': label  # 1 = True, 0 = False
            }
            all_samples.append(sample)

        # Limit samples if specified
        if max_samples and len(all_samples) >= max_samples:
            all_samples = all_samples[:max_samples]
            break

    print(f"Total samples collected: {len(all_samples)}")
    print(f"Skipped tables (CSV not found): {skipped_tables}")

    # Shuffle and split
    random.seed(seed)
    random.shuffle(all_samples)

    split_idx = int(len(all_samples) * train_ratio)
    train_samples = all_samples[:split_idx]
    dev_samples = all_samples[split_idx:]

    # Save to JSONL
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in dev_samples:  # Using dev split for evaluation
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"✓ Saved {len(dev_samples)} dev samples to {output_file}")

    # Also save train split
    train_output = output_path.parent / 'train.jsonl'
    with open(train_output, 'w', encoding='utf-8') as f:
        for sample in train_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"✓ Saved {len(train_samples)} train samples to {train_output}")

    return dev_samples


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess TabFact dataset")
    parser.add_argument("--json_file", type=str,
                       default="/media/liuyu/DataDrive/DASFAA-Table/data/tabfact/raw/Table-Fact-Checking/collected_data/r2_training_all.json",
                       help="Path to r2_training_all.json")
    parser.add_argument("--csv_dir", type=str,
                       default="/media/liuyu/DataDrive/DASFAA-Table/data/tabfact/raw/Table-Fact-Checking/data/all_csv",
                       help="Directory containing CSV files")
    parser.add_argument("--output", type=str,
                       default="/media/liuyu/DataDrive/DASFAA-Table/data/tabfact/processed/dev.jsonl",
                       help="Output JSONL file")
    parser.add_argument("--max_samples", type=int, default=1000,
                       help="Maximum samples to process (for testing)")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                       help="Train/dev split ratio")

    args = parser.parse_args()

    preprocess_tabfact(
        json_file=args.json_file,
        csv_dir=args.csv_dir,
        output_file=args.output,
        max_samples=args.max_samples,
        train_ratio=args.train_ratio
    )
