"""
Preprocess real WikiTQ dataset to JSONL format
"""

import json
import csv
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_table_from_csv(csv_path: Path) -> pd.DataFrame:
    """Load table from WikiTQ CSV file"""
    try:
        # WikiTQ CSV files use special format
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # First line is header
        header = lines[0].strip().split('\t') if '\t' in lines[0] else lines[0].strip().split(',')

        # Rest are rows
        rows = []
        for line in lines[1:]:
            row = line.strip().split('\t') if '\t' in line else line.strip().split(',')
            if len(row) == len(header):  # Only keep valid rows
                rows.append(row)

        df = pd.DataFrame(rows, columns=header)
        return df

    except Exception as e:
        logger.error(f"Error loading {csv_path}: {e}")
        return None


def process_wikitq_split(tsv_file: Path, csv_dir: Path, output_file: Path, max_samples: int = None):
    """Process WikiTQ TSV file to JSONL"""

    samples = []
    errors = 0

    with open(tsv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')

        for i, row in enumerate(reader):
            if max_samples and i >= max_samples:
                break

            try:
                # Get table CSV path
                csv_rel_path = row['context']
                # Remove 'csv/' prefix if it exists (path is relative to csv_dir)
                if csv_rel_path.startswith('csv/'):
                    csv_rel_path = csv_rel_path[4:]
                csv_path = csv_dir / csv_rel_path

                if not csv_path.exists():
                    logger.warning(f"Table file not found: {csv_path}")
                    errors += 1
                    continue

                # Load table
                table = load_table_from_csv(csv_path)
                if table is None or table.empty:
                    errors += 1
                    continue

                # Create sample
                sample = {
                    'id': row['id'],
                    'question': row['utterance'],
                    'table': {
                        'header': list(table.columns),
                        'rows': table.values.tolist()
                    },
                    'answer': row['targetValue']
                }

                samples.append(sample)

            except Exception as e:
                logger.error(f"Error processing row {i}: {e}")
                errors += 1
                continue

    # Write to JSONL
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    logger.info(f"✓ Processed {len(samples)} samples to {output_file}")
    if errors > 0:
        logger.warning(f"  {errors} samples had errors")

    return len(samples)


def main():
    """Process all WikiTQ splits"""

    # Paths
    base_dir = Path("data/wikitq/raw/WikiTableQuestions")
    csv_dir = base_dir / "csv"
    data_dir = base_dir / "data"
    output_dir = Path("data/wikitq")

    logger.info("Processing WikiTQ dataset...")
    logger.info(f"CSV directory: {csv_dir}")
    logger.info(f"Data directory: {data_dir}")

    # Process splits
    splits = {
        'train': ('training.tsv', None),  # Full training set
        'dev': ('pristine-unseen-tables.tsv', None),  # Standard dev set
        'test': ('pristine-unseen-tables.tsv', None),  # Use same as dev for now
    }

    total_samples = 0
    for split_name, (tsv_file, max_samples) in splits.items():
        logger.info(f"\nProcessing {split_name} split...")

        tsv_path = data_dir / tsv_file
        output_file = output_dir / f"{split_name}.jsonl"

        if not tsv_path.exists():
            logger.error(f"TSV file not found: {tsv_path}")
            continue

        n_samples = process_wikitq_split(
            tsv_path,
            csv_dir,
            output_file,
            max_samples=max_samples
        )

        total_samples += n_samples

    logger.info(f"\n✓ WikiTQ preprocessing complete!")
    logger.info(f"  Total samples: {total_samples}")
    logger.info(f"  Output directory: {output_dir}")


if __name__ == "__main__":
    main()
