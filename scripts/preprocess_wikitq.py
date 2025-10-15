"""
Preprocess WikiTQ dataset to JSONL format
"""

import json
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_wikitq():
    """Convert WikiTQ to standard JSONL format"""

    raw_dir = Path("data/wikitq/raw/WikiTableQuestions")
    output_dir = Path("data/wikitq")
    output_dir.mkdir(parents=True, exist_ok=True)

    # WikiTQ has train-test split
    splits = {
        'train': raw_dir / 'data' / 'random-split-1-train.tsv',
        'dev': raw_dir / 'data' / 'random-split-1-dev.tsv',
        'test': raw_dir / 'data' / 'pristine-unseen-tables.tsv'
    }

    for split_name, tsv_file in splits.items():
        if not tsv_file.exists():
            logger.warning(f"File not found: {tsv_file}")
            logger.info("Creating sample data instead...")
            create_sample_data(output_dir, split_name)
            continue

        logger.info(f"Processing {split_name}...")

        # Read TSV
        df = pd.read_csv(tsv_file, sep='\t')

        samples = []
        for idx, row in df.iterrows():
            try:
                # Parse table
                table_data = parse_wikitq_table(row)

                sample = {
                    'id': row.get('id', f'wikitq-{split_name}-{idx}'),
                    'question': row['utterance'],
                    'table': table_data,
                    'answer': row['targetValue']
                }
                samples.append(sample)

            except Exception as e:
                logger.warning(f"Error processing row {idx}: {e}")
                continue

        # Write JSONL
        output_file = output_dir / f"{split_name}.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        logger.info(f"✓ Wrote {len(samples)} samples to {output_file}")


def parse_wikitq_table(row):
    """Parse WikiTQ table format"""
    # WikiTQ stores tables in a specific format
    # This is a simplified version - actual parsing may need adjustment

    # For now, create a simple structure
    # You may need to adjust this based on actual WikiTQ format
    return {
        'header': ['Column1', 'Column2'],  # Placeholder
        'rows': [['Value1', 'Value2']]     # Placeholder
    }


def create_sample_data(output_dir, split_name):
    """Create sample data for testing when real data is not available"""

    samples = [
        {
            'id': f'sample-{split_name}-1',
            'question': 'What is the total population of Beijing and Shanghai?',
            'table': {
                'header': ['City', 'Population', 'GDP'],
                'rows': [
                    ['Beijing', '21.54', '4.0'],
                    ['Shanghai', '24.28', '4.3'],
                    ['Guangzhou', '15.3', '2.8']
                ]
            },
            'answer': '45.82'
        },
        {
            'id': f'sample-{split_name}-2',
            'question': 'Which city has the highest GDP?',
            'table': {
                'header': ['City', 'Population', 'GDP'],
                'rows': [
                    ['Beijing', '21.54', '4.0'],
                    ['Shanghai', '24.28', '4.3'],
                    ['Guangzhou', '15.3', '2.8']
                ]
            },
            'answer': 'Shanghai'
        },
        {
            'id': f'sample-{split_name}-3',
            'question': 'How many cities are in the table?',
            'table': {
                'header': ['City', 'Population', 'GDP'],
                'rows': [
                    ['Beijing', '21.54', '4.0'],
                    ['Shanghai', '24.28', '4.3'],
                    ['Guangzhou', '15.3', '2.8']
                ]
            },
            'answer': '3'
        }
    ]

    # Write sample data
    output_file = output_dir / f"{split_name}.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    logger.info(f"✓ Created {len(samples)} sample entries in {output_file}")


if __name__ == "__main__":
    preprocess_wikitq()
