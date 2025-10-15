"""
Preprocess TabFact dataset to JSONL format
"""

import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_tabfact():
    """Create sample TabFact data"""

    output_dir = Path("data/tabfact")
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = [
        {
            'id': 'tabfact-sample-1',
            'table': {
                'header': ['Year', 'Revenue', 'Profit'],
                'rows': [
                    ['2020', '100M', '10M'],
                    ['2021', '120M', '15M'],
                    ['2022', '150M', '20M']
                ]
            },
            'statement': 'The revenue increased every year from 2020 to 2022.',
            'label': 1  # True
        },
        {
            'id': 'tabfact-sample-2',
            'table': {
                'header': ['Year', 'Revenue', 'Profit'],
                'rows': [
                    ['2020', '100M', '10M'],
                    ['2021', '120M', '15M'],
                    ['2022', '150M', '20M']
                ]
            },
            'statement': 'The profit decreased in 2021.',
            'label': 0  # False
        }
    ]

    for split in ['train', 'dev', 'test']:
        output_file = output_dir / f"{split}.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        logger.info(f"✓ Created {len(samples)} sample entries in {output_file}")


if __name__ == "__main__":
    create_sample_tabfact()
