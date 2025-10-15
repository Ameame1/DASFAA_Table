"""
Preprocess FeTaQA dataset to JSONL format
"""

import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_fetaqa():
    """Create sample FeTaQA data"""

    output_dir = Path("data/fetaqa")
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = [
        {
            'id': 'fetaqa-sample-1',
            'table': {
                'header': ['Country', 'Capital', 'Population'],
                'rows': [
                    ['China', 'Beijing', '1.4B'],
                    ['USA', 'Washington D.C.', '330M'],
                    ['India', 'New Delhi', '1.38B']
                ]
            },
            'question': 'Which countries have populations over 1 billion?',
            'answer': 'China and India have populations over 1 billion. China has 1.4 billion people and India has 1.38 billion people.'
        }
    ]

    for split in ['train', 'dev', 'test']:
        output_file = output_dir / f"{split}.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        logger.info(f"✓ Created {len(samples)} sample entries in {output_file}")


if __name__ == "__main__":
    create_sample_fetaqa()
