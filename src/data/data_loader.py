"""
Data loader for Table QA datasets
Supports: WikiTQ, TabFact, FeTaQA, SemEval-2025 Task 8
"""

import json
import pandas as pd
import re
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean column names to remove special characters and emojis
    Inspired by AILS-NTUA preprocessing

    Args:
        df: DataFrame with potentially messy column names

    Returns:
        DataFrame with cleaned column names
    """
    def clean_name(name):
        name = str(name)
        # Remove emojis and non-ASCII characters (except common punctuation)
        name = re.sub(r"[^\w\s,.<>@()-]", "", name, flags=re.UNICODE)
        # Remove text enclosed in < >
        name = re.sub(r"<[^>]*>", "", name)
        # Remove text enclosed in ( )
        name = re.sub(r"\([^)]*\)", "", name)
        # Remove Twitter mentions (@username)
        name = re.sub(r"@\w+", "", name)
        # Remove extra whitespace
        name = " ".join(name.split())
        # Remove leading/trailing spaces and quotes
        name = name.strip().strip('"').strip("'")
        # If empty after cleaning, use placeholder
        if not name:
            return "col_unnamed"
        return name

    df.columns = [clean_name(col) for col in df.columns]
    return df


class TableQADataset:
    """Universal dataset loader for all Table QA benchmarks"""

    def __init__(
        self,
        file_path: str,
        dataset_name: str,
        max_samples: Optional[int] = None
    ):
        """
        Args:
            file_path: Path to JSONL data file
            dataset_name: One of ['wikitq', 'tabfact', 'fetaqa', 'semeval2025']
            max_samples: Maximum number of samples to load (for debugging)
        """
        self.file_path = Path(file_path)
        self.dataset_name = dataset_name.lower()
        self.max_samples = max_samples
        self.data = []

        # Validate dataset name
        valid_datasets = ['wikitq', 'tabfact', 'fetaqa', 'semeval2025']
        if self.dataset_name not in valid_datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}. Must be one of {valid_datasets}")

        # Load data
        self._load_data()
        logger.info(f"Loaded {len(self.data)} samples from {self.dataset_name}")

    def _load_data(self):
        """Load data from file"""
        if not self.file_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.file_path}")

        with open(self.file_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if self.max_samples and idx >= self.max_samples:
                    break

                sample = json.loads(line.strip())
                processed = self._process_sample(sample)
                if processed:
                    self.data.append(processed)

    def _process_sample(self, sample: Dict) -> Optional[Dict]:
        """Process a single sample based on dataset type"""
        try:
            if self.dataset_name == 'wikitq':
                return self._process_wikitq(sample)
            elif self.dataset_name == 'tabfact':
                return self._process_tabfact(sample)
            elif self.dataset_name == 'fetaqa':
                return self._process_fetaqa(sample)
            elif self.dataset_name == 'semeval2025':
                return self._process_semeval(sample)
        except Exception as e:
            logger.warning(f"Error processing sample: {e}")
            return None

    def _process_wikitq(self, sample: Dict) -> Dict:
        """
        WikiTQ format:
        {
            "id": "...",
            "question": "...",
            "table": {"header": [...], "rows": [[...], ...]},
            "answer": "..." or ["...", "..."]
        }
        """
        # Convert table to DataFrame
        table_data = sample['table']
        if isinstance(table_data, dict):
            df = pd.DataFrame(table_data['rows'], columns=table_data['header'])
        else:
            df = pd.DataFrame(table_data)

        # Clean column names
        df = clean_column_names(df)

        # Handle multiple possible answers
        answer = sample['answer']
        if isinstance(answer, list):
            answer = answer[0] if answer else ""

        return {
            'id': sample.get('id', ''),
            'question': sample['question'],
            'table': df,
            'answer': str(answer).strip(),
            'dataset': 'wikitq',
            'metadata': sample
        }

    def _process_tabfact(self, sample: Dict) -> Dict:
        """
        TabFact format:
        {
            "id": "...",
            "table": {"header": [...], "rows": [[...], ...]},
            "statement": "...",
            "label": 1 or 0
        }
        """
        # Convert table to DataFrame
        table_data = sample['table']
        if isinstance(table_data, dict):
            df = pd.DataFrame(table_data['rows'], columns=table_data['header'])
        else:
            df = pd.DataFrame(table_data)

        return {
            'id': sample.get('id', ''),
            'question': sample['statement'],
            'table': df,
            'answer': bool(sample['label']),  # True/False
            'dataset': 'tabfact',
            'metadata': sample
        }

    def _process_fetaqa(self, sample: Dict) -> Dict:
        """
        FeTaQA format:
        {
            "id": "...",
            "question": "...",
            "table": {"header": [...], "rows": [[...], ...]},
            "answer": "long answer text..."
        }
        """
        table_data = sample['table']
        if isinstance(table_data, dict):
            df = pd.DataFrame(table_data['rows'], columns=table_data['header'])
        else:
            df = pd.DataFrame(table_data)

        return {
            'id': sample.get('id', ''),
            'question': sample['question'],
            'table': df,
            'answer': sample['answer'],
            'dataset': 'fetaqa',
            'metadata': sample
        }

    def _process_semeval(self, sample: Dict) -> Dict:
        """
        SemEval-2025 Task 8 format:
        {
            "id": "...",
            "question": "...",
            "table": {"header": [...], "data": [[...], ...]},
            "answer": "..."
        }
        """
        table_data = sample['table']
        if 'header' in table_data and 'data' in table_data:
            df = pd.DataFrame(table_data['data'], columns=table_data['header'])
        elif 'header' in table_data and 'rows' in table_data:
            df = pd.DataFrame(table_data['rows'], columns=table_data['header'])
        else:
            df = pd.DataFrame(table_data)

        return {
            'id': sample.get('id', ''),
            'question': sample['question'],
            'table': df,
            'answer': str(sample['answer']),
            'dataset': 'semeval2025',
            'metadata': sample
        }

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        if idx < 0 or idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range [0, {len(self.data)})")
        return self.data[idx]

    def __iter__(self):
        return iter(self.data)

    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        stats = {
            'total_samples': len(self.data),
            'dataset_name': self.dataset_name,
            'avg_question_length': sum(len(s['question'].split()) for s in self.data) / len(self.data),
            'avg_table_rows': sum(len(s['table']) for s in self.data) / len(self.data),
            'avg_table_cols': sum(len(s['table'].columns) for s in self.data) / len(self.data),
        }

        # Answer type distribution
        if self.dataset_name == 'tabfact':
            stats['true_count'] = sum(1 for s in self.data if s['answer'] is True)
            stats['false_count'] = sum(1 for s in self.data if s['answer'] is False)

        return stats


def load_dataset(
    dataset_name: str,
    split: str = 'train',
    data_dir: str = 'data',
    max_samples: Optional[int] = None
) -> TableQADataset:
    """
    Convenience function to load a dataset

    Args:
        dataset_name: One of ['wikitq', 'tabfact', 'fetaqa', 'semeval2025']
        split: One of ['train', 'dev', 'test']
        data_dir: Base directory containing datasets
        max_samples: Maximum samples to load

    Returns:
        TableQADataset instance
    """
    file_path = Path(data_dir) / dataset_name / f"{split}.jsonl"
    return TableQADataset(str(file_path), dataset_name, max_samples)


if __name__ == "__main__":
    # Test loading
    import sys

    # Example: python -m src.data.data_loader
    print("Testing data loader...")

    # Create dummy test file
    test_data = [
        {
            "id": "test-1",
            "question": "What is the population of Beijing?",
            "table": {
                "header": ["City", "Population", "Area"],
                "rows": [
                    ["Beijing", "21.54", "16410"],
                    ["Shanghai", "24.28", "6340"]
                ]
            },
            "answer": "21.54"
        }
    ]

    test_file = Path("data/wikitq/test_sample.jsonl")
    test_file.parent.mkdir(parents=True, exist_ok=True)

    with open(test_file, 'w') as f:
        for sample in test_data:
            f.write(json.dumps(sample) + '\n')

    # Load and test
    dataset = TableQADataset(str(test_file), 'wikitq')
    print(f"\nLoaded {len(dataset)} samples")

    sample = dataset[0]
    print(f"\nSample ID: {sample['id']}")
    print(f"Question: {sample['question']}")
    print(f"Table:\n{sample['table']}")
    print(f"Answer: {sample['answer']}")

    stats = dataset.get_statistics()
    print(f"\nDataset statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\n✓ Data loader test passed!")
