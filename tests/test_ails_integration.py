"""
Test AILS-NTUA integration with code_generator
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.baselines.code_generator import QwenCodeGenerator
from examples.wikitq_fewshot_examples import WIKITQ_EXAMPLES
from examples.tabfact_fewshot_examples import TABFACT_EXAMPLES


def test_baseline_prompt():
    """Test original baseline prompt"""
    print("=" * 80)
    print("TEST 1: Original Baseline Prompt")
    print("=" * 80)

    df = pd.DataFrame({
        'year': [2015, 2016, 2017],
        'team': ['Team A', 'Team B', 'Team A'],
        'score': [95, 88, 92]
    })

    generator = QwenCodeGenerator(use_ails_prompt=False)
    prompt = generator._create_prompt(df, "What was the score in 2015?")
    print(prompt)
    print("\n")


def test_ails_zero_shot():
    """Test AILS zero-shot prompt"""
    print("=" * 80)
    print("TEST 2: AILS Zero-Shot Prompt")
    print("=" * 80)

    df = pd.DataFrame({
        'year': [2015, 2016, 2017],
        'team': ['Team A', 'Team B', 'Team A'],
        'score': [95, 88, 92]
    })

    generator = QwenCodeGenerator(use_ails_prompt=True)
    prompt = generator._create_prompt(df, "What was the score in 2015?")
    print(prompt)
    print("\n")


def test_ails_few_shot_wikitq():
    """Test AILS few-shot prompt with WikiTQ examples"""
    print("=" * 80)
    print("TEST 3: AILS Few-Shot Prompt (WikiTQ)")
    print("=" * 80)

    df = pd.DataFrame({
        'year': [2015, 2016, 2017, 2018, 2019],
        'winner': ['Team A', 'Team B', 'Team A', 'Team C', 'Team B'],
        'score': [95, 88, 92, 90, 93]
    })

    generator = QwenCodeGenerator(
        use_ails_prompt=True,
        few_shot_examples=WIKITQ_EXAMPLES
    )
    prompt = generator._create_prompt(df, "Who won in 2017?")
    print(prompt[:2000])  # Print first 2000 chars
    print("...")
    print(f"[Total length: {len(prompt)} chars]")
    print("\n")


def test_ails_few_shot_tabfact():
    """Test AILS few-shot prompt with TabFact examples"""
    print("=" * 80)
    print("TEST 4: AILS Few-Shot Prompt (TabFact)")
    print("=" * 80)

    df = pd.DataFrame({
        'year': [2018, 2019, 2020, 2021, 2022],
        'revenue': ['80M', '90M', '100M', '110M', '120M']
    })

    generator = QwenCodeGenerator(
        use_ails_prompt=True,
        few_shot_examples=TABFACT_EXAMPLES
    )
    prompt = generator._create_prompt(
        df,
        "Is the following statement true? The revenue in 2021 was 110M"
    )
    print(prompt[:2000])  # Print first 2000 chars
    print("...")
    print(f"[Total length: {len(prompt)} chars]")
    print("\n")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Testing AILS-NTUA Integration")
    print("=" * 80 + "\n")

    test_baseline_prompt()
    test_ails_zero_shot()
    test_ails_few_shot_wikitq()
    test_ails_few_shot_tabfact()

    print("=" * 80)
    print("All tests completed successfully!")
    print("=" * 80)
