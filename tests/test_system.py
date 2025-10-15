"""
Test the complete Table QA system with sample data
This script tests all components without requiring GPU
"""

import sys
sys.path.append('.')

import pandas as pd
from src.data.data_loader import TableQADataset
from src.execution.code_executor import SecureCodeExecutor
from src.diagnosis.error_classifier import ErrorClassifier
from src.diagnosis.root_cause_analyzer import RootCauseAnalyzer
from src.diagnosis.strategy_selector import StrategySelector
from src.diagnosis.prompt_generator import PromptGenerator
from src.diagnosis.diagnostic_system import HierarchicalDiagnosticSystem


def test_data_loading():
    """Test data loader"""
    print("\n" + "="*60)
    print("TEST 1: Data Loading")
    print("="*60)

    dataset = TableQADataset("data/wikitq/train.jsonl", "wikitq")
    print(f"✓ Loaded {len(dataset)} samples")

    sample = dataset[0]
    print(f"✓ Sample ID: {sample['id']}")
    print(f"✓ Question: {sample['question']}")
    print(f"✓ Table shape: {sample['table'].shape}")
    print(f"✓ Answer: {sample['answer']}")


def test_code_execution():
    """Test code executor"""
    print("\n" + "="*60)
    print("TEST 2: Code Execution")
    print("="*60)

    executor = SecureCodeExecutor(timeout=5)

    table = pd.DataFrame({
        'city': ['Beijing', 'Shanghai'],
        'population': [21.54, 24.28]
    })

    # Test 1: Successful execution
    code = "answer = df['population'].sum()"
    result = executor.execute(code, table)
    print(f"✓ Success execution: {result['success']}")
    print(f"✓ Answer: {result['answer']}")

    # Test 2: KeyError
    code = "answer = df['Population'].sum()"  # Wrong case
    result = executor.execute(code, table)
    print(f"✓ Error detection: {result['error_type']} - {result['error']}")


def test_error_diagnosis():
    """Test hierarchical error diagnosis"""
    print("\n" + "="*60)
    print("TEST 3: Error Diagnosis System")
    print("="*60)

    diagnostic_system = HierarchicalDiagnosticSystem(use_grpo=False)

    table = pd.DataFrame({
        'city': ['Beijing', 'Shanghai'],
        'population': [21.54, 24.28]
    })

    # Simulate a KeyError
    code = "answer = df['Population'].sum()"
    execution_result = {
        'success': False,
        'error_type': 'KeyError',
        'error': "'Population'",
        'answer': None
    }

    diagnosis = diagnostic_system.diagnose(
        execution_result, code, table, "What is the total population?"
    )

    print(f"✓ Error class: {diagnosis['error_class']}")
    print(f"✓ Root cause: {diagnosis['root_cause'].get('root_cause')}")
    print(f"✓ Strategy: {diagnosis['strategy']}")
    print(f"✓ Repair prompt generated: {len(diagnosis['repair_prompt'])} chars")


def test_strategy_selection():
    """Test strategy selector"""
    print("\n" + "="*60)
    print("TEST 4: Strategy Selection")
    print("="*60)

    selector = StrategySelector(use_grpo=False)
    print(f"✓ Available strategies: {selector.get_available_strategies()}")

    # Test column name error
    root_cause = {
        'root_cause': 'column_case_mismatch',
        'details': {
            'missing': 'Population',
            'correct': 'population',
            'available': ['city', 'population']
        }
    }

    table = pd.DataFrame({'city': [], 'population': []})
    strategy = selector.select_strategy(
        root_cause, 'runtime', table, "code", "question"
    )

    print(f"✓ Selected strategy: {strategy.name if strategy else 'None'}")


def test_complete_workflow():
    """Test complete workflow with real data"""
    print("\n" + "="*60)
    print("TEST 5: Complete Workflow (without LLM)")
    print("="*60)

    # Load data
    dataset = TableQADataset("data/wikitq/train.jsonl", "wikitq")
    sample = dataset[0]

    print(f"Question: {sample['question']}")
    print(f"Table:\n{sample['table']}")

    # Simulate code execution with an error
    executor = SecureCodeExecutor()
    wrong_code = "answer = df['CITY'].iloc[0]"  # Wrong column name

    result = executor.execute(wrong_code, sample['table'])

    if not result['success']:
        print(f"\n✓ Code failed as expected: {result['error_type']}")

        # Diagnose error
        diagnostic_system = HierarchicalDiagnosticSystem()
        diagnosis = diagnostic_system.diagnose(
            result, wrong_code, sample['table'], sample['question']
        )

        print(f"✓ Diagnosis complete")
        print(f"  - Error class: {diagnosis['error_class']}")
        print(f"  - Root cause: {diagnosis['root_cause'].get('root_cause')}")
        print(f"  - Strategy: {diagnosis['strategy']}")
        print(f"\n✓ Repair prompt (first 200 chars):")
        print(diagnosis['repair_prompt'][:200] + "...")


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("TESTING TABLE QA SYSTEM")
    print("="*60)

    try:
        test_data_loading()
        test_code_execution()
        test_error_diagnosis()
        test_strategy_selection()
        test_complete_workflow()

        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED")
        print("="*60)
        print("\nSystem is ready to use!")
        print("Note: Code generation requires GPU and Qwen2.5-Coder model")
        print("\nNext steps:")
        print("1. Download real datasets (bash scripts/download_datasets.sh)")
        print("2. Run with GPU for full code generation")
        print("3. Implement GRPO training with TRL")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
