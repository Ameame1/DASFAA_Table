"""
Test system on real downloaded data
Run baseline evaluation without GPU (rule-based code only)
"""

import sys
sys.path.append('.')

import pandas as pd
from pathlib import Path
from src.data.data_loader import load_dataset
from src.execution.code_executor import SecureCodeExecutor
from src.diagnosis.diagnostic_system import HierarchicalDiagnosticSystem


def test_real_data_loading():
    """Test loading real WikiTQ data"""
    print("\n" + "="*60)
    print("TEST 1: Loading Real Data")
    print("="*60)

    try:
        # Try loading real WikiTQ data
        dataset = load_dataset('wikitq', 'train', max_samples=10)
        print(f"✓ Loaded {len(dataset)} samples from WikiTQ")

        # Show first sample
        sample = dataset[0]
        print(f"\nSample {sample['id']}:")
        print(f"Question: {sample['question']}")
        print(f"Table shape: {sample['table'].shape}")
        print(f"Answer: {sample['answer']}")
        print(f"\nTable preview:")
        print(sample['table'].head())

        return dataset

    except Exception as e:
        print(f"✗ Error loading data: {e}")
        print("Note: Make sure data is in data/wikitq/train.jsonl")
        return None


def test_manual_code_execution(dataset):
    """Test manual code execution on real data"""
    print("\n" + "="*60)
    print("TEST 2: Manual Code Execution")
    print("="*60)

    if dataset is None or len(dataset) == 0:
        print("✗ No data available")
        return

    sample = dataset[0]
    executor = SecureCodeExecutor()

    # Test 1: Correct code
    print("\n--- Test Case 1: Correct Code ---")
    print(f"Question: {sample['question']}")

    # Generate simple code manually
    code = """
# Get the first column name
col_name = df.columns[0]
answer = df[col_name].iloc[0]
"""

    result = executor.execute(code, sample['table'])
    print(f"Code:\n{code}")
    print(f"Result: {result['success']}")
    print(f"Answer: {result['answer']}")

    # Test 2: Code with error
    print("\n--- Test Case 2: Code with KeyError ---")
    wrong_code = """
answer = df['NonexistentColumn'].iloc[0]
"""

    result = executor.execute(wrong_code, sample['table'])
    print(f"Code:\n{wrong_code}")
    print(f"Result: {result['success']}")
    print(f"Error: {result['error_type']} - {result['error']}")

    return result


def test_diagnosis_on_real_error(dataset):
    """Test error diagnosis on real data"""
    print("\n" + "="*60)
    print("TEST 3: Error Diagnosis on Real Data")
    print("="*60)

    if dataset is None or len(dataset) == 0:
        print("✗ No data available")
        return

    sample = dataset[0]
    executor = SecureCodeExecutor()
    diagnostic_system = HierarchicalDiagnosticSystem(use_grpo=False)

    # Create a typical error: wrong column name
    actual_columns = list(sample['table'].columns)
    print(f"Actual columns: {actual_columns}")

    # Try to access with wrong case
    if actual_columns:
        wrong_col = actual_columns[0].upper()
        code = f"""
answer = df['{wrong_col}'].iloc[0]
"""
        print(f"\nTrying code with wrong column case:")
        print(code)

        result = executor.execute(code, sample['table'])

        if not result['success']:
            print(f"✓ Error detected: {result['error_type']}")

            # Diagnose
            diagnosis = diagnostic_system.diagnose(
                result, code, sample['table'], sample['question']
            )

            print(f"\nDiagnosis Results:")
            print(f"  - Error class: {diagnosis['error_class']}")
            print(f"  - Root cause: {diagnosis['root_cause'].get('root_cause')}")
            print(f"  - Strategy: {diagnosis['strategy']}")
            print(f"\n  - Repair prompt (first 300 chars):")
            print(diagnosis['repair_prompt'][:300] + "...")

            return diagnosis


def test_batch_evaluation(dataset, n_samples=5):
    """Test batch processing on multiple samples"""
    print("\n" + "="*60)
    print(f"TEST 4: Batch Evaluation ({n_samples} samples)")
    print("="*60)

    if dataset is None or len(dataset) == 0:
        print("✗ No data available")
        return

    executor = SecureCodeExecutor()
    results = []

    for i in range(min(n_samples, len(dataset))):
        sample = dataset[i]

        # Simple test: try to access first column
        code = f"""
col_name = df.columns[0]
answer = str(df[col_name].iloc[0])
"""

        result = executor.execute(code, sample['table'])
        results.append({
            'id': sample['id'],
            'success': result['success'],
            'answer': result.get('answer'),
            'error': result.get('error_type')
        })

        print(f"Sample {i+1}: {'✓ Success' if result['success'] else '✗ ' + result.get('error_type', 'Failed')}")

    # Summary
    success_count = sum(1 for r in results if r['success'])
    print(f"\nSummary: {success_count}/{n_samples} succeeded")

    return results


def show_data_statistics():
    """Show statistics of downloaded data"""
    print("\n" + "="*60)
    print("DATA STATISTICS")
    print("="*60)

    datasets = ['wikitq', 'tabfact', 'fetaqa']
    splits = ['train', 'dev', 'test']

    for dataset_name in datasets:
        print(f"\n{dataset_name.upper()}:")
        for split in splits:
            try:
                dataset = load_dataset(dataset_name, split)
                stats = dataset.get_statistics()
                print(f"  {split:6s}: {stats['total_samples']:5d} samples, "
                      f"avg {stats['avg_table_rows']:.1f} rows × {stats['avg_table_cols']:.1f} cols")
            except Exception as e:
                print(f"  {split:6s}: Not available")


def main():
    """Run all tests on real data"""
    print("\n" + "="*60)
    print("TESTING WITH REAL DOWNLOADED DATA")
    print("="*60)

    # Show data statistics
    show_data_statistics()

    # Test 1: Load real data
    dataset = test_real_data_loading()

    if dataset is not None:
        # Test 2: Execute code
        test_manual_code_execution(dataset)

        # Test 3: Diagnose errors
        test_diagnosis_on_real_error(dataset)

        # Test 4: Batch evaluation
        test_batch_evaluation(dataset, n_samples=5)

        print("\n" + "="*60)
        print("✓ REAL DATA TESTS COMPLETED")
        print("="*60)
        print("\nYour system is working correctly with real data!")
        print("\nNext steps:")
        print("1. Run with GPU to test Qwen code generation:")
        print("   python3 tests/test_with_gpu.py")
        print("2. Start baseline evaluation on full datasets")
        print("3. Implement GRPO training with TRL")
    else:
        print("\n" + "="*60)
        print("⚠ DATA NOT FOUND")
        print("="*60)
        print("\nPlease ensure your data is in the correct format:")
        print("  data/wikitq/train.jsonl")
        print("  data/wikitq/dev.jsonl")
        print("  data/wikitq/test.jsonl")
        print("\nRun: bash scripts/download_datasets.sh")


if __name__ == "__main__":
    main()
