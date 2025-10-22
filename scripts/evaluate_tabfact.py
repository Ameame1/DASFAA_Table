#!/usr/bin/env python3
"""
Evaluate our Table QA system on TabFact dataset.

TabFact baseline: ~78%
Our target: 82-85%
"""

import os
import sys
import json
import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.system.table_qa_system import TableQASystem


def load_tabfact_sample(sample: dict) -> pd.DataFrame:
    """Convert TabFact sample to DataFrame."""
    table_data = sample['table']

    # TabFact format: {'header': [...], 'rows': [[...], ...]}
    headers = table_data['header']
    rows = table_data['rows']

    # Create DataFrame
    df = pd.DataFrame(rows, columns=headers)

    return df


def normalize_answer(answer, expected_type='bool'):
    """Normalize answer for comparison."""
    if answer is None:
        return None

    # For TabFact, answers are boolean (1/0 or True/False)
    answer_str = str(answer).strip().lower()

    # Convert to boolean
    if answer_str in ['true', '1', '1.0', 'yes']:
        return True
    elif answer_str in ['false', '0', '0.0', 'no']:
        return False

    # Try to parse as number
    try:
        num = float(answer_str)
        return num > 0
    except:
        pass

    return None


def evaluate_tabfact(
    data_file: str,
    num_samples: int = 100,
    output_file: str = None,
    verbose: bool = True
):
    """Evaluate on TabFact dataset."""

    print("=" * 70)
    print("TabFact Evaluation")
    print("=" * 70)
    print(f"\nDataset: {data_file}")
    print(f"Samples: {num_samples}")
    print(f"Baseline: ~78%")
    print(f"Our Target: 82-85%\n")

    # Load data
    with open(data_file, 'r', encoding='utf-8') as f:
        samples = [json.loads(line) for line in f][:num_samples]

    # Initialize system
    print("Loading Table QA System...")
    qa_system = TableQASystem(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        max_iterations=3
    )
    print("✓ System loaded\n")

    # Evaluate
    results = []
    correct = 0
    execution_success = 0
    total_iterations = 0
    skipped = 0

    print("Running evaluation...")
    for i, sample in enumerate(tqdm(samples, desc="Evaluating"), 1):
        statement = sample['statement']
        ground_truth_label = sample['label']  # 1 or 0
        ground_truth = bool(ground_truth_label)

        if verbose:
            print(f"\n{'=' * 70}")
            print(f"Sample {i}/{num_samples}")
            print(f"{'=' * 70}")
            print(f"Statement: {statement}")
            print(f"Ground Truth: {ground_truth} (label: {ground_truth_label})")

        # Load table
        try:
            table = load_tabfact_sample(sample)
        except Exception as e:
            if verbose:
                print(f"✗ Failed to load table: {e}")
            skipped += 1
            results.append({
                'sample_id': i,
                'statement': statement,
                'ground_truth': ground_truth,
                'predicted': None,
                'correct': False,
                'execution_success': False,
                'iterations': 0,
                'error': f'Table load failed: {e}'
            })
            continue

        if verbose:
            print(f"Table shape: {table.shape}")
            print(f"Columns: {list(table.columns)}")

        # Reformulate as a question for code generation
        question = f"Is the following statement true or false? {statement}"

        # Run QA system
        try:
            result = qa_system.answer_question(table, question)

            predicted = result['answer']
            is_success = result['success']
            iterations = result['iterations']

            # Normalize answer to boolean
            pred_bool = normalize_answer(predicted)

            # Check correctness
            is_correct = (pred_bool == ground_truth) if pred_bool is not None else False

            if is_success:
                execution_success += 1

            if is_correct:
                correct += 1

            total_iterations += iterations

            if verbose:
                print(f"Predicted: {predicted} -> {pred_bool}")
                print(f"Execution: {'✓ Success' if is_success else '✗ Failed'}")
                print(f"Iterations: {iterations}")
                print(f"Correctness: {'✓ Correct' if is_correct else '✗ Wrong'}")

            results.append({
                'sample_id': i,
                'statement': statement,
                'ground_truth': ground_truth,
                'predicted': pred_bool,
                'predicted_raw': str(predicted),
                'correct': is_correct,
                'execution_success': is_success,
                'iterations': iterations,
                'error': result.get('error', None) if not is_success else None
            })

        except Exception as e:
            if verbose:
                print(f"✗ System error: {e}")

            results.append({
                'sample_id': i,
                'statement': statement,
                'ground_truth': ground_truth,
                'predicted': None,
                'correct': False,
                'execution_success': False,
                'iterations': 0,
                'error': str(e)
            })

    # Calculate metrics
    valid_samples = num_samples - skipped
    exec_rate = 100 * execution_success / valid_samples if valid_samples > 0 else 0
    accuracy = 100 * correct / valid_samples if valid_samples > 0 else 0
    avg_iterations = total_iterations / execution_success if execution_success > 0 else 0

    # Print summary
    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)
    print(f"Total samples: {num_samples}")
    print(f"Skipped: {skipped}")
    print(f"Valid samples: {valid_samples}")
    print(f"\nExecution Success: {execution_success}/{valid_samples} ({exec_rate:.1f}%)")
    print(f"Answer Correctness: {correct}/{valid_samples} ({accuracy:.1f}%)")
    print(f"Average Iterations: {avg_iterations:.2f}")
    print(f"\n{'=' * 70}")
    print(f"vs Baseline (78%): {accuracy - 78:+.1f}%")
    print(f"Target (82-85%): {'✓ ACHIEVED!' if accuracy >= 82 else f'Gap: {82 - accuracy:.1f}%'}")
    print("=" * 70)

    # Save results
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'summary': {
                    'total_samples': num_samples,
                    'skipped': skipped,
                    'valid_samples': valid_samples,
                    'execution_success': execution_success,
                    'execution_rate': exec_rate,
                    'correct': correct,
                    'accuracy': accuracy,
                    'average_iterations': avg_iterations
                },
                'results': results
            }, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Results saved to: {output_file}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate on TabFact")
    parser.add_argument("--data_file", type=str,
                       default="/media/liuyu/DataDrive/DASFAA-Table/data/tabfact/processed/dev.jsonl",
                       help="TabFact data file")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of samples to evaluate")
    parser.add_argument("--output", type=str,
                       default="/media/liuyu/DataDrive/DASFAA-Table/results/tabfact_eval.json",
                       help="Output file for results")
    parser.add_argument("--verbose", action="store_true", default=False,
                       help="Verbose output")

    args = parser.parse_args()

    evaluate_tabfact(
        data_file=args.data_file,
        num_samples=args.num_samples,
        output_file=args.output,
        verbose=args.verbose
    )
