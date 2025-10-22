#!/usr/bin/env python3
"""
Evaluate our Table QA system on WikiTQ dataset.

WikiTQ baseline: ~54%
Our target: 60-65%
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


def load_wikitq_sample(sample: dict) -> pd.DataFrame:
    """Convert WikiTQ sample to DataFrame."""
    table_data = sample['table']

    # WikiTQ format: {'header': [...], 'rows': [[...], ...]}
    headers = table_data['header']
    rows = table_data['rows']

    # Clean headers (remove quotes)
    headers = [h.strip('"') for h in headers]

    # Clean rows (remove quotes)
    cleaned_rows = []
    for row in rows:
        cleaned_row = [cell.strip('"') if isinstance(cell, str) else cell for cell in row]
        cleaned_rows.append(cleaned_row)

    # Create DataFrame
    df = pd.DataFrame(cleaned_rows, columns=headers)

    return df


def normalize_answer(answer):
    """Normalize answer for comparison."""
    if answer is None:
        return None

    # Convert to string and lowercase
    answer_str = str(answer).strip().lower()

    # Remove common variations
    answer_str = answer_str.replace('_', ' ')

    return answer_str


def evaluate_wikitq(
    data_file: str,
    num_samples: int = 100,
    output_file: str = None,
    verbose: bool = True
):
    """Evaluate on WikiTQ dataset."""

    print("=" * 70)
    print("WikiTQ Evaluation")
    print("=" * 70)
    print(f"\nDataset: {data_file}")
    print(f"Samples: {num_samples}")
    print(f"Baseline: ~54%")
    print(f"Our Target: 60-65%\n")

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
        question = sample['question']
        ground_truth = sample['answer']

        if verbose:
            print(f"\n{'=' * 70}")
            print(f"Sample {i}/{num_samples}")
            print(f"{'=' * 70}")
            print(f"Question: {question}")
            print(f"Ground Truth: {ground_truth}")

        # Load table
        try:
            table = load_wikitq_sample(sample)
        except Exception as e:
            if verbose:
                print(f"✗ Failed to load table: {e}")
            skipped += 1
            results.append({
                'sample_id': i,
                'question': question,
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
            print(f"Columns: {list(table.columns)[:5]}{'...' if len(table.columns) > 5 else ''}")

        # Run QA system
        try:
            result = qa_system.answer_question(table, question)

            predicted = result['answer']
            is_success = result['success']
            iterations = result['iterations']

            # Normalize answers for comparison
            pred_norm = normalize_answer(predicted)
            gt_norm = normalize_answer(ground_truth)

            # Check correctness
            is_correct = (pred_norm == gt_norm)

            if is_success:
                execution_success += 1

            if is_correct:
                correct += 1

            total_iterations += iterations

            if verbose:
                print(f"Predicted: {predicted}")
                print(f"Execution: {'✓ Success' if is_success else '✗ Failed'}")
                print(f"Iterations: {iterations}")
                print(f"Correctness: {'✓ Correct' if is_correct else '✗ Wrong'}")

            results.append({
                'sample_id': i,
                'question': question,
                'ground_truth': ground_truth,
                'predicted': predicted,
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
                'question': question,
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
    print(f"vs Baseline (54%): {accuracy - 54:+.1f}%")
    print(f"Target (60-65%): {'✓ ACHIEVED!' if accuracy >= 60 else f'Gap: {60 - accuracy:.1f}%'}")
    print("=" * 70)

    # Save results
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy/pandas types to native Python types for JSON serialization
        def convert_to_native(obj):
            """Convert numpy/pandas types to native Python types"""
            import numpy as np
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            return obj

        # Convert all results
        serializable_results = convert_to_native(results)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'summary': {
                    'total_samples': int(num_samples),
                    'skipped': int(skipped),
                    'valid_samples': int(valid_samples),
                    'execution_success': int(execution_success),
                    'execution_rate': float(exec_rate),
                    'correct': int(correct),
                    'accuracy': float(accuracy),
                    'average_iterations': float(avg_iterations)
                },
                'results': serializable_results
            }, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Results saved to: {output_file}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate on WikiTQ")
    parser.add_argument("--data_file", type=str,
                       default="/media/liuyu/DataDrive/DASFAA-Table/data/wikitq/dev.jsonl",
                       help="WikiTQ data file")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of samples to evaluate")
    parser.add_argument("--output", type=str,
                       default="/media/liuyu/DataDrive/DASFAA-Table/results/wikitq_eval.json",
                       help="Output file for results")
    parser.add_argument("--verbose", action="store_true", default=False,
                       help="Verbose output")

    args = parser.parse_args()

    evaluate_wikitq(
        data_file=args.data_file,
        num_samples=args.num_samples,
        output_file=args.output,
        verbose=args.verbose
    )
