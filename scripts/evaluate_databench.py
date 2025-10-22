#!/usr/bin/env python3
"""
Evaluate our Table QA system on DataBench dataset.

DataBench baseline: 26-27%
Our target: 60-70%
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


def load_databench_sample(sample: dict) -> tuple:
    """Load table data for a DataBench sample."""

    dataset_id = sample['dataset']

    # Try to load table from HuggingFace
    try:
        # Load sample (20 rows) - faster for testing
        df = pd.read_parquet(f"hf://datasets/cardiffnlp/databench/data/{dataset_id}/sample.parquet")
        return df, True
    except Exception as e:
        print(f"  ✗ Failed to load table for {dataset_id}: {e}")
        return None, False


def normalize_answer(answer):
    """Normalize answer for comparison."""
    if answer is None:
        return None

    # Convert to string
    answer_str = str(answer).strip()

    # Lowercase for comparison
    return answer_str.lower()


def evaluate_databench(
    data_file: str,
    num_samples: int = 5,
    output_file: str = None,
    verbose: bool = True
):
    """Evaluate on DataBench dataset."""

    print("=" * 70)
    print("DataBench Evaluation")
    print("=" * 70)
    print(f"\nDataset: {data_file}")
    print(f"Samples: {num_samples}")
    print(f"Baseline: 26-27%")
    print(f"Our Target: 60-70%\n")

    # Load data
    with open(data_file, 'r', encoding='utf-8') as f:
        samples = [json.loads(line) for line in f][:num_samples]

    # Initialize system (using Coder model for code generation!)
    print("Loading Table QA System...")

    # Zero-shot configuration (55% accuracy)
    # To enable Few-shot, uncomment the lines below
    # from src.baselines.ails_prompt_generator import AILS_FEWSHOT_EXAMPLES
    # few_shot_examples = AILS_FEWSHOT_EXAMPLES
    few_shot_examples = []  # Zero-shot (recommended, 55% accuracy)

    qa_system = TableQASystem(
        model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
        max_iterations=3,
        use_ails_prompt=True,  # Enable AILS prompting
        use_ails_postprocessor=True,  # Enable AILS post-processor (CRITICAL!)
        few_shot_examples=few_shot_examples
    )

    if few_shot_examples:
        print(f"✓ System loaded (Coder model + AILS prompt + postprocessor + {len(few_shot_examples)}-shot)\n")
    else:
        print("✓ System loaded (Coder model + AILS prompt + postprocessor + Zero-shot)\n")

    # Evaluate
    results = []
    correct = 0
    execution_success = 0
    total_iterations = 0
    skipped = 0

    print("Running evaluation...")
    for i, sample in enumerate(tqdm(samples, desc="Evaluating"), 1):
        question = sample['question']
        ground_truth = sample['sample_answer']  # Use sample_answer for Subtask II (20-row sample)
        dataset_id = sample['dataset']
        answer_type = sample['type']

        if verbose:
            print(f"\n{'=' * 70}")
            print(f"Sample {i}/{num_samples}")
            print(f"{'=' * 70}")
            print(f"Dataset: {dataset_id}")
            print(f"Question: {question}")
            print(f"Ground Truth: {ground_truth} (type: {answer_type})")

        # Load table
        table, loaded = load_databench_sample(sample)

        if not loaded:
            print(f"  ✗ Skipped (table load failed)")
            skipped += 1
            results.append({
                'sample_id': i,
                'question': question,
                'ground_truth': ground_truth,
                'predicted': None,
                'correct': False,
                'execution_success': False,
                'iterations': 0,
                'error': 'Table load failed',
                'dataset': dataset_id,
                'type': answer_type
            })
            continue

        if verbose:
            print(f"Table shape: {table.shape}")
            print(f"Columns: {list(table.columns)[:5]}{'...' if len(table.columns) > 5 else ''}")

        # Run QA system
        try:
            result = qa_system.answer_question(table, question)

            predicted = result['answer']
            is_success = result['success']  # Changed from 'execution_success'
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
                'error': result.get('error', None),
                'dataset': dataset_id,
                'type': answer_type,
                'trajectory': result.get('trajectory', [])
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
                'error': str(e),
                'dataset': dataset_id,
                'type': answer_type
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
    print(f"vs Baseline (26%): {accuracy - 26:+.1f}%")
    print(f"Target (60-70%): {'✓ ACHIEVED!' if accuracy >= 60 else f'Gap: {60 - accuracy:.1f}%'}")
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
    parser = argparse.ArgumentParser(description="Evaluate on DataBench")
    parser.add_argument("--data_file", type=str,
                       default="/media/liuyu/DataDrive/DASFAA-Table/data/databench/dev.jsonl",
                       help="DataBench data file")
    parser.add_argument("--num_samples", type=int, default=5,
                       help="Number of samples to evaluate")
    parser.add_argument("--output", type=str,
                       default="/media/liuyu/DataDrive/DASFAA-Table/results/databench_eval.json",
                       help="Output file for results")
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="Verbose output")

    args = parser.parse_args()

    evaluate_databench(
        data_file=args.data_file,
        num_samples=args.num_samples,
        output_file=args.output,
        verbose=args.verbose
    )
