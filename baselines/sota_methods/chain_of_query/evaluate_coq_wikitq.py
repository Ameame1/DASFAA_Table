#!/usr/bin/env python3
"""
Evaluate Chain-of-Query (CoQ) on WikiTQ dataset

This script implements the CoQ approach (SQL generation) and compares it
with our Python code generation approach.
"""

import os
import sys
import json
import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm

# Add project root and CoQ module to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

from coq_sql_generator import ChainOfQueryGenerator
from coq_executor import CoQExecutor


def load_wikitq_sample(sample: dict) -> pd.DataFrame:
    """Convert WikiTQ sample to DataFrame"""
    table_data = sample['table']

    headers = table_data['header']
    rows = table_data['rows']

    # Clean headers
    headers = [h.strip('"') for h in headers]

    # Clean rows
    cleaned_rows = []
    for row in rows:
        cleaned_row = [cell.strip('"') if isinstance(cell, str) else cell for cell in row]
        cleaned_rows.append(cleaned_row)

    df = pd.DataFrame(cleaned_rows, columns=headers)

    return df


def normalize_answer(answer):
    """Normalize answer for comparison"""
    if answer is None:
        return None

    answer_str = str(answer).strip().lower()
    answer_str = answer_str.replace('_', ' ')

    return answer_str


def evaluate_coq_wikitq(
    data_file: str,
    num_samples: int = 100,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    use_clause_by_clause: bool = True,
    use_few_shot: bool = True,
    output_file: str = None,
    verbose: bool = True
):
    """
    Evaluate Chain-of-Query on WikiTQ

    Args:
        data_file: Path to WikiTQ data file
        num_samples: Number of samples to evaluate
        model_name: LLM model to use
        use_clause_by_clause: Whether to use clause-by-clause generation
        use_few_shot: Whether to use few-shot examples
        output_file: Path to save results
        verbose: Whether to print detailed output
    """

    print("=" * 70)
    print("Chain-of-Query (CoQ) Evaluation on WikiTQ")
    print("=" * 70)
    print(f"\nDataset: {data_file}")
    print(f"Samples: {num_samples}")
    print(f"Model: {model_name}")
    print(f"Clause-by-Clause: {use_clause_by_clause}")
    print(f"Few-shot: {use_few_shot}")
    print(f"\nExpected performance:")
    print(f"  - GPT-4: ~74.77% (CoQ paper)")
    print(f"  - Qwen 7B: ~40-50% (estimated)")
    print(f"  - Our Python approach: 25%\n")

    # Load data
    with open(data_file, 'r', encoding='utf-8') as f:
        samples = [json.loads(line) for line in f][:num_samples]

    # Initialize CoQ system
    print("Loading Chain-of-Query system...")
    generator = ChainOfQueryGenerator(
        model_name=model_name,
        use_few_shot=use_few_shot
    )
    coq_system = CoQExecutor(generator, max_retries=2)
    print("✓ System loaded\n")

    # Evaluate
    results = []
    correct = 0
    execution_success = 0
    invalid_sql = 0
    total_attempts = 0
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
                'sql': None,
                'correct': False,
                'execution_success': False,
                'attempts': 0,
                'error': f'Table load failed: {e}'
            })
            continue

        if verbose:
            print(f"Table shape: {table.shape}")
            print(f"Columns: {list(table.columns)[:5]}{'...' if len(table.columns) > 5 else ''}")

        # Run CoQ system
        try:
            result = coq_system.answer_question(question, table)

            predicted = result.get('answer')
            sql = result.get('sql')
            is_success = result['success']
            attempts = result['attempts']

            total_attempts += attempts

            # Normalize answers
            pred_norm = normalize_answer(predicted)
            gt_norm = normalize_answer(ground_truth)

            # Check correctness
            is_correct = (pred_norm == gt_norm) if pred_norm is not None else False

            if is_success:
                execution_success += 1
            else:
                # Check if it's invalid SQL
                error_msg = result.get('error', '')
                if 'Invalid SQL' in error_msg or 'Syntax' in error_msg:
                    invalid_sql += 1

            if is_correct:
                correct += 1

            if verbose:
                print(f"Generated SQL: {sql}")
                print(f"Predicted: {predicted}")
                print(f"Execution: {'✓ Success' if is_success else '✗ Failed'}")
                print(f"Attempts: {attempts}")
                print(f"Correctness: {'✓ Correct' if is_correct else '✗ Wrong'}")

            results.append({
                'sample_id': i,
                'question': question,
                'ground_truth': ground_truth,
                'predicted': predicted,
                'sql': sql,
                'correct': is_correct,
                'execution_success': is_success,
                'attempts': attempts,
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
                'sql': None,
                'correct': False,
                'execution_success': False,
                'attempts': 0,
                'error': str(e)
            })

    # Calculate metrics
    valid_samples = num_samples - skipped
    exec_rate = 100 * execution_success / valid_samples if valid_samples > 0 else 0
    accuracy = 100 * correct / valid_samples if valid_samples > 0 else 0
    invalid_sql_rate = 100 * invalid_sql / valid_samples if valid_samples > 0 else 0
    avg_attempts = total_attempts / valid_samples if valid_samples > 0 else 0

    # Print summary
    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)
    print(f"Total samples: {num_samples}")
    print(f"Skipped: {skipped}")
    print(f"Valid samples: {valid_samples}")
    print(f"\nExecution Success: {execution_success}/{valid_samples} ({exec_rate:.1f}%)")
    print(f"Invalid SQL Rate: {invalid_sql}/{valid_samples} ({invalid_sql_rate:.1f}%)")
    print(f"Answer Correctness: {correct}/{valid_samples} ({accuracy:.1f}%)")
    print(f"Average Attempts: {avg_attempts:.2f}")
    print(f"\n{'=' * 70}")
    print(f"vs Baseline (54%): {accuracy - 54:+.1f}%")
    print(f"vs Our Python (25%): {accuracy - 25:+.1f}%")
    print(f"vs CoQ Paper (74.77%): {accuracy - 74.77:+.1f}%")
    print("=" * 70)

    # Save results
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'summary': {
                    'total_samples': int(num_samples),
                    'skipped': int(skipped),
                    'valid_samples': int(valid_samples),
                    'execution_success': int(execution_success),
                    'execution_rate': float(exec_rate),
                    'invalid_sql': int(invalid_sql),
                    'invalid_sql_rate': float(invalid_sql_rate),
                    'correct': int(correct),
                    'accuracy': float(accuracy),
                    'average_attempts': float(avg_attempts),
                    'model': model_name,
                    'clause_by_clause': use_clause_by_clause,
                    'few_shot': use_few_shot
                },
                'results': results
            }, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Results saved to: {output_file}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CoQ on WikiTQ")
    parser.add_argument("--data_file", type=str,
                       default="/media/liuyu/DataDrive/DASFAA-Table/data/wikitq/dev.jsonl",
                       help="WikiTQ data file")
    parser.add_argument("--num_samples", type=int, default=10,
                       help="Number of samples to evaluate")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                       help="Model to use (Qwen/Qwen2.5-7B-Instruct, gpt-4, etc.)")
    parser.add_argument("--no_clause_by_clause", action="store_true",
                       help="Disable clause-by-clause generation")
    parser.add_argument("--no_few_shot", action="store_true",
                       help="Disable few-shot examples")
    parser.add_argument("--output", type=str,
                       default="/media/liuyu/DataDrive/DASFAA-Table/results/coq_wikitq_eval.json",
                       help="Output file for results")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")

    args = parser.parse_args()

    evaluate_coq_wikitq(
        data_file=args.data_file,
        num_samples=args.num_samples,
        model_name=args.model,
        use_clause_by_clause=not args.no_clause_by_clause,
        use_few_shot=not args.no_few_shot,
        output_file=args.output,
        verbose=args.verbose
    )
