#!/usr/bin/env python3
"""
Evaluate WikiTQ with AILS-NTUA prompts (10 samples for quick testing)

Tests three configurations:
1. Baseline (current method)
2. AILS Zero-Shot
3. AILS Few-Shot (with WikiTQ examples)
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.system.table_qa_system import TableQASystem
from examples.wikitq_fewshot_examples import WIKITQ_EXAMPLES


def load_wikitq_sample(sample: dict) -> pd.DataFrame:
    """Convert WikiTQ sample to DataFrame."""
    table_data = sample['table']
    headers = table_data['header']
    rows = table_data['rows']

    # Clean headers and rows
    headers = [h.strip('"') for h in headers]
    cleaned_rows = []
    for row in rows:
        cleaned_row = [cell.strip('"') if isinstance(cell, str) else cell for cell in row]
        cleaned_rows.append(cleaned_row)

    df = pd.DataFrame(cleaned_rows, columns=headers)
    return df


def normalize_answer(answer):
    """Normalize answer for comparison."""
    if answer is None:
        return None
    answer_str = str(answer).strip().lower()
    answer_str = answer_str.replace('_', ' ')
    return answer_str


def evaluate_configuration(samples, config_name, use_ails, few_shot_examples=None):
    """Evaluate a single configuration."""
    print(f"\n{'=' * 70}")
    print(f"Configuration: {config_name}")
    print(f"{'=' * 70}")

    # Initialize system with appropriate settings
    qa_system = TableQASystem(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        max_iterations=3,
        use_ails_prompt=use_ails,
        few_shot_examples=few_shot_examples
    )

    results = []
    correct = 0
    execution_success = 0
    total_iterations = 0

    for i, sample in enumerate(tqdm(samples, desc=config_name), 1):
        question = sample['question']
        ground_truth = sample['answer']

        try:
            table = load_wikitq_sample(sample)
        except Exception as e:
            results.append({
                'sample_id': i,
                'question': question,
                'ground_truth': ground_truth,
                'predicted': None,
                'correct': False,
                'execution_success': False,
                'error': str(e)
            })
            continue

        # Run QA
        try:
            result = qa_system.answer_question(table, question)
            predicted = result.get('answer')
            success = result.get('success', False)
            iterations = result.get('iterations_used', 0)

            total_iterations += iterations

            if success:
                execution_success += 1

            # Check correctness
            is_correct = False
            if predicted is not None and ground_truth is not None:
                pred_norm = normalize_answer(predicted)
                truth_norm = normalize_answer(ground_truth)

                # Check exact match or substring match
                is_correct = (pred_norm == truth_norm) or (pred_norm in truth_norm) or (truth_norm in pred_norm)

            if is_correct:
                correct += 1

            results.append({
                'sample_id': i,
                'question': question,
                'ground_truth': ground_truth,
                'predicted': predicted,
                'correct': is_correct,
                'execution_success': success,
                'iterations': iterations
            })

        except Exception as e:
            results.append({
                'sample_id': i,
                'question': question,
                'ground_truth': ground_truth,
                'predicted': None,
                'correct': False,
                'execution_success': False,
                'error': str(e)
            })

    # Calculate metrics
    total = len(samples)
    exec_rate = (execution_success / total) * 100
    acc_rate = (correct / total) * 100
    avg_iter = total_iterations / total if total > 0 else 0

    print(f"\n{config_name} Results:")
    print(f"  Execution Success: {execution_success}/{total} ({exec_rate:.1f}%)")
    print(f"  Answer Correct: {correct}/{total} ({acc_rate:.1f}%)")
    print(f"  Avg Iterations: {avg_iter:.2f}")

    return {
        'config': config_name,
        'execution_success': execution_success,
        'exec_rate': exec_rate,
        'correct': correct,
        'acc_rate': acc_rate,
        'avg_iterations': avg_iter,
        'results': results
    }


def main():
    data_file = "data/wikitq/dev.jsonl"
    num_samples = 10

    print("=" * 70)
    print("WikiTQ AILS-NTUA Comparison (10 samples)")
    print("=" * 70)
    print(f"Dataset: {data_file}")
    print(f"Samples: {num_samples}")
    print(f"Baseline (previous): 25% on 100 samples")
    print(f"Target: 35-40%\n")

    # Load samples
    with open(data_file, 'r', encoding='utf-8') as f:
        samples = [json.loads(line) for line in f][:num_samples]

    # Test 3 configurations
    configs = []

    # 1. Baseline
    print("\n" + "=" * 70)
    print("TEST 1/3: Baseline (current method)")
    print("=" * 70)
    config1 = evaluate_configuration(samples, "Baseline", use_ails=False)
    configs.append(config1)

    # 2. AILS Zero-Shot
    print("\n" + "=" * 70)
    print("TEST 2/3: AILS Zero-Shot")
    print("=" * 70)
    config2 = evaluate_configuration(samples, "AILS Zero-Shot", use_ails=True)
    configs.append(config2)

    # 3. AILS Few-Shot
    print("\n" + "=" * 70)
    print("TEST 3/3: AILS Few-Shot (3 examples)")
    print("=" * 70)
    config3 = evaluate_configuration(
        samples,
        "AILS Few-Shot",
        use_ails=True,
        few_shot_examples=WIKITQ_EXAMPLES
    )
    configs.append(config3)

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)
    print(f"{'Configuration':<20} {'Exec Success':<15} {'Accuracy':<15} {'Avg Iter':<10}")
    print("-" * 70)
    for config in configs:
        print(f"{config['config']:<20} {config['exec_rate']:>6.1f}% ({config['execution_success']:>2}/{num_samples}) "
              f"{config['acc_rate']:>6.1f}% ({config['correct']:>2}/{num_samples})  {config['avg_iterations']:>6.2f}")

    # Calculate improvements
    baseline_acc = configs[0]['acc_rate']
    zero_shot_acc = configs[1]['acc_rate']
    few_shot_acc = configs[2]['acc_rate']

    print("\n" + "=" * 70)
    print("IMPROVEMENTS vs Baseline")
    print("=" * 70)
    print(f"AILS Zero-Shot: {zero_shot_acc - baseline_acc:+.1f}%")
    print(f"AILS Few-Shot:  {few_shot_acc - baseline_acc:+.1f}%")

    # Save results
    output_file = "results/wikitq_10_ails_comparison.json"
    os.makedirs("results", exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(configs, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
