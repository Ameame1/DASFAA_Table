#!/usr/bin/env python3
"""
DataBench 100样本测试: Baseline vs AILS Zero-Shot
使用 Qwen2.5-Coder-7B-Instruct (代码专用模型)

根据AILS-NTUA论文,这是他们的主战场
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.system.table_qa_system import TableQASystem


def normalize_answer(answer):
    """Normalize answer for comparison."""
    if answer is None:
        return None
    answer_str = str(answer).strip().lower()
    return answer_str


def convert_to_native(obj):
    """Convert numpy types to native Python types."""
    import numpy as np
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_native(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(item) for item in obj]
    else:
        return obj


def evaluate_configuration(samples, config_name, model_name, use_ails):
    """Evaluate a single configuration."""
    print(f"\n{'=' * 70}")
    print(f"Configuration: {config_name}")
    print(f"Model: {model_name}")
    print(f"{'=' * 70}")

    qa_system = TableQASystem(
        model_name=model_name,
        max_iterations=3,
        use_ails_prompt=use_ails,
        few_shot_examples=None
    )

    results = []
    correct = 0
    execution_success = 0
    total_iterations = 0

    for i, sample in enumerate(tqdm(samples, desc=config_name), 1):
        question = sample['question']
        ground_truth = str(sample['answer'])

        # Convert table to DataFrame
        table_data = sample['table']
        try:
            df = pd.DataFrame(table_data['data'], columns=table_data['columns'])
        except Exception as e:
            results.append({
                'sample_id': i,
                'question': question,
                'ground_truth': ground_truth,
                'predicted': None,
                'correct': False,
                'execution_success': False,
                'error': f"Table load error: {str(e)}"
            })
            continue

        # Run QA
        try:
            result = qa_system.answer_question(df, question)
            predicted = result.get('answer')
            success = result.get('success', False)
            iterations = result.get('iterations_used', 1)

            total_iterations += iterations

            if success:
                execution_success += 1

            # Check correctness
            is_correct = False
            if predicted is not None and ground_truth is not None:
                pred_norm = normalize_answer(predicted)
                truth_norm = normalize_answer(ground_truth)
                is_correct = (pred_norm == truth_norm) or (pred_norm in truth_norm) or (truth_norm in pred_norm)

            if is_correct:
                correct += 1

            results.append({
                'sample_id': i,
                'question': question,
                'ground_truth': ground_truth,
                'predicted': str(predicted) if predicted is not None else None,
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
        'model': model_name,
        'execution_success': execution_success,
        'exec_rate': exec_rate,
        'correct': correct,
        'acc_rate': acc_rate,
        'avg_iterations': avg_iter,
        'results': results
    }


def main():
    num_samples = 100

    # 使用Coder模型!
    coder_model = "Qwen/Qwen2.5-Coder-7B-Instruct"

    print("=" * 70)
    print("DataBench 100-Sample: Baseline vs AILS (Qwen2.5-Coder-7B)")
    print("=" * 70)
    print(f"Model: {coder_model} (代码专用!)")
    print(f"Samples: {num_samples}")
    print(f"\nAILS-NTUA论文中:")
    print(f"  - Claude 3.5 Sonnet: ~85%")
    print(f"  - Qwen2.5-Coder-7B: 待测试")
    print(f"  - Baseline (估计): ~27%")
    print()

    # Load DataBench
    print("Loading DataBench dataset...")
    dataset = load_dataset("cardiffnlp/databench", "semeval", split="test")
    samples = list(dataset)[:num_samples]
    print(f"✓ Loaded {len(samples)} samples\n")

    configs = []

    # Test 1: Baseline (Coder模型)
    print("\n" + "=" * 70)
    print("TEST 1/2: Baseline (Qwen2.5-Coder-7B)")
    print("=" * 70)
    config1 = evaluate_configuration(
        samples,
        "Baseline (Coder)",
        coder_model,
        use_ails=False
    )
    configs.append(config1)

    # Test 2: AILS Zero-Shot (Coder模型)
    print("\n" + "=" * 70)
    print("TEST 2/2: AILS Zero-Shot (Qwen2.5-Coder-7B)")
    print("=" * 70)
    config2 = evaluate_configuration(
        samples,
        "AILS Zero-Shot (Coder)",
        coder_model,
        use_ails=True
    )
    configs.append(config2)

    # Summary
    print("\n" + "=" * 70)
    print("FINAL COMPARISON (DataBench 100 samples)")
    print("=" * 70)
    print(f"{'Configuration':<30} {'Exec Success':<15} {'Accuracy':<15} {'Avg Iter':<10}")
    print("-" * 70)
    for config in configs:
        print(f"{config['config']:<30} {config['exec_rate']:>6.1f}% ({config['execution_success']:>3}/{num_samples}) "
              f"{config['acc_rate']:>6.1f}% ({config['correct']:>3}/{num_samples})  {config['avg_iterations']:>6.2f}")

    # Calculate improvement
    baseline_acc = configs[0]['acc_rate']
    ails_acc = configs[1]['acc_rate']
    improvement = ails_acc - baseline_acc

    print("\n" + "=" * 70)
    print("IMPROVEMENT")
    print("=" * 70)
    print(f"AILS Zero-Shot vs Baseline: {improvement:+.1f}%")

    if improvement >= 10:
        print(f"✅ SIGNIFICANT IMPROVEMENT! (+{improvement:.1f}%)")
        print(f"   这符合AILS-NTUA论文的预期!")
    elif improvement >= 5:
        print(f"✓ Moderate improvement (+{improvement:.1f}%)")
    elif improvement > 0:
        print(f"⚠️ Small improvement (+{improvement:.1f}%)")
    else:
        print(f"⚠️ No improvement ({improvement:+.1f}%)")

    # Compare with previous WikiTQ Instruct results
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print("WikiTQ (Qwen2.5-7B-Instruct):")
    print("  Baseline: 33%, AILS: 33% → +0%")
    print("")
    print(f"DataBench (Qwen2.5-Coder-7B):")
    print(f"  Baseline: {baseline_acc:.1f}%, AILS: {ails_acc:.1f}% → {improvement:+.1f}%")

    # Save results
    output_file = "results/databench_100_coder_ails.json"
    os.makedirs("results", exist_ok=True)

    configs_serializable = convert_to_native(configs)

    with open(output_file, 'w') as f:
        json.dump(configs_serializable, f, indent=2)
    print(f"\n✓ Results saved to: {output_file}")


if __name__ == "__main__":
    main()
