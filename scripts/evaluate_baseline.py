"""
Evaluate system on WikiTQ dataset
"""

import sys
sys.path.append('.')

import json
from pathlib import Path
from src.system.table_qa_system import TableQASystem
from src.data.data_loader import load_dataset
from tqdm import tqdm


def normalize_answer(answer):
    """Normalize answer for comparison"""
    if answer is None:
        return ""
    answer = str(answer).lower().strip()
    # Remove quotes
    answer = answer.replace('"', '').replace("'", "")
    return answer


def evaluate_on_dataset(
    system,
    dataset_name='wikitq',
    split='dev',
    max_samples=100,
    output_file=None
):
    """Evaluate system on dataset"""

    print(f"\n{'='*60}")
    print(f"EVALUATING ON {dataset_name.upper()} - {split.upper()}")
    print(f"{'='*60}\n")

    # Load dataset
    dataset = load_dataset(dataset_name, split, max_samples=max_samples)
    print(f"Loaded {len(dataset)} samples\n")

    # Results
    results = []
    success_count = 0
    correct_count = 0
    total_iterations = 0

    # Evaluate
    for i, sample in enumerate(tqdm(dataset, desc="Evaluating")):
        try:
            result = system.answer_question(
                sample['table'],
                sample['question'],
                return_trajectory=False
            )

            # Normalize answers for comparison
            pred_answer = normalize_answer(result.get('answer'))
            gold_answer = normalize_answer(sample['answer'])

            # Check correctness (exact match or substring)
            is_correct = (
                pred_answer == gold_answer or
                pred_answer in gold_answer or
                gold_answer in pred_answer
            )

            if result['success']:
                success_count += 1
                total_iterations += result['iterations']

            if is_correct:
                correct_count += 1

            # Record result
            sample_result = {
                'id': sample['id'],
                'question': sample['question'],
                'gold_answer': sample['answer'],
                'pred_answer': result.get('answer'),
                'success': result['success'],
                'correct': is_correct,
                'iterations': result['iterations'],
                'final_code': result.get('final_code', '')
            }

            if not result['success'] and result.get('last_error'):
                sample_result['error_type'] = result['last_error'].get('error_type')

            results.append(sample_result)

            # Print progress every 20 samples
            if (i + 1) % 20 == 0:
                current_success = success_count / (i + 1) * 100
                current_correct = correct_count / (i + 1) * 100
                print(f"\n[{i+1}/{len(dataset)}] Success: {current_success:.1f}%, Correct: {current_correct:.1f}%")

        except Exception as e:
            print(f"\nError on sample {sample['id']}: {e}")
            results.append({
                'id': sample['id'],
                'question': sample['question'],
                'error': str(e),
                'success': False,
                'correct': False
            })

    # Summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_name} - {split}")
    print(f"Samples: {len(dataset)}")
    print(f"\nExecution Success Rate: {success_count}/{len(dataset)} ({success_count/len(dataset)*100:.2f}%)")
    print(f"Answer Correctness Rate: {correct_count}/{len(dataset)} ({correct_count/len(dataset)*100:.2f}%)")

    if success_count > 0:
        avg_iterations = total_iterations / success_count
        print(f"Average Iterations (success): {avg_iterations:.2f}")

    # Error analysis
    error_types = {}
    for r in results:
        if not r['success'] and 'error_type' in r:
            error_type = r['error_type']
            error_types[error_type] = error_types.get(error_type, 0) + 1

    if error_types:
        print(f"\nError Types:")
        for error_type, count in sorted(error_types.items(), key=lambda x: -x[1]):
            print(f"  {error_type}: {count}")

    # Save results
    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'dataset': dataset_name,
                'split': split,
                'total_samples': len(dataset),
                'success_count': success_count,
                'correct_count': correct_count,
                'success_rate': success_count / len(dataset),
                'correctness_rate': correct_count / len(dataset),
                'results': results
            }, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Results saved to {output_file}")

    return results


def main():
    """Run evaluation"""

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='wikitq', choices=['wikitq', 'tabfact', 'fetaqa'])
    parser.add_argument('--split', type=str, default='dev', choices=['train', 'dev', 'test'])
    parser.add_argument('--max_samples', type=int, default=100, help='Max samples to evaluate')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file')
    parser.add_argument('--max_iterations', type=int, default=3)
    args = parser.parse_args()

    # Initialize system
    print("Initializing system...")
    system = TableQASystem(
        model_name=args.model,
        use_grpo=False,
        max_iterations=args.max_iterations
    )
    print("✓ System initialized\n")

    # Set default output path
    if args.output is None:
        args.output = f"results/{args.dataset}_{args.split}_evaluation.json"

    # Evaluate
    results = evaluate_on_dataset(
        system,
        dataset_name=args.dataset,
        split=args.split,
        max_samples=args.max_samples,
        output_file=args.output
    )

    print(f"\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
