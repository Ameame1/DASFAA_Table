#!/usr/bin/env python3
"""
Monitor all three dataset evaluations
"""

import os
import subprocess
from pathlib import Path

def check_process(name_pattern):
    """Check if process is running."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", name_pattern],
            capture_output=True,
            text=True
        )
        return bool(result.stdout.strip())
    except:
        return False

def count_progress(log_file, total=100):
    """Count progress from log file."""
    if not os.path.exists(log_file):
        return None

    with open(log_file, 'r') as f:
        content = f.read()

    # Count completed samples
    lines = content.split('\n')
    completed = 0
    correct = 0
    exec_success = 0

    for line in lines:
        if 'Correctness: ✓ Correct' in line:
            correct += 1
        if 'Execution: ✓ Success' in line:
            exec_success += 1
        if f'/{total}' in line and 'Sample' in line:
            try:
                parts = line.split('Sample')[1].split('/')[0].strip()
                num = int(parts)
                if num > completed:
                    completed = num
            except:
                pass

    return {
        'completed': completed,
        'correct': correct,
        'exec_success': exec_success,
        'accuracy': 100 * correct / completed if completed > 0 else 0,
        'exec_rate': 100 * exec_success / completed if completed > 0 else 0
    }

def main():
    print("=" * 80)
    print("Multi-Dataset Evaluation Monitor")
    print("=" * 80)
    print()

    datasets = [
        {
            'name': 'DataBench',
            'log': 'logs/databench_100_eval.log',
            'pattern': 'evaluate_databench.py --num_samples 100',
            'baseline': 27,
            'target': '60-70%'
        },
        {
            'name': 'WikiTQ',
            'log': 'logs/wikitq_100_eval.log',
            'pattern': 'evaluate_wikitq.py',
            'baseline': 54,
            'target': '60-65%'
        },
        {
            'name': 'TabFact',
            'log': 'logs/tabfact_100_eval.log',
            'pattern': 'evaluate_tabfact.py',
            'baseline': 78,
            'target': '82-85%'
        }
    ]

    for ds in datasets:
        print(f"{'=' * 80}")
        print(f"{ds['name']} (Baseline: {ds['baseline']}%, Target: {ds['target']})")
        print(f"{'=' * 80}")

        # Check if running
        is_running = check_process(ds['pattern'])

        if not is_running:
            if os.path.exists(ds['log']):
                print("Status: ✓ Completed")
            else:
                print("Status: ✗ Not started")
        else:
            print("Status: ⏳ Running...")

        # Check progress
        progress = count_progress(ds['log'])

        if progress:
            completed = progress['completed']
            pct = completed
            accuracy = progress['accuracy']
            exec_rate = progress['exec_rate']

            print(f"Progress: {completed}/100 ({pct}%)")
            print(f"Execution: {progress['exec_success']}/{completed} ({exec_rate:.1f}%)")
            print(f"Accuracy: {progress['correct']}/{completed} ({accuracy:.1f}%)")

            # Progress bar
            bar_len = 40
            filled = int(bar_len * completed / 100)
            bar = '█' * filled + '░' * (bar_len - filled)
            print(f"[{bar}] {pct}%")

            if completed == 100:
                vs_baseline = accuracy - ds['baseline']
                print(f"\n✓ FINAL RESULT: {accuracy:.1f}% ({vs_baseline:+.1f}% vs baseline)")
        else:
            print("Progress: Not available yet")

        print()

    print("=" * 80)
    print("To view detailed logs:")
    print("  DataBench: tail -f logs/databench_100_eval.log")
    print("  WikiTQ:    tail -f logs/wikitq_100_eval.log")
    print("  TabFact:   tail -f logs/tabfact_100_eval.log")
    print("=" * 80)

if __name__ == "__main__":
    main()
