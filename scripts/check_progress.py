#!/usr/bin/env python3
"""
Simple progress monitor for DataBench evaluation
"""
import time
import subprocess
import os

LOG_FILE = "logs/databench_100_eval.log"
TOTAL_SAMPLES = 100

def check_progress():
    """Check current progress from log file."""

    if not os.path.exists(LOG_FILE):
        return None

    # Check if process is running
    try:
        result = subprocess.run(
            ["pgrep", "-f", "evaluate_databench.py --num_samples 100"],
            capture_output=True,
            text=True
        )
        if not result.stdout.strip():
            return {"status": "completed"}
    except:
        pass

    # Read log file
    with open(LOG_FILE, 'r') as f:
        content = f.read()

    # Count progress
    lines = content.split('\n')

    # Count samples with "Sample X/100" pattern
    completed = 0
    correct = 0
    exec_success = 0

    for line in lines:
        if 'Correctness: ✓ Correct' in line:
            correct += 1
        if 'Execution: ✓ Success' in line:
            exec_success += 1
        if '/100' in line and 'Sample' in line:
            # Extract sample number
            try:
                parts = line.split('Sample')[1].split('/')[0].strip()
                num = int(parts)
                if num > completed:
                    completed = num
            except:
                pass

    return {
        'status': 'running',
        'completed': completed,
        'correct': correct,
        'exec_success': exec_success,
        'accuracy': 100 * correct / completed if completed > 0 else 0,
        'exec_rate': 100 * exec_success / completed if completed > 0 else 0
    }

def main():
    print("=" * 60)
    print("DataBench 100-Sample Evaluation - Progress Monitor")
    print("=" * 60)

    progress = check_progress()

    if progress is None:
        print("✗ Log file not found")
        return

    if progress['status'] == 'completed':
        print("✓ Evaluation completed!")
        print("\nCheck results at: results/databench_100samples.json")
        return

    # Show progress
    completed = progress['completed']
    pct = 100 * completed / TOTAL_SAMPLES

    print(f"\nProgress: {completed}/{TOTAL_SAMPLES} ({pct:.1f}%)")
    print(f"Execution Success: {progress['exec_success']}/{completed} ({progress['exec_rate']:.1f}%)")
    print(f"Answer Correct: {progress['correct']}/{completed} ({progress['accuracy']:.1f}%)")

    # Progress bar
    bar_len = 40
    filled = int(bar_len * completed / TOTAL_SAMPLES)
    bar = '█' * filled + '░' * (bar_len - filled)
    print(f"\n[{bar}] {pct:.1f}%")

    # Estimate remaining time
    if completed > 5:
        # Assume ~3 seconds per sample average
        remaining_samples = TOTAL_SAMPLES - completed
        est_seconds = remaining_samples * 3
        est_minutes = est_seconds / 60
        print(f"\nEstimated time remaining: ~{est_minutes:.1f} minutes")

    print(f"\nTo monitor continuously: tail -f {LOG_FILE}")

if __name__ == "__main__":
    main()
