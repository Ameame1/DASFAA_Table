"""
Simplified GPU test for real WikiTQ data with Qwen2.5-7B-Instruct
"""

import sys
sys.path.append('.')

from src.system.table_qa_system import TableQASystem
from src.data.data_loader import load_dataset

print("="*60)
print("SIMPLIFIED GPU TEST - REAL DATA")
print("="*60)

# Initialize system
print("\n1. Initializing system...")
system = TableQASystem(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    use_grpo=False,
    max_iterations=3
)
print("   ✓ System loaded")

# Load real data
print("\n2. Loading WikiTQ samples...")
dataset = load_dataset('wikitq', 'train', max_samples=5)
print(f"   ✓ Loaded {len(dataset)} samples")

# Test on samples
print("\n3. Testing on real data...")
print("="*60)

success_count = 0
for i, sample in enumerate(dataset):
    print(f"\nSample {i+1}/{len(dataset)}")
    print(f"Question: {sample['question']}")
    print(f"Table shape: {sample['table'].shape}")
    print(f"Gold answer: {sample['answer']}")

    try:
        result = system.answer_question(
            sample['table'],
            sample['question'],
            return_trajectory=False
        )

        if result['success']:
            success_count += 1
            print(f"✓ SUCCESS - Answer: {result['answer']}")
            print(f"  Iterations: {result['iterations']}")
        else:
            print(f"✗ FAILED after {result['iterations']} iterations")
            if result.get('last_error'):
                print(f"  Last error: {result['last_error'].get('error_type')}")

    except Exception as e:
        print(f"✗ ERROR: {e}")

print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
print(f"Successful: {success_count}/{len(dataset)}")
print(f"Success rate: {success_count/len(dataset)*100:.1f}%")

if success_count > 0:
    print("\n✓ System is working! Ready for full evaluation.")
else:
    print("\n⚠ No successful executions. Check model prompts.")
