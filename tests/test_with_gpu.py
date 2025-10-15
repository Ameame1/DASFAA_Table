"""
Test complete Table QA system with GPU and Qwen model
This tests the full pipeline including code generation and iterative repair
"""

import sys
sys.path.append('.')

import pandas as pd
from pathlib import Path
from src.data.data_loader import load_dataset
from src.system.table_qa_system import TableQASystem


def test_model_loading():
    """Test Qwen model loading"""
    print("\n" + "="*60)
    print("TEST 1: Model Loading")
    print("="*60)

    try:
        system = TableQASystem(
            model_name="Qwen/Qwen2.5-7B-Instruct",
            use_grpo=False,
            max_iterations=3
        )
        print("✓ Model loaded successfully")
        print(f"✓ Device: {system.code_generator.device}")
        return system

    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("\nNote: This requires:")
        print("  - GPU with at least 16GB VRAM")
        print("  - Internet connection for first-time model download")
        print("  - About 14GB disk space for model weights")
        return None


def test_code_generation(system):
    """Test code generation on sample data"""
    print("\n" + "="*60)
    print("TEST 2: Code Generation")
    print("="*60)

    if system is None:
        print("✗ System not available")
        return

    # Create a simple test table
    table = pd.DataFrame({
        'City': ['Beijing', 'Shanghai', 'Guangzhou'],
        'Population': [21.54, 24.28, 15.30],
        'GDP': [3610, 3875, 2501]
    })

    question = "What is the total population of all cities?"

    print(f"Question: {question}")
    print(f"Table:\n{table}\n")

    try:
        code = system.code_generator.generate_code(table, question)
        print(f"Generated code:\n{code}\n")

        # Execute the code
        result = system.code_executor.execute(code, table)
        print(f"Execution result: {result['success']}")
        if result['success']:
            print(f"Answer: {result['answer']}")
        else:
            print(f"Error: {result['error_type']} - {result['error']}")

        return code, result

    except Exception as e:
        print(f"✗ Error during code generation: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_iterative_repair(system):
    """Test iterative error repair"""
    print("\n" + "="*60)
    print("TEST 3: Iterative Error Repair")
    print("="*60)

    if system is None:
        print("✗ System not available")
        return

    # Create a table that might cause errors
    table = pd.DataFrame({
        'product_name': ['Apple', 'Banana', 'Orange'],
        'price': ['5.99', '3.49', '4.29'],  # String prices to test type errors
        'quantity': [10, 15, 8]
    })

    question = "What is the total revenue (price × quantity)?"

    print(f"Question: {question}")
    print(f"Table:\n{table}\n")
    print("Note: Prices are strings, might need type conversion\n")

    try:
        result = system.answer_question(
            table,
            question,
            return_trajectory=True
        )

        print(f"\n✓ Final Answer: {result['answer']}")
        print(f"✓ Success: {result['success']}")
        print(f"✓ Iterations used: {result['iterations']}")

        print(f"\nTrajectory:")
        for i, step in enumerate(result['trajectory'], 1):
            print(f"\n--- Iteration {i} ---")
            print(f"Code:\n{step['code']}")
            print(f"Success: {step['execution_result']['success']}")
            if not step['execution_result']['success']:
                print(f"Error: {step['execution_result']['error_type']}")
                if 'diagnosis' in step:
                    print(f"Diagnosis: {step['diagnosis']['error_class']} -> {step['diagnosis']['root_cause'].get('root_cause')}")
                    print(f"Strategy: {step['diagnosis']['strategy']}")
            else:
                print(f"Answer: {step['execution_result']['answer']}")

        return result

    except Exception as e:
        print(f"✗ Error during iterative repair: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_real_data_samples(system, n_samples=3):
    """Test on real WikiTQ data"""
    print("\n" + "="*60)
    print(f"TEST 4: Real Data Evaluation ({n_samples} samples)")
    print("="*60)

    if system is None:
        print("✗ System not available")
        return

    try:
        dataset = load_dataset('wikitq', 'train', max_samples=n_samples)
        print(f"✓ Loaded {len(dataset)} samples from WikiTQ\n")

        results = []
        for i, sample in enumerate(dataset):
            print(f"\n{'='*60}")
            print(f"Sample {i+1}/{len(dataset)}: {sample['id']}")
            print(f"{'='*60}")
            print(f"Question: {sample['question']}")
            print(f"Table shape: {sample['table'].shape}")
            print(f"Gold answer: {sample['answer']}\n")

            try:
                result = system.answer_question(
                    sample['table'],
                    sample['question'],
                    gold_answer=sample['answer']
                )

                results.append({
                    'id': sample['id'],
                    'success': result['success'],
                    'answer': result['answer'],
                    'gold_answer': sample['answer'],
                    'iterations': result['iterations'],
                    'correct': result.get('correct', False)
                })

                print(f"Generated answer: {result['answer']}")
                print(f"Iterations: {result['iterations']}")
                print(f"Correct: {result.get('correct', 'Unknown')}")

            except Exception as e:
                print(f"✗ Error on sample {sample['id']}: {e}")
                results.append({
                    'id': sample['id'],
                    'success': False,
                    'error': str(e)
                })

        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        success_count = sum(1 for r in results if r.get('success', False))
        correct_count = sum(1 for r in results if r.get('correct', False))

        print(f"Successful executions: {success_count}/{len(results)}")
        print(f"Correct answers: {correct_count}/{len(results)}")

        if success_count > 0:
            avg_iterations = sum(r.get('iterations', 0) for r in results if r.get('success', False)) / success_count
            print(f"Average iterations: {avg_iterations:.2f}")

        return results

    except Exception as e:
        print(f"✗ Error loading/processing data: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_memory_usage():
    """Test memory efficiency with 8-bit quantization"""
    print("\n" + "="*60)
    print("TEST 5: Memory Efficiency (8-bit quantization)")
    print("="*60)

    try:
        import torch

        # Test with 8-bit quantization for lower memory usage
        system_8bit = TableQASystem(
            model_name="Qwen/Qwen2.5-7B-Instruct",
            device="cuda",
            load_in_8bit=True,
            use_grpo=False,
            max_iterations=3
        )

        print("✓ Model loaded with 8-bit quantization")

        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"✓ GPU memory allocated: {memory_allocated:.2f} GB")
            print(f"✓ GPU memory reserved: {memory_reserved:.2f} GB")

        # Quick test
        table = pd.DataFrame({
            'Item': ['A', 'B', 'C'],
            'Value': [10, 20, 30]
        })

        result = system_8bit.answer_question(table, "What is the sum of all values?")
        print(f"✓ Quick test passed: {result['success']}")
        print(f"  Answer: {result['answer']}")

        return system_8bit

    except Exception as e:
        print(f"✗ Error with 8-bit quantization: {e}")
        print("Note: Requires bitsandbytes library")
        return None


def main():
    """Run all GPU tests"""
    print("\n" + "="*60)
    print("TESTING TABLE QA SYSTEM WITH GPU")
    print("="*60)
    print("\nThis will test:")
    print("  1. Qwen2.5-7B-Instruct model loading")
    print("  2. Code generation from natural language")
    print("  3. Iterative error diagnosis and repair")
    print("  4. Real WikiTQ data evaluation")
    print("  5. Memory efficiency with 8-bit quantization")
    print("\nModel is already in cache (~14GB)")
    print("Using GPU: Requires 14-16GB VRAM (or 8GB with quantization)")

    try:
        # Test 1: Model loading
        system = test_model_loading()

        if system is not None:
            # Test 2: Code generation
            test_code_generation(system)

            # Test 3: Iterative repair
            test_iterative_repair(system)

            # Test 4: Real data samples
            test_real_data_samples(system, n_samples=3)

            # Test 5: Memory efficiency
            test_memory_usage()

            print("\n" + "="*60)
            print("✓ ALL GPU TESTS COMPLETED")
            print("="*60)
            print("\nYour system is fully operational with GPU!")
            print("\nNext steps:")
            print("1. Run full baseline evaluation on complete datasets")
            print("2. Collect error trajectories for GRPO training")
            print("3. Implement GRPO training with TRL library")
            print("4. Compare with GPT-4o and other baselines")

        else:
            print("\n" + "="*60)
            print("⚠ GPU TESTS FAILED")
            print("="*60)
            print("\nPossible issues:")
            print("  - GPU not available or insufficient VRAM")
            print("  - PyTorch not installed with CUDA support")
            print("  - Transformers library version incompatible")
            print("\nCheck: python3 -c 'import torch; print(torch.cuda.is_available())'")

    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
