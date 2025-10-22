"""
Complete Table QA System
Integrates all components: Code Generation, Execution, Diagnosis, Iteration
"""

import pandas as pd
from typing import Dict, Any, List, Optional
import logging

from ..execution.code_executor import SecureCodeExecutor
from ..diagnosis.diagnostic_system import HierarchicalDiagnosticSystem
from ..baselines.code_generator import QwenCodeGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TableQASystem:
    """
    Complete Table QA System with Iterative Error Correction

    Components:
    1. Code Generator (Qwen2.5-Coder-7B)
    2. Code Executor (Secure sandbox)
    3. Error Diagnoser (4-layer hierarchical)
    4. Iteration Controller (with dynamic stopping)
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        use_grpo: bool = False,
        grpo_model_path: Optional[str] = None,
        max_iterations: int = 3,
        timeout: int = 5,
        device: str = "cuda",
        load_in_8bit: bool = False,
        use_ails_prompt: bool = False,
        use_ails_postprocessor: bool = False,
        few_shot_examples: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Initialize Table QA system

        Args:
            model_name: Code generation model
            use_grpo: Whether to use GRPO for strategy selection
            grpo_model_path: Path to GRPO checkpoint
            max_iterations: Maximum repair iterations
            timeout: Code execution timeout
            device: Device for model
            load_in_8bit: Use 8-bit quantization
            use_ails_prompt: Whether to use AILS-NTUA style prompts
            use_ails_postprocessor: Whether to use AILS-NTUA post-processor
            few_shot_examples: Few-shot examples for AILS prompting
        """
        # Initialize components
        self.code_generator = QwenCodeGenerator(
            model_name,
            device,
            load_in_8bit,
            use_ails_prompt=use_ails_prompt,
            use_ails_postprocessor=use_ails_postprocessor,
            few_shot_examples=few_shot_examples
        )
        self.code_executor = SecureCodeExecutor(timeout=timeout)
        self.diagnostic_system = HierarchicalDiagnosticSystem(use_grpo, grpo_model_path)

        self.max_iterations = max_iterations

        logger.info("TableQASystem initialized")
        logger.info(f"Model: {model_name}")
        logger.info(f"Max iterations: {max_iterations}")
        logger.info(f"GRPO: {use_grpo}")

    def answer_question(
        self,
        table: pd.DataFrame,
        question: str,
        gold_answer: Optional[Any] = None,
        return_trajectory: bool = False
    ) -> Dict[str, Any]:
        """
        Answer a table question with iterative error correction

        Args:
            table: Input DataFrame
            question: Question to answer
            gold_answer: Gold answer for evaluation (optional)
            return_trajectory: Whether to return full execution trajectory

        Returns:
            Result dictionary:
            {
                'success': bool,
                'answer': Any,
                'iterations': int,
                'trajectory': List[Dict] (if return_trajectory=True),
                'final_code': str
            }
        """
        trajectory = []
        code = None

        for iteration in range(self.max_iterations):
            logger.info(f"Iteration {iteration + 1}/{self.max_iterations}")

            # Generate code
            if iteration == 0:
                # Initial generation
                code = self.code_generator.generate_code(table, question)
                logger.debug(f"Generated code:\n{code}")
            else:
                # Repair based on diagnosis
                diagnosis = self.diagnostic_system.diagnose(
                    exec_result, code, table, question
                )
                repair_prompt = diagnosis['repair_prompt']

                if repair_prompt is None:
                    logger.warning("No repair prompt generated, stopping")
                    break

                code = self.code_generator.generate_from_repair_prompt(repair_prompt)
                logger.debug(f"Repaired code:\n{code}")

            # Execute code
            exec_result = self.code_executor.execute(code, table)

            # Record trajectory
            step = {
                'iteration': iteration + 1,
                'code': code,
                'success': exec_result['success'],
                'answer': exec_result.get('answer'),
                'error': exec_result.get('error'),
                'error_type': exec_result.get('error_type')
            }
            trajectory.append(step)

            # Check if successful
            if exec_result['success']:
                logger.info(f"✓ Success at iteration {iteration + 1}")

                result = {
                    'success': True,
                    'answer': exec_result['answer'],
                    'iterations': iteration + 1,
                    'final_code': code
                }

                if return_trajectory:
                    result['trajectory'] = trajectory

                return result

            logger.warning(f"✗ Failed: {exec_result['error_type']}: {exec_result['error']}")

        # All iterations exhausted
        logger.error(f"Failed after {self.max_iterations} iterations")

        result = {
            'success': False,
            'answer': None,
            'iterations': self.max_iterations,
            'final_code': code,
            'last_error': trajectory[-1] if trajectory else None
        }

        if return_trajectory:
            result['trajectory'] = trajectory

        return result

    def batch_answer(
        self,
        samples: List[Dict[str, Any]],
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Answer multiple questions in batch

        Args:
            samples: List of samples with 'table' and 'question'
            show_progress: Whether to show progress bar

        Returns:
            List of results
        """
        results = []

        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(samples, desc="Answering questions")
            except ImportError:
                iterator = samples
        else:
            iterator = samples

        for sample in iterator:
            result = self.answer_question(
                sample['table'],
                sample['question'],
                sample.get('answer'),
                return_trajectory=False
            )
            results.append(result)

        return results


if __name__ == "__main__":
    print("✓ TableQASystem defined!")
    print("Note: Requires GPU and Qwen2.5-Coder model download")
