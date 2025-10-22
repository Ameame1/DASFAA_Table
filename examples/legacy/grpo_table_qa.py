"""
GRPO-Enhanced Iterative Table QA System
Author: [Your Name]
Date: 2025

This implementation combines:
1. Iterative error correction for table QA
2. Group Relative Policy Optimization (GRPO)
3. Multi-component reward function
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import traceback
import logging
from collections import Counter, defaultdict

# ============================================================================
# 1. Data Structures
# ============================================================================

class ErrorType(Enum):
    SYNTAX = "syntax"
    RUNTIME = "runtime"
    LOGIC = "logic"
    TIMEOUT = "timeout"
    NONE = "none"


@dataclass
class ExecutionResult:
    """Execution result container"""
    success: bool
    answer: Optional[str]
    error: Optional['ErrorInfo']
    execution_time: float
    code: str


@dataclass
class ErrorInfo:
    """Error information"""
    error_type: ErrorType
    message: str
    traceback: str
    line_number: Optional[int] = None


@dataclass
class Trajectory:
    """RL trajectory for GRPO training"""
    state: Dict  # Table + Question
    action: str  # Generated code
    reward: float
    log_prob: float
    success: bool
    num_iterations: int
    answer: Optional[str]
    gold_answer: str


# ============================================================================
# 2. Error Classifier
# ============================================================================

class ErrorClassifier:
    """Classify execution errors into categories"""

    def __init__(self):
        self.syntax_keywords = ["SyntaxError", "IndentationError", "TabError"]
        self.runtime_keywords = [
            "TypeError", "ValueError", "KeyError", "IndexError",
            "AttributeError", "NameError", "ZeroDivisionError"
        ]

    def classify(self, exception: Exception) -> ErrorType:
        """Classify error type"""
        error_name = type(exception).__name__

        if error_name in self.syntax_keywords:
            return ErrorType.SYNTAX
        elif error_name in self.runtime_keywords:
            return ErrorType.RUNTIME
        elif error_name == "TimeoutError":
            return ErrorType.TIMEOUT
        else:
            return ErrorType.RUNTIME

    def extract_error_info(self, exception: Exception, code: str) -> ErrorInfo:
        """Extract detailed error information"""
        error_type = self.classify(exception)
        tb = traceback.format_exc()

        # Try to extract line number
        line_number = None
        try:
            tb_lines = tb.split('\n')
            for line in tb_lines:
                if 'line' in line.lower():
                    parts = line.split('line')
                    if len(parts) > 1:
                        line_number = int(''.join(filter(str.isdigit, parts[1].split(',')[0])))
                        break
        except:
            pass

        return ErrorInfo(
            error_type=error_type,
            message=str(exception),
            traceback=tb,
            line_number=line_number
        )


# ============================================================================
# 3. Execution Engine
# ============================================================================

class ExecutionEngine:
    """Safe code execution with timeout"""

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.error_classifier = ErrorClassifier()

    def execute(self, code: str, table: pd.DataFrame) -> ExecutionResult:
        """Execute generated code safely"""
        import time
        import signal

        start_time = time.time()

        # Timeout handler
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Execution exceeded {self.timeout} seconds")

        # Set up timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self.timeout)

        try:
            # Create safe namespace
            namespace = {
                'pd': pd,
                'np': np,
                'table': table.copy(),
                'df': table.copy(),  # Alias
            }

            # Execute code
            exec(code, namespace)

            # Extract answer
            answer = namespace.get('answer', namespace.get('result', None))

            # Cancel timeout
            signal.alarm(0)

            execution_time = time.time() - start_time

            return ExecutionResult(
                success=True,
                answer=str(answer) if answer is not None else None,
                error=None,
                execution_time=execution_time,
                code=code
            )

        except Exception as e:
            signal.alarm(0)  # Cancel timeout
            execution_time = time.time() - start_time

            error_info = self.error_classifier.extract_error_info(e, code)

            return ExecutionResult(
                success=False,
                answer=None,
                error=error_info,
                execution_time=execution_time,
                code=code
            )


# ============================================================================
# 4. Code Generator (LLM Interface)
# ============================================================================

class CodeGenerator:
    """Generate and refine Python code for table QA"""

    def __init__(self, model_name: str = "gpt-4", max_iterations: int = 3):
        self.model_name = model_name
        self.max_iterations = max_iterations
        # Initialize LLM here (use OpenAI API, HuggingFace, etc.)
        self.model = self._load_model()

    def _load_model(self):
        """Load LLM model"""
        # TODO: Implement model loading
        # Example: return OpenAI(model=self.model_name)
        pass

    def generate_initial_code(self, table: pd.DataFrame, question: str) -> str:
        """Generate initial code"""
        prompt = self._build_initial_prompt(table, question)
        code = self.model.generate(prompt)
        return self._extract_code(code)

    def refine_code(
        self,
        table: pd.DataFrame,
        question: str,
        previous_code: str,
        error_info: ErrorInfo,
        iteration: int
    ) -> str:
        """Refine code based on error"""
        prompt = self._build_refinement_prompt(
            table, question, previous_code, error_info, iteration
        )
        code = self.model.generate(prompt)
        return self._extract_code(code)

    def _build_initial_prompt(self, table: pd.DataFrame, question: str) -> str:
        """Build initial code generation prompt"""
        table_markdown = table.head(10).to_markdown(index=False)

        prompt = f"""You are a table reasoning expert. Generate executable Python code to answer the question.

Table (showing first 10 rows):
{table_markdown}

Table info:
- Shape: {table.shape}
- Columns: {list(table.columns)}

Question: {question}

Requirements:
1. Use pandas DataFrame operations
2. The table is available as variable 'table' or 'df'
3. Store the final answer in variable 'answer'
4. Handle edge cases (empty results, type conversions)
5. Add brief comments explaining your logic

Generate Python code:
```python
"""
        return prompt

    def _build_refinement_prompt(
        self,
        table: pd.DataFrame,
        question: str,
        previous_code: str,
        error_info: ErrorInfo,
        iteration: int
    ) -> str:
        """Build error correction prompt"""
        table_markdown = table.head(5).to_markdown(index=False)

        # Error-specific hints
        hints = self._get_error_hints(error_info)

        prompt = f"""The previous code (Iteration {iteration-1}) failed. Please fix it.

Table (first 5 rows):
{table_markdown}

Question: {question}

Previous Code:
```python
{previous_code}
```

Error Information:
- Type: {error_info.error_type.value}
- Message: {error_info.message}
{f"- Line: {error_info.line_number}" if error_info.line_number else ""}

Debugging Hints:
{hints}

Please generate corrected code that fixes the error while maintaining the original logic.

Corrected Code:
```python
"""
        return prompt

    def _get_error_hints(self, error_info: ErrorInfo) -> str:
        """Get error-specific debugging hints"""
        hints = {
            ErrorType.SYNTAX: """
- Check indentation and syntax
- Ensure all brackets/parentheses are closed
- Check for missing colons after if/for/def statements
""",
            ErrorType.RUNTIME: """
- Verify column names exist in the table
- Check data types before operations
- Handle empty DataFrames or None values
- Use .fillna() for missing values
""",
            ErrorType.TIMEOUT: """
- Simplify complex operations
- Avoid nested loops on large datasets
- Use vectorized pandas operations
""",
        }
        return hints.get(error_info.error_type, "- Review the error message carefully")

    def _extract_code(self, response: str) -> str:
        """Extract Python code from LLM response"""
        # Remove markdown code blocks
        if '```python' in response:
            code = response.split('```python')[1].split('```')[0]
        elif '```' in response:
            code = response.split('```')[1].split('```')[0]
        else:
            code = response

        return code.strip()


# ============================================================================
# 5. Reward Function
# ============================================================================

class RewardFunction:
    """Multi-component reward function for GRPO"""

    def __init__(
        self,
        weights: Dict[str, float] = None
    ):
        self.weights = weights or {
            'execution': 0.4,
            'accuracy': 0.4,
            'efficiency': 0.1,
            'quality': 0.1
        }

    def compute_reward(
        self,
        execution_result: ExecutionResult,
        gold_answer: str,
        num_iterations: int,
        question: str
    ) -> float:
        """Compute total reward"""

        # Component 1: Execution Success
        r_exec = 1.0 if execution_result.success else -0.5

        # Component 2: Answer Accuracy
        r_accuracy = 0.0
        if execution_result.success and execution_result.answer:
            r_accuracy = self._compute_accuracy(
                execution_result.answer, gold_answer, question
            )

        # Component 3: Efficiency (fewer iterations better)
        r_efficiency = -0.1 * (num_iterations - 1)

        # Component 4: Code Quality
        r_quality = self._evaluate_code_quality(execution_result.code)

        # Weighted sum
        total_reward = (
            self.weights['execution'] * r_exec +
            self.weights['accuracy'] * r_accuracy +
            self.weights['efficiency'] * r_efficiency +
            self.weights['quality'] * r_quality
        )

        return total_reward

    def _compute_accuracy(self, pred: str, gold: str, question: str) -> float:
        """Compute answer accuracy"""
        # Normalize answers
        pred_norm = self._normalize_answer(pred)
        gold_norm = self._normalize_answer(gold)

        # Exact match
        if pred_norm == gold_norm:
            return 1.0

        # Partial match for list-type answers
        if ',' in gold_norm or ';' in gold_norm:
            pred_items = set(pred_norm.replace(';', ',').split(','))
            gold_items = set(gold_norm.replace(';', ',').split(','))

            if not gold_items:
                return 0.0

            overlap = len(pred_items & gold_items)
            return overlap / len(gold_items)

        # Numeric match with tolerance
        try:
            pred_num = float(pred_norm)
            gold_num = float(gold_norm)
            if abs(pred_num - gold_num) < 0.01:
                return 1.0
            elif abs(pred_num - gold_num) / abs(gold_num) < 0.05:
                return 0.8
        except:
            pass

        return 0.0

    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison"""
        if not answer:
            return ""

        answer = str(answer).lower().strip()
        # Remove common punctuation
        answer = answer.replace('.', '').replace('!', '').replace('?', '')
        # Remove extra whitespace
        answer = ' '.join(answer.split())

        return answer

    def _evaluate_code_quality(self, code: str) -> float:
        """Evaluate code quality (simple heuristics)"""
        score = 0.0

        # Has comments
        if '#' in code:
            score += 0.3

        # Uses pandas efficiently (no loops)
        if 'for ' not in code and 'while ' not in code:
            score += 0.3

        # Reasonable length (not too long)
        num_lines = len(code.split('\n'))
        if 5 <= num_lines <= 30:
            score += 0.2

        # Handles edge cases
        if any(keyword in code.lower() for keyword in ['fillna', 'dropna', 'isnull', 'empty']):
            score += 0.2

        return min(score, 1.0)


# ============================================================================
# 6. GRPO Trainer
# ============================================================================

class GRPOTrainer:
    """Group Relative Policy Optimization Trainer"""

    def __init__(
        self,
        policy_model,
        reference_model,
        learning_rate: float = 1e-6,
        clip_range: float = 0.2,
        kl_coef: float = 0.01,
        group_size: int = 4
    ):
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.clip_range = clip_range
        self.kl_coef = kl_coef
        self.group_size = group_size

        self.optimizer = AdamW(policy_model.parameters(), lr=learning_rate)
        self.reward_function = RewardFunction()

        # Metrics tracking
        self.training_stats = defaultdict(list)

    def compute_group_advantages(
        self,
        rewards: List[float],
        group_size: Optional[int] = None
    ) -> List[float]:
        """
        GRPO: Compute advantages using group-based baseline
        Instead of a learned value function, use group mean as baseline
        """
        if group_size is None:
            group_size = self.group_size

        advantages = []
        for i in range(0, len(rewards), group_size):
            group_rewards = rewards[i:i + group_size]
            group_mean = np.mean(group_rewards)
            group_std = np.std(group_rewards) + 1e-8

            # Normalize advantages
            group_advantages = [(r - group_mean) / group_std for r in group_rewards]
            advantages.extend(group_advantages)

        return advantages

    def compute_policy_loss(
        self,
        trajectories: List[Trajectory],
        advantages: List[float]
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute GRPO policy loss with clipping"""

        total_loss = 0.0
        total_kl = 0.0
        clipped_count = 0

        for traj, adv in zip(trajectories, advantages):
            # Get log probabilities
            log_prob_new = self.policy_model.compute_log_prob(
                traj.state, traj.action
            )
            log_prob_ref = self.reference_model.compute_log_prob(
                traj.state, traj.action
            )

            # Compute ratio
            ratio = torch.exp(log_prob_new - traj.log_prob)

            # Clipped surrogate objective
            adv_tensor = torch.tensor(adv, dtype=torch.float32)
            surr1 = ratio * adv_tensor
            surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * adv_tensor
            policy_loss = -torch.min(surr1, surr2)

            # KL divergence penalty
            kl_div = log_prob_new - log_prob_ref

            # Total loss
            loss = policy_loss + self.kl_coef * kl_div
            total_loss += loss
            total_kl += kl_div.item()

            # Track clipping
            if torch.abs(ratio - 1.0) > self.clip_range:
                clipped_count += 1

        avg_loss = total_loss / len(trajectories)

        stats = {
            'policy_loss': avg_loss.item(),
            'kl_divergence': total_kl / len(trajectories),
            'clip_fraction': clipped_count / len(trajectories)
        }

        return avg_loss, stats

    def train_step(
        self,
        batch_trajectories: List[Trajectory]
    ) -> Dict:
        """Single GRPO training step"""

        # 1. Compute rewards
        rewards = [traj.reward for traj in batch_trajectories]

        # 2. Compute group-based advantages
        advantages = self.compute_group_advantages(rewards)

        # 3. Compute policy loss
        loss, stats = self.compute_policy_loss(batch_trajectories, advantages)

        # 4. Backprop and update
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)

        self.optimizer.step()

        # 5. Update stats
        stats['avg_reward'] = np.mean(rewards)
        stats['max_reward'] = np.max(rewards)
        stats['min_reward'] = np.min(rewards)

        for key, value in stats.items():
            self.training_stats[key].append(value)

        return stats


# ============================================================================
# 7. Main Iterative System
# ============================================================================

class IterativeTableQASystem:
    """Main system combining code generation, execution, and GRPO"""

    def __init__(
        self,
        model_name: str = "gpt-4",
        max_iterations: int = 3,
        use_grpo: bool = True
    ):
        self.code_generator = CodeGenerator(model_name, max_iterations)
        self.execution_engine = ExecutionEngine(timeout=30)
        self.reward_function = RewardFunction()
        self.max_iterations = max_iterations
        self.use_grpo = use_grpo

        # GRPO trainer (initialized during training)
        self.grpo_trainer = None

        # Statistics
        self.stats = {
            'total_queries': 0,
            'success_at_k': {1: 0, 2: 0, 3: 0},
            'error_types': Counter(),
            'avg_iterations': []
        }

    def answer_question(
        self,
        table: pd.DataFrame,
        question: str,
        gold_answer: Optional[str] = None,
        return_trajectory: bool = False
    ) -> Dict:
        """
        Answer a table question with iterative refinement

        Returns:
            Dict containing answer, success, iterations, and optionally trajectory
        """
        trajectory_data = []

        # Generate initial code
        code = self.code_generator.generate_initial_code(table, question)

        for iteration in range(1, self.max_iterations + 1):
            # Execute code
            result = self.execution_engine.execute(code, table)

            # Store trajectory step
            trajectory_data.append({
                'iteration': iteration,
                'code': code,
                'result': result
            })

            # Success!
            if result.success:
                self.stats['success_at_k'][iteration] += 1
                self.stats['avg_iterations'].append(iteration)

                response = {
                    'answer': result.answer,
                    'success': True,
                    'iterations': iteration,
                    'final_code': code
                }

                if return_trajectory:
                    response['trajectory'] = trajectory_data

                # Compute reward if gold answer provided
                if gold_answer:
                    reward = self.reward_function.compute_reward(
                        result, gold_answer, iteration, question
                    )
                    response['reward'] = reward

                return response

            # Failure - try to fix
            self.stats['error_types'][result.error.error_type] += 1

            if iteration < self.max_iterations:
                # Refine code based on error
                code = self.code_generator.refine_code(
                    table, question, code, result.error, iteration
                )
            else:
                # Max iterations reached
                response = {
                    'answer': None,
                    'success': False,
                    'iterations': self.max_iterations,
                    'final_code': code,
                    'error': result.error
                }

                if return_trajectory:
                    response['trajectory'] = trajectory_data

                if gold_answer:
                    reward = self.reward_function.compute_reward(
                        result, gold_answer, iteration, question
                    )
                    response['reward'] = reward

                return response

    def train_with_grpo(
        self,
        train_dataset: List[Dict],
        val_dataset: List[Dict],
        num_epochs: int = 5,
        batch_size: int = 16
    ):
        """Train the system using GRPO"""

        # Initialize GRPO trainer
        if self.grpo_trainer is None:
            # Create reference model (frozen copy of policy model)
            reference_model = copy.deepcopy(self.code_generator.model)
            reference_model.eval()

            self.grpo_trainer = GRPOTrainer(
                policy_model=self.code_generator.model,
                reference_model=reference_model
            )

        logging.info(f"Starting GRPO training for {num_epochs} epochs")

        for epoch in range(num_epochs):
            # Shuffle training data
            np.random.shuffle(train_dataset)

            epoch_stats = []

            # Process in batches
            for batch_start in range(0, len(train_dataset), batch_size):
                batch = train_dataset[batch_start:batch_start + batch_size]

                # Collect trajectories
                trajectories = []

                for sample in batch:
                    table = sample['table']
                    question = sample['question']
                    gold_answer = sample['answer']

                    # Generate trajectory
                    result = self.answer_question(
                        table, question, gold_answer, return_trajectory=True
                    )

                    # Create trajectory object
                    traj = Trajectory(
                        state={'table': table, 'question': question},
                        action=result['final_code'],
                        reward=result.get('reward', 0.0),
                        log_prob=0.0,  # Would be computed by model
                        success=result['success'],
                        num_iterations=result['iterations'],
                        answer=result['answer'],
                        gold_answer=gold_answer
                    )
                    trajectories.append(traj)

                # GRPO update
                stats = self.grpo_trainer.train_step(trajectories)
                epoch_stats.append(stats)

            # Validation
            val_accuracy = self.evaluate(val_dataset)

            logging.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Avg Reward: {np.mean([s['avg_reward'] for s in epoch_stats]):.3f} - "
                f"Val Accuracy: {val_accuracy:.3f}"
            )

    def evaluate(self, test_dataset: List[Dict]) -> float:
        """Evaluate on test set"""
        correct = 0
        total = len(test_dataset)

        for sample in test_dataset:
            result = self.answer_question(
                sample['table'],
                sample['question'],
                sample['answer']
            )

            if result['success']:
                # Check accuracy
                pred = result['answer']
                gold = sample['answer']
                acc = self.reward_function._compute_accuracy(
                    pred, gold, sample['question']
                )
                if acc >= 0.9:  # Consider correct if >90% match
                    correct += 1

        return correct / total if total > 0 else 0.0


# ============================================================================
# 8. Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example table
    table_data = {
        'Country': ['USA', 'China', 'Japan', 'Germany', 'India'],
        'GDP_Trillion': [21.4, 14.3, 5.1, 3.8, 2.9],
        'Population_Million': [331, 1439, 126, 83, 1380]
    }
    table = pd.DataFrame(table_data)

    # Example question
    question = "Which country has the highest GDP per capita?"
    gold_answer = "USA"

    # Initialize system
    system = IterativeTableQASystem(
        model_name="gpt-4",
        max_iterations=3,
        use_grpo=False  # Set True for GRPO training
    )

    # Answer question
    result = system.answer_question(table, question, gold_answer)

    print(f"Question: {question}")
    print(f"Answer: {result['answer']}")
    print(f"Success: {result['success']}")
    print(f"Iterations: {result['iterations']}")
    print(f"Reward: {result.get('reward', 'N/A')}")
    print(f"\nGenerated Code:\n{result['final_code']}")
