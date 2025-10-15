"""
Complete Hierarchical Diagnostic System
Integrates all 4 layers of error diagnosis
"""

import pandas as pd
from typing import Dict, Any, Optional
import logging

from .error_classifier import ErrorClassifier
from .root_cause_analyzer import RootCauseAnalyzer
from .strategy_selector import StrategySelector
from .prompt_generator import PromptGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HierarchicalDiagnosticSystem:
    """
    Complete 4-layer hierarchical error diagnosis system

    Layer 1: Error Classification (Syntax/Runtime/Logic/Semantic)
    Layer 2: Root Cause Analysis (Specific cause identification)
    Layer 3: Strategy Selection (Choose repair strategy)
    Layer 4: Prompt Generation (Generate repair instruction)
    """

    def __init__(self, use_grpo: bool = False, grpo_model_path: Optional[str] = None):
        """
        Initialize diagnostic system

        Args:
            use_grpo: Whether to use GRPO for strategy selection
            grpo_model_path: Path to trained GRPO model
        """
        self.classifier = ErrorClassifier()
        self.root_cause_analyzer = RootCauseAnalyzer()
        self.strategy_selector = StrategySelector(use_grpo, grpo_model_path)
        self.prompt_generator = PromptGenerator()

        logger.info("HierarchicalDiagnosticSystem initialized")
        logger.info(f"GRPO mode: {use_grpo}")

    def diagnose(
        self,
        execution_result: Dict[str, Any],
        code: str,
        table: pd.DataFrame,
        question: str
    ) -> Dict[str, Any]:
        """
        Complete diagnosis pipeline

        Args:
            execution_result: Result from CodeExecutor
            code: The code that was executed
            table: Input table
            question: Original question

        Returns:
            Diagnosis result with repair prompt
        """
        # Layer 1: Classify error
        error_class = self.classifier.classify(execution_result)
        logger.debug(f"Layer 1 - Error class: {error_class}")

        # If success, no diagnosis needed
        if error_class == 'success':
            return {
                'error_class': 'success',
                'root_cause': None,
                'strategy': None,
                'repair_prompt': None
            }

        # Layer 2: Analyze root cause
        root_cause = self.root_cause_analyzer.analyze(
            execution_result, code, table, error_class
        )
        logger.debug(f"Layer 2 - Root cause: {root_cause.get('root_cause')}")

        # Layer 3: Select repair strategy
        strategy = self.strategy_selector.select_strategy(
            root_cause, error_class, table, code, question
        )
        strategy_name = strategy.name if strategy else 'None'
        logger.debug(f"Layer 3 - Strategy: {strategy_name}")

        # Layer 4: Generate repair prompt
        repair_prompt = self.prompt_generator.generate(
            strategy, root_cause, code, table, question, execution_result
        )

        return {
            'error_class': error_class,
            'root_cause': root_cause,
            'strategy': strategy_name,
            'repair_prompt': repair_prompt
        }


if __name__ == "__main__":
    print("✓ HierarchicalDiagnosticSystem ready!")
