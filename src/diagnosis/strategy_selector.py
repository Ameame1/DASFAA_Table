"""
Strategy Selector - Layer 3 of Hierarchical Error Diagnosis
Selects the best repair strategy based on root cause

TODO: GRPO训练接口预留，后续使用TRL训练
"""

import pandas as pd
from typing import Dict, Any, List, Optional
import logging

# Import all strategies
from .strategies.column_strategies import (
    ColumnNameCorrectionStrategy,
    ColumnDataTypeStrategy
)
from .strategies.type_aggregation_strategies import (
    TypeConversionStrategy,
    AggregationCorrectionStrategy,
    FilterRelaxationStrategy
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StrategySelector:
    """
    Layer 3: Strategy Selection

    Modes:
    1. Rule-based: Use pre-defined rules (no training needed)
    2. GRPO-based: Use learned policy (requires GRPO training)

    TODO: GRPO模式需要后续用TRL训练策略网络
    """

    def __init__(self, use_grpo: bool = False, grpo_model_path: Optional[str] = None):
        """
        Args:
            use_grpo: Whether to use GRPO-based selection
            grpo_model_path: Path to trained GRPO model (if use_grpo=True)
        """
        self.use_grpo = use_grpo
        self.grpo_model_path = grpo_model_path

        # Register all available strategies
        self.strategies = [
            ColumnNameCorrectionStrategy(),
            ColumnDataTypeStrategy(),
            TypeConversionStrategy(),
            AggregationCorrectionStrategy(),
            FilterRelaxationStrategy(),
        ]

        # Sort strategies by priority
        self.strategies.sort(key=lambda s: s.get_priority())

        logger.info(f"Initialized StrategySelector with {len(self.strategies)} strategies")
        logger.info(f"Mode: {'GRPO' if use_grpo else 'Rule-based'}")

        # GRPO model placeholder
        if use_grpo:
            if grpo_model_path:
                self.grpo_policy = self._load_grpo_model(grpo_model_path)
            else:
                logger.warning("GRPO mode enabled but no model path provided. Falling back to rule-based.")
                self.use_grpo = False
                self.grpo_policy = None

    def select_strategy(
        self,
        root_cause: Dict[str, Any],
        error_class: str,
        table: pd.DataFrame,
        code: str,
        question: str
    ):
        """
        Select the best repair strategy

        Args:
            root_cause: Root cause from RootCauseAnalyzer
            error_class: Error class from ErrorClassifier
            table: Input table
            code: Failed code
            question: Original question

        Returns:
            Selected RepairStrategy instance or None
        """
        if self.use_grpo:
            return self._grpo_based_selection(root_cause, error_class, table, code, question)
        else:
            return self._rule_based_selection(root_cause, error_class)

    def _rule_based_selection(
        self,
        root_cause: Dict[str, Any],
        error_class: str
    ):
        """
        Rule-based strategy selection (no training needed)

        Args:
            root_cause: Root cause dictionary
            error_class: Error classification

        Returns:
            First matching strategy or None
        """
        for strategy in self.strategies:
            if strategy.can_handle(root_cause, error_class):
                logger.info(f"Selected strategy: {strategy.name}")
                return strategy

        logger.warning("No matching strategy found")
        return None

    def _grpo_based_selection(
        self,
        root_cause: Dict[str, Any],
        error_class: str,
        table: pd.DataFrame,
        code: str,
        question: str
    ):
        """
        GRPO-based strategy selection (requires trained model)

        TODO: 实现GRPO策略选择
        - 特征编码（error_class, root_cause, table stats, code features）
        - 使用GRPO训练的策略网络预测
        - 返回概率最高的策略

        目前：回退到规则选择
        """
        logger.warning("GRPO selection not yet implemented, falling back to rule-based")
        return self._rule_based_selection(root_cause, error_class)

    def _load_grpo_model(self, model_path: str):
        """
        Load trained GRPO model

        TODO: 使用TRL加载训练好的策略网络
        - 加载checkpoint
        - 初始化策略网络
        - 返回可调用的模型

        Args:
            model_path: Path to GRPO model checkpoint

        Returns:
            Loaded model or None
        """
        logger.warning(f"GRPO model loading not implemented: {model_path}")
        return None

    def get_available_strategies(self) -> List[str]:
        """Get names of all registered strategies"""
        return [s.name for s in self.strategies]


if __name__ == "__main__":
    print("StrategySelector initialized!")
    selector = StrategySelector(use_grpo=False)
    print(f"Available strategies: {selector.get_available_strategies()}")
