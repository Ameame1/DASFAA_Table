"""
Base Strategy Class for Error Repair
All repair strategies inherit from this base class
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd


class RepairStrategy(ABC):
    """
    Abstract base class for all repair strategies

    Each strategy must implement:
    - can_handle(): Check if this strategy can handle the error
    - generate_repair_prompt(): Generate a repair instruction for the LLM
    """

    def __init__(self):
        self.name = self.__class__.__name__

    @abstractmethod
    def can_handle(self, root_cause: Dict[str, Any], error_class: str) -> bool:
        """
        Check if this strategy can handle the given error

        Args:
            root_cause: Root cause dictionary from RootCauseAnalyzer
            error_class: Error classification from ErrorClassifier

        Returns:
            True if this strategy can handle the error
        """
        pass

    @abstractmethod
    def generate_repair_prompt(
        self,
        root_cause: Dict[str, Any],
        original_code: str,
        table: pd.DataFrame,
        question: str,
        execution_result: Dict[str, Any]
    ) -> str:
        """
        Generate a repair prompt for the LLM

        Args:
            root_cause: Root cause dictionary
            original_code: The code that failed
            table: Input DataFrame
            question: Original question
            execution_result: Execution result with error

        Returns:
            Repair prompt string
        """
        pass

    def get_priority(self) -> int:
        """
        Get strategy priority (lower = higher priority)

        Returns:
            Priority value (0-10)
        """
        return 5  # Default medium priority


if __name__ == "__main__":
    print("Base RepairStrategy class defined successfully!")
