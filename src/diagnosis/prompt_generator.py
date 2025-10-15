"""
Prompt Generator - Layer 4 of Hierarchical Error Diagnosis
Generates high-quality repair prompts for LLM
"""

import pandas as pd
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PromptGenerator:
    """
    Layer 4: Repair Prompt Generation

    Takes strategy output and generates complete repair prompt
    """

    def __init__(self):
        pass

    def generate(
        self,
        strategy,
        root_cause: Dict[str, Any],
        original_code: str,
        table: pd.DataFrame,
        question: str,
        execution_result: Dict[str, Any]
    ) -> str:
        """
        Generate complete repair prompt

        Args:
            strategy: Selected RepairStrategy instance
            root_cause: Root cause dictionary
            original_code: Failed code
            table: Input table
            question: Original question
            execution_result: Execution result with error

        Returns:
            Complete repair prompt string
        """
        if strategy is None:
            # No specific strategy - generate generic repair prompt
            return self._generate_generic_prompt(
                original_code, table, question, execution_result
            )

        # Use strategy-specific prompt generation
        strategy_prompt = strategy.generate_repair_prompt(
            root_cause, original_code, table, question, execution_result
        )

        # Add context and formatting
        full_prompt = self._add_context(strategy_prompt, table, question)

        return full_prompt

    def _add_context(self, strategy_prompt: str, table: pd.DataFrame, question: str) -> str:
        """
        Add common context to all prompts
        Simplified based on AILS-NTUA approach
        """
        context = f"""You are a Python expert for Table Question Answering.

Fix the code to answer the question correctly.

Table Columns: {list(table.columns)}
Question: {question}

{strategy_prompt}

Generate the FIXED function:
```python
import pandas as pd

def answer(df: pd.DataFrame):
    df.columns = {list(table.columns)}
    # Your fixed solution
    result = ...
    return result
```

Return ONLY the fixed function code.
"""
        return context

    def _generate_generic_prompt(
        self,
        original_code: str,
        table: pd.DataFrame,
        question: str,
        execution_result: Dict[str, Any]
    ) -> str:
        """
        Generate generic repair prompt when no specific strategy matches
        Simplified AILS-NTUA style
        """
        error_msg = execution_result.get('error', 'Unknown error')
        error_type = execution_result.get('error_type', 'Unknown')

        prompt = f"""Previous Code:
```python
{original_code}
```

Error:
{error_type}: {error_msg}

Fix the code."""

        return self._add_context(prompt, table, question)


if __name__ == "__main__":
    print("PromptGenerator initialized!")
