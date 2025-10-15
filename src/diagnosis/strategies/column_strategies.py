"""
Column-related repair strategies
Handles column name errors (most common in table QA)
"""

import pandas as pd
from typing import Dict, Any
from .base_strategy import RepairStrategy


class ColumnNameCorrectionStrategy(RepairStrategy):
    """
    Strategy for correcting column name errors

    Handles:
    - Case mismatch (Population → population)
    - Typos with fuzzy matching
    - Column doesn't exist
    """

    def can_handle(self, root_cause: Dict[str, Any], error_class: str) -> bool:
        """Check if this is a column name error"""
        return root_cause.get('root_cause') in [
            'column_case_mismatch',
            'column_typo',
            'column_not_exist'
        ]

    def generate_repair_prompt(
        self,
        root_cause: Dict[str, Any],
        original_code: str,
        table: pd.DataFrame,
        question: str,
        execution_result: Dict[str, Any]
    ) -> str:
        """Generate repair prompt for column name errors"""
        details = root_cause.get('details', {})
        missing = details.get('missing', '')
        available = details.get('available', [])

        cause_type = root_cause.get('root_cause')

        if cause_type == 'column_case_mismatch':
            correct = details.get('correct', '')
            prompt = f"""The code failed with KeyError: '{missing}'

Problem: Column name case mismatch
- You used: '{missing}'
- Correct name: '{correct}' (different capitalization)

Available columns in table: {available}

Original code:
```python
{original_code}
```

Fix the code by replacing '{missing}' with '{correct}'. Ensure you:
1. Use the exact column name '{correct}'
2. Keep all other logic unchanged
3. Store final answer in 'answer' variable

Fixed code:
"""

        elif cause_type == 'column_typo':
            suggestions = details.get('suggestions', [])
            prompt = f"""The code failed with KeyError: '{missing}'

Problem: Column '{missing}' does not exist
- Most similar columns: {suggestions}
- All available columns: {available}

Original code:
```python
{original_code}
```

Fix the code by replacing '{missing}' with the most appropriate column from {suggestions}.
Choose based on the question context: "{question}"

Fixed code:
"""

        else:  # column_not_exist
            prompt = f"""The code failed with KeyError: '{missing}'

Problem: Column '{missing}' does not exist in the table
Available columns: {available}

This appears to be a hallucination. Please generate new code that:
1. Only uses columns that actually exist: {available}
2. Correctly answers the question: "{question}"
3. Stores the answer in 'answer' variable

Table preview:
{table.head(3).to_string()}

Generate correct code:
"""

        return prompt

    def get_priority(self) -> int:
        """High priority - most common error"""
        return 1


class ColumnDataTypeStrategy(RepairStrategy):
    """
    Strategy for handling column data type issues
    E.g., numeric column stored as string
    """

    def can_handle(self, root_cause: Dict[str, Any], error_class: str) -> bool:
        """Check if this is a data type error"""
        return root_cause.get('root_cause') in [
            'string_numeric_operation',
            'type_mismatch'
        ]

    def generate_repair_prompt(
        self,
        root_cause: Dict[str, Any],
        original_code: str,
        table: pd.DataFrame,
        question: str,
        execution_result: Dict[str, Any]
    ) -> str:
        """Generate repair prompt for type errors"""
        details = root_cause.get('details', {})
        suggestion = details.get('suggestion', '')

        prompt = f"""The code failed with TypeError

Problem: Type mismatch - likely trying to perform numeric operations on string data

Original code:
```python
{original_code}
```

Error: {execution_result.get('error', '')}

Fix the code by:
1. Converting string columns to numeric using pd.to_numeric(df['column'], errors='coerce')
2. Handle potential NaN values with .fillna(0) or .dropna()
3. Ensure all operations use the converted numeric values

Example:
```python
# Convert to numeric
df['column'] = pd.to_numeric(df['column'], errors='coerce')
# Then perform operations
answer = df['column'].sum()
```

Question: {question}

Fixed code:
"""
        return prompt

    def get_priority(self) -> int:
        """High priority"""
        return 2


if __name__ == "__main__":
    print("Column strategies defined successfully!")
