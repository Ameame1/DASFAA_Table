"""
Type conversion and aggregation repair strategies
"""

import pandas as pd
from typing import Dict, Any
from .base_strategy import RepairStrategy


class TypeConversionStrategy(RepairStrategy):
    """Strategy for type conversion errors"""

    def can_handle(self, root_cause: Dict[str, Any], error_class: str) -> bool:
        return root_cause.get('root_cause') in [
            'value_conversion_error',
            'string_numeric_operation'
        ]

    def generate_repair_prompt(
        self,
        root_cause: Dict[str, Any],
        original_code: str,
        table: pd.DataFrame,
        question: str,
        execution_result: Dict[str, Any]
    ) -> str:
        prompt = f"""The code failed due to type conversion error.

Original code:
```python
{original_code}
```

Error: {execution_result.get('error', '')}

Fix by converting data types properly:
- Use pd.to_numeric() for numeric conversions
- Use errors='coerce' to handle invalid values
- Handle NaN/None values appropriately

Question: {question}

Fixed code:
"""
        return prompt

    def get_priority(self) -> int:
        return 2


class AggregationCorrectionStrategy(RepairStrategy):
    """Strategy for wrong aggregation functions"""

    def can_handle(self, root_cause: Dict[str, Any], error_class: str) -> bool:
        # This requires logic error detection - placeholder for now
        return root_cause.get('root_cause') == 'wrong_aggregation'

    def generate_repair_prompt(
        self,
        root_cause: Dict[str, Any],
        original_code: str,
        table: pd.DataFrame,
        question: str,
        execution_result: Dict[str, Any]
    ) -> str:
        prompt = f"""The code may be using wrong aggregation function.

Question: {question}

Original code:
```python
{original_code}
```

Consider if the aggregation matches the question:
- "total/sum" → use .sum()
- "average/mean" → use .mean()
- "maximum/largest" → use .max()
- "minimum/smallest" → use .min()
- "count/number of" → use .count() or len()

Generate corrected code:
"""
        return prompt

    def get_priority(self) -> int:
        return 4


class FilterRelaxationStrategy(RepairStrategy):
    """Strategy for too strict filters returning empty results"""

    def can_handle(self, root_cause: Dict[str, Any], error_class: str) -> bool:
        # Placeholder - needs empty DataFrame detection
        return root_cause.get('root_cause') == 'empty_result'

    def generate_repair_prompt(
        self,
        root_cause: Dict[str, Any],
        original_code: str,
        table: pd.DataFrame,
        question: str,
        execution_result: Dict[str, Any]
    ) -> str:
        prompt = f"""The code returned an empty result - filters may be too strict.

Question: {question}

Original code:
```python
{original_code}
```

Relax the filtering conditions:
- Check for exact string matches → use .str.contains() or case-insensitive matching
- >= instead of >
- Consider partial matches

Table preview:
{table.head(3).to_string()}

Fixed code:
"""
        return prompt

    def get_priority(self) -> int:
        return 3


if __name__ == "__main__":
    print("Type and aggregation strategies defined!")
