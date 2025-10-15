"""
Root Cause Analyzer - Layer 2 of Hierarchical Error Diagnosis
Analyzes the specific root cause of errors
"""

import re
import pandas as pd
from typing import Dict, Any, List, Optional
from difflib import get_close_matches
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RootCauseAnalyzer:
    """
    Layer 2: Root Cause Analysis

    Analyzes the specific root cause for each error type:
    - KeyError → Column name mismatch, typo, hallucination
    - TypeError → Type conversion needed, incompatible operation
    - ValueError → Invalid value, parsing error
    - etc.
    """

    def __init__(self):
        pass

    def analyze(
        self,
        execution_result: Dict[str, Any],
        code: str,
        table: pd.DataFrame,
        error_class: str
    ) -> Dict[str, Any]:
        """
        Analyze root cause of error

        Args:
            execution_result: Result from CodeExecutor
            code: The code that was executed
            table: Input DataFrame
            error_class: Error classification from Layer 1

        Returns:
            Root cause dictionary with details
        """
        if error_class == 'syntax':
            return self._analyze_syntax_error(execution_result, code)

        elif error_class == 'runtime':
            error_type = execution_result.get('error_type', '')

            if error_type == 'KeyError':
                return self._analyze_keyerror(execution_result, code, table)
            elif error_type == 'TypeError':
                return self._analyze_typeerror(execution_result, code, table)
            elif error_type == 'ValueError':
                return self._analyze_valueerror(execution_result, code, table)
            elif error_type == 'AttributeError':
                return self._analyze_attributeerror(execution_result, code, table)
            elif error_type == 'IndexError':
                return self._analyze_indexerror(execution_result, code, table)
            else:
                return {'root_cause': 'unknown_runtime_error', 'details': {}}

        elif error_class == 'timeout':
            return self._analyze_timeout(execution_result, code, table)

        elif error_class == 'logic':
            return self._analyze_logic_error(execution_result, code, table)

        else:
            return {'root_cause': 'unknown', 'details': {}}

    def _analyze_syntax_error(
        self,
        execution_result: Dict,
        code: str
    ) -> Dict:
        """Analyze syntax errors"""
        error_msg = execution_result.get('error', '')
        error_type = execution_result.get('error_type', '')

        if error_type == 'IndentationError':
            return {
                'root_cause': 'indentation_error',
                'details': {
                    'description': 'Python code has incorrect indentation',
                    'error_message': error_msg
                }
            }

        elif error_type == 'NameError':
            # Extract undefined variable name
            match = re.search(r"name '(\w+)' is not defined", error_msg)
            undefined_var = match.group(1) if match else 'unknown'

            return {
                'root_cause': 'undefined_variable',
                'details': {
                    'variable': undefined_var,
                    'description': f"Variable '{undefined_var}' is not defined",
                    'error_message': error_msg
                }
            }

        else:
            return {
                'root_cause': 'general_syntax_error',
                'details': {
                    'description': 'Invalid Python syntax',
                    'error_message': error_msg
                }
            }

    def _analyze_keyerror(
        self,
        execution_result: Dict,
        code: str,
        table: pd.DataFrame
    ) -> Dict:
        """
        Analyze KeyError - most common error for table operations

        Possible causes:
        - Column name case mismatch
        - Column name typo
        - Column doesn't exist (hallucination)
        """
        error_msg = execution_result.get('error', '')

        # Extract column name from error message
        # KeyError: 'Population' → extract 'Population'
        match = re.search(r"['\"]([^'\"]+)['\"]", error_msg)
        if not match:
            return {
                'root_cause': 'keyerror_unknown',
                'details': {'error_message': error_msg}
            }

        missing_col = match.group(1)
        available_cols = list(table.columns)

        # Check if it's a case mismatch
        if missing_col.lower() in [c.lower() for c in available_cols]:
            correct_col = [c for c in available_cols if c.lower() == missing_col.lower()][0]
            return {
                'root_cause': 'column_case_mismatch',
                'details': {
                    'missing': missing_col,
                    'correct': correct_col,
                    'available': available_cols,
                    'description': f"Column '{missing_col}' not found. Did you mean '{correct_col}'?"
                }
            }

        # Check if it's a typo (fuzzy match)
        similar_cols = get_close_matches(missing_col, available_cols, n=3, cutoff=0.6)
        if similar_cols:
            return {
                'root_cause': 'column_typo',
                'details': {
                    'missing': missing_col,
                    'suggestions': similar_cols,
                    'available': available_cols,
                    'description': f"Column '{missing_col}' not found. Similar columns: {similar_cols}"
                }
            }

        # Column doesn't exist at all (hallucination)
        return {
            'root_cause': 'column_not_exist',
            'details': {
                'missing': missing_col,
                'available': available_cols,
                'description': f"Column '{missing_col}' does not exist in table"
            }
        }

    def _analyze_typeerror(
        self,
        execution_result: Dict,
        code: str,
        table: pd.DataFrame
    ) -> Dict:
        """
        Analyze TypeError

        Common causes:
        - Numeric operation on string column
        - Incompatible types in comparison
        - Wrong function arguments
        """
        error_msg = execution_result.get('error', '').lower()

        # Check for numeric operation on string
        if 'str' in error_msg and ('int' in error_msg or 'float' in error_msg):
            return {
                'root_cause': 'string_numeric_operation',
                'details': {
                    'description': 'Trying to perform numeric operation on string column',
                    'suggestion': 'Convert column to numeric using pd.to_numeric()',
                    'error_message': error_msg
                }
            }

        # Check for None type operation
        if 'nonetype' in error_msg:
            return {
                'root_cause': 'none_type_operation',
                'details': {
                    'description': 'Operation on None/null value',
                    'suggestion': 'Handle null values with fillna() or dropna()',
                    'error_message': error_msg
                }
            }

        # General type error
        return {
            'root_cause': 'type_mismatch',
            'details': {
                'description': 'Type mismatch in operation',
                'error_message': error_msg
            }
        }

    def _analyze_valueerror(
        self,
        execution_result: Dict,
        code: str,
        table: pd.DataFrame
    ) -> Dict:
        """Analyze ValueError"""
        error_msg = execution_result.get('error', '')

        if 'could not convert' in error_msg.lower():
            return {
                'root_cause': 'value_conversion_error',
                'details': {
                    'description': 'Failed to convert value to target type',
                    'suggestion': 'Use pd.to_numeric() with errors="coerce"',
                    'error_message': error_msg
                }
            }

        return {
            'root_cause': 'invalid_value',
            'details': {
                'description': 'Invalid value encountered',
                'error_message': error_msg
            }
        }

    def _analyze_attributeerror(
        self,
        execution_result: Dict,
        code: str,
        table: pd.DataFrame
    ) -> Dict:
        """Analyze AttributeError"""
        error_msg = execution_result.get('error', '')

        # Extract attribute name
        match = re.search(r"'(\w+)' object has no attribute '(\w+)'", error_msg)
        if match:
            obj_type = match.group(1)
            attr_name = match.group(2)

            return {
                'root_cause': 'attribute_not_exist',
                'details': {
                    'object_type': obj_type,
                    'attribute': attr_name,
                    'description': f"'{obj_type}' object has no attribute '{attr_name}'",
                    'error_message': error_msg
                }
            }

        return {
            'root_cause': 'attribute_error',
            'details': {'error_message': error_msg}
        }

    def _analyze_indexerror(
        self,
        execution_result: Dict,
        code: str,
        table: pd.DataFrame
    ) -> Dict:
        """Analyze IndexError"""
        return {
            'root_cause': 'index_out_of_bounds',
            'details': {
                'description': 'Index out of bounds',
                'table_size': len(table),
                'suggestion': 'Check index range or use .iloc[] instead of direct indexing'
            }
        }

    def _analyze_timeout(
        self,
        execution_result: Dict,
        code: str,
        table: pd.DataFrame
    ) -> Dict:
        """Analyze timeout errors"""
        # Check for loops in code
        has_loop = 'for ' in code or 'while ' in code

        return {
            'root_cause': 'execution_timeout',
            'details': {
                'description': 'Code execution exceeded time limit',
                'has_loop': has_loop,
                'table_size': len(table),
                'suggestion': 'Use vectorized operations instead of loops' if has_loop else 'Simplify computation'
            }
        }

    def _analyze_logic_error(
        self,
        execution_result: Dict,
        code: str,
        table: pd.DataFrame
    ) -> Dict:
        """Analyze logic errors (code runs but wrong answer)"""
        return {
            'root_cause': 'logic_error',
            'details': {
                'description': 'Code executed but may have logic issues',
                'suggestion': 'Review aggregation functions, filters, and column selections'
            }
        }


if __name__ == "__main__":
    # Test the analyzer
    print("Testing RootCauseAnalyzer...")

    analyzer = RootCauseAnalyzer()

    # Test table
    table = pd.DataFrame({
        'city': ['Beijing', 'Shanghai'],
        'population': ['21.54', '24.28'],  # String type!
        'gdp': [4.0, 4.3]
    })

    # Test 1: Column case mismatch
    print("\n=== Test 1: Column Case Mismatch ===")
    exec_result = {
        'success': False,
        'error_type': 'KeyError',
        'error': "'Population'"
    }
    root_cause = analyzer.analyze(exec_result, "df['Population']", table, 'runtime')
    print(f"Root cause: {root_cause['root_cause']}")
    print(f"Details: {root_cause['details']}")
    assert root_cause['root_cause'] == 'column_case_mismatch'
    assert root_cause['details']['correct'] == 'population'

    # Test 2: Column typo
    print("\n=== Test 2: Column Typo ===")
    exec_result = {
        'success': False,
        'error_type': 'KeyError',
        'error': "'popul'"
    }
    root_cause = analyzer.analyze(exec_result, "df['popul']", table, 'runtime')
    print(f"Root cause: {root_cause['root_cause']}")
    print(f"Suggestions: {root_cause['details'].get('suggestions', [])}")
    assert root_cause['root_cause'] == 'column_typo'
    assert 'population' in root_cause['details']['suggestions']

    # Test 3: String/Numeric operation
    print("\n=== Test 3: Type Error ===")
    exec_result = {
        'success': False,
        'error_type': 'TypeError',
        'error': "can only concatenate str (not \"int\") to str"
    }
    root_cause = analyzer.analyze(exec_result, "df['population'].sum()", table, 'runtime')
    print(f"Root cause: {root_cause['root_cause']}")
    print(f"Suggestion: {root_cause['details'].get('suggestion', '')}")
    assert root_cause['root_cause'] == 'string_numeric_operation'

    # Test 4: Indentation error
    print("\n=== Test 4: Indentation Error ===")
    exec_result = {
        'success': False,
        'error_type': 'IndentationError',
        'error': 'unexpected indent'
    }
    root_cause = analyzer.analyze(exec_result, "  answer = 42", table, 'syntax')
    print(f"Root cause: {root_cause['root_cause']}")
    assert root_cause['root_cause'] == 'indentation_error'

    print("\n✓ All root cause analyzer tests passed!")
