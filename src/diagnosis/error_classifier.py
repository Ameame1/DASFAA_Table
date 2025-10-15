"""
Error Classifier - Layer 1 of Hierarchical Error Diagnosis
Classifies execution errors into 4 major categories
"""

from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ErrorClassifier:
    """
    Layer 1: Error Classification

    Classifies errors into 4 categories:
    - Syntax: SyntaxError, IndentationError, NameError
    - Runtime: KeyError, TypeError, ValueError, AttributeError, IndexError
    - Logic: Code executes but returns wrong answer
    - Semantic: Code doesn't match question intent
    """

    def __init__(self):
        # Define error type mappings
        self.syntax_errors = {
            'SyntaxError', 'IndentationError', 'TabError', 'NameError'
        }

        self.runtime_errors = {
            'KeyError', 'TypeError', 'ValueError', 'AttributeError',
            'IndexError', 'ZeroDivisionError', 'OverflowError'
        }

        self.timeout_errors = {
            'TimeoutError', 'MemoryError'
        }

    def classify(self, execution_result: Dict[str, Any]) -> str:
        """
        Classify error into one of 4 categories

        Args:
            execution_result: Result dictionary from CodeExecutor

        Returns:
            Error class: 'syntax', 'runtime', 'timeout', 'logic', 'semantic', or 'unknown'
        """
        # If execution succeeded, check if answer is correct (logic error)
        if execution_result.get('success', False):
            # If answer exists but might be wrong, classify as potential logic error
            if execution_result.get('answer') is not None:
                return 'success'  # Actually succeeded, will be validated later
            else:
                return 'logic'  # No answer produced despite success

        # Get error type
        error_type = execution_result.get('error_type', '')

        # Classify based on error type
        if error_type in self.syntax_errors:
            return 'syntax'

        if error_type in self.runtime_errors:
            return 'runtime'

        if error_type in self.timeout_errors:
            return 'timeout'

        # Check error message for semantic clues
        error_msg = execution_result.get('error', '').lower()

        # Keywords that suggest semantic misunderstanding
        semantic_keywords = [
            'column', 'schema', 'structure', 'header',
            'does not exist', 'not found', 'invalid column'
        ]

        if any(keyword in error_msg for keyword in semantic_keywords):
            # Could be runtime or semantic, default to runtime for now
            # Semantic classification requires deeper analysis
            return 'runtime'

        # Default classification
        if error_type:
            return 'runtime'  # Most unlisted errors are runtime

        return 'unknown'

    def get_error_severity(self, error_class: str) -> str:
        """
        Get error severity level

        Args:
            error_class: Error classification

        Returns:
            Severity: 'low', 'medium', 'high'
        """
        severity_map = {
            'syntax': 'low',      # Easy to fix
            'runtime': 'medium',  # Moderate difficulty
            'timeout': 'medium',  # May need algorithm change
            'logic': 'high',      # Hard to diagnose
            'semantic': 'high',   # Requires understanding intent
            'unknown': 'high'
        }

        return severity_map.get(error_class, 'high')

    def get_typical_causes(self, error_class: str) -> list:
        """
        Get typical causes for an error class

        Args:
            error_class: Error classification

        Returns:
            List of typical causes
        """
        causes = {
            'syntax': [
                'Missing or mismatched parentheses/brackets',
                'Incorrect indentation',
                'Invalid Python syntax',
                'Undefined variable names'
            ],
            'runtime': [
                'Column name mismatch (case sensitivity)',
                'Column does not exist in table',
                'Type mismatch (string vs numeric)',
                'Index out of bounds',
                'Division by zero',
                'Null/None values'
            ],
            'timeout': [
                'Infinite loop',
                'Computationally expensive operation',
                'Large data processing'
            ],
            'logic': [
                'Wrong aggregation function',
                'Incorrect filter condition',
                'Wrong column selected',
                'Misunderstood question intent'
            ],
            'semantic': [
                'Hallucinated column names',
                'Misinterpreted table schema',
                'Wrong understanding of question'
            ]
        }

        return causes.get(error_class, ['Unknown causes'])


if __name__ == "__main__":
    # Test the classifier
    print("Testing ErrorClassifier...")

    classifier = ErrorClassifier()

    # Test 1: Syntax Error
    print("\n=== Test 1: Syntax Error ===")
    result = {
        'success': False,
        'error_type': 'SyntaxError',
        'error': 'invalid syntax'
    }
    error_class = classifier.classify(result)
    severity = classifier.get_error_severity(error_class)
    print(f"Class: {error_class} (Severity: {severity})")
    print(f"Typical causes: {classifier.get_typical_causes(error_class)}")
    assert error_class == 'syntax'
    assert severity == 'low'

    # Test 2: KeyError (Runtime)
    print("\n=== Test 2: KeyError (Runtime) ===")
    result = {
        'success': False,
        'error_type': 'KeyError',
        'error': "'Population'"
    }
    error_class = classifier.classify(result)
    severity = classifier.get_error_severity(error_class)
    print(f"Class: {error_class} (Severity: {severity})")
    assert error_class == 'runtime'
    assert severity == 'medium'

    # Test 3: Success
    print("\n=== Test 3: Success ===")
    result = {
        'success': True,
        'answer': 42
    }
    error_class = classifier.classify(result)
    print(f"Class: {error_class}")
    assert error_class == 'success'

    # Test 4: Logic Error (success but no answer)
    print("\n=== Test 4: Logic Error ===")
    result = {
        'success': True,
        'answer': None
    }
    error_class = classifier.classify(result)
    severity = classifier.get_error_severity(error_class)
    print(f"Class: {error_class} (Severity: {severity})")
    assert error_class == 'logic'
    assert severity == 'high'

    # Test 5: Timeout
    print("\n=== Test 5: Timeout ===")
    result = {
        'success': False,
        'error_type': 'TimeoutError',
        'error': 'Code execution exceeded 5 seconds'
    }
    error_class = classifier.classify(result)
    print(f"Class: {error_class}")
    assert error_class == 'timeout'

    print("\n✓ All classifier tests passed!")
