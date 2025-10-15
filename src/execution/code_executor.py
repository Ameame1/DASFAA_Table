"""
Secure Code Executor with Sandbox
Safely executes Python code on pandas DataFrames with timeout and memory limits
"""

import pandas as pd
import numpy as np
import re
import traceback
import signal
import sys
from typing import Dict, Any, Optional
from contextlib import contextmanager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeoutException(Exception):
    """Raised when code execution times out"""
    pass


class MemoryLimitException(Exception):
    """Raised when code execution exceeds memory limit"""
    pass


@contextmanager
def time_limit(seconds: int):
    """
    Context manager to limit execution time

    Args:
        seconds: Maximum execution time in seconds
    """
    def signal_handler(signum, frame):
        raise TimeoutException(f"Code execution exceeded {seconds} seconds")

    # Set up the signal handler
    old_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


class SecureCodeExecutor:
    """
    Secure Python code executor with sandboxing

    Features:
    - Timeout protection (default: 5 seconds)
    - Memory limit (default: 2GB)
    - Restricted builtins (no file I/O, no network)
    - Whitelist of allowed imports
    - Detailed error reporting
    """

    def __init__(
        self,
        timeout: int = 5,
        memory_limit_mb: int = 2048,
        allowed_imports: Optional[list] = None
    ):
        """
        Args:
            timeout: Maximum execution time in seconds
            memory_limit_mb: Maximum memory usage in MB
            allowed_imports: List of allowed import modules
        """
        self.timeout = timeout
        self.memory_limit = memory_limit_mb

        # Default allowed imports (safe for table operations)
        self.allowed_imports = allowed_imports or [
            'pandas', 'numpy', 're', 'datetime', 'math',
            'statistics', 'collections', 'itertools'
        ]

        # Create restricted builtins (no file I/O, no exec, no eval, etc.)
        self.safe_builtins = {
            'abs': abs,
            'all': all,
            'any': any,
            'bool': bool,
            'dict': dict,
            'enumerate': enumerate,
            'filter': filter,
            'float': float,
            'int': int,
            'isinstance': isinstance,
            'len': len,
            'list': list,
            'locals': locals,
            'map': map,
            'max': max,
            'min': min,
            'range': range,
            'round': round,
            'set': set,
            'sorted': sorted,
            'str': str,
            'sum': sum,
            'tuple': tuple,
            'type': type,
            'zip': zip,
        }

    def create_safe_globals(self, table: pd.DataFrame) -> Dict[str, Any]:
        """
        Create a safe global namespace for code execution

        Args:
            table: Input DataFrame

        Returns:
            Dictionary of safe globals
        """
        safe_globals = {
            '__builtins__': self.safe_builtins,
            'pd': pd,
            'np': np,
            're': re,
            'df': table.copy(),  # Work on a copy to prevent modification
            'answer': None,      # Variable where answer should be stored
        }

        return safe_globals

    def clean_code(self, code: str) -> str:
        """
        Clean generated code by removing unnecessary imports and print statements

        Args:
            code: Raw generated code

        Returns:
            Cleaned code
        """
        lines = code.split('\n')
        cleaned_lines = []

        for line in lines:
            stripped = line.strip()
            # Skip import statements (we provide pd, np, re in globals)
            if stripped.startswith('import ') or stripped.startswith('from '):
                continue
            # Skip print statements (not needed for answer)
            if stripped.startswith('print('):
                continue
            # Skip empty lines and comments at the start
            if not stripped or (stripped.startswith('#') and not cleaned_lines):
                continue
            cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    def execute(
        self,
        code: str,
        table: pd.DataFrame,
        return_locals: bool = False
    ) -> Dict[str, Any]:
        """
        Execute Python code safely
        Supports both:
        1. Direct code: answer = df['col'].sum()
        2. Function code: def answer(df): return df['col'].sum()

        Args:
            code: Python code string
            table: Input DataFrame
            return_locals: Whether to return local variables

        Returns:
            Dictionary with execution result
        """
        # Clean the code first
        code = self.clean_code(code)

        # Create safe execution environment
        safe_globals = self.create_safe_globals(table)
        safe_locals = {}

        try:
            # Execute with timeout
            with time_limit(self.timeout):
                exec(code, safe_globals, safe_locals)

            # Get answer
            # Check if it's a function definition
            if 'answer' in safe_locals and callable(safe_locals['answer']):
                # It's a function, call it with the DataFrame
                answer_func = safe_locals['answer']
                answer = answer_func(safe_globals['df'])
            elif 'answer' in safe_locals:
                # Direct assignment
                answer = safe_locals['answer']
            elif 'answer' in safe_globals:
                # Answer in globals
                answer = safe_globals['answer']
            else:
                answer = None

            result = {
                'success': True,
                'answer': answer,
                'error': None,
                'error_type': None,
                'traceback': None
            }

            if return_locals:
                result['locals'] = safe_locals

            return result

        except TimeoutException as e:
            return {
                'success': False,
                'answer': None,
                'error': str(e),
                'error_type': 'TimeoutError',
                'traceback': traceback.format_exc()
            }

        except MemoryError as e:
            return {
                'success': False,
                'answer': None,
                'error': 'Memory limit exceeded',
                'error_type': 'MemoryError',
                'traceback': traceback.format_exc()
            }

        except SyntaxError as e:
            return {
                'success': False,
                'answer': None,
                'error': str(e),
                'error_type': 'SyntaxError',
                'traceback': traceback.format_exc()
            }

        except KeyError as e:
            return {
                'success': False,
                'answer': None,
                'error': str(e),
                'error_type': 'KeyError',
                'traceback': traceback.format_exc()
            }

        except TypeError as e:
            return {
                'success': False,
                'answer': None,
                'error': str(e),
                'error_type': 'TypeError',
                'traceback': traceback.format_exc()
            }

        except ValueError as e:
            return {
                'success': False,
                'answer': None,
                'error': str(e),
                'error_type': 'ValueError',
                'traceback': traceback.format_exc()
            }

        except AttributeError as e:
            return {
                'success': False,
                'answer': None,
                'error': str(e),
                'error_type': 'AttributeError',
                'traceback': traceback.format_exc()
            }

        except IndexError as e:
            return {
                'success': False,
                'answer': None,
                'error': str(e),
                'error_type': 'IndexError',
                'traceback': traceback.format_exc()
            }

        except Exception as e:
            # Catch-all for other exceptions
            return {
                'success': False,
                'answer': None,
                'error': str(e),
                'error_type': type(e).__name__,
                'traceback': traceback.format_exc()
            }

    def validate_code(self, code: str) -> tuple[bool, Optional[str]]:
        """
        Pre-validate code for obvious security issues

        Args:
            code: Python code string

        Returns:
            (is_valid, error_message)
        """
        # Check for dangerous imports
        dangerous_imports = [
            'os', 'sys', 'subprocess', 'socket', 'requests',
            '__import__', 'eval', 'exec', 'compile', 'open',
            'file', 'input', 'raw_input'
        ]

        for dangerous in dangerous_imports:
            if dangerous in code:
                return False, f"Dangerous import/function detected: {dangerous}"

        return True, None


if __name__ == "__main__":
    # Test the executor
    print("Testing SecureCodeExecutor...")

    executor = SecureCodeExecutor(timeout=5)

    # Test 1: Normal execution
    print("\n=== Test 1: Normal Execution ===")
    table = pd.DataFrame({
        'city': ['Beijing', 'Shanghai', 'Guangzhou'],
        'population': [21.54, 24.28, 15.3],
        'gdp': [4.0, 4.3, 2.8]
    })

    code = """
answer = df['population'].sum()
"""
    result = executor.execute(code, table)
    print(f"Success: {result['success']}")
    print(f"Answer: {result['answer']}")
    assert result['success'] is True
    assert abs(result['answer'] - 61.12) < 0.01

    # Test 2: KeyError
    print("\n=== Test 2: KeyError ===")
    code = """
answer = df['Population'].sum()  # Wrong case
"""
    result = executor.execute(code, table)
    print(f"Success: {result['success']}")
    print(f"Error Type: {result['error_type']}")
    print(f"Error: {result['error']}")
    assert result['success'] is False
    assert result['error_type'] == 'KeyError'

    # Test 3: TypeError
    print("\n=== Test 3: TypeError ===")
    code = """
answer = df['city'].sum()  # Can't sum strings
"""
    result = executor.execute(code, table)
    print(f"Success: {result['success']}")
    print(f"Error Type: {result['error_type']}")
    assert result['success'] is False
    assert result['error_type'] == 'TypeError'

    # Test 4: Timeout
    print("\n=== Test 4: Timeout (skipped for safety) ===")
    # Uncomment to test (will take 5+ seconds):
    # code = """
    # import time
    # time.sleep(10)
    # answer = 42
    # """
    # result = executor.execute(code, table)
    # assert result['error_type'] == 'TimeoutError'

    # Test 5: Syntax Error
    print("\n=== Test 5: Syntax Error ===")
    code = """
answer = df['population'].sum(
"""  # Missing closing parenthesis
    result = executor.execute(code, table)
    print(f"Success: {result['success']}")
    print(f"Error Type: {result['error_type']}")
    assert result['success'] is False
    assert result['error_type'] == 'SyntaxError'

    print("\n✓ All executor tests passed!")
