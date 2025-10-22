"""
SQL Query Executor for Chain-of-Query

Executes SQL queries on pandas DataFrames using pandasql
"""

import pandas as pd
import pandasql as psql
from typing import Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SQLExecutor:
    """Execute SQL queries on pandas DataFrames"""

    def __init__(self, timeout: int = 10):
        """
        Initialize SQL executor

        Args:
            timeout: Execution timeout in seconds
        """
        self.timeout = timeout

    def execute_query(self, sql: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Execute SQL query on DataFrame

        Args:
            sql: SQL query string
            df: Input DataFrame

        Returns:
            Dict with 'result', 'success', and optional 'error'
        """
        try:
            # Rename DataFrame to 'table' for SQL execution
            table = df.copy()

            # Execute SQL using pandasql
            logger.info(f"Executing SQL: {sql}")
            result_df = psql.sqldf(sql, locals())

            # Extract answer from result
            answer = self._extract_answer(result_df)

            logger.info(f"SQL execution successful. Answer: {answer}")

            return {
                'result': answer,
                'result_df': result_df,
                'success': True
            }

        except Exception as e:
            error_msg = str(e)
            logger.error(f"SQL execution failed: {error_msg}")

            return {
                'result': None,
                'success': False,
                'error': error_msg,
                'error_type': type(e).__name__
            }

    def _extract_answer(self, result_df: pd.DataFrame) -> Any:
        """
        Extract answer from result DataFrame

        Args:
            result_df: Result DataFrame from SQL query

        Returns:
            Extracted answer (scalar value, list, or formatted string)
        """
        if result_df is None or len(result_df) == 0:
            return None

        # If single cell result, return scalar
        if result_df.shape == (1, 1):
            return result_df.iloc[0, 0]

        # If single row, return as dict or first value
        if len(result_df) == 1:
            if len(result_df.columns) == 1:
                return result_df.iloc[0, 0]
            else:
                return result_df.iloc[0].to_dict()

        # If single column, return as list
        if len(result_df.columns) == 1:
            return result_df.iloc[:, 0].tolist()

        # Multiple rows and columns - return as formatted string or first value
        if len(result_df) <= 5:
            # Small result - return as formatted string
            return result_df.to_string(index=False)
        else:
            # Large result - return first value
            return result_df.iloc[0, 0]

    def validate_sql(self, sql: str) -> Dict[str, Any]:
        """
        Validate SQL query syntax

        Args:
            sql: SQL query string

        Returns:
            Dict with 'valid' and optional 'error'
        """
        try:
            # Basic syntax validation
            sql_upper = sql.upper().strip()

            # Check for required SELECT
            if not sql_upper.startswith("SELECT"):
                return {
                    'valid': False,
                    'error': "SQL must start with SELECT"
                }

            # Check for FROM clause
            if "FROM" not in sql_upper:
                return {
                    'valid': False,
                    'error': "SQL must contain FROM clause"
                }

            # Check for dangerous operations
            dangerous_keywords = ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'CREATE']
            for keyword in dangerous_keywords:
                if keyword in sql_upper:
                    return {
                        'valid': False,
                        'error': f"Dangerous operation not allowed: {keyword}"
                    }

            return {'valid': True}

        except Exception as e:
            return {
                'valid': False,
                'error': str(e)
            }


class CoQExecutor:
    """
    Complete Chain-of-Query execution system

    Combines SQL generation and execution with error recovery
    """

    def __init__(self, generator, max_retries: int = 2):
        """
        Initialize CoQ executor

        Args:
            generator: ChainOfQueryGenerator instance
            max_retries: Maximum number of retry attempts on failure
        """
        self.generator = generator
        self.sql_executor = SQLExecutor()
        self.max_retries = max_retries

    def answer_question(self, question: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Answer a question using Chain-of-Query

        Args:
            question: Natural language question
            df: Input DataFrame

        Returns:
            Dict with answer, SQL, success status, and metadata
        """
        attempts = 0
        last_error = None

        while attempts <= self.max_retries:
            attempts += 1

            logger.info(f"Attempt {attempts}/{self.max_retries + 1}")

            # Generate SQL
            gen_result = self.generator.generate_query(question, df)

            if not gen_result['success']:
                last_error = gen_result.get('error', 'Unknown generation error')
                continue

            sql = gen_result['sql']

            # Validate SQL
            validation = self.sql_executor.validate_sql(sql)
            if not validation['valid']:
                last_error = validation['error']
                logger.warning(f"Invalid SQL: {last_error}")
                continue

            # Execute SQL
            exec_result = self.sql_executor.execute_query(sql, df)

            if exec_result['success']:
                return {
                    'answer': exec_result['result'],
                    'sql': sql,
                    'success': True,
                    'attempts': attempts,
                    'result_df': exec_result.get('result_df')
                }
            else:
                last_error = exec_result.get('error', 'Unknown execution error')
                logger.warning(f"Execution failed: {last_error}")

        # All attempts failed
        return {
            'answer': None,
            'sql': None,
            'success': False,
            'attempts': attempts,
            'error': last_error
        }


if __name__ == "__main__":
    # Test the executor
    import sys
    sys.path.insert(0, '/media/liuyu/DataDrive/DASFAA-Table/baselines/sota_methods/chain_of_query')
    from coq_sql_generator import ChainOfQueryGenerator

    # Create sample table
    df = pd.DataFrame({
        'year': [2015, 2016, 2017, 2018],
        'winner': ['Team A', 'Team B', 'Team A', 'Team C'],
        'score': [95, 88, 92, 90]
    })

    # Initialize system
    print("Loading model...")
    generator = ChainOfQueryGenerator(model_name="Qwen/Qwen2.5-7B-Instruct")
    executor = CoQExecutor(generator)

    # Test questions
    questions = [
        "Who won in 2015?",
        "How many times did Team A win?",
        "What is the highest score?"
    ]

    for q in questions:
        print(f"\n{'=' * 60}")
        print(f"Question: {q}")
        result = executor.answer_question(q, df)
        print(f"SQL: {result.get('sql')}")
        print(f"Answer: {result.get('answer')}")
        print(f"Success: {result['success']}")
        print(f"Attempts: {result['attempts']}")
