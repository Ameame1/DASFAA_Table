"""
AILS-NTUA Style Post-processor

This module implements the critical post-processing step from AILS-NTUA:
extracting code until the first return statement and assembling complete functions.

References:
- AILS-NTUA paper: arXiv:2503.00435
- GitHub: https://github.com/AILS-NTUA/tabularqa
"""

import re
from typing import Optional, List


class TillReturnPostProcessor:
    """
    Extract code until the first return statement.

    This mimics AILS-NTUA's TillReturnLinePostProcessorMultipleIndents class.
    The key insight is that Coder models work best with fill-in-the-blank style
    prompting, where they complete a partial function rather than generating
    a complete function from scratch.

    Workflow:
    1. Prompt gives function header (def answer(df): ...)
    2. Model generates code snippet
    3. Post-processor extracts until first 'return' statement
    4. Assemble complete executable function
    """

    def __init__(
        self,
        base_indent: int = 4,
        return_indent: int = 4,
        first_prefix: str = ""
    ):
        """
        Args:
            base_indent: Base indentation level for function body (default 4 spaces)
            return_indent: Expected indentation of return statement (default 4 spaces)
            first_prefix: Optional prefix for first line (e.g., column comment)
        """
        self.base_indent = base_indent
        self.return_indent = return_indent
        self.first_prefix = first_prefix

    def extract_until_return(self, response: str) -> str:
        """
        Extract code until first return statement.

        This is the core post-processing step. It:
        1. Splits response into lines
        2. Finds the first line starting with 'return'
        3. Returns all lines up to and including that return statement

        Args:
            response: Raw model output (code snippet)

        Returns:
            Processed code snippet (ends at first return statement)

        Example:
            Input:
                "    result = df['col'].sum()\n    return result\n    print('extra')"
            Output:
                "    result = df['col'].sum()\n    return result"
        """
        lines = response.split("\n")
        extracted_lines = []
        indents = []

        for i, line in enumerate(lines):
            # Calculate indent level
            indent = len(line) - len(line.lstrip())
            indents.append(indent)

            # Strip and store
            stripped = line.strip()

            # Check if this is a return statement at expected indent level
            if line.startswith((' ' * self.return_indent) + "return"):
                # Found return, include this line and stop
                extracted_lines.append(line)
                break

            # Not a return statement, include and continue
            extracted_lines.append(line)

        return "\n".join(extracted_lines)

    def assemble_function(
        self,
        code_snippet: str,
        columns: List[str],
        function_name: str = "answer"
    ) -> str:
        """
        Assemble complete executable function from code snippet.

        Takes the extracted code snippet (which should end with a return statement)
        and wraps it in a proper function definition with column assignment.

        Args:
            code_snippet: Extracted code snippet (output of extract_until_return)
            columns: List of column names for the DataFrame
            function_name: Function name (default "answer")

        Returns:
            Complete executable function as a string

        Example:
            Input:
                code_snippet = "    result = df['year'].max()\n    return result"
                columns = ['year', 'team', 'score']
            Output:
                def answer(df: pd.DataFrame):
                    df.columns = ['year', 'team', 'score']
                    result = df['year'].max()
                    return result
        """
        # Add prefix to first line if specified
        snippet_lines = code_snippet.split("\n")
        if self.first_prefix and snippet_lines:
            snippet_lines[0] = self.first_prefix + snippet_lines[0].strip()

        # Ensure proper indentation for all lines
        indented_lines = []
        for line in snippet_lines:
            if line.strip():  # Non-empty line
                # Ensure it has at least base_indent spaces
                if not line.startswith(' ' * self.base_indent):
                    indented_lines.append(' ' * self.base_indent + line.lstrip())
                else:
                    indented_lines.append(line)
            else:
                indented_lines.append("")  # Keep empty lines

        indented_code = "\n".join(indented_lines)

        # Assemble complete function
        function = f"""def {function_name}(df: pd.DataFrame):
    df.columns = {columns}
{indented_code}
"""
        return function

    def process(
        self,
        model_output: str,
        columns: List[str],
        function_name: str = "answer"
    ) -> str:
        """
        Complete pipeline: extract + assemble.

        This is the main entry point combining both steps.

        Args:
            model_output: Raw model output
            columns: DataFrame columns
            function_name: Function name

        Returns:
            Complete executable function
        """
        # Step 1: Extract until return
        extracted = self.extract_until_return(model_output)

        # Step 2: Assemble function
        complete = self.assemble_function(extracted, columns, function_name)

        return complete


def clean_model_output(output: str) -> str:
    """
    Clean model output before processing.

    Removes common artifacts:
    - Code fence markers (```python, ```)
    - Extra whitespace
    - Function definition if model generated it (we'll add our own)

    Args:
        output: Raw model output

    Returns:
        Cleaned output
    """
    # Remove code fences
    output = re.sub(r'```python\s*', '', output)
    output = re.sub(r'```\s*', '', output)

    # Remove function definition if present (we'll add our own)
    output = re.sub(r'def\s+answer\s*\([^)]*\)\s*:\s*\n?', '', output)

    # Remove import statements
    lines = output.split('\n')
    cleaned_lines = [line for line in lines if not line.strip().startswith('import ')]

    return '\n'.join(cleaned_lines)


# Example usage and tests
if __name__ == "__main__":
    import pandas as pd

    print("=" * 70)
    print("AILS Post-Processor Test")
    print("=" * 70)

    processor = TillReturnPostProcessor(
        base_indent=4,
        return_indent=4,
        first_prefix="    # The columns used to answer the question: ['year', 'team']\n"
    )

    # Test 1: Basic extraction
    print("\n[Test 1] Basic extraction until return")
    print("-" * 70)

    model_output_1 = """    result = df[df['year'] == 2015]['team'].iloc[0]
    return result
    # This line should be ignored
    print("extra code that appears after return")
"""

    extracted_1 = processor.extract_until_return(model_output_1)
    print("Model output:")
    print(model_output_1)
    print("\nExtracted (until return):")
    print(extracted_1)

    # Test 2: Complete assembly
    print("\n\n[Test 2] Complete function assembly")
    print("-" * 70)

    complete_1 = processor.assemble_function(
        extracted_1,
        columns=['year', 'team', 'score', 'wins']
    )
    print("Complete function:")
    print(complete_1)

    # Test 3: One-step processing
    print("\n\n[Test 3] One-step process()")
    print("-" * 70)

    model_output_2 = """    # Count unique values
    result = df['team'].nunique()
    return result
"""

    complete_2 = processor.process(
        model_output_2,
        columns=['year', 'team', 'score']
    )
    print("Model output:")
    print(model_output_2)
    print("\nComplete function:")
    print(complete_2)

    # Test 4: Execute assembled function
    print("\n\n[Test 4] Execution test")
    print("-" * 70)

    # Create test DataFrame
    df = pd.DataFrame({
        'year': [2015, 2016, 2017, 2018, 2019],
        'team': ['Team A', 'Team B', 'Team A', 'Team C', 'Team A'],
        'score': [95, 88, 92, 90, 93]
    })

    print("Test DataFrame:")
    print(df)

    # Execute the function
    try:
        exec(complete_2)
        result = locals()['answer'](df)
        print(f"\nFunction result: {result}")
        print(f"Expected: 3 unique teams")
        print(f"Match: {result == 3}")
    except Exception as e:
        print(f"\nExecution error: {e}")

    # Test 5: Clean model output
    print("\n\n[Test 5] Cleaning model output")
    print("-" * 70)

    messy_output = """```python
import pandas as pd

def answer(df: pd.DataFrame):
    result = df['year'].max()
    return result
```
"""

    print("Messy output:")
    print(messy_output)

    cleaned = clean_model_output(messy_output)
    print("\nCleaned:")
    print(cleaned)

    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)
