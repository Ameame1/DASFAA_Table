"""
AILS-NTUA Style Prompt Generator

Implements the detailed schema info and Chain-of-Thought prompting
strategy from AILS-NTUA.
"""

import pandas as pd
from typing import List, Dict, Any, Optional


# Simplified Few-shot examples (based on AILS-NTUA annotations_cot.json)
# We use 5 examples to balance performance and context length
AILS_FEWSHOT_EXAMPLES = [
    {
        "question": "How many players have position ST?",
        "columns_used": ["Position"],
        "column_types": ["category"],
        "answer_type": "number",
        "code": """    # Create a boolean mask for rows where Position is 'ST'
    is_st_position = df['Position'] == 'ST'

    # Count the rows where the mask is True
    st_player_count = df[is_st_position].shape[0]

    # Return the count of players with 'ST' position
    return st_player_count"""
    },
    {
        "question": "How many unique customers are there?",
        "columns_used": ["CustomerID"],
        "column_types": ["number"],
        "answer_type": "number",
        "code": """    # Count the number of unique values in the 'CustomerID' column
    unique_customers = df['CustomerID'].nunique()
    # Return the count of unique customers
    return unique_customers"""
    },
    {
        "question": "What are the top 3 scores?",
        "columns_used": ["score"],
        "column_types": ["number"],
        "answer_type": "list[number]",
        "code": """    # Retrieve the 'score' column, sort it in descending order, and get the top 3 scores
    top_scores = df['score'].nlargest(3).tolist()
    # Return the list of the top 3 scores
    return top_scores"""
    },
    {
        "question": "Is there any speed greater than 100 mph?",
        "columns_used": ["speed_mph"],
        "column_types": ["number"],
        "answer_type": "boolean",
        "code": """    # Check if any value in the 'speed_mph' column is greater than 100, and return True if so
    return (df['speed_mph'] > 100).any()"""
    },
    {
        "question": "What is the most common day of the week for incidents?",
        "columns_used": ["Incident Day of Week"],
        "column_types": ["category"],
        "answer_type": "category",
        "code": """    # Count occurencies of each day of the week
    days_counts = df['Incident Day of Week'].value_counts()

    # Return the day with the highest count
    return days_counts.idxmax()"""
    }
]


def get_detailed_schema_info(df: pd.DataFrame) -> str:
    """
    Generate detailed schema information in AILS-NTUA style.

    Format:
    #,Column,Non-Null Count,Dtype,Types of Elements,Values,Are all values unique?

    Args:
        df: Input DataFrame

    Returns:
        Formatted schema information string
    """
    output = ["#,Column,Non-Null Count,Dtype,Types of Elements,Values,Are all values unique?"]

    for i, col in enumerate(df.columns):
        # Basic info
        non_null_count = df[col].notnull().sum()
        dtype = str(df[col].dtype)

        # Python types
        try:
            python_types = df[col].apply(lambda x: type(x).__name__).unique().tolist()
            python_types_str = str(python_types)
        except:
            python_types_str = "mixed"

        # Sample values
        values_str = ""
        try:
            if df[col].dtype.name in ['object', 'category', 'string']:
                # For string/categorical columns, show unique values
                unique_values = df[col].dropna().unique()
                if len(unique_values) <= 5:
                    values_str = f"All values: {list(unique_values)}"
                else:
                    sample_values = list(unique_values[:5])
                    values_str = f"5 example values: {sample_values}"
            elif pd.api.types.is_numeric_dtype(df[col]):
                # For numeric columns, show range or examples
                if df[col].dropna().nunique() <= 10:
                    unique_values = sorted(df[col].dropna().unique())
                    values_str = f"Values: {list(unique_values)}"
        except:
            values_str = ""

        # Check uniqueness
        try:
            are_all_unique = df[col].nunique() == df[col].count()
        except:
            are_all_unique = "Unknown"

        # Append row
        output.append(f"{i},{col},{non_null_count},{dtype},{python_types_str},{values_str},{are_all_unique}")

    return "\n        ".join(output)


def get_data_preview(df: pd.DataFrame, num_rows: int = 5) -> str:
    """
    Get formatted data preview (first N rows).

    Args:
        df: Input DataFrame
        num_rows: Number of rows to show

    Returns:
        Formatted data preview string
    """
    preview = df.head(num_rows).to_string(max_colwidth=100)
    # Indent each line
    lines = preview.split('\n')
    indented = '\n        '.join(lines)
    return indented


def generate_ails_prompt(
    question: str,
    df: pd.DataFrame,
    num_rows: int = 5
) -> str:
    """
    Generate AILS-NTUA style prompt with detailed schema and data.

    Args:
        question: The question to answer
        df: Input DataFrame
        num_rows: Number of data rows to show

    Returns:
        Complete prompt string
    """
    schema_info = get_detailed_schema_info(df)
    data_preview = get_data_preview(df, num_rows)
    columns_list = list(df.columns)

    # Create a complete example to guide the model
    prompt = f'''You are an expert Python programmer. Generate a complete function to answer the question about the dataframe.

Question: {question}

Table Schema:
{schema_info}

First {num_rows} rows:
{data_preview}

Generate ONLY the Python function code below. The function must:
1. Use pandas operations on the df parameter
2. Return the final answer (number, string, boolean, or list)
3. Be complete and executable

```python
import pandas as pd

def answer(df: pd.DataFrame):
    df.columns = {columns_list}

    # Your solution here

    return result  # Must return the answer
```'''

    return prompt


def generate_ails_prompt_incomplete(
    question: str,
    df: pd.DataFrame,
    num_rows: int = 5,
    use_fewshot: bool = False,
    num_shots: int = 3
) -> str:
    """
    Generate AILS-NTUA style INCOMPLETE prompt (function header only).

    This is the CORRECT way to use AILS prompting with Coder models.
    The model completes the function body, then a post-processor
    extracts code until the first return statement.

    This mimics AILS-NTUA's actual implementation:
    https://github.com/AILS-NTUA/tabularqa

    Args:
        question: The question to answer
        df: Input DataFrame
        num_rows: Number of data rows to show
        use_fewshot: Whether to include Few-shot examples
        num_shots: Number of Few-shot examples to include (default 3)

    Returns:
        Incomplete prompt string (function header + schema, no body)
    """
    schema_info = get_detailed_schema_info(df)
    data_preview = get_data_preview(df, num_rows)
    columns_list = list(df.columns)

    prompt_parts = []

    # Add Few-shot examples if requested
    if use_fewshot:
        selected_examples = AILS_FEWSHOT_EXAMPLES[:min(num_shots, len(AILS_FEWSHOT_EXAMPLES))]

        for idx, ex in enumerate(selected_examples, 1):
            # Create example prompt (simplified, no actual df/schema)
            example_block = f'''# Example {idx}:
# TODO: complete the following function. It should give the answer to: {ex['question']}
def answer(df: pd.DataFrame):
    """
    DataFrame with relevant columns for this question.
    """

    df.columns = [...]  # Column names from the table

    # The columns used to answer the question: {ex['columns_used']}
    # The types of the columns used: {ex['column_types']}
    # The type of the answer: {ex['answer_type']}

{ex['code']}

'''
            prompt_parts.append(example_block)

    # Add the actual question (incomplete)
    # IMPORTANT: This prompt is INCOMPLETE by design
    # The model is expected to complete the function body
    # A post-processor will then extract until the first return statement
    actual_prompt = f'''# TODO: complete the following function. It should give the answer to: {question}
def answer(df: pd.DataFrame):
    """
        {schema_info}

        The first {num_rows} rows from the dataframe:
        {data_preview}
    """

    df.columns = {columns_list}

    # The columns used to answer the question:'''

    prompt_parts.append(actual_prompt)

    return '\n'.join(prompt_parts)


def generate_ails_fewshot_prompt(
    question: str,
    df: pd.DataFrame,
    examples: List[Dict[str, Any]],
    num_rows: int = 5
) -> str:
    """
    Generate few-shot prompt with examples in AILS-NTUA style.

    Args:
        question: The question to answer
        df: Input DataFrame
        examples: List of example dicts with keys:
            - 'question': example question
            - 'df': example dataframe
            - 'columns_used': columns used in solution
            - 'column_types': types of columns used
            - 'answer_type': type of the answer
            - 'code': solution code (body only, without def/return)
        num_rows: Number of data rows to show

    Returns:
        Complete few-shot prompt string
    """
    prompt_parts = []

    # Add examples
    for idx, ex in enumerate(examples, 1):
        ex_schema = get_detailed_schema_info(ex['df'])
        ex_preview = get_data_preview(ex['df'], num_rows)
        ex_columns = list(ex['df'].columns)

        example_prompt = f'''# Example {idx}:
# TODO: complete the following function. It should give the answer to: {ex['question']}
def answer(df: pd.DataFrame):
    """
        {ex_schema}

        The first {num_rows} rows from the dataframe:
        {ex_preview}
    """

    df.columns = {ex_columns}

    # The columns used to answer the question: {ex['columns_used']}
    # The types of the columns used: {ex['column_types']}
    # The type of the answer: {ex['answer_type']}

{ex['code']}
'''
        prompt_parts.append(example_prompt)

    # Add the actual question
    schema_info = get_detailed_schema_info(df)
    data_preview = get_data_preview(df, num_rows)
    columns_list = list(df.columns)

    actual_prompt = f'''# Now your turn:
# TODO: complete the following function. It should give the answer to: {question}
def answer(df: pd.DataFrame):
    """
        {schema_info}

        The first {num_rows} rows from the dataframe:
        {data_preview}
    """

    df.columns = {columns_list}

    # The columns used to answer the question:
    # Write your code here to answer the question
    # Make sure to return the answer at the end
    '''

    prompt_parts.append(actual_prompt)

    return '\n\n'.join(prompt_parts)


def generate_ails_error_fixing_prompt(
    question: str,
    df: pd.DataFrame,
    failed_code: str,
    error_msg: str,
    num_rows: int = 5
) -> str:
    """
    Generate error-fixing prompt in AILS-NTUA style.

    Args:
        question: The original question
        df: Input DataFrame
        failed_code: The code that failed
        error_msg: Error message from execution
        num_rows: Number of data rows to show

    Returns:
        Error-fixing prompt string
    """
    schema_info = get_detailed_schema_info(df)
    data_preview = get_data_preview(df, num_rows)
    columns_list = list(df.columns)

    # Generate original prompt
    original_prompt = f'''# TODO: complete the following function. It should give the answer to: {question}
def answer(df: pd.DataFrame):
    """
        {schema_info}

        The first {num_rows} rows from the dataframe:
        {data_preview}
    """

    df.columns = {columns_list}

    # Write your code here to answer the question
    # Make sure to return the answer at the end
    '''

    # Error-fixing prompt
    prompt = f'''# Help me fix the code error of the following function by rewriting it.
# Try to parse columns with list types yourself instead of using the `eval` function.
# Some lists may be written without the necessary '' to be parsed correctly.
# If rare or special characters are included as values, test equality by substring detection e.g. "query" in df[col].
# The function should return the answer to the question in the TODO comment below:

{original_prompt}

{failed_code}


# The function outputs the following error:
# {error_msg}

# Please rewrite the complete function:
{original_prompt}'''

    return prompt


if __name__ == "__main__":
    # Test the functions
    import pandas as pd

    # Create sample DataFrame
    df = pd.DataFrame({
        'year': [2015, 2016, 2017, 2018, 2019],
        'team': ['Team A', 'Team B', 'Team A', 'Team C', 'Team A'],
        'score': [95, 88, 92, 90, 93],
        'wins': [12, 10, 11, 9, 13]
    })

    print("=== Test 1: Detailed Schema Info ===")
    print(get_detailed_schema_info(df))

    print("\n=== Test 2: Zero-Shot Prompt ===")
    prompt = generate_ails_prompt("Who won in 2015?", df)
    print(prompt)

    print("\n=== Test 3: Few-Shot Prompt ===")
    examples = [
        {
            'question': "How many unique teams are there?",
            'df': df,
            'columns_used': ['team'],
            'column_types': ['object'],
            'answer_type': 'int',
            'code': "    return df['team'].nunique()"
        }
    ]
    fewshot = generate_ails_fewshot_prompt("What is the highest score?", df, examples)
    print(fewshot)

    print("\n=== Test 4: Error-Fixing Prompt ===")
    failed_code = "    result = df[df['Year'] == 2015]['team'].iloc[0]  # Wrong column name"
    error = "KeyError: 'Year'"
    fix_prompt = generate_ails_error_fixing_prompt("Who won in 2015?", df, failed_code, error)
    print(fix_prompt)
