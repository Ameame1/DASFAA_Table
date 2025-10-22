"""
TabFact Few-Shot Examples for AILS-NTUA Style Prompting

These examples demonstrate common TabFact verification patterns:
1. Single condition verification
2. Comparison/trend verification
3. Count-based verification

Each example includes:
- question: The statement to verify (True/False)
- df: Sample DataFrame
- columns_used: Which columns are needed
- column_types: Types of those columns
- answer_type: Always 'bool' for TabFact
- code: The solution code (function body only)
"""

import pandas as pd

# Example 1: Single condition verification
EXAMPLE_1 = {
    'question': "Is the following statement true? The revenue in 2020 was 100M",
    'df': pd.DataFrame({
        'year': [2018, 2019, 2020, 2021, 2022],
        'revenue': ['80M', '90M', '100M', '110M', '120M'],
        'profit': ['20M', '25M', '30M', '35M', '40M']
    }),
    'columns_used': ['year', 'revenue'],
    'column_types': ['int64', 'object'],
    'answer_type': 'bool',
    'code': """    result = (df[df['year'] == 2020]['revenue'].iloc[0] == '100M')
    return result"""
}

# Example 2: Trend verification (all values increasing)
EXAMPLE_2 = {
    'question': "Is the following statement true? Revenue increased every year",
    'df': pd.DataFrame({
        'year': [2018, 2019, 2020, 2021, 2022],
        'revenue': ['80M', '90M', '100M', '110M', '120M'],
        'profit': ['20M', '25M', '30M', '35M', '40M']
    }),
    'columns_used': ['revenue'],
    'column_types': ['object'],
    'answer_type': 'bool',
    'code': """    # Parse revenue to numbers
    revenues = df['revenue'].str.replace('M', '').astype(float)
    # Check if all differences are positive (increasing)
    result = all(revenues.diff()[1:] > 0)
    return result"""
}

# Example 3: Count-based verification
EXAMPLE_3 = {
    'question': "Is the following statement true? There are exactly 3 years with profit above 30M",
    'df': pd.DataFrame({
        'year': [2018, 2019, 2020, 2021, 2022],
        'revenue': ['80M', '90M', '100M', '110M', '120M'],
        'profit': ['20M', '25M', '30M', '35M', '40M']
    }),
    'columns_used': ['profit'],
    'column_types': ['object'],
    'answer_type': 'bool',
    'code': """    # Parse profit to numbers
    profits = df['profit'].str.replace('M', '').astype(float)
    # Count how many are above 30
    count = (profits > 30).sum()
    result = (count == 3)
    return result"""
}

# All examples in a list for easy iteration
TABFACT_EXAMPLES = [EXAMPLE_1, EXAMPLE_2, EXAMPLE_3]


if __name__ == "__main__":
    """Test the examples"""
    print("=== TabFact Few-Shot Examples ===\n")

    for i, example in enumerate(TABFACT_EXAMPLES, 1):
        print(f"Example {i}: {example['question']}")
        print(f"Columns used: {example['columns_used']}")
        print(f"Answer type: {example['answer_type']}")
        print(f"\nCode:")
        print(example['code'])

        # Execute the code to verify it works
        df = example['df']
        code_to_exec = f"def answer(df):\n{example['code']}\n\nresult = answer(df)"
        local_vars = {'df': df, 'pd': pd}
        exec(code_to_exec, local_vars)
        print(f"\nExpected output: {local_vars['result']}")
        print(f"Type: {type(local_vars['result'])}")
        print("\n" + "="*60 + "\n")
