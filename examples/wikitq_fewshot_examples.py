"""
WikiTQ Few-Shot Examples for AILS-NTUA Style Prompting

These examples demonstrate common WikiTQ question patterns:
1. Simple WHERE condition query
2. Aggregation (COUNT)
3. MAX/MIN query

Each example includes:
- question: The natural language question
- df: Sample DataFrame
- columns_used: Which columns are needed
- column_types: Types of those columns
- answer_type: Expected answer type
- code: The solution code (function body only)
"""

import pandas as pd

# Example 1: Simple WHERE condition query
EXAMPLE_1 = {
    'question': "What was the attendance in 2015?",
    'df': pd.DataFrame({
        'year': [2013, 2014, 2015, 2016, 2017],
        'team': ['Team A', 'Team B', 'Team A', 'Team C', 'Team B'],
        'attendance': [45000, 42000, 48000, 46000, 44000],
        'result': ['Win', 'Loss', 'Win', 'Win', 'Loss']
    }),
    'columns_used': ['year', 'attendance'],
    'column_types': ['int64', 'int64'],
    'answer_type': 'int',
    'code': """    result = df[df['year'] == 2015]['attendance'].iloc[0]
    return result"""
}

# Example 2: Aggregation query (COUNT)
EXAMPLE_2 = {
    'question': "How many times did Team A appear?",
    'df': pd.DataFrame({
        'year': [2013, 2014, 2015, 2016, 2017],
        'team': ['Team A', 'Team B', 'Team A', 'Team C', 'Team A'],
        'attendance': [45000, 42000, 48000, 46000, 44000],
        'result': ['Win', 'Loss', 'Win', 'Win', 'Loss']
    }),
    'columns_used': ['team'],
    'column_types': ['object'],
    'answer_type': 'int',
    'code': """    result = (df['team'] == 'Team A').sum()
    return result"""
}

# Example 3: MAX query with WHERE
EXAMPLE_3 = {
    'question': "What is the highest attendance?",
    'df': pd.DataFrame({
        'year': [2013, 2014, 2015, 2016, 2017],
        'team': ['Team A', 'Team B', 'Team A', 'Team C', 'Team B'],
        'attendance': [45000, 42000, 48000, 46000, 44000],
        'result': ['Win', 'Loss', 'Win', 'Win', 'Loss']
    }),
    'columns_used': ['attendance'],
    'column_types': ['int64'],
    'answer_type': 'int',
    'code': """    result = df['attendance'].max()
    return result"""
}

# All examples in a list for easy iteration
WIKITQ_EXAMPLES = [EXAMPLE_1, EXAMPLE_2, EXAMPLE_3]


if __name__ == "__main__":
    """Test the examples"""
    print("=== WikiTQ Few-Shot Examples ===\n")

    for i, example in enumerate(WIKITQ_EXAMPLES, 1):
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
