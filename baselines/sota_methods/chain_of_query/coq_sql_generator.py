"""
Chain-of-Query SQL Generator

Simplified implementation of the Chain-of-Query method for WikiTQ.
Based on: "Chain-of-Query: Unleashing the Power of LLMs in SQL-Aided Table Understanding"

Key features:
1. Generates SQL queries instead of Python code
2. Clause-by-Clause generation strategy
3. Natural language schema description
4. Error recovery with re-generation
"""

import pandas as pd
from typing import Dict, List, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChainOfQueryGenerator:
    """SQL query generator using Chain-of-Query strategy"""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", use_few_shot: bool = True):
        """
        Initialize CoQ generator

        Args:
            model_name: LLM model to use
            use_few_shot: Whether to use few-shot examples
        """
        self.model_name = model_name
        self.use_few_shot = use_few_shot

        # Initialize model
        if "gpt" in model_name.lower() or "claude" in model_name.lower():
            logger.info(f"Using API model: {model_name}")
            self.model = None  # Will use API calls
            self.use_api = True
        else:
            logger.info(f"Loading local model: {model_name}")
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.use_api = False
            logger.info("Model loaded successfully")

    def create_schema_description(self, df: pd.DataFrame) -> str:
        """
        Create natural language description of table schema

        This is a key innovation in CoQ - representing schema in natural language
        instead of raw column names.
        """
        schema_desc = f"This table has {len(df)} rows and {len(df.columns)} columns.\n\n"
        schema_desc += "Columns:\n"

        for col in df.columns:
            # Get column type
            dtype = df[col].dtype

            # Get sample values (first 3 unique non-null values)
            sample_values = df[col].dropna().unique()[:3].tolist()
            sample_str = ", ".join([str(v) for v in sample_values])

            # Infer semantic meaning from column name
            col_desc = col.replace('_', ' ').title()

            schema_desc += f"  - '{col}': {col_desc} ({dtype})\n"
            if sample_values:
                schema_desc += f"    Example values: {sample_str}\n"

        return schema_desc

    def get_few_shot_examples(self) -> str:
        """Get few-shot examples for SQL generation"""
        if not self.use_few_shot:
            return ""

        examples = """
Here are some examples of how to convert questions to SQL queries:

Example 1:
Question: "Who won in 2015?"
Table columns: ['year', 'winner', 'score']
SQL Query:
SELECT winner FROM table WHERE year = 2015

Example 2:
Question: "How many games were played in each year?"
Table columns: ['year', 'game', 'result']
SQL Query:
SELECT year, COUNT(*) as count FROM table GROUP BY year

Example 3:
Question: "What is the highest score?"
Table columns: ['player', 'score', 'year']
SQL Query:
SELECT MAX(score) FROM table

"""
        return examples

    def generate_sql_clause_by_clause(self, question: str, schema: str) -> str:
        """
        Generate SQL query clause by clause

        This is the core CoQ strategy: generate SELECT, FROM, WHERE, etc. incrementally
        """

        # Step 1: Generate SELECT clause
        select_prompt = f"""
{self.get_few_shot_examples()}
Given this question and table schema, generate the SELECT clause for a SQL query.

{schema}

Question: {question}

What columns should be selected? Generate ONLY the SELECT clause.
Format: SELECT <columns>

SELECT clause:
"""
        select_clause = self._call_llm(select_prompt).strip()

        # Ensure it starts with SELECT
        if not select_clause.upper().startswith("SELECT"):
            select_clause = "SELECT " + select_clause

        # Step 2: FROM clause (always "FROM table")
        from_clause = "FROM table"

        # Step 3: Generate WHERE clause (if needed)
        where_prompt = f"""
Given this question and the SELECT clause we've generated, do we need a WHERE clause to filter rows?

Question: {question}
Current SQL: {select_clause} {from_clause}

If a WHERE clause is needed, generate it. If not, respond with "NO_WHERE".

WHERE clause (or NO_WHERE):
"""
        where_response = self._call_llm(where_prompt).strip()

        if "NO_WHERE" not in where_response.upper():
            where_clause = where_response
            if not where_clause.upper().startswith("WHERE"):
                where_clause = "WHERE " + where_clause
        else:
            where_clause = ""

        # Step 4: Generate GROUP BY / ORDER BY / LIMIT if needed
        agg_prompt = f"""
Given this question and current SQL, do we need GROUP BY, ORDER BY, or LIMIT?

Question: {question}
Current SQL: {select_clause} {from_clause} {where_clause}

If needed, generate additional clauses. If not, respond with "NO_AGG".

Additional clauses (or NO_AGG):
"""
        agg_response = self._call_llm(agg_prompt).strip()

        if "NO_AGG" not in agg_response.upper():
            agg_clause = agg_response
        else:
            agg_clause = ""

        # Combine all clauses
        full_sql = f"{select_clause} {from_clause} {where_clause} {agg_clause}".strip()

        return full_sql

    def generate_sql_direct(self, question: str, schema: str) -> str:
        """
        Direct SQL generation (without clause-by-clause)
        Simpler but potentially less accurate
        """
        prompt = f"""
{self.get_few_shot_examples()}
Convert this natural language question into a SQL query.

Table Schema:
{schema}

Question: {question}

Important:
- Use "table" as the table name
- Generate a single, valid SQL query
- Return ONLY the SQL query, no explanations

SQL Query:
"""
        sql = self._call_llm(prompt).strip()

        # Clean up the SQL
        sql = sql.replace("```sql", "").replace("```", "").strip()

        return sql

    def _call_llm(self, prompt: str, max_new_tokens: int = 256) -> str:
        """Call LLM (API or local model)"""
        if self.use_api:
            # TODO: Implement API calls for GPT-4/Claude
            raise NotImplementedError("API models not yet implemented. Use local models.")
        else:
            # Use local model
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.1,  # Low temperature for more deterministic output
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract only the generated part (after the prompt)
            response = response[len(prompt):].strip()

            return response

    def generate_query(
        self,
        question: str,
        df: pd.DataFrame,
        use_clause_by_clause: bool = True
    ) -> Dict[str, Any]:
        """
        Generate SQL query for a question

        Args:
            question: Natural language question
            df: Input DataFrame
            use_clause_by_clause: Whether to use clause-by-clause generation

        Returns:
            Dict with 'sql', 'success', and optional 'error'
        """
        try:
            # Create schema description
            schema = self.create_schema_description(df)

            # Generate SQL
            if use_clause_by_clause:
                logger.info("Generating SQL clause-by-clause")
                sql = self.generate_sql_clause_by_clause(question, schema)
            else:
                logger.info("Generating SQL directly")
                sql = self.generate_sql_direct(question, schema)

            logger.info(f"Generated SQL: {sql}")

            return {
                'sql': sql,
                'success': True,
                'schema': schema
            }

        except Exception as e:
            logger.error(f"Error generating SQL: {e}")
            return {
                'sql': None,
                'success': False,
                'error': str(e)
            }


if __name__ == "__main__":
    # Test the generator
    import pandas as pd

    # Create sample table
    df = pd.DataFrame({
        'year': [2015, 2016, 2017, 2018],
        'winner': ['Team A', 'Team B', 'Team A', 'Team C'],
        'score': [95, 88, 92, 90]
    })

    # Initialize generator
    generator = ChainOfQueryGenerator(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        use_few_shot=True
    )

    # Test questions
    questions = [
        "Who won in 2015?",
        "How many times did Team A win?",
        "What is the highest score?"
    ]

    for q in questions:
        print(f"\nQuestion: {q}")
        result = generator.generate_query(q, df, use_clause_by_clause=True)
        print(f"SQL: {result['sql']}")
