"""
Code Generator using Qwen2.5-Coder-7B-Instruct
Generates Python code for table question answering
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QwenCodeGenerator:
    """
    Code generator using Qwen2.5-Coder-7B-Instruct model
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        load_in_8bit: bool = False
    ):
        """
        Initialize code generator

        Args:
            model_name: HuggingFace model name
            device: Device to load model on
            load_in_8bit: Whether to load model in 8-bit precision
        """
        self.model_name = model_name
        self.device = device
        self.load_in_8bit = load_in_8bit

        logger.info(f"Loading model: {model_name}")
        logger.info(f"Device: {device}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # Load model
        if load_in_8bit:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                load_in_8bit=True,
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True
            )

        self.model.eval()
        logger.info("Model loaded successfully")

    def generate_code(
        self,
        table: pd.DataFrame,
        question: str,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.95
    ) -> str:
        """
        Generate Python code for the given table and question

        Args:
            table: Input DataFrame
            question: Question to answer
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling

        Returns:
            Generated Python code as string
        """
        prompt = self._create_prompt(table, question)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode and extract code
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        code = self._extract_code(generated_text, prompt)

        return code

    def generate_from_repair_prompt(
        self,
        repair_prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.2
    ) -> str:
        """
        Generate code from a repair prompt (for error fixing)

        Args:
            repair_prompt: Repair prompt from PromptGenerator
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated Python code
        """
        inputs = self.tokenizer(repair_prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        code = self._extract_code(generated_text, repair_prompt)

        return code

    def _create_prompt(self, table: pd.DataFrame, question: str) -> str:
        """
        Create initial code generation prompt
        Improved with techniques from AILS-NTUA:
        - Column selection
        - Unique values
        - Function template
        """
        # 1. Column selection (simple keyword matching)
        question_lower = question.lower()
        question_words = set(question_lower.split())

        selected_columns = []
        for col in table.columns:
            col_words = set(str(col).lower().split())
            if question_words & col_words:  # Intersection
                selected_columns.append(col)

        # If no columns matched, use all columns
        if not selected_columns:
            selected_columns = list(table.columns)

        # 2. Get unique values for selected columns (max 10 values per column, max 3 columns)
        unique_values_info = []
        for col in selected_columns[:3]:
            try:
                unique_vals = table[col].unique()[:10].tolist()
                unique_values_info.append(f"# {col}: {unique_vals}")
            except:
                continue

        unique_values_str = "\n    ".join(unique_values_info) if unique_values_info else "# No unique values info"

        # 3. Table preview (smaller, only selected columns if possible)
        try:
            if len(selected_columns) <= 5:
                preview_df = table[selected_columns].head(3)
            else:
                preview_df = table.head(3)
            table_preview = preview_df.to_string(index=False)
        except:
            table_preview = table.head(3).to_string(index=False)

        # 4. Create improved prompt with function template
        prompt = f"""You are a Python expert for Table Question Answering.

Table Information:
Columns: {list(table.columns)}
Selected Columns (relevant to question): {selected_columns}

Unique Values (sample):
    {unique_values_str}

Table Preview:
{table_preview}

Question: {question}

Generate a Python function that answers the question:

```python
import pandas as pd

def answer(df: pd.DataFrame):
    # Set column names explicitly
    df.columns = {list(table.columns)}

    # Your solution here
    result = ...

    return result
```

Requirements:
1. Return ONLY the function code, no explanations
2. Use exact column names as provided
3. Return answer as: number, string, boolean, or list
4. Keep code concise and correct

Python code:
```python
"""
        return prompt

    def _extract_code(self, generated_text: str, prompt: str) -> str:
        """
        Extract code from generated text
        Now expects: def answer(df: pd.DataFrame): ...

        Args:
            generated_text: Full generated text
            prompt: Original prompt

        Returns:
            Extracted Python code
        """
        # Remove prompt from output
        if prompt in generated_text:
            code = generated_text.split(prompt)[-1]
        else:
            code = generated_text

        # Extract code from markdown if present
        if "```python" in code:
            code = code.split("```python")[-1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0] if code.count("```") >= 2 else code.split("```")[0]

        # Clean up
        code = code.strip()

        # Check if it's a function definition
        if "def answer(" in code:
            # Extract the function definition
            lines = code.split('\n')
            # Keep the function
            return code
        else:
            # Old style code without function, wrap it
            logger.warning("Generated code doesn't contain 'def answer()', attempting to use as-is")
            # Still check for answer variable
            if "answer" not in code and "result" not in code:
                logger.warning("Generated code doesn't contain 'answer' or 'result' variable")

        return code


if __name__ == "__main__":
    print("✓ QwenCodeGenerator defined!")
    print("Note: Model loading requires GPU and will download ~14GB model")
