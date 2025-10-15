"""
Quick GPU test - minimal version to check if model loads
"""

import sys
sys.path.append('.')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

print("="*60)
print("QUICK GPU TEST")
print("="*60)

# Check CUDA
print(f"\n1. CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   Device: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Load tokenizer
print("\n2. Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    trust_remote_code=True
)
print("   ✓ Tokenizer loaded")

# Load model
print("\n3. Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    device_map="cuda:0",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
model.eval()
print("   ✓ Model loaded")

if torch.cuda.is_available():
    memory_allocated = torch.cuda.memory_allocated() / 1024**3
    print(f"   GPU memory used: {memory_allocated:.2f} GB")

# Quick generation test
print("\n4. Testing code generation...")
table = pd.DataFrame({
    'City': ['Beijing', 'Shanghai'],
    'Population': [21.54, 24.28]
})

question = "What is the total population?"

table_str = table.to_string(index=False)
prompt = f"""Generate Python code using pandas to answer the question.

Table:
{table_str}

Question: {question}

Code (use variable 'df' for the table, store answer in 'answer'):
```python
"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

print(f"   Input tokens: {inputs['input_ids'].shape[1]}")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.2,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Extract code
code = generated_text.split(prompt)[-1].strip()
if "```" in code:
    code = code.split("```")[0]

print(f"   ✓ Generated code:")
print("   " + "\n   ".join(code.split("\n")))

# Test execution
print("\n5. Testing code execution...")
try:
    df = table.copy()
    exec_globals = {'df': df, 'pd': pd}
    exec_locals = {}
    exec(code, exec_globals, exec_locals)
    answer = exec_locals.get('answer')
    print(f"   ✓ Code executed successfully")
    print(f"   Answer: {answer}")
    print(f"   Expected: 45.82")
except Exception as e:
    print(f"   ✗ Execution failed: {e}")

print("\n" + "="*60)
print("✓ QUICK GPU TEST COMPLETE")
print("="*60)
