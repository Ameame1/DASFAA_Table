"""
GPU test with offline mode (use only local cache)
"""

import sys
sys.path.append('.')

import os
os.environ['HF_HUB_OFFLINE'] = '1'  # Force offline mode
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

print("="*60)
print("OFFLINE GPU TEST (using local cache only)")
print("="*60)

# Check CUDA
print(f"\n1. CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   Device: {torch.cuda.get_device_name(0)}")

# Load tokenizer
print("\n2. Loading tokenizer (offline mode)...")
try:
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        trust_remote_code=True,
        local_files_only=True
    )
    print("   ✓ Tokenizer loaded from cache")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Load model
print("\n3. Loading model (offline mode)...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True
    )
    model.eval()
    print("   ✓ Model loaded from cache")

    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        print(f"   GPU memory used: {memory_allocated:.2f} GB")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

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

print(f"   Generated code:")
print("   " + code.replace("\n", "\n   "))

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
except Exception as e:
    print(f"   ✗ Execution failed: {e}")

print("\n" + "="*60)
print("✓ OFFLINE GPU TEST COMPLETE")
print("="*60)
print("\nThe model is working! You can now test with real data.")
