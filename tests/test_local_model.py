"""
Quick test with local Qwen2.5-7B-Instruct model
"""

import sys
sys.path.append('.')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

print("="*60)
print("TESTING WITH LOCAL QWEN2.5-7B-INSTRUCT")
print("="*60)

# Check CUDA
print(f"\n1. CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   Device: {torch.cuda.get_device_name(0)}")
    print(f"   Total memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Load model
print("\n2. Loading Qwen2.5-7B-Instruct from local cache...")
try:
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        trust_remote_code=True
    )
    print("   ✓ Tokenizer loaded")

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model.eval()
    print("   ✓ Model loaded")

    if torch.cuda.is_available():
        memory_used = torch.cuda.memory_allocated() / 1024**3
        print(f"   GPU memory used: {memory_used:.2f} GB")

except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test code generation
print("\n3. Testing code generation...")
table = pd.DataFrame({
    'City': ['Beijing', 'Shanghai', 'Guangzhou'],
    'Population': [21.54, 24.28, 15.30]
})

question = "What is the total population?"

# Create prompt
table_str = table.to_string(index=False)
prompt = f"""You are a Python expert. Generate Python code using pandas to answer the question.

Table:
{table_str}

Columns: {list(table.columns)}

Question: {question}

Generate Python code that:
1. Uses pandas DataFrame 'df' (already loaded)
2. Stores the final answer in variable 'answer'

Python code:
```python
"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

print(f"   Generating code...")

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

code = code.strip()

print(f"\n   Generated code:")
for line in code.split('\n'):
    print(f"   {line}")

# Execute code
print("\n4. Executing code...")
try:
    df = table.copy()
    exec_globals = {'df': df, 'pd': pd}
    exec_locals = {}
    exec(code, exec_globals, exec_locals)
    answer = exec_locals.get('answer')
    print(f"   ✓ Execution successful")
    print(f"   Answer: {answer}")
    print(f"   Expected: 61.12")

    if answer is not None and abs(float(answer) - 61.12) < 0.01:
        print(f"   ✓ Answer is CORRECT!")

except Exception as e:
    print(f"   ✗ Execution failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("✓ LOCAL MODEL TEST COMPLETE")
print("="*60)
print("\n您的 Qwen2.5-7B-Instruct 模型已就绪!")
print("现在可以运行完整的系统测试了。")
