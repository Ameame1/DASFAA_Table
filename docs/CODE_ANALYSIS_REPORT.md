# 现有代码库分析报告

## 📚 已下载的代码库

### 1. OpenCodeInterpreter ✅
- **位置**: `/media/liuyu/DataDrive/OpenCodeInterpreter/`
- **Star**: ~2.6k GitHub stars
- **功能**: 代码生成 + 执行 + 迭代refinement

### 2. Tabular_QA (AILS-NTUA/LyS Team) ✅
- **位置**: `/media/liuyu/DataDrive/Tabular_QA/`
- **功能**: SemEval 2025 Task 8 参赛代码
- **附带**: 论文PDF (`Tabular_QA.pdf`)

---

## 🔍 关键发现和对比

### OpenCodeInterpreter 的关键设计

#### 1. 安全执行机制 (`utils/const.py`)

```python
# 他们的GUARD_CODE - 禁用危险操作
os.kill = lambda *args: raise PermissionError
os.system = lambda *args: raise PermissionError
os.remove = lambda *args: raise PermissionError
# ... 禁用了50+个危险函数
```

**vs 我们的实现**:
```python
# 我们的safe_builtins - 白名单机制
self.safe_builtins = {
    'abs': abs, 'all': all, ..., 'isinstance': isinstance
}
```

**对比**:
- ✅ **他们**: 黑名单 (禁用危险函数) - 使用Jupyter notebook执行
- ✅ **我们**: 白名单 (只允许安全函数) - 使用exec()执行
- 📊 **建议**: 我们的方法更安全，但应该添加代码清理功能

#### 2. System Prompt (`utils/const.py`)

```python
CODE_INTERPRETER_SYSTEM_PROMPT = """You are an AI code interpreter.
Your goal is to help users do a variety of jobs by executing Python code.

You should:
1. Comprehend the user's requirements carefully & to the letter.
2. Give a brief description for what you plan to do & call the provided function to run code.
3. Provide results analysis based on the execution output.
4. If error occurred, try to fix it.  # ← 迭代修复
5. Response in the same language as the user."""
```

**vs 我们的Prompt** (`src/baselines/code_generator.py:142`):
```python
prompt = f"""You are a Python expert. Generate Python code using pandas...
Generate Python code that:
1. Uses pandas DataFrame 'df' (already loaded)
2. Answers the question accurately
3. Stores the final answer in variable 'answer'
4. Handles edge cases and errors
"""
```

**对比**:
- ✅ **他们**: 强调迭代修复 "If error occurred, try to fix it"
- ⚠️ **我们**: 没有在prompt中明确错误修复
- 📊 **建议**: 在初始生成prompt中就告诉模型"代码可能会被多次修复"

---

### AILS-NTUA (Tabular_QA) 的关键设计

#### 1. 错误修复策略 (`src/code_fixer.py`)

他们的CodeFixer非常简单直接:

```python
def code_fix(self, response: str, error: str):
    messages = [
        {"role": "system", "content": """
            You are a Python-powered Tabular Data Question-Answering System.

            Task: Fix the Python code to address a query

            Input:
                code: The Python code that needs to be fixed
                error: The error message

            Output:
                Return only the Python code (no explanations)
        """},
        {"role": "user", "content": f"""
            Code: {response}
            Error: {error}
        """}
    ]

    output = self.pipe(messages, max_new_tokens=2048)
    return output
```

**vs 我们的诊断系统** (4层):
```python
# Layer 1: Error Classification
error_class = self.classifier.classify(execution_result)

# Layer 2: Root Cause Analysis
root_cause = self.root_cause_analyzer.analyze(...)

# Layer 3: Strategy Selection
strategy = self.strategy_selector.select_strategy(...)

# Layer 4: Prompt Generation
repair_prompt = self.prompt_generator.generate(...)
```

**对比**:
- ✅ **他们**: 简单直接 - 直接将错误+代码给LLM修复
- ✅ **我们**: 复杂细致 - 4层诊断，规则based策略
- 📊 **优劣**:
  - 他们: 简单但依赖LLM能力，可能重复相同错误
  - 我们: 复杂但更系统化，可以针对性修复
- 📊 **建议**: **保留我们的4层系统** (这是创新点！)，但借鉴他们的简洁prompt

#### 2. 迭代逻辑 (`main.py:145-175`)

```python
def example_postprocess(response: str, dataset: str):
    try:
        result = execute_answer_code(response, df)
        return (response, result)
    except Exception as e:
        # 只修复一次！
        code_fixer = CodeFixer(pipe)
        response_fixed = code_fixer.code_fix(response, str(e))
        try:
            result = execute_answer_code(response_fixed, df)
            return (response_fixed, result)
        except Exception as code_error:
            return (response_fixed, f"__CODE_ERROR__: {code_error}")
```

**关键发现**:
- ⚠️ **他们只迭代1次修复**（原始代码 → 修复1次 → 失败就返回错误）
- 论文中说"最多2次迭代"，但代码只有1次修复

**vs 我们的迭代** (`src/system/table_qa_system.py`):
```python
for iteration in range(self.max_iterations):  # 默认3次
    if iteration == 0:
        code = self.code_generator.generate_code(table, question)
    else:
        diagnosis = self.diagnostic_system.diagnose(...)
        code = self.code_generator.generate_from_repair_prompt(...)

    exec_result = self.code_executor.execute(code, table)
    if exec_result['success']:
        return result
```

**对比**:
- ✅ **他们**: 最多1次修复 (论文说2次迭代 = 1次生成+1次修复)
- ✅ **我们**: 最多3次迭代
- 📊 **建议**: 我们的更灵活，但可能需要优化停止条件

#### 3. Prompt模板 (`main.py:62-96`)

他们的完整prompt:

```python
def _format_prompt(row, df, selected_columns, columns_unique):
    return f"""
    Role and Context:
    You are a Python-powered Tabular Data Question-Answering System.

    Task: Generate Python code to address a query based on the provided dataset.

    Output must:
    - Use the dataset as given
    - Adhere to strict Python syntax
    - Retain original column names  # ← 重要！

    Code Template:
    import pandas as pd
    def answer(df: pd.DataFrame) -> None:
        df.columns = {list(df.columns)}  # 显式列名
        # The columns used: {selected_columns}  # ← 列选择
        {columns_unique}  # ← 列的unique values
        # Your solution goes here
        ...

    Question: {row["question"]}
    """
```

**关键技巧**:
1. **显式列名**: `df.columns = ['col1', 'col2', ...]` - 避免列名问题
2. **列选择**: 只提供相关的列 - 减少token和混淆
3. **Unique values**: 提供列的唯一值 - 帮助理解数据
4. **函数模板**: 要求生成`def answer(df)` - 规范输出

**vs 我们的Prompt**:
```python
prompt = f"""Generate Python code using pandas...
Table:
{table.head(5).to_string(index=False)}

Columns: {list(table.columns)}
Data types: {dict(table.dtypes)}

Question: {question}

Python code:
```python
"""
```

**对比**:
- ✅ **他们**:
  - 使用ColumnSelector选择相关列
  - 提供unique values
  - 要求def answer()函数格式
- ⚠️ **我们**:
  - 提供完整表格preview
  - 没有列选择
  - 没有函数模板
- 📊 **建议**: **借鉴他们的prompt技巧**，特别是:
  - 添加列选择
  - 提供unique values
  - 使用函数模板

#### 4. 列清理 (`main.py:23-37`)

```python
def clean_column_names(df):
    def clean_name(name):
        # Remove emojis
        name = re.sub(r"[^\w\s,.<>@]", "", name, flags=re.UNICODE)
        # Remove text in < >
        name = re.sub(r"<[^>]*>", "", name)
        # Remove Twitter mentions
        name = re.sub(r"@\w+", "", name)
        return name.strip()

    df.columns = [clean_name(col) for col in df.columns]
    return df
```

**vs 我们的实现**: ❌ 我们没有列名清理

**建议**: **添加列名清理** - WikiTQ数据可能有特殊字符

---

## 📊 我们的优势和劣势

### ✅ 我们的优势

1. **4层诊断系统** - 比AILS更系统化
   - Layer 1: Error Classification (4大类错误)
   - Layer 2: Root Cause Analysis (9种根因)
   - Layer 3: Strategy Selection (5个策略，可扩展20个)
   - Layer 4: Prompt Generation (结构化修复提示)

2. **模块化设计** - 易于扩展和测试
   - 每个组件独立
   - GRPO接口预留

3. **更安全的执行** - 白名单机制
   - 限制builtins
   - 代码清理 (移除import)

4. **灵活的迭代** - 可配置迭代次数
   - 默认3次 vs 他们的1次

### ⚠️ 我们的劣势

1. **Prompt不够优化**
   - ❌ 没有列选择
   - ❌ 没有unique values
   - ❌ 没有函数模板
   - ❌ 没有显式列名

2. **缺少列名清理**
   - ❌ WikiTQ数据可能有emoji、特殊字符

3. **错误修复prompt过于复杂**
   - ⚠️ 我们生成很长的诊断信息
   - ⚠️ AILS直接简单: "Code: ..., Error: ..."

4. **没有列选择器**
   - ❌ AILS有ColumnSelector (只提供相关列)
   - ❌ 我们提供完整表格 (token浪费)

---

## 🎯 改进建议 (优先级排序)

### 🔥 立即改进 (本周)

#### 1. 优化代码生成Prompt ⭐⭐⭐
借鉴AILS的技巧:

```python
# 添加到 src/baselines/code_generator.py

def _create_prompt(self, table: pd.DataFrame, question: str) -> str:
    # 1. 列选择 (简单版: 基于关键词)
    question_words = set(question.lower().split())
    selected_columns = [col for col in table.columns
                       if any(word in col.lower() for word in question_words)]
    if not selected_columns:
        selected_columns = list(table.columns)

    # 2. Unique values (前10个)
    unique_values = {}
    for col in selected_columns[:5]:  # 最多5列
        unique_values[col] = table[col].unique()[:10].tolist()

    # 3. 新的prompt模板
    prompt = f"""You are a Python expert for Table QA.

Table Columns: {list(table.columns)}
Selected Columns: {selected_columns}
Unique Values: {unique_values}

Question: {question}

Generate a Python function:
```python
import pandas as pd

def answer(df: pd.DataFrame):
    df.columns = {list(table.columns)}
    # Your solution here
    result = ...
    return result
```

Return ONLY the function code, no explanations.
"""
    return prompt
```

#### 2. 添加列名清理 ⭐⭐
```python
# 添加到 src/data/data_loader.py

import re

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    def clean_name(name):
        # Remove emojis and special chars
        name = re.sub(r"[^\w\s,.<>@-]", "", str(name), flags=re.UNICODE)
        # Remove text in brackets
        name = re.sub(r"<[^>]*>", "", name)
        name = re.sub(r"\([^)]*\)", "", name)
        return name.strip()

    df.columns = [clean_name(col) for col in df.columns]
    return df

# 在load后调用
df = clean_column_names(df)
```

#### 3. 简化错误修复Prompt ⭐
```python
# 修改 src/diagnosis/prompt_generator.py

def generate(self, ...):
    # 简单版本 (参考AILS)
    prompt = f"""Fix the Python code for table question answering.

Original Question: {question}
Table Columns: {list(table.columns)}

Previous Code:
{code}

Error:
{execution_result['error_type']}: {execution_result['error']}

Generate the FIXED code:
```python
def answer(df: pd.DataFrame):
    df.columns = {list(table.columns)}
    # Fixed solution
    ...
```

Return ONLY the fixed function code.
"""
    return prompt
```

### 📅 中期改进 (下周)

#### 4. 实现ColumnSelector ⭐⭐
参考AILS的`src/column_selector.py`:
- 使用LLM选择相关列
- 减少token使用
- 提高准确度

#### 5. 改进代码清理 ⭐
目前我们的clean_code只移除import，应该:
- 提取函数定义
- 验证语法
- 标准化格式

### 🔮 长期改进 (月底)

#### 6. A/B测试不同Prompt
- 简单版 (AILS风格)
- 详细版 (我们的4层诊断)
- 对比性能

#### 7. 实现动态迭代停止
- 检测重复错误
- 学习停止时机

---

## 💡 创新点保留

### 我们应该保留的优势:

1. **4层诊断系统** ⭐⭐⭐
   - 这是我们的**核心创新**
   - AILS只有简单的code_fix
   - 可以在论文中强调: "We propose a hierarchical diagnostic system with 4 layers..."

2. **多策略修复** ⭐⭐
   - 我们有5个strategy，可扩展到20个
   - AILS只有1个通用修复
   - 论文可说: "We design 20 specialized repair strategies..."

3. **GRPO接口** ⭐⭐
   - 我们预留了GRPO训练接口
   - 结合Table-R1的思想
   - 这是研究方向的核心

### 我们应该借鉴的:

1. **简洁的Prompt** - AILS的prompt更高效
2. **列选择** - 减少token，提高准确度
3. **列名清理** - 处理真实数据的噪声
4. **函数模板** - 规范输出格式

---

## 📋 行动计划

### 今天完成:
1. ✅ 下载OpenCodeInterpreter
2. ✅ 下载Tabular_QA
3. ✅ 分析关键代码
4. ⬜ **改进代码生成Prompt** (借鉴AILS)
5. ⬜ **添加列名清理**

### 明天完成:
1. ⬜ 简化错误修复Prompt
2. ⬜ 在WikiTQ上重新测试
3. ⬜ 对比改进前后性能

### 本周完成:
1. ⬜ 实现ColumnSelector
2. ⬜ 运行50-100样本评估
3. ⬜ 撰写初步实验结果

---

## 📝 论文中如何说明

在论文中我们应该这样表述:

> "We build upon the code execution framework from OpenCodeInterpreter [cite] and adopt the language-to-code approach from AILS-NTUA [cite]. However, unlike AILS-NTUA's single-step error correction, we propose a **4-layer hierarchical diagnostic system** that systematically classifies errors, analyzes root causes, selects repair strategies, and generates targeted repair prompts. This hierarchical approach enables more precise error correction and can be further optimized through GRPO reinforcement learning [cite Table-R1]."

这样既承认了借鉴，又突出了我们的创新。

---

## 🎉 总结

### 关键发现:
1. ✅ AILS的代码**很简单** (只有2个核心文件)
2. ✅ 他们的Prompt**很有效** (列选择、unique values、函数模板)
3. ✅ 我们的4层诊断系统**是创新点**，应该保留
4. ⚠️ 我们的Prompt**需要优化**

### 下一步:
1. **立即**: 改进Prompt (借鉴AILS)
2. **立即**: 添加列名清理
3. **短期**: 重新测试，对比性能
4. **中期**: 完整评估，撰写论文

我们**不需要推倒重来**，只需要**借鉴他们的最佳实践**，同时**保留我们的创新点**！
