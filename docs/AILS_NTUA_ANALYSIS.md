# AILS-NTUA 方法深度解析

**基于代码**: https://github.com/andrewevag/SemEval2025-Task8-TQA

---

## 核心架构

AILS-NTUA 使用**两模块架构**:

```
Main Module (主生成模块)
    ↓ 生成代码
    ↓ 执行
    ↓ (如果失败)
Error-Fixing Module (错误修复模块)
    ↓ 修复代码
    ↓ 重新执行
```

---

## 关键技术1: Chain-of-Thought Prompting (分步推理)

### 我们当前的Prompt (简单版)
```python
prompt = f"""
Question: {question}
Table columns: {columns}

Generate Python code to answer the question.
"""
```

### AILS-NTUA的Prompt (Chain-of-Thought版)

#### **Zero-Shot Prompt结构**

```python
prompt = f"""
# TODO: complete the following function. It should give the answer to: {question}
def answer(df: pd.DataFrame):
    \"\"\"
    # 表格schema信息 (详细!)
    #,Column,Non-Null Count,Dtype,Types of Elements,Values,Are all values unique?
    0,year,100,int64,[<class 'int'>],,False
    1,winner,95,object,[<class 'str'>],5 example values are ['Team A', 'Team B', ...],False
    2,score,100,int64,[<class 'int'>],,False

    # 前5行数据 (实际数据!)
    The first 5 rows from the dataframe:
       year winner  score
    0  2015 Team A     95
    1  2016 Team B     88
    2  2017 Team A     92
    3  2018 Team C     90
    4  2019 Team A     93
    \"\"\"

    df.columns = ['year', 'winner', 'score']  # 显式设置列名

```

**关键点**:
1. ✅ **详细的schema信息**: 包括类型、null值、唯一值、示例值
2. ✅ **实际数据展示**: 前5行完整数据
3. ✅ **显式列名设置**: `df.columns = [...]`
4. ✅ **TODO形式**: 让模型"完成"函数而非"生成"函数

#### **Few-Shot Prompt结构** (更强!)

```python
prompt = f"""
# Example 1:
# TODO: complete the following function. It should give the answer to: How many teams won?
def answer(df: pd.DataFrame):
    \"\"\"
    {schema_info}
    {data_preview}
    \"\"\"
    df.columns = ['year', 'winner', 'score']

    # The columns used to answer the question: ['winner']
    # The types of the columns used to answer the question: ['object']
    # The type of the answer: int

    return df['winner'].nunique()

# Example 2:
...

# Now your turn:
# TODO: complete the following function. It should give the answer to: {question}
def answer(df: pd.DataFrame):
    \"\"\"
    {schema_info}
    {data_preview}
    \"\"\"
    df.columns = {columns}

    # The columns used to answer the question:
```

**关键点**:
1. ✅ **显式中间推理**:
   - 使用的列
   - 列的类型
   - 答案的类型
2. ✅ **示例引导**: 1-2个完整示例
3. ✅ **引导模型思考**: 让模型预测要用哪些列、什么类型

---

## 关键技术2: Error-Fixing Prompt (错误修复)

### 我们当前的修复Prompt (基于诊断)
```python
# 我们的方法: 基于错误分类生成修复策略
if error_type == "NameError":
    strategy = ColumnCorrectionStrategy()
    repair_prompt = strategy.generate_repair_prompt(error)
```

### AILS-NTUA的修复Prompt (直接重写)

```python
# User message:
f"""
# Help me fix the code error of the following function by rewriting it.
# Try to parse columns with list types yourself instead of using the `eval` function.
# Some lists may be written without the necessary '' to be parsed correctly.
# If rare or special characters are included as values, test equality by substring detection e.g. "query" in df[col].
# The function should return the answer to the question in the TODO comment below:

{原始prompt_with_schema_and_data}

{之前生成的代码}

# The function outputs the following error:
# {error_msg}
"""

# Assistant message (预填充):
f"""
{原始prompt_with_schema_and_data的开头}
"""
```

**关键点**:
1. ✅ **给出具体修复建议**:
   - 不要用`eval`，自己解析列表
   - 用substring匹配而非精确匹配
   - 处理特殊字符
2. ✅ **提供完整上下文**:
   - 原始表格信息
   - 失败的代码
   - 错误消息
3. ✅ **预填充assistant消息**: 引导模型重写整个函数

---

## 关键技术3: 丰富的Schema信息

### 我们当前的Schema信息 (简单)
```python
columns = ['year', 'winner', 'score']
sample_values = {
    'year': [2015, 2016, 2017],
    'winner': ['Team A', 'Team B', 'Team A']
}
```

### AILS-NTUA的Schema信息 (详细)

```python
def custom_info_csv_5(df):
    """
    生成详细的列信息表格:
    #,Column,Non-Null Count,Dtype,Types of Elements,Values,Are all values unique?
    """
    for i, col in enumerate(df.columns):
        non_null_count = df[col].notnull().sum()
        dtype = df[col].dtype
        python_inner_types = df[col].apply(lambda x: type(x)).unique()

        # 对category和object类型，显示示例值
        if dtype.name == "category" or (dtype.name == 'object' and has_strings):
            unique_values = df[col].unique()
            if len(unique_values) <= 5:
                values = f"All values are {unique_values}"
            else:
                values = f"5 example values are {unique_values[:5]}"

        are_all_values_unique = df[col].nunique() == df[col].count()

        output.append(f"{i},{col},{non_null_count},{dtype},{python_inner_types},{values},{are_all_values_unique}")
```

**提供的信息**:
- ✅ Non-Null Count (多少非空值)
- ✅ Dtype (pandas数据类型)
- ✅ Types of Elements (Python类型,如`<class 'int'>`)
- ✅ Values (示例值或所有值)
- ✅ Are all values unique? (是否所有值唯一)

---

## 关键技术4: 显式设置列名

### 我们当前的做法
```python
# 不设置，直接使用
result = df[df['year'] == 2015]['winner'].iloc[0]
```

### AILS-NTUA的做法
```python
# 在每个生成的函数开头显式设置
df.columns = ['year', 'winner', 'score']

# 然后再使用
result = df[df['year'] == 2015]['winner'].iloc[0]
```

**好处**:
- ✅ 避免列名大小写问题
- ✅ 避免特殊字符问题
- ✅ 确保列名一致性

---

## 关键差异对比

| 方面 | 我们的方法 | AILS-NTUA | 优势方 |
|------|-----------|-----------|--------|
| **Prompt结构** | 简单直接 | Chain-of-Thought | AILS |
| **Schema信息** | 基础(列名+示例) | 详细(类型+null+唯一性+示例) | AILS |
| **数据展示** | 部分 | 前5行完整数据 | AILS |
| **Few-shot** | 无 | 1-2个完整示例 | AILS |
| **中间推理** | 无 | 预测列/类型/答案类型 | AILS |
| **错误修复** | 基于诊断分类 | 直接重写+具体建议 | 我们(更系统) |
| **错误修复上下文** | 错误信息+代码 | 错误+代码+表格信息 | AILS |
| **列名处理** | 自动推断 | 显式设置 | AILS |
| **模型** | Qwen 7B | Claude 3.5 (200B) | AILS |

---

## 我们可以借鉴的改进

### 立即可做 (1-2天)

#### 1. **改进Prompt结构** (+5-10%)
```python
# 当前
prompt = f"Question: {q}\nColumns: {cols}\nGenerate code."

# 改进为AILS-NTUA风格
prompt = f"""
# TODO: complete the following function. It should give the answer to: {question}
def answer(df: pd.DataFrame):
    \"\"\"
    {detailed_schema_info()}  # 详细schema

    The first 5 rows from the dataframe:
    {df.head(5).to_string()}
    \"\"\"

    df.columns = {list(df.columns)}
"""
```

#### 2. **添加Few-shot Examples** (+5-10%)
```python
# 在prompt开头添加1-2个示例
examples = [
    {
        'question': "How many unique teams?",
        'code': "return df['team'].nunique()",
        'columns_used': ['team'],
        'answer_type': 'int'
    }
]
```

#### 3. **丰富Schema信息** (+3-5%)
```python
# 实现custom_info_csv_5
# 提供: dtype, null count, unique values, sample values
```

#### 4. **改进错误修复Prompt** (+3-5%)
```python
# 当前: 简短的修复提示
# 改进: 添加具体的修复建议
"""
Help me fix this code error.
Specific suggestions:
- Don't use eval() for parsing lists
- Use substring matching for special characters
- Check column types before operations

Original table info:
{schema}

Failed code:
{code}

Error:
{error}
"""
```

### 中期可做 (1周)

#### 5. **实现Chain-of-Thought中间推理** (+10-15%)
```python
# 让模型先预测要用哪些列、什么类型
# 然后再生成代码
prompt = f"""
{question}

First, think step by step:
1. Which columns are needed?
2. What are their types?
3. What type should the answer be?

Now generate the code:
"""
```

---

## 预期提升

| 改进 | 难度 | 时间 | DataBench | WikiTQ | TabFact |
|------|------|------|-----------|--------|---------|
| **改进Prompt结构** | 低 | 1天 | +5-10% | +3-5% | +5-8% |
| **Few-shot Examples** | 低 | 1天 | +5-10% | +5-10% | +3-5% |
| **丰富Schema** | 低 | 1天 | +3-5% | +2-3% | +3-5% |
| **改进修复Prompt** | 低 | 1天 | +3-5% | +2-3% | +3-5% |
| **Chain-of-Thought** | 中 | 3天 | +10-15% | +5-8% | +8-10% |
| **总计** | - | 1-2周 | **+25-40%** | **+15-25%** | **+20-30%** |

### 预期最终结果

| 数据集 | 当前 | 改进后 | SOTA | vs SOTA |
|--------|------|--------|------|---------|
| DataBench | 67% | **82-92%** | 85% | **接近/超过!** |
| WikiTQ | 25% | **40-50%** | 75% | 仍有差距 |
| TabFact | 68% | **78-88%** | 85% | **接近!** |

---

## 实施计划

### Day 1: Prompt改进
- ✅ 实现detailed_schema_info()
- ✅ 改进主生成prompt为TODO格式
- ✅ 添加显式df.columns设置

### Day 2: Few-shot
- ✅ 准备2-3个high-quality examples
- ✅ 实现few-shot prompt builder
- ✅ 测试few-shot vs zero-shot

### Day 3: 错误修复改进
- ✅ 改进error-fixing prompt
- ✅ 添加具体修复建议
- ✅ 提供完整上下文(表格+代码+错误)

### Day 4-7: Chain-of-Thought
- ✅ 实现中间推理步骤
- ✅ 让模型预测列/类型/答案类型
- ✅ 测试CoT vs non-CoT

### Day 8-10: 评估和调优
- ✅ 在三个数据集上全面评估
- ✅ 对比改进前后
- ✅ 分析剩余错误

---

## 与我们诊断系统的结合

AILS-NTUA和我们的系统可以**互补**:

| 阶段 | AILS-NTUA | 我们的系统 | 结合 |
|------|-----------|-----------|------|
| **生成** | 丰富prompt + CoT | 简单prompt | 用AILS的prompt |
| **执行** | 直接执行 | 执行 | 相同 |
| **错误修复** | 通用重写建议 | 4层诊断分类 | **结合两者!** |

### 最佳组合策略

```
生成 (AILS-NTUA风格prompt)
    ↓
执行
    ↓ (如果失败)
诊断 (我们的4层系统)
    ↓
选择修复策略
    ↓
生成修复prompt (AILS建议 + 我们的诊断)
    ↓
重新生成
```

**优势**:
- ✅ 生成阶段: 用AILS的丰富prompt减少初始错误
- ✅ 修复阶段: 用我们的诊断精确定位问题
- ✅ 最佳组合: 预防(AILS) + 精确修复(我们)

---

## 总结

AILS-NTUA的成功主要来自:

1. **丰富的Prompt工程**
   - Detailed schema
   - 实际数据展示
   - Chain-of-Thought推理

2. **强大的模型** (Claude 3.5)
   - 200B参数
   - 强语义理解

3. **实用的错误修复建议**
   - 具体而非通用
   - 完整上下文

**我们的优势**:
- 4层诊断系统更系统化
- 适合小模型

**结合策略**:
- 学习AILS的prompt engineering
- 保留我们的诊断系统
- 两者结合 → 最佳效果！
