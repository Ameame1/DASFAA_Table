# AILS-NTUA 复现失败分析与正确方案

## 日期: 2025-10-22

## 复现尝试总结

### 测试结果

| 测试 | 模型 | Prompt | 样本数 | 执行成功率 | 准确率 | 结论 |
|------|------|--------|--------|-----------|--------|------|
| WikiTQ Baseline | Qwen2.5-7B-Instruct | 标准 | 100 | 93% | 33% | Baseline |
| WikiTQ + AILS | Qwen2.5-7B-Instruct | AILS v1 | 10 | 100% | 30% | +10% (统计噪音) |
| WikiTQ + AILS | Qwen2.5-7B-Instruct | AILS v1 | 100 | 93% | **33%** | **+0%** ❌ |
| DataBench + Coder + AILS | Qwen2.5-Coder-7B | AILS v1 | 39 | 100% | **~8%** | **远低于baseline** ❌ |
| DataBench + Coder + AILS | Qwen2.5-Coder-7B | AILS v2 | 20 | 100% | **0%** | **完全失败** ❌ |

### 失败原因分析

#### 1. 核心问题: 后处理缺失

**AILS-NTUA的工作流**:
```
Prompt (函数开头)
  → 模型生成代码片段
  → 后处理器截取到return语句  ⭐ 关键!
  → 组装完整函数
  → 执行
```

**我们的工作流** (错误):
```
Prompt (函数开头)
  → 模型生成代码片段
  → (❌ 没有后处理!)
  → 直接执行 → 失败 (缺少函数定义/return)
```

#### 2. 关键组件缺失

从AILS-NTUA的配置文件 (`config/qwen2.5-coder-7B.yaml`) 发现:

```yaml
postprocessor:
  class_name: postprocessors.TillReturnLinePostProcessorMultipleIndents
  arguments:
    loader: !LOADER_PLACEHOLDER
    prefix: 4
    first_prefix: "    # The columns used to answer the question: "
```

**后处理器作用**:
```python
def __call__(self, response: str, dataset: Optional[str]=None) -> str:
    lines = response.split("\n")
    xs = []; indents = []
    for i, line in enumerate(lines):
        indent = len(line) - len(line.lstrip())
        indents.append(indent)
        xs.append(line.strip())
        if line.startswith(((' ' * self.return_indent) + "return")):  # 找到return语句就停止
            break
    # ... 处理缩进 ...
    return "\n".join(lines[:i+1])  # 只返回到return语句为止的代码
```

**功能**:
- 从模型生成的文本中提取代码
- **截取到第一个`return`语句为止**
- 调整缩进使其成为函数体
- 添加前缀(如列注释)

#### 3. Prompt设计差异

**AILS-NTUA的Prompt**:
```python
# TODO: complete the following function. It should give the answer to: {question}
def answer(df: pd.DataFrame):
    """
    Schema info...
    """
    df.columns = [...]
```

**期望模型生成**:
```python
    # The columns used to answer the question: ['col1', 'col2']
    result = df['col1'].nunique()
    return result
```

**后处理后变成**:
```python
def answer(df: pd.DataFrame):
    df.columns = [...]
    # The columns used to answer the question: ['col1', 'col2']
    result = df['col1'].nunique()
    return result
```

#### 4. 我们的错误尝试

**尝试1**: 添加注释 "# Make sure to return the answer at the end"
- **结果**: 模型仍不生成return (8.6%准确率)

**尝试2**: 给出完整函数模板
```python
def answer(df: pd.DataFrame):
    df.columns = [...]
    # Your solution here
    return result  # Must return the answer
```
- **结果**: 模型当作上下文,不生成函数定义 (0%准确率)

**根本问题**: **Coder模型期望填空式补全,而不是生成完整函数**

## 正确的复现方案

### 方案1: 使用AILS-NTUA官方代码 (推荐)

```bash
cd baselines/sota_methods/ails_ntua

# 1. 安装Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. 下载模型
ollama pull qwen2.5-coder:7b

# 3. 准备DataBench数据 (已完成)
# data/databench/dev.jsonl

# 4. 运行官方评估
python main.py --pipeline config/qwen2.5-coder-7B.yaml --lite

# 5. 查看结果
ls results/
```

**预期结果** (根据论文):
- Qwen2.5-Coder-7B on DataBench: **待测试**
- Claude 3.5 Sonnet: ~85%

### 方案2: 实现后处理器 (如果要集成到我们系统)

需要实现:

1. **代码提取器**: 从模型输出中提取到第一个return为止
2. **函数组装器**: 组装成完整函数
3. **修改Code Generator**:
   - Prompt只给出函数开头(不给完整模板)
   - 添加后处理步骤

**伪代码**:
```python
# 1. Generate code snippet
prompt = generate_ails_prompt_incomplete(table, question)  # 只给函数开头
code_snippet = model.generate(prompt)

# 2. Post-process: Extract until return
processed_code = extract_until_return(code_snippet)

# 3. Assemble complete function
complete_function = f"""
def answer(df: pd.DataFrame):
    df.columns = {columns}
{processed_code}
"""

# 4. Execute
result = executor.execute(complete_function, table)
```

### 方案3: 切换到支持的模型

AILS-NTUA测试过的模型:
- Claude 3.5 Sonnet (最佳: ~85%)
- Llama 3.1-405B Instruct
- Llama 3.3-70B
- Llama 3.1-8B
- Qwen2.5-Coder-7B (需要Ollama)

我们当前的模型 (Qwen2.5-7B-Instruct) **不在他们的测试列表中**,可能不兼容。

## 经验教训

### 1. 复现SOTA需要完整pipeline
- ❌ 只复现prompt不够
- ✅ 需要复现: prompt + 后处理 + 执行器

### 2. 小模型和大模型行为不同
- 大模型(Claude): 能理解"生成完整函数"的指令
- 小模型(Qwen-7B): 倾向于代码补全,需要后处理

### 3. 10样本测试不可靠
- WikiTQ 10样本: +10%改进 (虚假信号)
- WikiTQ 100样本: +0%改进 (真实结果)
- **结论**: 至少需要100样本才能得出可靠结论

### 4. 配置细节很重要
- 后处理器: 决定性作用
- 温度参数: Main=0.0, Error fixing=1.0
- Max tokens: 300 (main), 1000 (error fixing)

## 下一步行动

### 立即可行:
1. ✅ **使用AILS-NTUA官方代码测试** (推荐)
   - 安装Ollama
   - 运行他们的pipeline
   - 验证论文结果

2. **使用我们的Baseline** (已有46%准确率)
   - 继续改进baseline
   - 不依赖AILS

### 长期方案:
1. **实现完整的后处理pipeline**
   - 需要1-2天开发
   - 参考AILS-NTUA的postprocessors.py

2. **使用更大的模型**
   - Qwen2.5-14B或32B
   - 可能不需要复杂的后处理

## 结论

AILS-NTUA的prompting技术**需要特定的后处理流程**才能工作,简单复制prompt不够。我们的复现失败是因为:

1. ❌ 缺少后处理器(提取到return)
2. ❌ 模型不在AILS测试列表
3. ❌ 误以为prompt是唯一关键

**建议**:
- **短期**: 使用AILS官方代码验证
- **长期**: 继续改进我们自己的baseline方法(更可控,已有46%准确率)
