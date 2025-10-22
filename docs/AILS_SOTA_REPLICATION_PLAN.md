# AILS-NTUA SOTA复现完整方案

**日期**: 2025-10-22
**目标**: 正确复现AILS-NTUA的Table QA系统,验证其SOTA性能

---

## 当前状态总结

### 我们的复现尝试失败原因

经过深入分析,发现AILS-NTUA的工作流程包含一个**关键的后处理组件**,我们之前只复制了prompt,没有复制完整pipeline:

```
正确的AILS-NTUA工作流:
  Prompt (函数头)
    ↓
  LLM生成代码片段
    ↓
  ⭐ 后处理器: 提取到第一个return语句为止
    ↓
  组装完整函数
    ↓
  执行

我们的错误工作流:
  Prompt (函数头 或 完整函数)
    ↓
  LLM生成代码
    ↓
  直接执行 → 失败 (缺少return或函数定义)
```

### 测试结果对比

| 测试 | 模型 | 样本数 | 执行成功率 | 准确率 | 结论 |
|------|------|--------|-----------|--------|------|
| WikiTQ Baseline | Qwen2.5-7B-Instruct | 100 | 93% | **33%** | 基线 |
| WikiTQ + AILS v1 | Qwen2.5-7B-Instruct | 100 | 93% | **33%** | +0% ❌ |
| DataBench + AILS v1 | Qwen2.5-Coder-7B | 39 | 100% | **~8%** | 严重退化 ❌ |
| DataBench + AILS v2 | Qwen2.5-Coder-7B | 20 | 100% | **0%** | 完全失败 ❌ |

---

## 方案1: 使用AILS-NTUA官方代码 (推荐)

这是**最可靠**的复现方式,直接使用他们经过验证的实现。

### 前置条件

1. **安装Ollama** (需要sudo权限):
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

2. **下载模型**:
```bash
ollama pull qwen2.5-coder:7b
```

3. **启动Ollama服务** (后台运行,监听11434端口):
```bash
ollama serve
```

### 准备数据

我们已经有了DataBench数据:
```
data/databench/
  ├── dev.jsonl      (已下载)
  └── test.jsonl     (如需测试集,需从competition.zip获取)
```

AILS-NTUA默认从HuggingFace加载DataBench,我们的数据已经准备好。

### 运行评估

```bash
cd baselines/sota_methods/ails_ntua

# 在lite版本上测试 (小规模数据集)
python main.py --pipeline config/qwen2.5-coder-7B.yaml --lite

# 在完整dev集上测试
python main.py --pipeline config/qwen2.5-coder-7B.yaml

# 结果保存在 results/ 目录
ls results/
```

### 预期结果

根据AILS-NTUA论文 (arXiv:2503.00435):
- **Qwen2.5-Coder-7B on DataBench Lite**: 预期 > 60%
- **Claude 3.5 Sonnet on DataBench**: ~85% (最佳)

### 验证检查项

运行后检查:
1. ✅ 是否使用了后处理器 (`TillReturnLinePostProcessorMultipleIndents`)
2. ✅ 生成的代码是否包含return语句
3. ✅ 执行成功率是否 > 90%
4. ✅ 答案准确率是否 > 60%

---

## 方案2: 实现后处理器并集成到我们系统

如果无法安装Ollama,可以实现AILS-NTUA的后处理器,集成到我们现有系统中。

### 2.1 核心组件: 后处理器

创建 `src/baselines/ails_postprocessor.py`:

```python
"""
AILS-NTUA Style Post-processor
Extracts code until the first return statement
"""

import re
from typing import Optional

class TillReturnPostProcessor:
    """
    Extract code until the first return statement.

    This mimics AILS-NTUA's TillReturnLinePostProcessorMultipleIndents.
    """

    def __init__(self, base_indent: int = 4, first_prefix: str = ""):
        """
        Args:
            base_indent: Base indentation level (default 4 spaces)
            first_prefix: Prefix to add before first line (e.g., column comment)
        """
        self.base_indent = base_indent
        self.first_prefix = first_prefix

    def extract(self, response: str) -> str:
        """
        Extract code until first return statement.

        Args:
            response: Raw model output

        Returns:
            Processed code snippet (ends at return statement)
        """
        lines = response.split("\n")
        extracted_lines = []

        for i, line in enumerate(lines):
            # Calculate current indent
            indent = len(line) - len(line.lstrip())

            # Store line
            extracted_lines.append(line)

            # Check if this is a return statement
            stripped = line.strip()
            if stripped.startswith("return "):
                # Found return statement, stop here
                break

        # Add prefix to first line if specified
        if self.first_prefix and extracted_lines:
            extracted_lines[0] = self.first_prefix + extracted_lines[0].strip()

        return "\n".join(extracted_lines)

    def assemble_function(
        self,
        code_snippet: str,
        columns: list,
        function_name: str = "answer"
    ) -> str:
        """
        Assemble complete function from code snippet.

        Args:
            code_snippet: Extracted code snippet (with return)
            columns: List of column names
            function_name: Function name (default "answer")

        Returns:
            Complete executable function
        """
        # Ensure proper indentation
        indented_lines = []
        for line in code_snippet.split("\n"):
            if line.strip():  # Non-empty line
                indented_lines.append("    " + line if not line.startswith("    ") else line)
            else:
                indented_lines.append("")

        indented_code = "\n".join(indented_lines)

        # Assemble function
        function = f"""def {function_name}(df: pd.DataFrame):
    df.columns = {columns}
{indented_code}
"""
        return function


# Example usage
if __name__ == "__main__":
    processor = TillReturnPostProcessor(
        first_prefix="    # The columns used to answer the question: ['year', 'team']\n"
    )

    # Simulate model output
    model_output = """    result = df[df['year'] == 2015]['team'].iloc[0]
    return result
    # This line should be ignored
    print("extra code")
"""

    # Extract
    extracted = processor.extract(model_output)
    print("Extracted code:")
    print(extracted)
    print()

    # Assemble
    complete = processor.assemble_function(extracted, ['year', 'team', 'score'])
    print("Complete function:")
    print(complete)
```

### 2.2 修改 Code Generator

修改 `src/baselines/code_generator.py`:

```python
from src.baselines.ails_postprocessor import TillReturnPostProcessor

class CodeGenerator:
    def __init__(self, ...):
        ...
        self.postprocessor = TillReturnPostProcessor()

    def generate_code(self, table, question, use_ails_with_postprocessing=False):
        if use_ails_with_postprocessing:
            # 1. Generate incomplete prompt (function header only)
            prompt = self._generate_incomplete_ails_prompt(table, question)

            # 2. Get model output (code snippet)
            code_snippet = self.model.generate(prompt)

            # 3. Post-process: extract until return
            processed_snippet = self.postprocessor.extract(code_snippet)

            # 4. Assemble complete function
            columns = list(table.columns)
            complete_code = self.postprocessor.assemble_function(
                processed_snippet,
                columns
            )

            return complete_code
        else:
            # Original method
            ...

    def _generate_incomplete_ails_prompt(self, table, question):
        """
        Generate AILS-style prompt with incomplete function.
        Model is expected to complete the function body.
        """
        schema_info = get_detailed_schema_info(table)
        data_preview = get_data_preview(table, num_rows=5)
        columns = list(table.columns)

        prompt = f"""# TODO: complete the following function. It should give the answer to: {question}
def answer(df: pd.DataFrame):
    \"\"\"
        {schema_info}

        The first 5 rows from the dataframe:
        {data_preview}
    \"\"\"

    df.columns = {columns}

    # The columns used to answer the question:"""

        return prompt
```

### 2.3 测试流程

```bash
# 测试后处理器
python src/baselines/ails_postprocessor.py

# 在DataBench上测试完整pipeline
python scripts/evaluate_databench.py \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --num_samples 100 \
    --use_ails_postprocessor \
    --output results/databench_ails_with_postprocessor.json
```

### 2.4 预期改进

- **执行成功率**: 90% → 95%+ (后处理确保代码格式正确)
- **答案准确率**:
  - Qwen2.5-7B-Instruct: 33% → 40%+ (更好的prompt)
  - Qwen2.5-Coder-7B: 0% → 60%+ (Coder模型现在能正常工作)

---

## 方案3: 使用更大模型 (备选)

如果以上方案不可行,可以尝试更大的模型:

### 3.1 切换到更大的Qwen模型

```python
# 使用Qwen2.5-14B-Instruct或32B-Instruct
qa_system = TableQASystem(
    model_name="Qwen/Qwen2.5-14B-Instruct",  # 更大模型
    max_iterations=3
)
```

**优势**:
- 更大模型理解能力更强,可能不需要复杂后处理
- 可能更好地遵循"生成完整函数"指令

**劣势**:
- 需要更多GPU内存 (~28GB for 14B)
- 推理速度更慢

### 3.2 使用商业API (Claude/GPT-4)

AILS-NTUA的最佳结果使用Claude 3.5 Sonnet (~85%准确率):

```python
# 需要配置API key
qa_system = TableQASystem(
    model_name="claude-3.5-sonnet",  # 通过API调用
    max_iterations=3
)
```

---

## 实施建议

### 短期 (今天完成)

**选择方案1 (官方代码)**:
1. ✅ 已安装AILS-NTUA依赖
2. ⏳ **需要用户手动安装Ollama** (需要sudo)
3. ⏳ 运行官方评估脚本
4. ⏳ 记录结果并与我们baseline对比

**如果无法安装Ollama,选择方案2**:
1. 实现 `ails_postprocessor.py` (1小时)
2. 修改 `code_generator.py` (30分钟)
3. 测试完整pipeline (1小时)

### 中期 (1-2天)

1. 完善后处理器集成
2. 在多个数据集上测试 (WikiTQ, TabFact, DataBench)
3. 调优超参数 (temperature, max_tokens)
4. 实现Few-shot版本

### 长期 (1-2周)

1. 实现完整的error-fixing pipeline (温度=1.0)
2. 添加LLM-based列选择 (AILS论文Section 3.2)
3. 实现self-consistency (多次采样取最常见答案)
4. GRPO训练优化策略选择

---

## 关键配置参数

从AILS-NTUA的 `config/qwen2.5-coder-7B.yaml`:

```yaml
# 主要代码生成
answerer:
  temperature: 0.0          # 确定性生成
  max_gen_len: 300         # 较短,只生成核心代码

# 错误修复
error_fix_pipeline:
  answerer:
    temperature: 1.0        # 探索性生成
    max_gen_len: 1000       # 更长,允许重写

# ⭐ 后处理器 (关键!)
postprocessor:
  class_name: postprocessors.TillReturnLinePostProcessorMultipleIndents
  arguments:
    prefix: 4                                           # 基础缩进
    first_prefix: "    # The columns used to answer the question: "  # 首行前缀
```

---

## 验证检查清单

运行任何方案后,检查以下指标:

### 代码质量
- [ ] 生成的代码包含 `def answer(df: pd.DataFrame):` 定义
- [ ] 代码包含 `return` 语句
- [ ] 代码缩进正确
- [ ] 没有多余的import/print语句

### 性能指标
- [ ] 执行成功率 ≥ 90%
- [ ] 答案准确率 (DataBench):
  - Qwen2.5-Coder-7B: ≥ 60%
  - Claude 3.5 Sonnet: ≥ 85%
- [ ] 首次成功率 ≥ 80%
- [ ] 平均迭代次数 ≤ 1.5

### 对比我们的baseline
- [ ] 准确率提升 ≥ 10个百分点
- [ ] 执行成功率提升 ≥ 5个百分点

---

## 下一步行动

### 立即可行 (需要用户操作)

**方案1: 官方AILS代码**
```bash
# 用户需要运行 (需要sudo权限):
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5-coder:7b
ollama serve &  # 后台运行

# 然后运行评估:
cd baselines/sota_methods/ails_ntua
python main.py --pipeline config/qwen2.5-coder-7B.yaml --lite
```

**方案2: 实现后处理器**
```bash
# 我可以立即开始实现:
# 1. 创建 src/baselines/ails_postprocessor.py
# 2. 修改 src/baselines/code_generator.py
# 3. 创建测试脚本 scripts/test_ails_with_postprocessor.py
```

---

## 参考资料

1. **AILS-NTUA论文**: [arXiv:2503.00435](https://arxiv.org/abs/2503.00435)
2. **GitHub仓库**: [AILS-NTUA/tabularqa](https://github.com/AILS-NTUA/tabularqa)
3. **DataBench数据集**: [HuggingFace](https://huggingface.co/datasets/cardiffnlp/databench)
4. **SemEval 2025 Task 8**: [Competition Page](https://www.codabench.org/competitions/3360/)

---

## 结论

要正确复现AILS-NTUA的SOTA性能,我们有两个可行路径:

1. **方案1 (推荐)**: 使用官方代码 + Ollama → 最可靠,但需要sudo安装
2. **方案2 (备选)**: 实现后处理器 → 可立即开始,集成到我们系统

**关键教训**: SOTA方法的复现不仅需要复制prompt,还需要复制完整的pipeline,包括后处理、配置参数、模型选择等所有细节。

**建议**: 如果用户有sudo权限,优先选择方案1验证论文结果;如果没有,则实施方案2将后处理器集成到我们现有系统中。
