# AILS-NTUA 正确复现计划

## 当前问题分析

你说得对!我发现了几个关键差异:

### 1. 模型差异 ⚠️ (最重要!)

**我的实现**:
- 模型: `Qwen/Qwen2.5-7B-Instruct` (通用对话模型)
- 用途: 通用指令遵循

**AILS-NTUA官方**:
- 模型: `Qwen2.5-Coder-7B` (代码专用模型)
- 用途: **专门为代码生成优化**
- 配置文件: `config/qwen2.5-coder-7B.yaml`

**影响**: Coder模型可能在代码生成任务上表现显著更好!

### 2. 数据集差异

**我的测试**:
- 数据集: WikiTQ (Wikipedia表格,复杂推理)
- Baseline: ~54%

**AILS-NTUA论文**:
- 主要数据集: DataBench (SemEval 2025 Task 8)
- Baseline: ~27%
- 论文中小模型的表现: 需要查看论文具体数字

### 3. 实现细节可能的差异

从他们的配置文件看到:

**Few-shot设置**:
```yaml
prompt_generator:
  class_name: FewShotChainOfThoughtBuilderWithTypesRowsAndPredictingResultingTypesVol2
  shots: !EXEMPLARS_PLACEHOLDER
```

**Error fixing**:
```yaml
error_fix_pipeline:
  num_attempts: 2
  max_timeout: 600  # 10分钟超时!
```

**Temperature**:
- Main: 0.0 (deterministic)
- Error fix: 1.0 (更随机,增加修复成功率)

## 正确复现步骤

### Phase 1: 使用正确的模型 (Qwen2.5-Coder-7B)

1. **下载Qwen2.5-Coder-7B模型**
   ```bash
   # 使用Ollama (AILS-NTUA官方方式)
   ollama pull qwen2.5-coder:7b
   
   # 或HuggingFace
   # Qwen/Qwen2.5-Coder-7B-Instruct
   ```

2. **修改代码使用Coder模型**
   ```python
   model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
   # 而不是 "Qwen/Qwen2.5-7B-Instruct"
   ```

3. **在WikiTQ上重新测试100样本**
   - 预期: 可能会看到改进!

### Phase 2: 在DataBench上测试 (论文的主战场)

1. **准备DataBench数据** (已有)
   ```bash
   # 已下载: data/databench/
   ```

2. **运行Baseline vs AILS对比**
   - DataBench baseline更低 (~27%)
   - 小模型可能表现更好

3. **对比论文中Qwen2.5-Coder-7B的结果**

### Phase 3: 完全复现AILS-NTUA配置

如果前两步仍然没改进,考虑:

1. **使用他们的完整pipeline**
   ```bash
   cd baselines/sota_methods/ails_ntua
   python main.py --pipeline config/qwen2.5-coder-7B.yaml
   ```

2. **对比他们的prompt生成器**
   - 他们的: `ZeroShotDetailedTypesRowsExValuesNullsOneLineRowsVol2`
   - 我的: 简化版AILS prompt

3. **复制他们的few-shot examples**
   - 他们使用什么examples?
   - 有多少个examples?

## 预期结果

### 如果使用Qwen2.5-Coder-7B:

**WikiTQ预期**:
- Baseline: 33% (当前)
- AILS Zero-Shot: 40-45%? (Coder模型应该更好)
- 改进: +7-12%

**DataBench预期** (更容易的数据集):
- Baseline: 27%
- AILS Zero-Shot: 50-60%? (根据论文)
- 改进: +23-33%

### 如果仍然无改进:

可能原因:
1. Prompt生成细节不同
2. Few-shot examples选择不同
3. Error fixing策略不同
4. 需要更多迭代次数

## 立即行动

**建议顺序**:

1. **切换到Qwen2.5-Coder-7B模型** ⭐ (最重要!)
   - 时间: 5分钟(如果模型已缓存) 或 30分钟(需要下载)
   - 代码修改: 1行
   - 重新测试WikiTQ 100样本: 40分钟

2. **测试DataBench**
   - 准备DataBench数据: 已完成
   - 运行100样本测试: 40分钟
   - 对比论文结果

3. **如果需要,使用AILS-NTUA官方代码**
   - 直接运行他们的pipeline
   - 对比输出差异

## 关键问题待确认

1. **论文中Qwen2.5-Coder-7B的具体表现是多少?**
   - 需要查看论文或GitHub的results
   - 可能在issues或README中

2. **他们用多少few-shot examples?**
   - 配置文件中是`!EXEMPLARS_PLACEHOLDER`
   - 需要看main.py如何加载

3. **WikiTQ vs DataBench哪个是主战场?**
   - 论文标题是SemEval 2025 Task 8 (DataBench)
   - WikiTQ可能只是额外测试

---

**结论**: 你的质疑非常正确!我应该:
1. 先用**Qwen2.5-Coder-7B**而不是Instruct模型
2. 在**DataBench**而不是WikiTQ上测试
3. 查看论文中小模型的具体数字

让我立即切换模型并重新测试!
