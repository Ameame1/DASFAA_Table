# 创新点对比：我们 vs AILS-NTUA

## 📊 核心对比

### AILS-NTUA的贡献 (SemEval 2025冠军)

| 方面 | 他们的方法 |
|------|-----------|
| **错误修复** | 简单直接的单步修复 |
| **迭代次数** | 固定2次迭代（1次生成+1次修复） |
| **Prompt技巧** | ✅ 列选择、unique values、函数模板、列名清理 |
| **模型** | Claude-3.5-Sonnet (专有模型) |
| **结果** | 85.63% (SemEval-2025 DataBench) |

**他们的代码** (`code_fixer.py`):
```python
def code_fix(self, response: str, error: str):
    messages = [
        {"role": "system", "content": "Fix the Python code..."},
        {"role": "user", "content": f"Code: {response}\nError: {error}"}
    ]
    return self.pipe(messages)
```

**特点**:
- ✅ **简单有效** - 直接将错误和代码给LLM修复
- ❌ **缺乏系统性** - 没有错误分类、根因分析
- ❌ **固定迭代** - 总是2次（1次生成+1次修复），无法动态调整

---

### 我们的贡献

| 方面 | 我们的方法 | 创新程度 |
|------|-----------|---------|
| **错误诊断** | **4层分层系统** | ⭐⭐⭐ **核心创新** |
| **迭代控制** | 动态迭代（1-3次，平均1.28） | ⭐⭐ **有创新** |
| **修复策略** | 5种专用策略（可扩展20种） | ⭐⭐ **有创新** |
| **Prompt技巧** | 借鉴AILS（列选择、unique values等） | ⭐ **增量改进** |
| **模型** | Qwen2.5-7B-Instruct (开源) | ⭐ **开源友好** |
| **结果** | 46% (WikiTQ 50样本) | ⚠️ **需要提升** |

**我们的代码** (`diagnostic_system.py`):
```python
def diagnose(self, execution_result, code, table, question):
    # Layer 1: 错误分类
    error_class = self.classifier.classify(execution_result)

    # Layer 2: 根因分析
    root_cause = self.root_cause_analyzer.analyze(...)

    # Layer 3: 策略选择
    strategy = self.strategy_selector.select_strategy(...)

    # Layer 4: 提示生成
    repair_prompt = self.prompt_generator.generate(...)

    return {'repair_prompt': repair_prompt}
```

**特点**:
- ✅ **系统化** - 4层诊断，逐层深入
- ✅ **可扩展** - 策略库可以扩展到20+种
- ✅ **理论支持** - 借鉴编译器错误诊断思想
- ⚠️ **复杂度高** - 可能过度设计

---

## 🎯 创新性评估

### ✅ 我们**确实有创新**的地方：

1. **4层分层诊断系统** ⭐⭐⭐
   - AILS: 简单的code_fix(code, error)
   - 我们: Error Classification → Root Cause → Strategy → Prompt
   - **这是真正的创新点**

2. **多策略修复** ⭐⭐
   - AILS: 1个通用修复prompt
   - 我们: 5个专用策略（可扩展20个）
   - **有一定创新**

3. **动态迭代控制** ⭐
   - AILS: 固定2次
   - 我们: 1-3次，平均1.28（更高效）
   - **小创新**

### ❌ 我们**没有创新**的地方：

1. **Prompt工程技巧**
   - 列选择、unique values、函数模板、列名清理
   - **完全借鉴自AILS-NTUA**

2. **Language-to-Code范式**
   - 生成Python代码执行
   - **AILS已经做了**

3. **代码执行框架**
   - 安全沙箱、超时保护
   - **借鉴OpenCodeInterpreter**

---

## ⚠️ 严峻的现实

### 问题1: 数据集不同，无法直接对比

| 方面 | AILS-NTUA | 我们 |
|------|-----------|------|
| 数据集 | SemEval-2025 DataBench | WikiTQ |
| 准确率 | 85.63% | 46% |
| 可比性 | ❌ **不可直接对比** | |

**原因**:
- SemEval-2025和WikiTQ是**不同的数据集**
- 难度、问题类型、评估方式都不同
- **我们不能说我们比AILS差40%，因为数据集不同**

### 问题2: 我们在WikiTQ上也不如AILS

如果AILS-NTUA在WikiTQ上运行（我们的文档估计）：
- AILS (估计): ~80%
- 我们: 46%
- **差距: -34%**

**这说明**:
- ✅ 我们的4层诊断系统**有效** (92%执行成功)
- ❌ 但**答案正确率仍然很低** (46%)
- ❌ 主要是**模型能力不足**，不是系统问题

---

## 📝 论文发表策略

### 情况A: 强调"系统设计"创新

**优势**:
- ✅ 4层诊断是真创新
- ✅ 92%执行成功率与AILS持平
- ✅ 1.28迭代次数更少

**劣势**:
- ❌ 46%准确率太低
- ❌ 低于WikiTQ baseline (54%)

**适合会议**: CCF-C (NLPCC, CCL)

**论文标题**: *"Hierarchical Error Diagnosis for Table Question Answering via Code Generation"*

**核心卖点**:
> "Unlike AILS-NTUA's single-step error correction, we propose a **4-layer hierarchical diagnostic system** achieving **92% execution success** with **1.28 average iterations** on WikiTQ."

---

### 情况B: 先提升到60%+再投稿

**必须做的**:
1. 使用14B/32B模型 → 58%
2. 添加Few-shot examples → 60%
3. 完整评估2000+样本

**时间**: 2-3周

**适合会议**: CCF-B (COLING, EMNLP Findings)

---

## 🎓 诚实的结论

### 我们**有创新**吗？

**有，但不够强**:
- ✅ **4层诊断系统是真创新** (相比AILS的单步修复)
- ✅ **系统设计更模块化** (便于扩展和研究)
- ⚠️ **但实际效果不如AILS** (46% vs 80%估计)

### 为什么效果不如AILS？

1. **模型能力**:
   - AILS: Claude-3.5-Sonnet (最强专有模型)
   - 我们: Qwen2.5-7B (开源小模型)

2. **Prompt优化**:
   - AILS: 在SemEval-2025上专门调优
   - 我们: 使用通用prompt

3. **数据集**:
   - AILS: 在更简单的SemEval数据上
   - 我们: 在更难的WikiTQ上

### 可以发表吗？

**当前状态: ❌ 不建议直接投稿**

**原因**:
- 46%低于WikiTQ baseline (54%)
- 无法证明我们的方法优于现有方法

**改进后: ✅ 可以投CCF-C/B**

**需要**:
1. 提升到60%+ (使用14B模型)
2. 在SemEval-2025上复现对比
3. 完整评估

---

## 💡 建议

### 选项1: 强调"系统设计"，弱化准确率
- 投稿CCF-C workshop
- 标题强调"Diagnostic System"
- 承认准确率不足，强调未来工作

### 选项2: 花2-3周改进到60%+
- 使用14B模型
- Few-shot examples
- 投稿CCF-B

### 选项3: 全力做GRPO，冲击68-72%
- 实现完整GRPO训练
- 投稿CCF-A
- 时间: 1-2个月

---

**最后更新**: 2025-10-20
**结论**: 有创新但需要提升准确率
