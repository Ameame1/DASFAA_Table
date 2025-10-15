# GRPO-Enhanced Iterative Table QA System

## 1. 研究动机

### 1.1 现有方法的局限
- **Chain-of-Table**: 仅使用预定义操作,缺乏错误恢复机制
- **Dater/Binder**: 单次生成,无法从执行错误中学习
- **传统CoT**: 在复杂表格推理中容易出错,无反馈循环

### 1.2 我们的核心创新
1. **迭代式错误修正**: 通过执行反馈自动修复代码错误
2. **GRPO强化学习**: 利用执行成功/失败作为奖励信号优化策略
3. **混合推理策略**: 结合表格操作链和代码执行的优势

---

## 2. 系统架构

### 2.1 整体Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    Input: (Table, Question)                  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: Initial Code Generation                            │
│  ┌─────────────────────────────────────────┐                │
│  │ LLM generates Python/SQL code           │                │
│  │ - Pandas operations for table QA        │                │
│  │ - Chain-of-Table style operations       │                │
│  └─────────────────────────────────────────┘                │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 2: Execution & Error Detection                        │
│  ┌─────────────────────────────────────────┐                │
│  │ Execute code in sandbox                 │                │
│  │ ├─ Success → Extract answer             │                │
│  │ └─ Error → Capture error trace          │                │
│  └─────────────────────────────────────────┘                │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 3: Iterative Refinement (Max 3 iterations)            │
│  ┌─────────────────────────────────────────┐                │
│  │ If error:                               │                │
│  │  1. Error Classification                │                │
│  │     - Syntax Error                      │                │
│  │     - Runtime Error                     │                │
│  │     - Logic Error                       │                │
│  │  2. Feedback Generation                 │                │
│  │     - Error message + code snippet      │                │
│  │     - Hint generation (optional)        │                │
│  │  3. Code Regeneration with LLM          │                │
│  │  4. Re-execute → Loop or Exit           │                │
│  └─────────────────────────────────────────┘                │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 4: GRPO Training (Offline RL)                         │
│  ┌─────────────────────────────────────────┐                │
│  │ Reward Signal:                          │                │
│  │  r(s,a) = α·r_exec + β·r_accuracy +     │                │
│  │           γ·r_efficiency                │                │
│  │                                         │                │
│  │ Group-based Advantage Estimation:       │                │
│  │  A_i = r_i - mean(r_group)              │                │
│  │                                         │                │
│  │ Policy Update:                          │                │
│  │  L_GRPO = -E[min(ratio·A, clip(ratio)·A)]│                │
│  └─────────────────────────────────────────┘                │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    Output: Final Answer                       │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 关键组件设计

#### 2.2.1 Code Generator Module
```python
class CodeGenerator:
    def __init__(self, model_name, max_iterations=3):
        self.model = load_llm(model_name)
        self.max_iterations = max_iterations
        self.operation_pool = [
            "f_select_row", "f_select_column",
            "f_group_by", "f_sort_by", "f_add_column"
        ]

    def generate_initial_code(self, table, question):
        """Generate initial Python code for table QA"""
        prompt = self._build_prompt(table, question, error_history=None)
        code = self.model.generate(prompt)
        return code

    def refine_code(self, table, question, error_info, previous_code):
        """Refine code based on error feedback"""
        prompt = self._build_refinement_prompt(
            table, question, error_info, previous_code
        )
        refined_code = self.model.generate(prompt)
        return refined_code
```

#### 2.2.2 Execution Engine
```python
class ExecutionEngine:
    def __init__(self, timeout=30):
        self.timeout = timeout
        self.error_classifier = ErrorClassifier()

    def execute(self, code, table):
        """Execute code in sandbox and return result or error"""
        try:
            # Create isolated environment
            namespace = {"pd": pd, "table": table}
            exec(code, namespace)
            result = namespace.get("answer", None)
            return ExecutionResult(
                success=True,
                answer=result,
                error=None
            )
        except Exception as e:
            error_type = self.error_classifier.classify(e)
            return ExecutionResult(
                success=False,
                answer=None,
                error=ErrorInfo(type=error_type, message=str(e), traceback=...)
            )
```

#### 2.2.3 GRPO Trainer
```python
class GRPOTrainer:
    def __init__(self, model, ref_model, beta=0.01):
        self.policy_model = model
        self.reference_model = ref_model
        self.beta = beta  # KL penalty coefficient

    def compute_rewards(self, trajectories):
        """Compute multi-component rewards"""
        rewards = []
        for traj in trajectories:
            r_exec = 1.0 if traj.success else -0.5
            r_accuracy = self._compute_accuracy(traj.answer, traj.gold_answer)
            r_efficiency = -0.1 * traj.num_iterations  # Penalty for more iterations

            total_reward = 0.5 * r_exec + 0.4 * r_accuracy + 0.1 * r_efficiency
            rewards.append(total_reward)
        return rewards

    def compute_group_advantage(self, rewards, group_size=4):
        """GRPO: Group-based advantage estimation"""
        advantages = []
        for i in range(0, len(rewards), group_size):
            group_rewards = rewards[i:i+group_size]
            group_mean = np.mean(group_rewards)
            group_advantages = [r - group_mean for r in group_rewards]
            advantages.extend(group_advantages)
        return advantages

    def update_policy(self, trajectories, advantages):
        """GRPO policy update with clipped objective"""
        for traj, adv in zip(trajectories, advantages):
            # Compute probability ratio
            log_prob_old = traj.log_prob
            log_prob_new = self.policy_model.compute_log_prob(traj.state, traj.action)
            ratio = torch.exp(log_prob_new - log_prob_old)

            # Clipped surrogate objective
            clip_ratio = torch.clamp(ratio, 1-0.2, 1+0.2)
            loss = -torch.min(ratio * adv, clip_ratio * adv).mean()

            # Add KL divergence penalty
            kl_div = self._compute_kl(traj.state)
            total_loss = loss + self.beta * kl_div

            # Backprop and update
            total_loss.backward()
            self.optimizer.step()
```

---

## 3. 奖励函数设计

### 3.1 多维度奖励

```python
def compute_reward(execution_result, gold_answer, num_iterations):
    """
    Multi-component reward function
    """
    # Component 1: Execution Success (Binary)
    r_exec = 1.0 if execution_result.success else -0.5

    # Component 2: Answer Accuracy
    if execution_result.success:
        if is_exact_match(execution_result.answer, gold_answer):
            r_accuracy = 1.0
        elif is_partial_match(execution_result.answer, gold_answer):
            r_accuracy = 0.5
        else:
            r_accuracy = 0.0
    else:
        r_accuracy = 0.0

    # Component 3: Efficiency (Fewer iterations = better)
    r_efficiency = -0.1 * num_iterations

    # Component 4: Code Quality (Optional)
    r_quality = evaluate_code_quality(execution_result.code)

    # Weighted sum
    total_reward = (
        0.4 * r_exec +
        0.4 * r_accuracy +
        0.1 * r_efficiency +
        0.1 * r_quality
    )

    return total_reward
```

### 3.2 奖励权重调整策略

| 训练阶段 | r_exec | r_accuracy | r_efficiency | 策略 |
|---------|--------|-----------|--------------|------|
| Early (0-30%) | 0.6 | 0.3 | 0.1 | 优先保证代码能执行 |
| Mid (30-70%) | 0.4 | 0.5 | 0.1 | 平衡执行和准确率 |
| Late (70-100%) | 0.3 | 0.5 | 0.2 | 优化效率和准确率 |

---

## 4. 实验设计

### 4.1 数据集

| 数据集 | 任务类型 | 样本数 | 用途 | 特点 |
|--------|---------|-------|------|------|
| **WikiTQ** | Table QA | 22,033 | 主评估 | 短答案,复杂推理 |
| **TabFact** | Fact Verification | 117,854 | 主评估 | 二分类,逻辑推理 |
| **FeTaQA** | Free-form QA | 10,000 | 补充评估 | 长答案生成 |
| **SemEval 2025 Task 8** | Table QA | ~5,000 | 最新benchmark | 真实场景 |

**数据划分**:
- Training: 70%
- Validation: 15% (用于GRPO训练监控)
- Test: 15% (最终评估)

### 4.2 Baseline方法

#### 4.2.1 Tier 1: 基础方法
1. **Direct QA** (Zero-shot)
   - 直接用GPT-4/Claude回答
   - 无代码生成,无迭代

2. **Few-shot CoT**
   - Chain-of-Thought prompting
   - 5-shot demonstrations

#### 4.2.2 Tier 2: 代码生成方法
3. **Text-to-SQL** (Rajkumar et al., 2022)
   - 生成SQL查询
   - 无错误修复

4. **Binder** (Cheng et al., 2022)
   - SQL/Python + LLM API调用
   - 单次生成,无迭代

5. **Dater** (Ye et al., 2023)
   - 表格分解 + SQL生成
   - 固定流程,无自适应

#### 4.2.3 Tier 3: 表格推理方法
6. **Chain-of-Table** (ICLR 2024)
   - 表格操作链
   - 动态规划,但无错误恢复

7. **TabSQLify** (NAACL 2024)
   - 表格分解 + SQL
   - WikiTQ: 64.7%, TabFact: 79.5%

#### 4.2.4 Tier 4: 迭代/RL方法
8. **AILS-NTUA** (SemEval 2025 Winner)
   - Language-to-Code + Error Fixing
   - 最多2次迭代
   - **我们的主要对比对象**

9. **Table-R1** (2025)
   - Region-based RL
   - GRPO训练
   - **我们的RL对比对象**

#### 4.2.5 我们的方法变体
10. **Ours-NoGRPO**
    - 仅迭代修正,无RL训练
    - 用于消融实验

11. **Ours-GRPO** (Full Model)
    - 迭代修正 + GRPO训练
    - **我们的完整方法**

### 4.3 模型配置

| 模型 | 参数量 | 用途 | 备注 |
|------|--------|------|------|
| **GPT-4o** | - | Baseline对比 | API调用 |
| **Claude 3.5 Sonnet** | - | Baseline对比 | API调用 |
| **Llama-3.1-70B-Instruct** | 70B | 主实验模型 | 开源,可微调 |
| **Qwen2.5-Coder-32B** | 32B | 代码生成专用 | 对比实验 |
| **DeepSeek-Coder-33B** | 33B | 支持GRPO训练 | 参考Table-R1 |

---

## 5. 评估指标

### 5.1 准确率指标

#### 5.1.1 主指标
| 指标 | 定义 | 计算方式 | 适用数据集 |
|------|------|---------|-----------|
| **Exact Match (EM)** | 完全匹配准确率 | EM = \|{pred == gold}\| / N | WikiTQ, TabFact |
| **Denotation Accuracy** | 语义等价准确率 | 考虑不同表述的等价答案 | WikiTQ |
| **F1 Score** | 精确率和召回率调和平均 | 2·P·R/(P+R) | FeTaQA |

#### 5.1.2 生成质量指标 (FeTaQA)
| 指标 | 说明 |
|------|------|
| **BLEU** | n-gram重叠度 |
| **ROUGE-1/2/L** | 召回率为主的文本相似度 |
| **BERTScore** | 基于语义的相似度 |

### 5.2 效率指标

| 指标 | 定义 | 目标 |
|------|------|------|
| **Avg Iterations** | 平均迭代次数 | ≤ 2.0 |
| **Success@1** | 首次执行成功率 | ≥ 60% |
| **Success@3** | 3次内执行成功率 | ≥ 90% |
| **Avg Execution Time** | 平均执行时间(秒) | ≤ 5s |
| **API Calls per Query** | 每个查询的LLM调用次数 | ≤ 3 |

### 5.3 错误分析指标

| 错误类型 | 定义 | 统计方法 |
|---------|------|---------|
| **Syntax Error Rate** | 语法错误比例 | Count / Total |
| **Runtime Error Rate** | 运行时错误比例 | Count / Total |
| **Logic Error Rate** | 逻辑错误(执行成功但答案错) | Count / Total |
| **Timeout Rate** | 超时比例 | Count / Total |
| **Recovery Rate** | 从错误中恢复的比例 | Fixed / Total Errors |

### 5.4 GRPO训练指标

| 指标 | 说明 | 监控频率 |
|------|------|---------|
| **Average Reward** | 平均奖励值 | 每100 steps |
| **KL Divergence** | 策略偏离度 | 每100 steps |
| **Policy Entropy** | 策略熵(探索度) | 每100 steps |
| **Value Loss** | 不适用(GRPO无value function) | - |
| **Gradient Norm** | 梯度范数 | 每step |

---

## 6. 预期结果

### 6.1 性能目标

| 数据集 | SOTA | 我们的目标 | 改进幅度 |
|--------|------|-----------|---------|
| **WikiTQ** | 67.31% (CoT, PaLM2) | **70-72%** | +2.7-4.7% |
| **TabFact** | 86.61% (CoT, PaLM2) | **88-90%** | +1.4-3.4% |
| **FeTaQA (BLEU)** | 32.61 | **35-37** | +2.4-4.4 |
| **SemEval 2025 Task 8** | 86.21% (Ensemble) | **87-89%** | +0.8-2.8% |

### 6.2 消融实验预期

| 变体 | WikiTQ | TabFact | 说明 |
|------|--------|---------|------|
| Direct QA (GPT-4) | 60.5% | 77.9% | Baseline |
| + Code Generation | 64.2% | 82.3% | +代码生成 |
| + Error Correction (1 iter) | 67.8% | 85.1% | +1次迭代 |
| + Error Correction (3 iter) | 69.5% | 86.8% | +3次迭代 |
| + GRPO (Ours Full) | **71.2%** | **88.5%** | +RL优化 |

### 6.3 效率对比

| 方法 | Avg Iterations | Avg Time (s) | Success@1 |
|------|---------------|--------------|-----------|
| Chain-of-Table | 3.2 | 4.5 | - |
| AILS-NTUA | 1.6 | 3.2 | 58% |
| **Ours-GRPO** | **1.8** | **3.5** | **65%** |

---

## 7. 实验细节

### 7.1 训练配置

```python
# GRPO Training Config
training_config = {
    "model": "Llama-3.1-70B-Instruct",
    "learning_rate": 1e-6,
    "batch_size": 16,
    "group_size": 4,  # GRPO group size
    "max_iterations": 3,
    "num_epochs": 5,
    "warmup_steps": 1000,
    "clip_range": 0.2,
    "kl_coef": 0.01,
    "reward_weights": {
        "execution": 0.4,
        "accuracy": 0.4,
        "efficiency": 0.1,
        "quality": 0.1
    }
}
```

### 7.2 Prompt设计

#### Initial Code Generation Prompt
```
You are a table reasoning expert. Given a table and a question,
generate executable Python code using pandas to answer the question.

Table:
{table_markdown}

Question: {question}

Requirements:
1. Use pandas DataFrame operations
2. Store final answer in variable 'answer'
3. Handle edge cases (empty results, type errors)
4. Add comments explaining your logic

Code:
```python
```

#### Error Correction Prompt
```
The previous code failed with the following error:

Error Type: {error_type}
Error Message: {error_message}
Traceback: {traceback}

Previous Code:
```python
{previous_code}
```

Please analyze the error and generate corrected code.
Focus on fixing the {error_type} while maintaining the original logic.

Corrected Code:
```python
```

### 7.3 评估协议

```python
def evaluate_model(model, test_dataset):
    results = {
        "accuracy": [],
        "iterations": [],
        "success_at_k": {1: 0, 2: 0, 3: 0},
        "error_types": Counter()
    }

    for sample in test_dataset:
        table, question, gold_answer = sample

        # Generate and execute with iteration
        for iter_num in range(1, 4):
            code = model.generate_code(table, question, iter_num)
            exec_result = execute_code(code, table)

            if exec_result.success:
                results["success_at_k"][iter_num] += 1
                is_correct = check_answer(exec_result.answer, gold_answer)
                results["accuracy"].append(is_correct)
                results["iterations"].append(iter_num)
                break
            else:
                results["error_types"][exec_result.error_type] += 1
                if iter_num == 3:
                    results["accuracy"].append(0)
                    results["iterations"].append(3)

    return compute_metrics(results)
```

---

## 8. 时间计划

| 阶段 | 任务 | 时间 | 输出 |
|------|------|------|------|
| **Phase 1** | 数据准备与Baseline复现 | 2周 | Baseline结果 |
| **Phase 2** | 迭代修正系统开发 | 3周 | Ours-NoGRPO |
| **Phase 3** | GRPO训练流程实现 | 3周 | Ours-GRPO |
| **Phase 4** | 完整实验与消融研究 | 2周 | 实验结果 |
| **Phase 5** | 论文撰写 | 2周 | 初稿 |
| **总计** | | **12周** | 可投稿论文 |

---

## 9. 预期贡献

### 9.1 技术贡献
1. **首个将GRPO应用于表格QA迭代修正的工作**
2. **多维度奖励函数设计** (execution + accuracy + efficiency)
3. **错误分类与针对性修复策略**

### 9.2 实验贡献
1. **4个标准数据集的全面评估**
2. **与9个SOTA方法的系统对比**
3. **详细的消融实验和错误分析**

### 9.3 开源贡献
- 完整代码实现 (GitHub)
- GRPO训练检查点
- 错误修复案例数据集

---

## 10. 参考文献

### 核心参考
1. Chain-of-Table (ICLR 2024)
2. AILS-NTUA (SemEval 2025 Winner) - arXiv:2503.00435
3. Table-R1 (2025) - arXiv:2505.12415
4. DeepSeekMath GRPO (2024)
5. TabSQLify (NAACL 2024)

### 补充参考
6. Binder (ICLR 2022)
7. Dater (2023)
8. OpenCodeInterpreter (2024)
9. Self-Refine (2023)
10. STRuCT-LLM (2025)
