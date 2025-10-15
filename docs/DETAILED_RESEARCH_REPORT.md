# 深度调研报告:基于AILS-NTUA的GRPO增强实现方案

> **调研日期**: 2025-01
> **目标**: 为Table QA任务提供详细的技术实现路线

---

## 一、核心工作深度分析

### 🔍 **1. AILS-NTUA (SemEval 2025冠军)**

#### **架构分析**
```python
# 论文提到的双模块架构

┌────────────────────────────────────────┐
│  Main Module: Query → Python Code      │
│  ┌──────────────────────────────────┐  │
│  │ Input: Table + Question          │  │
│  │ ↓                                │  │
│  │ Step 1: Decompose Task           │  │
│  │   - 将复杂问题分解为子问题          │  │
│  │ ↓                                │  │
│  │ Step 2: Generate Python Function │  │
│  │   - 使用LLM prompting生成代码      │  │
│  │ ↓                                │  │
│  │ Step 3: Execute Code             │  │
│  │   - 在sandbox中执行                │  │
│  └──────────────────────────────────┘  │
└────────────────────────────────────────┘
          ↓ (如果执行失败)
┌────────────────────────────────────────┐
│  Error-Fixing Module: Code Refinement  │
│  ┌──────────────────────────────────┐  │
│  │ Input: Error Message + Old Code  │  │
│  │ ↓                                │  │
│  │ Self-Correction Mechanism        │  │
│  │   - 分析错误                      │  │
│  │   - 生成修复提示                  │  │
│  │   - 重新生成代码                  │  │
│  │ ↓                                │  │
│  │ Re-execute (最多2次迭代)          │  │
│  └──────────────────────────────────┘  │
└────────────────────────────────────────┘
```

#### **关键技术细节**
1. **Task Decomposition**: 将问题分解为子任务
2. **LLM Prompting**: 精心设计的prompt templates
3. **Self-Correction**: 简单的error message反馈
4. **迭代上限**: 固定2次

#### **代码可用性**
- ✅ 论文提到代码在GitHub (但具体repo未找到)
- ✅ 团队来自NTUA (National Technical University of Athens)
- ✅ 可以尝试联系作者: Andreas Evangelatos等

#### **我们的改进点**
```
AILS-NTUA的局限           →  我们的改进
────────────────────────────────────────────
简单error message反馈      →  4层分层诊断
固定2次迭代                →  动态1-5次迭代
无结构化修复策略           →  20+专用策略库
无RL优化                  →  GRPO训练迭代策略
```

---

### 🔍 **2. Table-R1 (GRPO for Table QA)**

#### **核心创新: TARPO算法**

```python
# Table-Aware Region Policy Optimization

class TARPO:
    """
    Table-R1的核心:扩展GRPO用于表格理解
    """

    def __init__(self):
        # Stage 1: Region-Enhanced SFT
        self.re_sft = RegionEnhancedSFT()

        # Stage 2: GRPO训练
        self.grpo_trainer = GRPOTrainer()

    def train(self, table, question):
        # Step 1: 识别相关表格区域
        regions = self.identify_regions(table, question)

        # Step 2: 基于区域生成推理
        reasoning_steps = self.generate_reasoning(table, regions, question)

        # Step 3: 计算mixed reward
        reward = self.compute_mixed_reward(
            region_accuracy=regions.accuracy,
            answer_correctness=reasoning_steps.answer_correct,
            consistency=regions.consistency
        )

        return reward

    def compute_mixed_reward(self, region_accuracy, answer_correctness, consistency):
        """
        Mixed Reward System (Table-R1的创新)
        """
        # Decaying region rewards (训练后期降低region权重)
        alpha = self.decay_schedule(epoch)

        reward = (
            alpha * region_accuracy +        # 区域准确率
            (1 - alpha) * answer_correctness + # 答案正确率
            -0.1 * (1 - consistency)         # 一致性惩罚
        )

        return reward
```

#### **GRPO实现细节 (基于DeepSeek)**

```python
class GRPOTrainer:
    """
    Group Relative Policy Optimization
    核心:用group mean作为baseline,不需要critic network
    """

    def __init__(self, group_size=4):
        self.group_size = group_size

    def train_step(self, prompts, policy_model, ref_model):
        # Step 1: 对每个prompt生成多个responses (group_size个)
        responses = []
        for prompt in prompts:
            group_responses = []
            for _ in range(self.group_size):
                response = policy_model.generate(prompt)
                group_responses.append(response)
            responses.append(group_responses)

        # Step 2: 计算每个response的reward
        rewards = []
        for group in responses:
            group_rewards = [compute_reward(r) for r in group]
            rewards.append(group_rewards)

        # Step 3: Group-based advantage estimation
        advantages = []
        for group_rewards in rewards:
            group_mean = np.mean(group_rewards)
            group_std = np.std(group_rewards) + 1e-8

            # 关键:用group mean作为baseline!
            group_advantages = [
                (r - group_mean) / group_std
                for r in group_rewards
            ]
            advantages.extend(group_advantages)

        # Step 4: Policy update with PPO-style clipping
        for response, adv in zip(flatten(responses), advantages):
            # 计算probability ratio
            log_prob_new = policy_model.log_prob(response)
            log_prob_old = response.log_prob  # 存储的old log prob

            ratio = torch.exp(log_prob_new - log_prob_old)

            # Clipped objective
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * adv
            loss = -torch.min(surr1, surr2)

            # KL divergence penalty
            kl = log_prob_new - ref_model.log_prob(response)
            total_loss = loss + beta * kl

            # Backprop
            total_loss.backward()

        # Update
        optimizer.step()

        return {
            'avg_reward': np.mean(flatten(rewards)),
            'avg_advantage': np.mean(advantages),
            'kl_div': kl.mean().item()
        }
```

#### **TARPO vs 标准GRPO的区别**

| 特性 | 标准GRPO | TARPO (Table-R1) | 我们的改进 |
|------|----------|------------------|-----------|
| **奖励组件** | 单一reward | region + answer + consistency | **+ repair quality + efficiency** |
| **应用对象** | 单次生成 | 单次生成+区域识别 | **迭代修复过程** |
| **训练策略** | 均匀采样 | Region-aware | **Curriculum learning** |
| **效率优化** | 无 | 减少67.5% token | **动态迭代预算** |

---

### 🔍 **3. OpenCodeInterpreter (代码执行框架)**

#### **核心架构**

```python
# OpenCodeInterpreter的execution feedback循环

class CodeInterpreter:
    """
    核心:代码生成 + 执行 + 反馈循环
    """

    def __init__(self, model):
        self.model = model
        self.executor = CodeExecutor()

    def generate_with_feedback(self, prompt, max_iterations=3):
        code = self.model.generate(prompt)
        history = []

        for iteration in range(max_iterations):
            # Execute code
            result = self.executor.execute(code)

            history.append({
                'code': code,
                'result': result,
                'iteration': iteration
            })

            # Success
            if result.success:
                return code, result

            # Failure - generate feedback
            feedback = self.generate_feedback(result.error, code)

            # Refine code based on feedback
            refinement_prompt = f"""
Previous code:
{code}

Execution error:
{result.error}

Feedback:
{feedback}

Generate corrected code:
"""
            code = self.model.generate(refinement_prompt)

        return code, result

    def generate_feedback(self, error, code):
        """
        生成结构化反馈 (我们可以增强这里!)
        """
        # 基础版本:简单error message
        feedback = f"Error: {str(error)}"

        # 我们的增强版本:分层诊断
        # feedback = self.error_diagnoser.diagnose(error, code)

        return feedback
```

#### **可复用组件**

1. **代码执行器**
   ```python
   class CodeExecutor:
       def execute(self, code, timeout=30):
           # Sandbox execution
           # Exception handling
           # Result extraction
   ```

2. **反馈生成器**
   ```python
   class FeedbackGenerator:
       def generate(self, error, code):
           # Error analysis
           # Suggestion generation
   ```

3. **迭代控制器**
   ```python
   class IterationController:
       def should_continue(self, history):
           # Decide whether to continue iteration
   ```

**我们直接借鉴**: 执行引擎 + 基础迭代框架
**我们增强**: 反馈生成器 (4层诊断) + GRPO控制

---

### 🔍 **4. Chain-of-Table (表格操作)**

#### **操作定义**

```python
# Chain-of-Table的5个核心操作

class TableOperations:
    """
    预定义的表格操作
    """

    @staticmethod
    def f_select_row(table, row_indices):
        """
        选择特定行
        Example: f_select_row(table, [1, 3, 5])
        """
        return table.iloc[row_indices]

    @staticmethod
    def f_select_column(table, column_names):
        """
        选择特定列
        Example: f_select_column(table, ['Name', 'Age'])
        """
        return table[column_names]

    @staticmethod
    def f_add_column(table, column_name, values):
        """
        添加新列
        Example: f_add_column(table, 'GDP_per_capita', gdp/population)
        """
        table[column_name] = values
        return table

    @staticmethod
    def f_group_by(table, column_name):
        """
        分组统计
        Example: f_group_by(table, 'Country')
        Returns: DataFrame with counts
        """
        grouped = table.groupby(column_name).size().reset_index(name='Count')
        return grouped

    @staticmethod
    def f_sort_by(table, column_name, ascending=True):
        """
        排序
        Example: f_sort_by(table, 'GDP', ascending=False)
        """
        return table.sort_values(by=column_name, ascending=ascending)
```

#### **Dynamic Planning机制**

```python
class DynamicPlanner:
    """
    Chain-of-Table的核心:动态选择下一个操作
    """

    def __init__(self, llm):
        self.llm = llm
        self.operation_pool = [
            'f_select_row',
            'f_select_column',
            'f_add_column',
            'f_group_by',
            'f_sort_by'
        ]

    def plan_next_operation(self, table, question, operation_history):
        """
        选择下一个操作
        """
        prompt = f"""
Current table:
{table.to_markdown()}

Question: {question}

Previous operations: {operation_history}

Available operations: {self.operation_pool}

Choose the next operation:
"""

        next_op = self.llm.generate(prompt)
        return next_op

    def execute_operation_chain(self, table, question):
        """
        执行完整的operation chain
        """
        operation_history = []
        current_table = table.copy()

        while True:
            # Plan next operation
            next_op = self.plan_next_operation(
                current_table, question, operation_history
            )

            if next_op == '[END]':
                break

            # Execute operation
            current_table = self.execute(current_table, next_op)
            operation_history.append(next_op)

        return current_table, operation_history
```

#### **我们的魔改方案**

```python
class EnhancedTableOperations(TableOperations):
    """
    扩展Chain-of-Table操作,加入错误恢复
    """

    @staticmethod
    def f_rollback(table, operation_history, steps=1):
        """
        🆕 回滚操作 (Chain-of-Table没有!)
        """
        # 回退到steps步之前的状态
        pass

    @staticmethod
    def f_python_code(table, code_snippet):
        """
        🆕 执行灵活的Python代码 (突破固定操作限制)
        """
        namespace = {'df': table.copy(), 'pd': pd, 'np': np}
        exec(code_snippet, namespace)
        return namespace['df']

    @staticmethod
    def f_fuzzy_match_column(table, column_pattern):
        """
        🆕 模糊匹配列名 (处理列名错误)
        """
        from difflib import get_close_matches
        matches = get_close_matches(column_pattern, table.columns)
        return matches[0] if matches else None

    @staticmethod
    def f_data_cleaning(table, column_name):
        """
        🆕 数据清洗 (处理空值、类型错误)
        """
        # 填充空值
        table[column_name] = table[column_name].fillna(method='ffill')
        # 类型转换
        table[column_name] = pd.to_numeric(table[column_name], errors='coerce')
        return table
```

---

## 二、我们的具体实现方案

### 🎯 **Step 1: 基础框架 (Week 1-2)**

#### **1.1 复现AILS-NTUA Baseline**

```python
# ails_ntua_baseline.py

class AILS_NTUA_Baseline:
    """
    AILS-NTUA的基础实现
    """

    def __init__(self, model_name="gpt-4"):
        self.llm = OpenAI(model=model_name)
        self.executor = CodeExecutor()  # 借用OpenCodeInterpreter
        self.max_iterations = 2  # AILS-NTUA固定2次

    def answer(self, table, question):
        # Step 1: Task Decomposition
        subtasks = self.decompose_task(table, question)

        # Step 2: Generate Python code
        code = self.generate_code(table, subtasks)

        # Step 3: Execute with error fixing
        for iteration in range(self.max_iterations):
            result = self.executor.execute(code, table)

            if result.success:
                return result.answer

            # Error fixing (简单版本)
            code = self.fix_code(code, result.error)

        return None

    def decompose_task(self, table, question):
        """简单的任务分解"""
        prompt = f"""
Decompose the question into subtasks:
Question: {question}
Table columns: {table.columns.tolist()}

Subtasks (3-5 steps):
"""
        subtasks = self.llm.generate(prompt)
        return subtasks

    def generate_code(self, table, subtasks):
        """生成Python代码"""
        prompt = f"""
Generate Python code to solve:
Table: {table.head().to_markdown()}
Subtasks: {subtasks}

Code:
```python
"""
        code = self.llm.generate(prompt)
        return extract_code(code)

    def fix_code(self, code, error):
        """简单的错误修复 (AILS-NTUA风格)"""
        prompt = f"""
Fix the error:
Code: {code}
Error: {error}

Corrected code:
```python
"""
        fixed_code = self.llm.generate(prompt)
        return extract_code(fixed_code)
```

**测试目标**: WikiTQ ~65% (与AILS-NTUA论文一致)

---

### 🎯 **Step 2: 智能错误诊断 (Week 3-4)**

#### **2.1 4层诊断系统实现**

```python
# intelligent_diagnoser.py

class IntelligentErrorDiagnoser:
    """
    我们的核心创新:4层错误诊断
    """

    def __init__(self):
        self.error_classifier = ErrorClassifier()
        self.root_cause_analyzer = RootCauseAnalyzer()
        self.strategy_selector = StrategySelector()
        self.prompt_generator = RepairPromptGenerator()

    def diagnose(self, error, code, table, question):
        """完整诊断流程"""
        # Level 1: Classification
        error_type = self.error_classifier.classify(error)

        # Level 2: Root Cause Analysis
        root_cause = self.root_cause_analyzer.analyze(
            error, error_type, code, table
        )

        # Level 3: Strategy Selection
        strategy = self.strategy_selector.select(root_cause)

        # Level 4: Prompt Generation
        repair_prompt = self.prompt_generator.generate(
            error, root_cause, strategy, code, table, question
        )

        return DiagnosisResult(
            error_type=error_type,
            root_cause=root_cause,
            strategy=strategy,
            repair_prompt=repair_prompt
        )


class ErrorClassifier:
    """Level 1: 错误分类"""

    def classify(self, error):
        error_map = {
            'KeyError': ErrorType.MISSING_COLUMN,
            'TypeError': ErrorType.TYPE_MISMATCH,
            'ValueError': ErrorType.INVALID_VALUE,
            'IndexError': ErrorType.INDEX_ERROR,
            'SyntaxError': ErrorType.SYNTAX_ERROR,
            'AttributeError': ErrorType.ATTRIBUTE_ERROR,
        }

        error_name = type(error).__name__
        return error_map.get(error_name, ErrorType.UNKNOWN)


class RootCauseAnalyzer:
    """Level 2: 根因分析"""

    def analyze(self, error, error_type, code, table):
        if error_type == ErrorType.MISSING_COLUMN:
            return self.analyze_missing_column(error, table)
        elif error_type == ErrorType.TYPE_MISMATCH:
            return self.analyze_type_mismatch(error, code, table)
        # ... 其他类型

    def analyze_missing_column(self, error, table):
        """分析缺失列错误"""
        # 提取错误中的列名
        missing_col = extract_column_from_error(error)

        # Fuzzy matching
        from difflib import get_close_matches
        suggestions = get_close_matches(
            missing_col,
            table.columns.tolist(),
            n=3,
            cutoff=0.6
        )

        return RootCause(
            type='missing_column',
            missing_column=missing_col,
            available_columns=table.columns.tolist(),
            suggestions=suggestions
        )


class StrategySelector:
    """Level 3: 策略选择"""

    def __init__(self):
        self.strategies = {
            'missing_column': ColumnNameCorrectionStrategy(),
            'type_mismatch': TypeConversionStrategy(),
            'invalid_value': ValueValidationStrategy(),
            'index_error': BoundaryCheckStrategy(),
            # ... 20+策略
        }

    def select(self, root_cause):
        strategy_map = {
            'missing_column': 'missing_column',
            'type_mismatch': 'type_mismatch',
            'invalid_value': 'invalid_value',
        }

        strategy_id = strategy_map.get(root_cause.type, 'generic')
        return self.strategies[strategy_id]


class RepairPromptGenerator:
    """Level 4: Prompt生成"""

    def generate(self, error, root_cause, strategy, code, table, question):
        templates = {
            'missing_column': self.missing_column_template,
            'type_mismatch': self.type_mismatch_template,
            # ... 其他模板
        }

        template = templates.get(root_cause.type, self.generic_template)
        return template(error, root_cause, strategy, code, table, question)

    def missing_column_template(self, error, root_cause, strategy, code, table, question):
        return f"""
## Error Analysis

**Error Type**: Missing Column
**Root Cause**: Column '{root_cause.missing_column}' does not exist

**Available Columns**: {root_cause.available_columns}
**Suggested Matches**: {root_cause.suggestions}

## Repair Strategy

Strategy: {strategy.name}
Recommended Action:
1. Replace '{root_cause.missing_column}' with '{root_cause.suggestions[0]}'
2. Or use case-insensitive matching
3. Or check if column needs to be created

## Previous Code

```python
{code}
```

## Your Task

Generate corrected code that fixes the KeyError.
Use column '{root_cause.suggestions[0]}' instead of '{root_cause.missing_column}'.

Corrected Code:
```python
"""
```

**测试目标**: 错误修复成功率从50% → 80%+

---

### 🎯 **Step 3: GRPO实现 (Week 7-9)**

#### **3.1 GRPO Trainer实现**

```python
# grpo_trainer.py

class GRPOTrainer:
    """
    基于DeepSeek GRPO实现
    """

    def __init__(
        self,
        policy_model,
        ref_model,
        group_size=4,
        clip_eps=0.2,
        kl_coef=0.01
    ):
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.group_size = group_size
        self.clip_eps = clip_eps
        self.kl_coef = kl_coef

    def train_step(self, batch_data):
        """
        单步GRPO训练
        """
        all_trajectories = []
        all_rewards = []

        # 对每个样本生成group_size个trajectory
        for sample in batch_data:
            group_trajectories = []
            group_rewards = []

            for _ in range(self.group_size):
                # 执行迭代修复
                result = self.system.iterative_repair(
                    sample.table,
                    sample.question,
                    sample.gold_answer
                )

                # 计算trajectory reward
                reward = self.compute_trajectory_reward(
                    result['trajectory'],
                    sample.gold_answer
                )

                group_trajectories.append(result['trajectory'])
                group_rewards.append(reward)

            all_trajectories.extend(group_trajectories)
            all_rewards.append(group_rewards)

        # Group-based advantage estimation
        advantages = self.compute_group_advantages(all_rewards)

        # Policy update
        loss = self.compute_policy_loss(all_trajectories, advantages)
        loss.backward()
        self.optimizer.step()

        return {
            'loss': loss.item(),
            'avg_reward': np.mean(flatten(all_rewards)),
            'avg_advantage': np.mean(advantages)
        }

    def compute_group_advantages(self, all_rewards):
        """
        Group-based advantage (GRPO核心)
        """
        advantages = []

        for group_rewards in all_rewards:
            # 用group mean作为baseline
            group_mean = np.mean(group_rewards)
            group_std = np.std(group_rewards) + 1e-8

            # Normalize
            group_adv = [
                (r - group_mean) / group_std
                for r in group_rewards
            ]

            advantages.extend(group_adv)

        return advantages

    def compute_trajectory_reward(self, trajectory, gold_answer):
        """
        计算trajectory的总reward
        """
        final_result = trajectory[-1]['result']

        # Component 1: Execution
        r_exec = 1.0 if final_result.success else -0.5

        # Component 2: Accuracy
        if final_result.success:
            r_acc = compute_accuracy(final_result.answer, gold_answer)
        else:
            r_acc = 0.0

        # Component 3: Efficiency
        num_iter = len(trajectory)
        r_eff = 1.0 / num_iter

        # Component 4: Repair Quality (我们的创新!)
        r_repair = 0.0
        for i in range(1, len(trajectory)):
            prev = trajectory[i-1]['result']
            curr = trajectory[i]['result']

            if not prev.success and curr.success:
                r_repair += 0.5  # 成功修复
            elif not prev.success and not curr.success:
                if is_improving(prev.error, curr.error):
                    r_repair += 0.2  # 错误在改善

        # Component 5: Code Quality
        final_code = trajectory[-1]['code']
        r_quality = evaluate_code_quality(final_code)

        # Weighted sum
        total_reward = (
            0.3 * r_exec +
            0.4 * r_acc +
            0.1 * r_eff +
            0.1 * r_repair +
            0.1 * r_quality
        )

        return total_reward
```

---

## 三、代码仓库结构

```
table-qa-grpo/
├── README.md
├── requirements.txt
├── setup.py
│
├── data/
│   ├── wikitq/
│   ├── tabfact/
│   └── fetaqa/
│
├── src/
│   ├── __init__.py
│   │
│   ├── baselines/
│   │   ├── ails_ntua_baseline.py      # AILS-NTUA复现
│   │   ├── chain_of_table.py          # Chain-of-Table复现
│   │   └── direct_qa.py               # Direct QA baseline
│   │
│   ├── core/
│   │   ├── code_generator.py          # 代码生成器
│   │   ├── code_executor.py           # 执行引擎 (借用OpenCodeInterpreter)
│   │   ├── table_operations.py        # 表格操作 (借用Chain-of-Table)
│   │   └── iteration_controller.py    # 迭代控制
│   │
│   ├── diagnosis/
│   │   ├── error_classifier.py        # Level 1: 分类
│   │   ├── root_cause_analyzer.py     # Level 2: 根因分析
│   │   ├── strategy_selector.py       # Level 3: 策略选择
│   │   ├── prompt_generator.py        # Level 4: Prompt生成
│   │   └── strategies/                # 20+修复策略
│   │       ├── column_name_correction.py
│   │       ├── type_conversion.py
│   │       └── ...
│   │
│   ├── grpo/
│   │   ├── grpo_trainer.py           # GRPO训练器
│   │   ├── reward_function.py        # Reward计算
│   │   └── curriculum_learning.py    # 课程学习
│   │
│   └── system/
│       ├── hybrid_qa_system.py       # 完整系统
│       └── config.py                 # 配置
│
├── experiments/
│   ├── run_baseline.py               # 运行baseline
│   ├── run_ablation.py               # 消融实验
│   ├── run_grpo_training.py          # GRPO训练
│   └── evaluate.py                   # 评估脚本
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_analysis.ipynb
│   └── 03_error_pattern_analysis.ipynb
│
└── tests/
    ├── test_diagnoser.py
    ├── test_executor.py
    └── test_grpo.py
```

---

## 四、12周详细实施计划

### **Week 1: 环境搭建 + 数据准备**
```bash
# Day 1-2: 环境配置
conda create -n table-qa python=3.10
conda activate table-qa
pip install torch transformers openai pandas scikit-learn

# Day 3-4: 数据下载
python scripts/download_wikitq.py
python scripts/download_tabfact.py
python scripts/download_fetaqa.py

# Day 5-7: 数据预处理
python scripts/preprocess_data.py --dataset wikitq
python scripts/preprocess_data.py --dataset tabfact
```

### **Week 2: Baseline复现**
```bash
# AILS-NTUA Baseline
python experiments/run_baseline.py --method ails_ntua --dataset wikitq
# 目标: ~65% accuracy

# Direct QA Baseline
python experiments/run_baseline.py --method direct_qa --dataset wikitq
# 目标: ~60% accuracy

# Chain-of-Table Baseline
python experiments/run_baseline.py --method chain_of_table --dataset wikitq
# 目标: ~67% accuracy
```

### **Week 3-4: 错误诊断系统**
```python
# 实现顺序
1. ErrorClassifier (Day 1-2)
2. RootCauseAnalyzer (Day 3-5)
3. 10个核心策略 (Day 6-10)
4. PromptGenerator (Day 11-12)
5. 测试诊断准确率 (Day 13-14)

# 测试
python tests/test_diagnoser.py
# 目标: 诊断准确率>85%
```

### **Week 5-6: 混合推理框架**
```python
# 集成Chain-of-Table操作
1. TableOperations扩展 (Day 1-3)
2. HybridCodeGenerator实现 (Day 4-7)
3. 整合测试 (Day 8-10)
4. 对比实验 (Day 11-14)

# 评估
python experiments/run_ablation.py --component hybrid_reasoning
# 目标: WikiTQ 68.8% (+1.5% vs baseline)
```

### **Week 7-9: GRPO实现**
```python
# Week 7: GRPO Trainer基础
1. Trajectory记录 (Day 1-2)
2. Reward计算 (Day 3-4)
3. Group advantage (Day 5-7)

# Week 8: Policy update
1. Policy loss (Day 1-3)
2. KL divergence (Day 4-5)
3. Optimizer配置 (Day 6-7)

# Week 9: Curriculum Learning
1. 难度分级 (Day 1-2)
2. 训练循环 (Day 3-5)
3. 完整训练 (Day 6-7)

# 训练
python experiments/run_grpo_training.py --epochs 5
# 目标: WikiTQ 71.2% (+2.4% vs no-GRPO)
```

### **Week 10: 完整实验**
```bash
# 4个数据集全面评估
python evaluate.py --dataset all --method all

# 消融实验
python experiments/run_ablation.py --all-components

# 错误分析
python analysis/error_pattern_mining.py
```

### **Week 11-12: 论文撰写**
```
Day 1-3: Introduction + Related Work
Day 4-7: Method (3页)
Day 8-10: Experiments (2页)
Day 11-12: Analysis + Conclusion
Day 13-14: 图表制作 + 校对
```

---

## 五、关键代码示例

### **完整系统使用**

```python
# main.py

from src.system import HybridTableQASystem
from src.grpo import GRPOTrainer
from data import load_wikitq

# 初始化系统
system = HybridTableQASystem(
    model_name="gpt-4",
    use_grpo=True,
    use_diagnosis=True,    # 使用智能诊断
    use_hybrid=True,       # 使用混合推理
    max_iterations=5
)

# 加载数据
train_data = load_wikitq(split='train')
val_data = load_wikitq(split='dev')
test_data = load_wikitq(split='test')

# GRPO训练
trainer = GRPOTrainer(system)
trainer.train(
    train_data=train_data,
    val_data=val_data,
    num_epochs=5,
    batch_size=16
)

# 评估
accuracy = system.evaluate(test_data)
print(f"Test Accuracy: {accuracy:.2%}")

# 单个样本推理
table = test_data[0]['table']
question = test_data[0]['question']

result = system.answer(table, question)
print(f"Answer: {result['answer']}")
print(f"Iterations: {result['iterations']}")
print(f"Success: {result['success']}")
```

---

## 六、总结

### ✅ **我们做的是什么?**

1. **基础**: AILS-NTUA (Language-to-Code + 简单Error Fixing)
2. **增强1**: 4层智能错误诊断 (vs 简单error message)
3. **增强2**: 混合推理 (Chain-of-Table操作 + 灵活代码)
4. **增强3**: GRPO优化迭代策略 (vs 固定策略)
5. **增强4**: 动态迭代预算 (vs 固定2次)

### 🎯 **预期成果**

- **WikiTQ**: 71.2% (SOTA 67.3%, **+3.9%**)
- **TabFact**: 88.5% (SOTA 86.6%, **+1.9%**)
- **效率**: 平均1.8次迭代 (vs 2.0次)
- **代码**: 完全开源,可复现

### 🚀 **立即开始!**

```bash
git clone https://github.com/your-username/table-qa-grpo
cd table-qa-grpo
pip install -r requirements.txt
python scripts/setup.py
python experiments/run_baseline.py
```

**完整方案已准备就绪,现在就开始实现吧!** 🎉
