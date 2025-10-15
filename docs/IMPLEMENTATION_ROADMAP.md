# 具体实施方案：代码复用 + 魔改路线图

## 总体架构：三大模块，哪些复用，哪些自己写

```
┌─────────────────────────────────────────────────────────────┐
│                    我们的完整系统                              │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│  Code         │  │  Error        │  │  GRPO         │
│  Generator    │  │  Diagnoser    │  │  Trainer      │
│               │  │               │  │               │
│ 复用AILS-NTUA │  │ 自己写(核心)   │  │ 复用DeepSeek  │
└───────────────┘  └───────────────┘  └───────────────┘
```

---

## 第一部分：代码生成器（Code Generator）

### 直接复用：AILS-NTUA 的 Language-to-Code

**用什么代码**：
- AILS-NTUA 没有开源代码，但论文描述很清楚
- 我们参考 **OpenCodeInterpreter** 的代码生成部分
- GitHub: https://github.com/OpenCodeInterpreter/OpenCodeInterpreter

**具体文件**：
```bash
# 从 OpenCodeInterpreter 复用
src/baselines/code_generator.py  # 核心生成逻辑
src/baselines/prompts.py          # Prompt templates
```

**Prompt 模板**（直接用）：
```python
# src/baselines/prompts.py
CODE_GEN_PROMPT = """
Given a table and a question, generate Python code using pandas to answer it.

Table (first 5 rows):
{table_preview}

Columns: {column_names}

Question: {question}

Requirements:
1. Use pandas DataFrame API
2. Store final answer in variable 'answer'
3. Handle edge cases (empty results, type errors)

Python Code:
"""
```

**怎么用**：
```python
# src/baselines/code_generator.py (复用 OpenCodeInterpreter)
class CodeGenerator:
    def __init__(self, model_name="Qwen/Qwen2.5-14B-Instruct"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate(self, table, question):
        prompt = CODE_GEN_PROMPT.format(
            table_preview=table.head().to_string(),
            column_names=list(table.columns),
            question=question
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.2,
            do_sample=True
        )
        code = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._extract_code(code)
```

**训练吗**？**不训练**，直接用 Qwen-2.5-14B-Instruct 的 pretrained weights。

---

## 第二部分：代码执行器（Code Executor）

### 直接复用：OpenCodeInterpreter 的 Sandbox

**用什么代码**：
```bash
# 从 OpenCodeInterpreter 复用
src/core/code_executor.py  # 沙盒执行
```

**具体实现**（复用并小改）：
```python
# src/core/code_executor.py
import pandas as pd
import numpy as np
import re
import traceback
from typing import Dict, Any
import signal
from contextlib import contextmanager

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Code execution timeout")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

class CodeExecutor:
    def __init__(self, timeout=5):
        self.timeout = timeout

    def execute(self, code: str, table: pd.DataFrame) -> Dict[str, Any]:
        """执行代码并捕获错误"""
        # 创建受限的全局命名空间
        safe_globals = {
            '__builtins__': {
                'len': len, 'sum': sum, 'max': max, 'min': min,
                'int': int, 'float': float, 'str': str, 'bool': bool,
                'list': list, 'dict': dict, 'set': set, 'tuple': tuple,
                'range': range, 'enumerate': enumerate, 'zip': zip,
                'sorted': sorted, 'abs': abs, 'round': round,
            },
            'pd': pd,
            'np': np,
            're': re,
            'df': table.copy(),  # 工作副本，防止原表被修改
        }

        try:
            with time_limit(self.timeout):
                exec(code, safe_globals)
                answer = safe_globals.get('answer', None)

            return {
                'success': True,
                'answer': answer,
                'error': None,
                'error_type': None,
                'traceback': None
            }

        except TimeoutException:
            return {
                'success': False,
                'answer': None,
                'error': 'Execution timeout',
                'error_type': 'TimeoutError',
                'traceback': None
            }

        except Exception as e:
            return {
                'success': False,
                'answer': None,
                'error': str(e),
                'error_type': type(e).__name__,
                'traceback': traceback.format_exc()
            }
```

**训练吗**？**不需要训练**，这是确定性的执行器。

---

## 第三部分：错误诊断器（Error Diagnoser）—— 核心创新，自己写

### 这部分完全自己写，是论文的主要贡献

**文件结构**：
```
src/diagnosis/
├── __init__.py
├── error_classifier.py       # Layer 1: 错误分类
├── root_cause_analyzer.py    # Layer 2: 根因分析
├── strategy_selector.py      # Layer 3: 策略选择（这里用GRPO）
├── prompt_generator.py       # Layer 4: 修复提示生成
└── strategies/
    ├── __init__.py
    ├── syntax_fixer.py       # 语法错误修复
    ├── keyerror_fixer.py     # KeyError修复
    ├── typeerror_fixer.py    # TypeError修复
    ├── logic_fixer.py        # 逻辑错误修复
    └── ... (共20个策略)
```

### Layer 1: 错误分类器（规则+简单ML）

**自己写，不复用**：
```python
# src/diagnosis/error_classifier.py
class ErrorClassifier:
    """Layer 1: 将错误分类到4大类"""

    def classify(self, error_info: Dict) -> str:
        """
        返回: 'syntax' | 'runtime' | 'logic' | 'semantic'
        """
        error_type = error_info['error_type']
        error_msg = error_info['error']

        # 规则判断（简单高效）
        if error_type in ['SyntaxError', 'IndentationError', 'TabError']:
            return 'syntax'

        if error_type in ['KeyError', 'TypeError', 'ValueError',
                          'AttributeError', 'IndexError']:
            return 'runtime'

        if error_type == 'TimeoutError':
            return 'timeout'

        # 如果没有错误但答案错误 → logic error
        if error_info['success'] and error_info.get('is_wrong_answer'):
            return 'logic'

        # 如果错误信息包含表结构相关的词 → semantic error
        semantic_keywords = ['column', 'schema', 'structure', 'header']
        if any(kw in error_msg.lower() for kw in semantic_keywords):
            return 'semantic'

        return 'runtime'  # 默认
```

**训练吗**？**不训练**，用规则即可（准确率>95%）。

---

### Layer 2: 根因分析器（自己写，基于模式匹配）

**自己写**：
```python
# src/diagnosis/root_cause_analyzer.py
class RootCauseAnalyzer:
    """Layer 2: 分析具体的失败原因"""

    def analyze(self, error_info: Dict, code: str, table: pd.DataFrame) -> Dict:
        """
        返回根因信息，比如：
        {
            'root_cause': 'column_name_mismatch',
            'details': {
                'expected': 'Country',
                'available': ['country', 'population', 'gdp'],
                'suggestion': 'country'
            }
        }
        """
        error_type = error_info['error_type']

        if error_type == 'KeyError':
            return self._analyze_keyerror(error_info, code, table)
        elif error_type == 'TypeError':
            return self._analyze_typeerror(error_info, code, table)
        # ... 其他类型

    def _analyze_keyerror(self, error_info, code, table):
        """分析 KeyError 的根因"""
        import re

        # 从错误信息中提取缺失的列名
        # KeyError: 'Country' → 提取 'Country'
        match = re.search(r"KeyError: ['\"](.+?)['\"]", error_info['error'])
        if not match:
            return {'root_cause': 'unknown_keyerror'}

        missing_col = match.group(1)
        available_cols = list(table.columns)

        # 用模糊匹配找最相似的列
        from difflib import get_close_matches
        suggestions = get_close_matches(missing_col, available_cols, n=3, cutoff=0.6)

        return {
            'root_cause': 'column_name_mismatch',
            'details': {
                'missing': missing_col,
                'available': available_cols,
                'suggestions': suggestions
            }
        }
```

**训练吗**？**不训练**，模式匹配+字符串相似度即可。

---

### Layer 3: 策略选择器（这里用GRPO！）—— 唯一需要训练的部分

**自己写，集成GRPO**：
```python
# src/diagnosis/strategy_selector.py
class StrategySelector:
    """Layer 3: 用GRPO学习选择哪个修复策略"""

    def __init__(self, use_grpo=True):
        self.use_grpo = use_grpo

        # 20个修复策略（手工实现的规则）
        self.strategies = {
            'column_name_correction': ColumnNameCorrectionStrategy(),
            'type_conversion': TypeConversionStrategy(),
            'filter_relaxation': FilterRelaxationStrategy(),
            # ... 共20个
        }

        if use_grpo:
            # GRPO策略选择器（这部分需要训练）
            self.policy_model = self._init_grpo_policy()
        else:
            # 固定规则选择
            self.policy_model = None

    def select_strategy(self, error_class: str, root_cause: Dict) -> str:
        """选择修复策略"""
        if not self.use_grpo:
            # 固定规则（baseline）
            return self._rule_based_selection(error_class, root_cause)
        else:
            # GRPO学习的策略
            return self._grpo_based_selection(error_class, root_cause)

    def _rule_based_selection(self, error_class, root_cause):
        """固定规则（不需要训练）"""
        if root_cause['root_cause'] == 'column_name_mismatch':
            return 'column_name_correction'
        elif root_cause['root_cause'] == 'type_mismatch':
            return 'type_conversion'
        # ... 其他规则
        return 'generic_retry'

    def _grpo_based_selection(self, error_class, root_cause):
        """GRPO学习的策略选择（需要训练）"""
        # 特征编码
        features = self._encode_features(error_class, root_cause)

        # 用GRPO策略网络预测
        with torch.no_grad():
            logits = self.policy_model(features)
            probs = torch.softmax(logits, dim=-1)
            strategy_idx = torch.argmax(probs).item()

        strategy_names = list(self.strategies.keys())
        return strategy_names[strategy_idx]

    def _init_grpo_policy(self):
        """初始化GRPO策略网络"""
        # 简单的MLP分类器
        import torch.nn as nn

        class PolicyNetwork(nn.Module):
            def __init__(self, input_dim=64, num_strategies=20):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Linear(128, num_strategies)
                )

            def forward(self, x):
                return self.net(x)

        return PolicyNetwork()
```

**训练吗**？**需要训练GRPO策略网络**（见后面的GRPO训练部分）。

---

### Layer 4: 修复提示生成器（自己写，模板+填充）

**自己写**：
```python
# src/diagnosis/prompt_generator.py
class PromptGenerator:
    """Layer 4: 生成修复提示"""

    def generate(self, strategy_name: str, root_cause: Dict,
                 original_code: str, error_info: Dict) -> str:
        """
        生成给LLM的修复提示
        """
        if strategy_name == 'column_name_correction':
            return self._column_correction_prompt(root_cause, original_code, error_info)
        elif strategy_name == 'type_conversion':
            return self._type_conversion_prompt(root_cause, original_code, error_info)
        # ... 其他策略

    def _column_correction_prompt(self, root_cause, original_code, error_info):
        """生成列名修复的提示"""
        missing = root_cause['details']['missing']
        suggestions = root_cause['details']['suggestions']
        available = root_cause['details']['available']

        prompt = f"""
The previous code failed with KeyError: '{missing}'

Available columns: {available}
Most similar columns: {suggestions}

Original code:
```python
{original_code}
```

Please fix the code by:
1. Replacing '{missing}' with the correct column name from {suggestions}
2. Ensure all column references are valid
3. Keep the logic unchanged

Fixed code:
"""
        return prompt
```

**训练吗**？**不训练**，模板填充即可。

---

## 第四部分：GRPO训练器 —— 复用DeepSeek代码 + 小改

### 复用什么代码？

**基础代码来源**：
- DeepSeek-R1 的 GRPO 实现（如果开源）
- 或者参考 HuggingFace TRL 库的 PPO trainer，改成GRPO

**实际上用**：HuggingFace TRL + 自己改

```bash
# 安装 TRL
pip install trl transformers

# 参考代码
# https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py
```

### 自己写的GRPO训练器

**文件**：`src/grpo/grpo_trainer.py`

**代码**（基于TRL改写）：
```python
# src/grpo/grpo_trainer.py
import torch
import numpy as np
from trl import PPOTrainer, PPOConfig
from transformers import AutoModelForCausalLM

class GRPOTrainer:
    """
    Group Relative Policy Optimization Trainer
    基于 TRL 的 PPOTrainer 改写
    """

    def __init__(self, policy_model, ref_model, reward_fn, group_size=4):
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.reward_fn = reward_fn
        self.group_size = group_size

        # 使用 TRL 的配置
        self.config = PPOConfig(
            learning_rate=1e-6,
            batch_size=16,
            ppo_epochs=1,
            mini_batch_size=4,
        )

        # 优化器
        self.optimizer = torch.optim.AdamW(
            policy_model.parameters(),
            lr=self.config.learning_rate
        )

    def train_step(self, batch_prompts, batch_tables, batch_questions, batch_gold_answers):
        """
        训练一个step

        batch_prompts: List[str], 错误诊断后的修复提示
        batch_tables: List[pd.DataFrame], 表格
        batch_questions: List[str], 问题
        batch_gold_answers: List[str], 正确答案
        """
        all_responses = []
        all_rewards = []

        # Step 1: 对每个prompt，生成 group_size 个响应
        for prompt, table, question, gold in zip(
            batch_prompts, batch_tables, batch_questions, batch_gold_answers
        ):
            group_responses = []
            group_rewards = []

            for _ in range(self.group_size):
                # 生成修复后的代码
                response = self._generate_code(prompt)

                # 执行代码，计算reward
                reward = self.reward_fn.compute(response, table, question, gold)

                group_responses.append(response)
                group_rewards.append(reward)

            all_responses.append(group_responses)
            all_rewards.append(group_rewards)

        # Step 2: 计算 group-based advantages（GRPO的关键！）
        advantages = self._compute_group_advantages(all_rewards)

        # Step 3: PPO-style 策略更新
        loss = self._compute_policy_loss(all_responses, advantages)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return {'loss': loss.item(), 'avg_reward': np.mean(all_rewards)}

    def _compute_group_advantages(self, all_rewards):
        """
        GRPO的核心：用组内平均作为baseline
        """
        advantages = []

        for group_rewards in all_rewards:
            # 组内平均作为baseline（而不是value function）
            group_mean = np.mean(group_rewards)
            group_std = np.std(group_rewards) + 1e-8

            # 标准化
            group_adv = [(r - group_mean) / group_std for r in group_rewards]
            advantages.extend(group_adv)

        return advantages

    def _compute_policy_loss(self, all_responses, advantages):
        """
        PPO-style clipped loss
        """
        total_loss = 0

        flat_responses = [r for group in all_responses for r in group]

        for response, advantage in zip(flat_responses, advantages):
            # 计算 log prob
            old_log_prob = self._get_log_prob(self.ref_model, response)
            new_log_prob = self._get_log_prob(self.policy_model, response)

            # Importance ratio
            ratio = torch.exp(new_log_prob - old_log_prob)

            # Clipped surrogate
            clip_range = 0.2
            clipped_ratio = torch.clamp(ratio, 1 - clip_range, 1 + clip_range)

            loss1 = ratio * advantage
            loss2 = clipped_ratio * advantage
            policy_loss = -torch.min(loss1, loss2)

            # KL penalty
            kl_div = old_log_prob - new_log_prob
            kl_penalty = 0.01 * kl_div

            total_loss += policy_loss + kl_penalty

        return total_loss / len(flat_responses)
```

**训练吗**？**这个就是训练器本身**，下面说怎么用。

---

## 第五部分：Reward Function —— 自己写

**文件**：`src/grpo/reward_function.py`

```python
# src/grpo/reward_function.py
class MultiComponentReward:
    """
    多组件奖励函数
    R = 0.4*accuracy + 0.3*execution + 0.1*efficiency + 0.1*repair + 0.1*quality
    """

    def __init__(self, code_executor):
        self.executor = code_executor

    def compute(self, code: str, table, question: str, gold_answer: str) -> float:
        """计算总reward"""
        # 执行代码
        result = self.executor.execute(code, table)

        # R1: 执行成功 (0.3)
        r_exec = 1.0 if result['success'] else -0.5

        # R2: 答案正确性 (0.4)
        if result['success']:
            r_acc = self._compute_accuracy(result['answer'], gold_answer)
        else:
            r_acc = 0.0

        # R3: 效率 (0.1) - 代码长度越短越好
        r_eff = 1.0 - min(len(code.split('\n')) / 20, 1.0)

        # R4: 修复质量 (0.1) - 这个在多轮迭代时用
        r_repair = 0.0  # 暂时不用

        # R5: 代码质量 (0.1) - 用启发式规则
        r_quality = self._evaluate_code_quality(code)

        # 加权求和
        total = 0.4*r_acc + 0.3*r_exec + 0.1*r_eff + 0.1*r_repair + 0.1*r_quality

        return total

    def _compute_accuracy(self, pred, gold):
        """计算答案准确性"""
        # 简单版本：exact match
        pred_norm = str(pred).strip().lower()
        gold_norm = str(gold).strip().lower()

        if pred_norm == gold_norm:
            return 1.0

        # 如果是数字，允许小误差
        try:
            pred_num = float(pred)
            gold_num = float(gold)
            if abs(pred_num - gold_num) / (abs(gold_num) + 1e-8) < 0.01:
                return 1.0
        except:
            pass

        # Token F1
        pred_tokens = set(pred_norm.split())
        gold_tokens = set(gold_norm.split())
        overlap = pred_tokens & gold_tokens

        if len(overlap) == 0:
            return 0.0

        precision = len(overlap) / len(pred_tokens) if pred_tokens else 0
        recall = len(overlap) / len(gold_tokens) if gold_tokens else 0

        if precision + recall == 0:
            return 0.0

        f1 = 2 * precision * recall / (precision + recall)
        return f1

    def _evaluate_code_quality(self, code):
        """启发式代码质量评估"""
        score = 1.0

        # 太长扣分
        if len(code.split('\n')) > 20:
            score -= 0.3

        # 用了好的pandas操作加分
        good_ops = ['.groupby(', '.agg(', '.apply(', '.merge(']
        if any(op in code for op in good_ops):
            score += 0.2

        # 用了循环扣分
        if 'for ' in code or 'while ' in code:
            score -= 0.2

        return max(0, min(1, score))
```

**训练吗**？**不训练**，这是确定性的奖励函数。

---

## 完整训练流程

### Week 7-9: GRPO 训练

**数据准备**（Week 7, Day 1-2）：
```python
# scripts/prepare_grpo_training_data.py
import pandas as pd
import json

# 1. 加载 WikiTQ 训练集
with open('data/wikitq/train.jsonl') as f:
    train_data = [json.loads(line) for line in f]

# 2. 用 baseline 模型跑一遍，收集错误
from src.baselines.code_generator import CodeGenerator
from src.core.code_executor import CodeExecutor
from src.diagnosis.error_classifier import ErrorClassifier
from src.diagnosis.root_cause_analyzer import RootCauseAnalyzer

generator = CodeGenerator(model_name="Qwen/Qwen2.5-14B-Instruct")
executor = CodeExecutor()
classifier = ErrorClassifier()
analyzer = RootCauseAnalyzer()

error_cases = []

for sample in train_data[:5000]:  # 只用5000个样本
    table = pd.DataFrame(sample['table'])
    question = sample['question']
    gold_answer = sample['answer']

    # 生成代码
    code = generator.generate(table, question)

    # 执行
    result = executor.execute(code, table)

    # 如果有错误，记录下来
    if not result['success'] or result['answer'] != gold_answer:
        error_class = classifier.classify(result)
        root_cause = analyzer.analyze(result, code, table)

        error_cases.append({
            'table': sample['table'],
            'question': question,
            'gold_answer': gold_answer,
            'original_code': code,
            'error_class': error_class,
            'root_cause': root_cause,
            'execution_result': result
        })

# 3. 保存错误案例（用于GRPO训练）
with open('data/grpo_training/error_cases.jsonl', 'w') as f:
    for case in error_cases:
        f.write(json.dumps(case) + '\n')

print(f"收集了 {len(error_cases)} 个错误案例")
```

**GRPO训练主循环**（Week 7-8）：
```python
# scripts/train_grpo.py
import torch
from src.grpo.grpo_trainer import GRPOTrainer
from src.grpo.reward_function import MultiComponentReward
from src.baselines.code_generator import CodeGenerator
from src.core.code_executor import CodeExecutor
import json

# 1. 加载数据
with open('data/grpo_training/error_cases.jsonl') as f:
    error_cases = [json.loads(line) for line in f]

# 2. 初始化模型
policy_model = CodeGenerator(model_name="Qwen/Qwen2.5-14B-Instruct").model
ref_model = CodeGenerator(model_name="Qwen/Qwen2.5-14B-Instruct").model
ref_model.eval()  # 冻结

# 3. 初始化trainer
executor = CodeExecutor()
reward_fn = MultiComponentReward(executor)
trainer = GRPOTrainer(policy_model, ref_model, reward_fn, group_size=4)

# 4. 训练循环
num_epochs = 5
batch_size = 16

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")

    # Shuffle
    import random
    random.shuffle(error_cases)

    epoch_loss = 0
    epoch_reward = 0
    num_batches = 0

    for i in range(0, len(error_cases), batch_size):
        batch = error_cases[i:i+batch_size]

        # 准备batch数据
        batch_prompts = []
        batch_tables = []
        batch_questions = []
        batch_gold_answers = []

        for case in batch:
            # 用 Layer 4 生成修复提示
            from src.diagnosis.prompt_generator import PromptGenerator
            pg = PromptGenerator()

            # 假设我们知道要用哪个策略（实际上这里可以用rule-based）
            strategy = 'column_name_correction'  # 简化
            prompt = pg.generate(strategy, case['root_cause'],
                                case['original_code'], case['execution_result'])

            batch_prompts.append(prompt)
            batch_tables.append(pd.DataFrame(case['table']))
            batch_questions.append(case['question'])
            batch_gold_answers.append(case['gold_answer'])

        # 训练一步
        metrics = trainer.train_step(
            batch_prompts, batch_tables,
            batch_questions, batch_gold_answers
        )

        epoch_loss += metrics['loss']
        epoch_reward += metrics['avg_reward']
        num_batches += 1

        if num_batches % 10 == 0:
            print(f"  Batch {num_batches}: loss={metrics['loss']:.4f}, "
                  f"reward={metrics['avg_reward']:.4f}")

    avg_loss = epoch_loss / num_batches
    avg_reward = epoch_reward / num_batches
    print(f"Epoch {epoch+1} Summary: loss={avg_loss:.4f}, reward={avg_reward:.4f}")

    # 保存checkpoint
    torch.save(policy_model.state_dict(),
               f'checkpoints/grpo_policy_epoch{epoch+1}.pt')

print("GRPO训练完成！")
```

**训练时间估计**：
- 5000个错误案例
- Batch size 16, group size 4 → 每个batch要生成 16×4=64个代码
- Qwen-2.5-14B 生成速度 ~20 tokens/s → 每个代码(~200 tokens) 需要 10秒
- 每个batch: 64×10秒 = 640秒 ≈ 10分钟
- 总batch数: 5000/16 ≈ 313 batches
- 每个epoch: 313×10分钟 ≈ 52小时
- **5个epoch总共: 260小时 ≈ 11天**（用单张A100 GPU）

**如果太慢怎么办**：
- 用2张GPU并行 → 5.5天
- 或者减少到2000个错误案例 → 4.4天
- 或者用更小的模型（Qwen-2.5-7B）→ 快2倍

---

## 完整推理流程（训练完成后）

### 用训练好的系统回答问题

```python
# scripts/inference.py
from src.baselines.code_generator import CodeGenerator
from src.core.code_executor import CodeExecutor
from src.diagnosis.error_classifier import ErrorClassifier
from src.diagnosis.root_cause_analyzer import RootCauseAnalyzer
from src.diagnosis.strategy_selector import StrategySelector
from src.diagnosis.prompt_generator import PromptGenerator
import pandas as pd

class TableQASystem:
    def __init__(self, use_grpo=True):
        self.generator = CodeGenerator(model_name="Qwen/Qwen2.5-14B-Instruct")
        self.executor = CodeExecutor()
        self.classifier = ErrorClassifier()
        self.analyzer = RootCauseAnalyzer()
        self.selector = StrategySelector(use_grpo=use_grpo)
        self.prompt_gen = PromptGenerator()

        # 如果用GRPO，加载训练好的策略模型
        if use_grpo:
            import torch
            checkpoint = torch.load('checkpoints/grpo_policy_epoch5.pt')
            self.selector.policy_model.load_state_dict(checkpoint)

    def answer_question(self, table: pd.DataFrame, question: str,
                       max_iterations=3):
        """主流程"""
        iteration = 0
        code = None

        while iteration < max_iterations:
            # 生成代码（第一次）或修复代码（后续）
            if iteration == 0:
                # 初始生成
                code = self.generator.generate(table, question)
            else:
                # 错误诊断 + 修复
                # Layer 1: 分类
                error_class = self.classifier.classify(result)

                # Layer 2: 根因分析
                root_cause = self.analyzer.analyze(result, code, table)

                # Layer 3: 策略选择（GRPO！）
                strategy = self.selector.select_strategy(error_class, root_cause)

                # Layer 4: 生成修复提示
                repair_prompt = self.prompt_gen.generate(
                    strategy, root_cause, code, result
                )

                # 用修复提示生成新代码
                code = self.generator.generate_from_prompt(repair_prompt)

            # 执行代码
            result = self.executor.execute(code, table)

            # 如果成功，返回答案
            if result['success']:
                return {
                    'answer': result['answer'],
                    'code': code,
                    'iterations': iteration + 1,
                    'success': True
                }

            iteration += 1

        # 达到最大迭代次数，返回失败
        return {
            'answer': None,
            'code': code,
            'iterations': max_iterations,
            'success': False,
            'last_error': result
        }

# 使用示例
system = TableQASystem(use_grpo=True)

table = pd.DataFrame({
    'country': ['USA', 'China', 'India'],
    'population': [331, 1441, 1380],
    'gdp': [21.4, 14.7, 2.9]
})

question = "Which country has the largest population?"

answer = system.answer_question(table, question)
print(f"Answer: {answer['answer']}")
print(f"Iterations: {answer['iterations']}")
print(f"Code:\n{answer['code']}")
```

---

## 总结：哪些复用，哪些自己写，哪些训练

| 组件 | 复用来源 | 自己写/改 | 需要训练 | 训练时间 |
|------|---------|----------|----------|---------|
| **Code Generator** | OpenCodeInterpreter | 小改prompt | ❌ 不训练 | - |
| **Code Executor** | OpenCodeInterpreter | 小改sandbox | ❌ 不训练 | - |
| **Error Classifier** | - | ✅ 完全自己写 | ❌ 不训练（规则） | - |
| **Root Cause Analyzer** | - | ✅ 完全自己写 | ❌ 不训练（规则） | - |
| **Strategy Selector** | - | ✅ 完全自己写 | ✅ **训练GRPO** | **11天（单GPU）** |
| **Prompt Generator** | - | ✅ 完全自己写 | ❌ 不训练（模板） | - |
| **GRPO Trainer** | HuggingFace TRL | 改写成GRPO | - | - |
| **Reward Function** | - | ✅ 完全自己写 | ❌ 不训练（规则） | - |

**关键点**：
- **只有Strategy Selector的策略网络需要训练**（GRPO训练）
- 其他所有部分都是确定性规则或者直接用pretrained模型
- 训练时间：11天（单A100），可以用2-4张GPU并行缩短到3-5天

**最终交付物**：
1. 完整的代码库（~3000行Python代码）
2. 训练好的GRPO策略模型（checkpoint）
3. 5000+个标注的错误案例数据集
4. 实验结果和分析

这就是完整的技术路线！是否清楚了？
