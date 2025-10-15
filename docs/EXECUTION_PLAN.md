# Table QA 项目执行计划

## 项目信息

- **项目名称**: Open-Source LLM Error Diagnosis for Table QA
- **目标会议**: ACL/EMNLP/NAACL 2025
- **项目周期**: 12周（2025年某月-某月）
- **主要模型**: Qwen-2.5-7B/14B/32B, Llama-3.1-8B/70B
- **主要数据集**: WikiTQ, TabFact, FeTaQA, SemEval-2025 Task 8

---

## Week 1-2: 环境搭建与Baseline验证

### Week 1: 环境搭建 (Day 1-7)

#### Day 1-2: 基础环境
```bash
# 1. 创建conda环境
conda create -n table-qa python=3.10
conda activate table-qa

# 2. 安装依赖
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.36.0
pip install pandas numpy scipy
pip install datasets evaluate
pip install openai anthropic  # API baselines
pip install wandb  # 实验跟踪
pip install jupyter notebook
pip install pytest black flake8

# 3. 验证安装
python -c "import torch; print(torch.cuda.is_available())"
python -c "from transformers import AutoModelForCausalLM; print('OK')"
```

#### Day 3-4: 数据下载与预处理
```bash
# 创建数据目录
mkdir -p data/{wikitq,tabfact,fetaqa,semeval2025}

# 下载WikiTQ
wget https://github.com/ppasupat/WikiTableQuestions/archive/refs/heads/master.zip
unzip master.zip -d data/wikitq/
python scripts/preprocess_wikitq.py  # 需要编写

# 下载TabFact
git clone https://github.com/wenhuchen/Table-Fact-Checking.git data/tabfact

# 下载FeTaQA
git clone https://github.com/Yale-LILY/FeTaQA.git data/fetaqa

# 下载SemEval-2025 Task 8 (DataBench)
# 需要从官方获取：https://www.codabench.org/competitions/3360/
```

**输出文件**:
```
data/
├── wikitq/
│   ├── train.jsonl  # 11,321 samples
│   ├── dev.jsonl    # 2,831 samples
│   └── test.jsonl   # 4,344 samples
├── tabfact/
│   ├── train.jsonl  # 92,283 samples
│   ├── dev.jsonl    # 12,792 samples
│   └── test.jsonl   # 12,779 samples
├── fetaqa/
│   ├── train.jsonl  # 8,007 samples
│   ├── dev.jsonl    # 1,000 samples
│   └── test.jsonl   # 1,731 samples
└── semeval2025/
    ├── train.jsonl
    └── test.jsonl
```

#### Day 5-6: 数据加载器实现
创建文件 `src/data/data_loader.py`:
```python
class TableQADataset:
    def __init__(self, file_path, dataset_name):
        """
        参数:
        - file_path: jsonl文件路径
        - dataset_name: 'wikitq', 'tabfact', 'fetaqa', 'semeval2025'
        """
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        """
        返回:
        {
            'table': pd.DataFrame,
            'question': str,
            'answer': str or bool,
            'metadata': dict
        }
        """
        pass
```

**测试代码**:
```python
# tests/test_data_loader.py
dataset = TableQADataset('data/wikitq/train.jsonl', 'wikitq')
print(f"Total samples: {len(dataset)}")
sample = dataset[0]
print(sample['table'].head())
print(f"Question: {sample['question']}")
print(f"Answer: {sample['answer']}")
```

#### Day 7: 代码执行沙盒实现
创建文件 `src/execution/sandbox.py`:
```python
class SecureCodeExecutor:
    def __init__(self, timeout=5, memory_limit_mb=2048):
        """
        参数:
        - timeout: 执行超时(秒)
        - memory_limit_mb: 内存限制(MB)
        """
        self.timeout = timeout
        self.memory_limit = memory_limit_mb
        self.allowed_imports = ['pandas', 'numpy', 're', 'datetime']

    def execute(self, code: str, table: pd.DataFrame) -> dict:
        """
        执行代码并返回结果

        返回:
        {
            'success': bool,
            'answer': any,
            'error': {
                'type': str,
                'message': str,
                'traceback': str
            } or None
        }
        """
        pass
```

**测试用例**:
```python
# tests/test_executor.py
executor = SecureCodeExecutor(timeout=5)

# 测试1: 正常执行
code = """
answer = df['column1'].sum()
"""
result = executor.execute(code, test_table)
assert result['success'] == True

# 测试2: KeyError
code = """
answer = df['nonexistent_column'].sum()
"""
result = executor.execute(code, test_table)
assert result['success'] == False
assert result['error']['type'] == 'KeyError'

# 测试3: 超时
code = """
import time
time.sleep(10)
"""
result = executor.execute(code, test_table)
assert result['error']['type'] == 'TimeoutError'
```

---

### Week 2: Baseline验证 (Day 8-14)

#### Day 8-9: GPT-4o Baseline
创建文件 `src/baselines/gpt4_baseline.py`:
```python
class GPT4Baseline:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def generate_code(self, table: pd.DataFrame, question: str) -> str:
        """
        生成Python代码

        Prompt模板:
        You are a Python expert. Given a table and a question,
        write Python code to answer the question.

        Table:
        {table.to_string()}

        Question: {question}

        Write code that:
        1. Uses pandas DataFrame 'df'
        2. Stores the final answer in variable 'answer'
        3. Only uses: pandas, numpy, re, datetime

        Code:
        """
        pass

    def answer_question(self, table, question):
        code = self.generate_code(table, question)
        result = executor.execute(code, table)
        return result
```

**运行评估**:
```bash
# 在WikiTQ dev set上评估（2831个样本，采样100个快速测试）
python scripts/eval_baseline.py \
    --model gpt-4o \
    --dataset wikitq \
    --split dev \
    --n_samples 100 \
    --output results/gpt4o_wikitq_dev_100.json

# 预期输出
# Accuracy: ~58-62%
# Avg time: ~3.5s per sample
# Total cost: ~$2.50 (100 samples)
```

#### Day 10-11: Qwen-2.5-14B Baseline (Zero-Shot)
创建文件 `src/baselines/qwen_baseline.py`:
```python
class QwenBaseline:
    def __init__(self, model_name="Qwen/Qwen2.5-14B-Instruct"):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate_code(self, table, question):
        """
        使用相同的prompt格式，但用Qwen生成
        """
        pass
```

**运行评估**:
```bash
# 同样在100个样本上测试
python scripts/eval_baseline.py \
    --model qwen-2.5-14b \
    --dataset wikitq \
    --split dev \
    --n_samples 100 \
    --output results/qwen14b_wikitq_dev_100.json

# 预期输出
# Accuracy: ~52-56%
# Avg time: ~2.1s per sample (本地GPU)
```

#### Day 12-13: AILS-NTUA Method复现
创建文件 `src/baselines/ails_ntua.py`:
```python
class AILSNTUABaseline:
    def __init__(self, model, max_iterations=2):
        self.model = model
        self.max_iterations = max_iterations

    def generate_code_with_error_fixing(self, table, question, gold_answer=None):
        """
        实现AILS-NTUA的迭代修复机制

        步骤:
        1. 生成初始代码
        2. 执行代码
        3. 如果出错，将错误信息反馈给LLM
        4. LLM生成修复后的代码
        5. 重复2-4，最多max_iterations次
        """
        code = self.model.generate_code(table, question)

        for iteration in range(self.max_iterations):
            result = executor.execute(code, table)
            if result['success']:
                return result

            # 错误修复prompt
            error_msg = result['error']['message']
            code = self.model.fix_code(
                table, question, code, error_msg
            )

        return result
```

**运行评估**:
```bash
python scripts/eval_baseline.py \
    --model qwen-2.5-14b \
    --method ails-ntua \
    --max_iterations 2 \
    --dataset wikitq \
    --split dev \
    --n_samples 100 \
    --output results/qwen14b_ails_wikitq_dev_100.json

# 预期输出
# Accuracy: ~58-62%
# Avg iterations: ~1.95
# Avg time: ~4.2s per sample
```

#### Day 14: 结果汇总与分析
创建notebook `notebooks/01_baseline_analysis.ipynb`:
```python
import pandas as pd
import json

# 加载所有baseline结果
gpt4_results = json.load(open('results/gpt4o_wikitq_dev_100.json'))
qwen_zero = json.load(open('results/qwen14b_wikitq_dev_100.json'))
qwen_ails = json.load(open('results/qwen14b_ails_wikitq_dev_100.json'))

# 对比表格
comparison = pd.DataFrame({
    'Method': ['GPT-4o (Zero-Shot)', 'Qwen-14B (Zero-Shot)', 'Qwen-14B (AILS-NTUA)'],
    'Accuracy': [
        gpt4_results['accuracy'],
        qwen_zero['accuracy'],
        qwen_ails['accuracy']
    ],
    'Avg Time (s)': [...],
    'Avg Iterations': [1.0, 1.0, 1.95]
})

print(comparison)
```

**Milestone检查点**:
- [ ] 数据加载器可以正确加载4个数据集
- [ ] 代码执行沙盒可以安全执行代码并捕获错误
- [ ] GPT-4o baseline在WikiTQ上达到58-62%
- [ ] Qwen-2.5-14B zero-shot在52-56%
- [ ] Qwen-2.5-14B + AILS-NTUA在58-62%

---

## Week 3-4: 错误数据收集与分类

### Week 3: 错误案例收集 (Day 15-21)

#### Day 15-16: 批量运行收集错误
创建脚本 `scripts/collect_errors.py`:
```python
def collect_errors(model, dataset, n_samples=5000):
    """
    运行模型并收集所有错误案例

    返回:
    errors = [
        {
            'sample_id': int,
            'table': pd.DataFrame,
            'question': str,
            'generated_code': str,
            'error_type': str,  # 'KeyError', 'TypeError', etc.
            'error_message': str,
            'traceback': str,
            'iteration': int,  # 第几次迭代出的错
        },
        ...
    ]
    """
    pass
```

**执行计划**:
```bash
# WikiTQ训练集 (11,321 samples) - 使用Qwen-2.5-14B
python scripts/collect_errors.py \
    --model qwen-2.5-14b \
    --dataset wikitq \
    --split train \
    --output data/errors/wikitq_qwen14b_errors.jsonl

# TabFact训练集 (随机采样10,000) - 使用Qwen-2.5-14B
python scripts/collect_errors.py \
    --model qwen-2.5-14b \
    --dataset tabfact \
    --split train \
    --n_samples 10000 \
    --output data/errors/tabfact_qwen14b_errors.jsonl

# 预期收集到2500-3500个错误案例
```

#### Day 17-18: 初步错误分类
创建notebook `notebooks/02_error_classification.ipynb`:
```python
import json
import pandas as pd

# 加载错误数据
errors = []
with open('data/errors/wikitq_qwen14b_errors.jsonl') as f:
    for line in f:
        errors.append(json.loads(line))

# 按error_type统计
error_counts = pd.Series([e['error_type'] for e in errors]).value_counts()
print(error_counts)

# 预期分布:
# KeyError          ~40%
# TypeError         ~20%
# ValueError        ~12%
# AttributeError    ~8%
# SyntaxError       ~8%
# IndexError        ~6%
# Other             ~6%
```

#### Day 19-20: 人工标注错误类型
创建文件 `data/errors/annotation_guidelines.md`:
```markdown
# 错误分类标注指南

## 1. Syntax Error (语法错误)
- IndentationError: 缩进错误
- SyntaxError: 无效Python语法
- NameError: 变量名错误

## 2. Runtime Error (运行时错误)
### 2.1 Schema Mismatch (模式不匹配)
- KeyError: 列名不存在
- AttributeError: 方法或属性不存在

### 2.2 Type Error (类型错误)
- TypeError: 类型不匹配（如字符串不能做数值运算）
- ValueError: 值错误（如int('abc')）

### 2.3 Index Error (索引错误)
- IndexError: 索引越界
- Empty DataFrame操作

## 3. Logic Error (逻辑错误)
- Wrong Aggregation: 用了sum但应该用mean
- Wrong Filter: 过滤条件错误
- Wrong Column: 选错列

## 4. Semantic Error (语义错误)
- Hallucinated Column: 模型生成了不存在的列名
- Misunderstood Question: 理解错问题意图
```

**标注任务**:
```bash
# 随机采样500个错误进行详细标注
python scripts/sample_for_annotation.py \
    --input data/errors/wikitq_qwen14b_errors.jsonl \
    --n_samples 500 \
    --output data/errors/annotation_samples_500.jsonl

# 使用Label Studio进行标注
# 或者创建简单的标注脚本
python scripts/annotate_errors.py \
    --input data/errors/annotation_samples_500.jsonl \
    --output data/errors/annotated_500.jsonl
```

#### Day 21: 计算标注一致性
创建脚本 `scripts/compute_agreement.py`:
```python
from sklearn.metrics import cohen_kappa_score

# 两位标注员独立标注相同的100个样本
annotator1 = load_annotations('data/errors/annotator1_100.jsonl')
annotator2 = load_annotations('data/errors/annotator2_100.jsonl')

# 计算Kappa
labels1 = [a['error_category'] for a in annotator1]
labels2 = [a['error_category'] for a in annotator2]
kappa = cohen_kappa_score(labels1, labels2)

print(f"Cohen's Kappa: {kappa:.3f}")
# 目标: kappa > 0.75
```

---

### Week 4: Error Taxonomy设计 (Day 22-28)

#### Day 22-23: 设计Taxonomy结构
创建文件 `src/diagnosis/error_taxonomy.py`:
```python
class ErrorTaxonomy:
    """
    4层错误分类体系

    Layer 1: Error Classification
    - Syntax, Runtime, Logic, Semantic

    Layer 2: Root Cause
    - Schema Mismatch, Type Error, Index Error, etc.

    Layer 3: Failure Mode
    - Missing Column, Wrong Data Type, etc.

    Layer 4: Repair Strategy
    - Column Name Correction, Type Conversion, etc.
    """

    def __init__(self):
        self.taxonomy = {
            'Syntax': {
                'IndentationError': {
                    'failure_mode': 'Incorrect indentation',
                    'repair_strategy': 'FixIndentationStrategy'
                },
                'SyntaxError': {
                    'failure_mode': 'Invalid Python syntax',
                    'repair_strategy': 'FixSyntaxStrategy'
                }
            },
            'Runtime': {
                'KeyError': {
                    'ColumnNotExist': {
                        'failure_mode': 'Column name mismatch',
                        'repair_strategy': 'ColumnNameCorrectionStrategy'
                    },
                    'IndexNotExist': {
                        'failure_mode': 'Row index not exist',
                        'repair_strategy': 'IndexCorrectionStrategy'
                    }
                },
                'TypeError': {
                    'StringNumericOperation': {
                        'failure_mode': 'Numeric operation on string',
                        'repair_strategy': 'TypeConversionStrategy'
                    },
                    'NoneTypeOperation': {
                        'failure_mode': 'Operation on None value',
                        'repair_strategy': 'NullHandlingStrategy'
                    }
                }
            },
            'Logic': {
                'WrongAggregation': {
                    'failure_mode': 'Used sum instead of mean',
                    'repair_strategy': 'AggregationCorrectionStrategy'
                },
                'WrongFilter': {
                    'failure_mode': 'Filter condition too strict',
                    'repair_strategy': 'FilterRelaxationStrategy'
                }
            },
            'Semantic': {
                'HallucinatedColumn': {
                    'failure_mode': 'Generated non-existent column',
                    'repair_strategy': 'SchemaGroundingStrategy'
                }
            }
        }

    def classify(self, error_info):
        """分类错误并返回修复策略"""
        pass
```

#### Day 24-25: 实现Error Classifier
创建文件 `src/diagnosis/error_classifier.py`:
```python
class ErrorClassifier:
    """Layer 1: 错误分类"""

    def classify(self, error_info: dict) -> str:
        """
        输入:
        {
            'error_type': 'KeyError',
            'error_message': "KeyError: 'Population'",
            'traceback': '...',
            'code': 'df["Population"].sum()'
        }

        输出:
        'Runtime' or 'Syntax' or 'Logic' or 'Semantic'
        """
        error_type = error_info['error_type']

        if error_type in ['SyntaxError', 'IndentationError', 'NameError']:
            return 'Syntax'
        elif error_type in ['KeyError', 'TypeError', 'ValueError', 'AttributeError', 'IndexError']:
            return 'Runtime'
        else:
            # 需要更复杂的逻辑判断Logic vs Semantic
            return self._classify_logic_or_semantic(error_info)
```

#### Day 26-27: 实现Root Cause Analyzer
创建文件 `src/diagnosis/root_cause_analyzer.py`:
```python
class RootCauseAnalyzer:
    """Layer 2: 根因分析"""

    def analyze(self, error_info: dict, error_class: str, table: pd.DataFrame) -> str:
        """
        分析错误根因

        例如：KeyError可能是:
        - ColumnNotExist (列名不存在)
        - ColumnCaseMismatch (列名大小写不匹配)
        - ColumnTypo (列名拼写错误)
        """
        if error_class == 'Runtime':
            if error_info['error_type'] == 'KeyError':
                return self._analyze_key_error(error_info, table)
            elif error_info['error_type'] == 'TypeError':
                return self._analyze_type_error(error_info, table)

        return 'Unknown'

    def _analyze_key_error(self, error_info, table):
        """
        分析KeyError的具体原因
        """
        # 从错误消息中提取列名
        missing_col = self._extract_column_name(error_info['error_message'])
        available_cols = table.columns.tolist()

        # 检查是否是大小写问题
        if missing_col.lower() in [c.lower() for c in available_cols]:
            return 'ColumnCaseMismatch'

        # 检查是否是拼写错误 (Levenshtein distance)
        from difflib import get_close_matches
        similar = get_close_matches(missing_col, available_cols, n=1, cutoff=0.7)
        if similar:
            return 'ColumnTypo'

        return 'ColumnNotExist'
```

#### Day 28: Taxonomy覆盖率验证
创建脚本 `scripts/validate_taxonomy.py`:
```python
def validate_taxonomy_coverage():
    """
    在新的1000个错误样本上测试taxonomy覆盖率
    """
    # 加载新的错误样本
    test_errors = load_errors('data/errors/test_errors_1000.jsonl')

    classifier = ErrorClassifier()
    root_cause_analyzer = RootCauseAnalyzer()

    covered = 0
    uncovered = []

    for error in test_errors:
        error_class = classifier.classify(error)
        root_cause = root_cause_analyzer.analyze(error, error_class, error['table'])

        if root_cause != 'Unknown':
            covered += 1
        else:
            uncovered.append(error)

    coverage = covered / len(test_errors)
    print(f"Taxonomy Coverage: {coverage:.2%}")
    print(f"Uncovered cases: {len(uncovered)}")

    # 目标: coverage > 95%
    return coverage, uncovered
```

**Milestone检查点**:
- [ ] 收集到5000+错误案例
- [ ] 500个样本人工标注完成
- [ ] 标注一致性 Kappa > 0.75
- [ ] Taxonomy覆盖率 > 95%

---

## Week 5-6: 修复策略实现

### Week 5: 策略开发 (Day 29-35)

#### Day 29-30: Strategy接口设计
创建文件 `src/diagnosis/strategies/base_strategy.py`:
```python
from abc import ABC, abstractmethod

class RepairStrategy(ABC):
    """修复策略基类"""

    @abstractmethod
    def can_handle(self, error_info: dict, root_cause: str) -> bool:
        """判断该策略是否能处理此错误"""
        pass

    @abstractmethod
    def generate_repair_prompt(
        self,
        error_info: dict,
        original_code: str,
        table: pd.DataFrame,
        question: str
    ) -> str:
        """生成修复prompt"""
        pass

    def repair(self, llm, error_info, original_code, table, question):
        """执行修复"""
        prompt = self.generate_repair_prompt(
            error_info, original_code, table, question
        )
        repaired_code = llm.generate(prompt)
        return repaired_code
```

#### Day 31-32: 实现Top-5核心策略
```python
# 1. ColumnNameCorrectionStrategy
class ColumnNameCorrectionStrategy(RepairStrategy):
    def can_handle(self, error_info, root_cause):
        return (
            error_info['error_type'] == 'KeyError' and
            root_cause in ['ColumnNotExist', 'ColumnCaseMismatch', 'ColumnTypo']
        )

    def generate_repair_prompt(self, error_info, original_code, table, question):
        missing_col = self._extract_column_name(error_info['error_message'])
        available_cols = table.columns.tolist()
        similar_cols = self._find_similar_columns(missing_col, available_cols)

        prompt = f"""
The code failed with error: {error_info['error_message']}

Problem: Column '{missing_col}' does not exist.
Available columns: {available_cols}
Most similar columns: {similar_cols}

Original code:
```python
{original_code}
```

Generate corrected code by replacing '{missing_col}' with the correct column name.

Corrected code:
"""
        return prompt

# 2. TypeConversionStrategy
class TypeConversionStrategy(RepairStrategy):
    def can_handle(self, error_info, root_cause):
        return error_info['error_type'] == 'TypeError'

    def generate_repair_prompt(self, error_info, original_code, table, question):
        prompt = f"""
The code failed with TypeError: {error_info['error_message']}

This usually means you're performing numeric operations on string data.

Original code:
```python
{original_code}
```

Generate corrected code by:
1. Converting string columns to numeric using pd.to_numeric()
2. Handling non-numeric values with errors='coerce'

Corrected code:
"""
        return prompt

# 3. AggregationCorrectionStrategy
# 4. FilterRelaxationStrategy
# 5. NullHandlingStrategy
```

#### Day 33-34: 实现剩余15个策略
按照相同模式实现:
- IndexCorrectionStrategy
- SyntaxFixStrategy
- IndentationFixStrategy
- AttributeCorrectionStrategy
- ValueErrorHandlingStrategy
- SchemaGroundingStrategy
- ...

创建目录结构:
```
src/diagnosis/strategies/
├── base_strategy.py
├── __init__.py
├── column_strategies.py      # 列相关策略 (5个)
├── type_strategies.py         # 类型相关策略 (4个)
├── aggregation_strategies.py  # 聚合相关策略 (3个)
├── filter_strategies.py       # 过滤相关策略 (3个)
├── syntax_strategies.py       # 语法相关策略 (3个)
└── semantic_strategies.py     # 语义相关策略 (2个)
```

#### Day 35: 策略测试
创建文件 `tests/test_strategies.py`:
```python
import pytest

def test_column_name_correction():
    """测试列名纠正策略"""
    strategy = ColumnNameCorrectionStrategy()

    error_info = {
        'error_type': 'KeyError',
        'error_message': "KeyError: 'Population'"
    }
    table = pd.DataFrame({
        'population': [100, 200],  # 注意小写
        'city': ['A', 'B']
    })
    code = "df['Population'].sum()"

    # 测试can_handle
    assert strategy.can_handle(error_info, 'ColumnCaseMismatch')

    # 测试prompt生成
    prompt = strategy.generate_repair_prompt(error_info, code, table, "What is the total population?")
    assert 'population' in prompt.lower()
    assert 'Population' in prompt

# 为每个策略编写类似测试
```

---

### Week 6: 集成诊断系统 (Day 36-42)

#### Day 36-37: 策略选择器
创建文件 `src/diagnosis/strategy_selector.py`:
```python
class StrategySelector:
    """Layer 3: 策略选择"""

    def __init__(self):
        # 注册所有策略
        self.strategies = [
            ColumnNameCorrectionStrategy(),
            TypeConversionStrategy(),
            AggregationCorrectionStrategy(),
            FilterRelaxationStrategy(),
            NullHandlingStrategy(),
            # ... 其他15个策略
        ]

    def select(self, error_info: dict, root_cause: str) -> RepairStrategy:
        """
        选择最合适的修复策略
        """
        for strategy in self.strategies:
            if strategy.can_handle(error_info, root_cause):
                return strategy

        # 如果没有匹配的策略，返回通用策略
        return GenericRepairStrategy()
```

#### Day 38-39: Prompt生成器
创建文件 `src/diagnosis/prompt_generator.py`:
```python
class PromptGenerator:
    """Layer 4: Prompt生成"""

    def generate(
        self,
        error_info: dict,
        root_cause: str,
        strategy: RepairStrategy,
        original_code: str,
        table: pd.DataFrame,
        question: str
    ) -> str:
        """
        生成完整的修复prompt
        """
        # 调用策略的prompt生成方法
        strategy_prompt = strategy.generate_repair_prompt(
            error_info, original_code, table, question
        )

        # 添加通用的上下文信息
        full_prompt = f"""
You are a Python code debugging expert.

Table Schema:
Columns: {table.columns.tolist()}
Dtypes: {table.dtypes.to_dict()}
Shape: {table.shape}

Question: {question}

{strategy_prompt}

Important:
1. Only output the corrected Python code
2. Use variable 'df' for the DataFrame
3. Store final answer in variable 'answer'
4. Do not include explanations
"""
        return full_prompt
```

#### Day 40-41: 完整诊断系统
创建文件 `src/diagnosis/diagnostic_system.py`:
```python
class HierarchicalDiagnosticSystem:
    """4层诊断系统集成"""

    def __init__(self):
        self.classifier = ErrorClassifier()
        self.root_cause_analyzer = RootCauseAnalyzer()
        self.strategy_selector = StrategySelector()
        self.prompt_generator = PromptGenerator()

    def diagnose(
        self,
        error_info: dict,
        original_code: str,
        table: pd.DataFrame,
        question: str
    ) -> dict:
        """
        完整的诊断流程

        返回:
        {
            'error_class': str,       # Layer 1
            'root_cause': str,        # Layer 2
            'strategy': RepairStrategy,  # Layer 3
            'repair_prompt': str      # Layer 4
        }
        """
        # Layer 1: 分类
        error_class = self.classifier.classify(error_info)

        # Layer 2: 根因分析
        root_cause = self.root_cause_analyzer.analyze(
            error_info, error_class, table
        )

        # Layer 3: 策略选择
        strategy = self.strategy_selector.select(error_info, root_cause)

        # Layer 4: Prompt生成
        repair_prompt = self.prompt_generator.generate(
            error_info, root_cause, strategy,
            original_code, table, question
        )

        return {
            'error_class': error_class,
            'root_cause': root_cause,
            'strategy': strategy.__class__.__name__,
            'repair_prompt': repair_prompt
        }
```

#### Day 42: 诊断系统测试
创建文件 `tests/test_diagnostic_system.py`:
```python
def test_end_to_end_diagnosis():
    """端到端测试诊断系统"""
    system = HierarchicalDiagnosticSystem()

    # 测试用例1: KeyError
    error_info = {
        'error_type': 'KeyError',
        'error_message': "KeyError: 'Population'",
        'traceback': '...'
    }
    table = pd.DataFrame({'population': [100, 200]})
    code = "df['Population'].sum()"
    question = "What is the total population?"

    result = system.diagnose(error_info, code, table, question)

    assert result['error_class'] == 'Runtime'
    assert result['root_cause'] in ['ColumnCaseMismatch', 'ColumnNotExist']
    assert 'ColumnNameCorrectionStrategy' in result['strategy']
    assert 'population' in result['repair_prompt']
```

**Milestone检查点**:
- [ ] 20个修复策略全部实现
- [ ] 每个策略有单元测试
- [ ] 4层诊断系统集成完成
- [ ] 端到端测试通过

---

## Week 7-8: 迭代系统实现

### Week 7: Iteration Controller (Day 43-49)

#### Day 43-44: 基础迭代控制器
创建文件 `src/iteration/iteration_controller.py`:
```python
class IterationController:
    """迭代控制器"""

    def __init__(self, max_iterations=5):
        self.max_iterations = max_iterations
        self.diagnostic_system = HierarchicalDiagnosticSystem()
        self.executor = SecureCodeExecutor()

    def solve_with_iteration(
        self,
        llm,
        table: pd.DataFrame,
        question: str,
        gold_answer=None,
        return_trajectory=False
    ) -> dict:
        """
        迭代求解

        返回:
        {
            'success': bool,
            'answer': any,
            'iterations': int,
            'trajectory': list,  # 如果return_trajectory=True
            'error': dict or None
        }
        """
        # 第一次生成
        code = llm.generate_code(table, question)
        trajectory = []

        for iteration in range(self.max_iterations):
            # 执行代码
            result = self.executor.execute(code, table)

            # 记录轨迹
            trajectory.append({
                'iteration': iteration,
                'code': code,
                'result': result
            })

            # 如果成功，返回
            if result['success']:
                return {
                    'success': True,
                    'answer': result['answer'],
                    'iterations': iteration + 1,
                    'trajectory': trajectory if return_trajectory else None
                }

            # 如果失败，诊断并修复
            diagnosis = self.diagnostic_system.diagnose(
                result['error'], code, table, question
            )

            # 使用诊断的prompt生成修复代码
            code = llm.generate(diagnosis['repair_prompt'])

        # 超过最大迭代次数
        return {
            'success': False,
            'answer': None,
            'iterations': self.max_iterations,
            'trajectory': trajectory if return_trajectory else None,
            'error': result['error']
        }
```

#### Day 45-46: 动态迭代预算
创建文件 `src/iteration/dynamic_budget.py`:
```python
class DynamicIterationBudget:
    """根据错误严重程度和进展动态调整迭代次数"""

    def __init__(self, min_iter=1, max_iter=5):
        self.min_iter = min_iter
        self.max_iter = max_iter

    def determine_budget(
        self,
        error_severity: str,
        progress: float,
        iteration: int
    ) -> int:
        """
        确定剩余迭代预算

        参数:
        - error_severity: 'low', 'medium', 'high'
        - progress: 0.0-1.0, 错误改善程度
        - iteration: 当前迭代次数

        返回:
        剩余可用迭代次数
        """
        base_budget = {
            'low': 2,     # Syntax error: 快速修复
            'medium': 3,  # Runtime error: 中等难度
            'high': 5     # Logic error: 需要更多尝试
        }[error_severity]

        # 如果进展良好，允许更多迭代
        if progress > 0.5:
            base_budget += 1
        elif progress < 0.2:
            base_budget -= 1

        # 确保在范围内
        remaining = max(
            self.min_iter,
            min(base_budget - iteration, self.max_iter)
        )

        return remaining

    def compute_progress(self, prev_error, current_error):
        """
        计算错误改善程度

        返回 0.0-1.0
        """
        severity_score = {
            'SyntaxError': 1,
            'KeyError': 2,
            'TypeError': 2,
            'LogicError': 3,
            'SemanticError': 4
        }

        prev_score = severity_score.get(prev_error['error_type'], 3)
        curr_score = severity_score.get(current_error['error_type'], 3)

        if curr_score < prev_score:
            # 错误严重程度降低
            progress = (prev_score - curr_score) / prev_score
        else:
            # 错误未改善
            progress = 0.0

        return progress
```

#### Day 47-48: 完整系统集成
创建文件 `src/system/table_qa_system.py`:
```python
class TableQASystem:
    """完整的Table QA系统"""

    def __init__(
        self,
        model_name="Qwen/Qwen2.5-14B-Instruct",
        use_diagnosis=True,
        use_dynamic_budget=True
    ):
        self.llm = QwenModel(model_name)
        self.executor = SecureCodeExecutor()
        self.use_diagnosis = use_diagnosis

        if use_diagnosis:
            self.diagnostic_system = HierarchicalDiagnosticSystem()

        if use_dynamic_budget:
            self.iteration_controller = IterationController(
                max_iterations=5,
                dynamic_budget=True
            )
        else:
            self.iteration_controller = IterationController(
                max_iterations=2  # 固定2次，类似AILS-NTUA
            )

    def answer_question(self, table, question, gold_answer=None):
        """回答问题"""
        if self.use_diagnosis:
            return self.iteration_controller.solve_with_iteration(
                self.llm, table, question, gold_answer
            )
        else:
            # 简单的zero-shot
            code = self.llm.generate_code(table, question)
            result = self.executor.execute(code, table)
            return result
```

#### Day 49: 系统测试
创建文件 `tests/test_table_qa_system.py`:
```python
def test_system_with_diagnosis():
    """测试带诊断的系统"""
    system = TableQASystem(
        model_name="Qwen/Qwen2.5-14B-Instruct",
        use_diagnosis=True
    )

    table = pd.DataFrame({
        'city': ['Beijing', 'Shanghai'],
        'population': [21.54, 24.28]  # 单位：百万
    })
    question = "What is the total population?"

    result = system.answer_question(table, question)

    assert result['success']
    assert abs(result['answer'] - 45.82) < 0.01
    print(f"Solved in {result['iterations']} iterations")
```

---

### Week 8: 评估与分析 (Day 50-56)

#### Day 50-51: 完整评估脚本
创建文件 `scripts/full_evaluation.py`:
```python
def evaluate_system(system, dataset, split='dev', n_samples=None):
    """
    完整评估

    返回:
    {
        'accuracy': float,
        'avg_iterations': float,
        'success_at_1': float,
        'success_at_2': float,
        'success_at_3': float,
        'error_recovery_rate': float,
        'error_breakdown': dict
    }
    """
    results = []

    for sample in tqdm(dataset):
        result = system.answer_question(
            sample['table'],
            sample['question'],
            sample['answer'],
            return_trajectory=True
        )
        results.append(result)

    # 计算指标
    metrics = compute_metrics(results, dataset)
    return metrics
```

**运行评估**:
```bash
# Qwen-2.5-14B + 完整诊断系统
python scripts/full_evaluation.py \
    --model qwen-2.5-14b \
    --use_diagnosis \
    --use_dynamic_budget \
    --dataset wikitq \
    --split dev \
    --output results/qwen14b_full_system_wikitq_dev.json

# 预期结果:
# Accuracy: 68-72%
# Avg iterations: 1.6-1.9
# Success@1: ~55%
# Success@2: ~68%
# Success@3: ~71%
# Error recovery: ~85%
```

#### Day 52-53: Ablation Studies
```bash
# 1. No diagnosis (baseline)
python scripts/full_evaluation.py \
    --model qwen-2.5-14b \
    --no_diagnosis \
    --dataset wikitq \
    --output results/ablation_no_diagnosis.json

# 2. Fixed 2 iterations (AILS-NTUA style)
python scripts/full_evaluation.py \
    --model qwen-2.5-14b \
    --use_diagnosis \
    --max_iterations 2 \
    --no_dynamic_budget \
    --dataset wikitq \
    --output results/ablation_fixed_2iter.json

# 3. No dynamic budget
python scripts/full_evaluation.py \
    --model qwen-2.5-14b \
    --use_diagnosis \
    --max_iterations 5 \
    --no_dynamic_budget \
    --dataset wikitq \
    --output results/ablation_no_dynamic.json

# 4. Full system
python scripts/full_evaluation.py \
    --model qwen-2.5-14b \
    --use_diagnosis \
    --use_dynamic_budget \
    --dataset wikitq \
    --output results/ablation_full.json
```

#### Day 54-55: 错误分析
创建notebook `notebooks/03_error_analysis.ipynb`:
```python
# 分析哪些类型的错误被成功修复
import json

results = json.load(open('results/qwen14b_full_system_wikitq_dev.json'))

# 统计错误修复情况
error_recovery = {}
for result in results['trajectories']:
    if len(result) > 1:  # 有迭代
        first_error = result[0]['error']['error_type']
        final_success = result[-1]['success']

        if first_error not in error_recovery:
            error_recovery[first_error] = {'total': 0, 'recovered': 0}

        error_recovery[first_error]['total'] += 1
        if final_success:
            error_recovery[first_error]['recovered'] += 1

# 计算每种错误的恢复率
for error_type, stats in error_recovery.items():
    rate = stats['recovered'] / stats['total']
    print(f"{error_type}: {rate:.2%} ({stats['recovered']}/{stats['total']})")
```

#### Day 56: Baseline对比
创建对比表格:
```python
import pandas as pd

comparison = pd.DataFrame({
    'Method': [
        'Qwen-14B Zero-Shot',
        'Qwen-14B Few-Shot',
        'Qwen-14B AILS-NTUA',
        'Qwen-14B Ours (No Diagnosis)',
        'Qwen-14B Ours (Fixed 2 Iter)',
        'Qwen-14B Ours (Full)',
        'Llama-70B AILS-NTUA'
    ],
    'WikiTQ Acc': [54.2, 58.1, 60.3, 63.5, 66.8, 69.7, 66.2],
    'Avg Iter': [1.0, 1.0, 2.0, 1.5, 2.0, 1.8, 2.0],
    'Recovery Rate': [0, 0, 73, 78, 82, 86, 75]
})

print(comparison.to_markdown(index=False))
```

**Milestone检查点**:
- [ ] 迭代控制器实现完成
- [ ] 动态预算机制实现
- [ ] 完整系统集成测试通过
- [ ] WikiTQ dev set评估完成，准确率达到68-72%

---

## Week 9-10: GRPO训练

### Week 9: GRPO实现 (Day 57-63)

#### Day 57-58: Reward Function
创建文件 `src/grpo/reward_function.py`:
```python
class MultiComponentReward:
    """多组件奖励函数"""

    def __init__(
        self,
        w_exec=0.3,      # 执行成功
        w_acc=0.4,       # 答案准确性
        w_eff=0.1,       # 效率
        w_repair=0.1,    # 修复质量
        w_quality=0.1    # 代码质量
    ):
        self.w_exec = w_exec
        self.w_acc = w_acc
        self.w_eff = w_eff
        self.w_repair = w_repair
        self.w_quality = w_quality

    def compute(self, trajectory, gold_answer):
        """
        计算轨迹的总奖励

        参数:
        trajectory: [
            {'iteration': 0, 'code': '...', 'result': {...}},
            {'iteration': 1, 'code': '...', 'result': {...}},
            ...
        ]
        """
        final_result = trajectory[-1]['result']

        # R1: 执行成功 (0.3权重)
        r_exec = 1.0 if final_result['success'] else -0.5

        # R2: 答案准确性 (0.4权重)
        if final_result['success']:
            r_acc = self._compute_accuracy(
                final_result['answer'],
                gold_answer
            )
        else:
            r_acc = 0.0

        # R3: 效率 (0.1权重)
        num_iters = len(trajectory)
        r_eff = 1.0 - (num_iters - 1) / 5.0  # 归一化到[0, 1]

        # R4: 修复质量 (0.1权重)
        if num_iters > 1:
            r_repair = self._compute_repair_quality(trajectory)
        else:
            r_repair = 0.0

        # R5: 代码质量 (0.1权重)
        final_code = trajectory[-1]['code']
        r_quality = self._evaluate_code_quality(final_code)

        # 总奖励
        total = (
            self.w_exec * r_exec +
            self.w_acc * r_acc +
            self.w_eff * r_eff +
            self.w_repair * r_repair +
            self.w_quality * r_quality
        )

        return total

    def _compute_accuracy(self, pred, gold):
        """计算答案准确性"""
        if self._exact_match(pred, gold):
            return 1.0
        else:
            # 部分匹配 (F1)
            return self._compute_f1(pred, gold)

    def _compute_repair_quality(self, trajectory):
        """计算修复质量 - 错误是否在改善"""
        if len(trajectory) < 2:
            return 0.0

        # 错误严重程度评分
        severity_scores = []
        for step in trajectory:
            if not step['result']['success']:
                error_type = step['result']['error']['error_type']
                severity = {
                    'SyntaxError': 1,
                    'KeyError': 2,
                    'TypeError': 2,
                    'LogicError': 3,
                    'SemanticError': 4
                }.get(error_type, 3)
                severity_scores.append(severity)
            else:
                severity_scores.append(0)  # 成功

        # 计算改善程度
        if severity_scores[0] == 0:
            return 0.0  # 第一次就成功

        improvement = (severity_scores[0] - severity_scores[-1]) / severity_scores[0]
        return max(0.0, improvement)

    def _evaluate_code_quality(self, code):
        """评估代码质量"""
        score = 1.0

        # 惩罚过长代码
        if len(code.split('\n')) > 20:
            score -= 0.2

        # 奖励向量化操作
        vectorized_ops = ['.sum()', '.mean()', '.max()', '.min()', '.groupby(']
        if any(op in code for op in vectorized_ops):
            score += 0.1

        # 惩罚循环 (低效)
        if 'for ' in code or 'while ' in code:
            score -= 0.1

        return max(0.0, min(1.0, score))
```

#### Day 59-60: Group Sampler
创建文件 `src/grpo/group_sampler.py`:
```python
class GroupSampler:
    """Group-based采样器"""

    def __init__(self, group_size=4):
        self.group_size = group_size

    def sample_group(self, model, table, question, gold_answer):
        """
        为一个问题生成group_size个不同的解决轨迹

        返回:
        [
            {'trajectory': [...], 'reward': float},
            {'trajectory': [...], 'reward': float},
            {'trajectory': [...], 'reward': float},
            {'trajectory': [...], 'reward': float}
        ]
        """
        group_samples = []

        for i in range(self.group_size):
            # 使用temperature采样生成不同的解
            result = model.solve_with_iteration(
                table, question, gold_answer,
                return_trajectory=True,
                temperature=0.8  # 增加多样性
            )

            # 计算奖励
            reward = self.reward_function.compute(
                result['trajectory'],
                gold_answer
            )

            group_samples.append({
                'trajectory': result['trajectory'],
                'reward': reward
            })

        return group_samples
```

#### Day 61-62: GRPO Trainer
创建文件 `src/grpo/grpo_trainer.py`:
```python
class GRPOTrainer:
    """Group Relative Policy Optimization训练器"""

    def __init__(
        self,
        model,
        reward_function,
        group_size=4,
        learning_rate=1e-6,
        clip_range=0.2,
        kl_coef=0.01
    ):
        self.model = model
        self.reward_function = reward_function
        self.group_size = group_size
        self.clip_range = clip_range
        self.kl_coef = kl_coef

        # 保存初始模型作为reference
        self.ref_model = deepcopy(model)
        self.ref_model.eval()

        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate
        )

    def train_step(self, batch):
        """
        一个训练步骤

        参数:
        batch: [
            {'table': ..., 'question': ..., 'gold_answer': ...},
            ...
        ]
        """
        all_trajectories = []
        all_rewards = []

        # Step 1: 为每个样本生成group_size个解
        for sample in batch:
            group_samples = self.sample_group(
                self.model,
                sample['table'],
                sample['question'],
                sample['gold_answer']
            )

            trajectories = [s['trajectory'] for s in group_samples]
            rewards = [s['reward'] for s in group_samples]

            all_trajectories.extend(trajectories)
            all_rewards.extend(rewards)

        # Step 2: 计算group-based advantages
        advantages = self._compute_group_advantages(
            all_rewards,
            self.group_size
        )

        # Step 3: PPO-style policy update
        loss = self._compute_policy_loss(
            all_trajectories,
            advantages
        )

        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {
            'loss': loss.item(),
            'avg_reward': np.mean(all_rewards),
            'avg_advantage': np.mean(advantages)
        }

    def _compute_group_advantages(self, all_rewards, group_size):
        """
        使用组平均作为baseline计算advantage

        GRPO核心创新: 不需要value function
        """
        advantages = []

        for i in range(0, len(all_rewards), group_size):
            group_rewards = all_rewards[i:i+group_size]
            group_mean = np.mean(group_rewards)
            group_std = np.std(group_rewards) + 1e-8

            # 归一化advantage
            group_adv = [
                (r - group_mean) / group_std
                for r in group_rewards
            ]
            advantages.extend(group_adv)

        return advantages

    def _compute_policy_loss(self, trajectories, advantages):
        """
        计算PPO-style的clipped loss
        """
        total_loss = 0

        for traj, adv in zip(trajectories, advantages):
            # 获取轨迹的log probability
            log_probs = self.model.compute_log_prob(traj)
            ref_log_probs = self.ref_model.compute_log_prob(traj)

            # 计算ratio
            ratio = torch.exp(log_probs - ref_log_probs)

            # Clipped surrogate objective
            clipped_ratio = torch.clamp(
                ratio,
                1 - self.clip_range,
                1 + self.clip_range
            )

            # Policy loss
            policy_loss = -torch.min(
                ratio * adv,
                clipped_ratio * adv
            )

            # KL penalty
            kl_div = ref_log_probs - log_probs
            kl_penalty = self.kl_coef * kl_div

            total_loss += policy_loss + kl_penalty

        return total_loss / len(trajectories)

    def train(self, train_dataset, val_dataset, num_epochs=5, batch_size=16):
        """
        完整训练循环
        """
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")

            # 训练
            train_metrics = []
            for batch in tqdm(DataLoader(train_dataset, batch_size=batch_size)):
                metrics = self.train_step(batch)
                train_metrics.append(metrics)

            # 验证
            val_metrics = self.evaluate(val_dataset)

            print(f"Train Loss: {np.mean([m['loss'] for m in train_metrics]):.4f}")
            print(f"Train Reward: {np.mean([m['avg_reward'] for m in train_metrics]):.4f}")
            print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")

            # 保存checkpoint
            torch.save(
                self.model.state_dict(),
                f'checkpoints/model_epoch_{epoch}.pt'
            )
```

#### Day 63: GRPO训练脚本
创建文件 `scripts/train_grpo.py`:
```bash
python scripts/train_grpo.py \
    --model qwen-2.5-14b \
    --train_data data/wikitq/train.jsonl \
    --val_data data/wikitq/dev.jsonl \
    --num_epochs 5 \
    --batch_size 8 \
    --group_size 4 \
    --learning_rate 1e-6 \
    --output_dir checkpoints/grpo_qwen14b

# 预期训练时间:
# - 单卡A100: ~24-30小时
# - 4卡A100: ~6-8小时
```

---

### Week 10: 训练与调优 (Day 64-70)

#### Day 64-66: 实际训练
```bash
# 启动训练 (需要4×A100 40GB)
export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun --nproc_per_node=4 scripts/train_grpo.py \
    --model Qwen/Qwen2.5-14B-Instruct \
    --train_data data/wikitq/train.jsonl \
    --val_data data/wikitq/dev.jsonl \
    --num_epochs 5 \
    --batch_size 4 \  # per GPU
    --group_size 4 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 2 \
    --warmup_ratio 0.1 \
    --logging_steps 10 \
    --eval_steps 500 \
    --save_steps 1000 \
    --output_dir checkpoints/grpo_qwen14b_wikitq \
    --wandb_project table-qa-grpo

# 监控训练
wandb login
# 访问 https://wandb.ai/your-username/table-qa-grpo
```

#### Day 67-68: 超参数调优
```bash
# 尝试不同的group_size
for group_size in 2 4 8; do
    python scripts/train_grpo.py \
        --group_size $group_size \
        --output_dir checkpoints/grpo_groupsize_${group_size}
done

# 尝试不同的learning_rate
for lr in 5e-7 1e-6 2e-6; do
    python scripts/train_grpo.py \
        --learning_rate $lr \
        --output_dir checkpoints/grpo_lr_${lr}
done

# 尝试不同的reward权重
python scripts/train_grpo.py \
    --reward_exec 0.4 \
    --reward_acc 0.4 \
    --reward_eff 0.1 \
    --reward_repair 0.05 \
    --reward_quality 0.05 \
    --output_dir checkpoints/grpo_reward_balanced
```

#### Day 69-70: 模型评估
```bash
# 评估最佳checkpoint
python scripts/evaluate.py \
    --model checkpoints/grpo_qwen14b_wikitq/best_model \
    --dataset wikitq \
    --split test \
    --output results/grpo_qwen14b_wikitq_test.json

# 对比未训练的模型
python scripts/evaluate.py \
    --model Qwen/Qwen2.5-14B-Instruct \
    --use_diagnosis \
    --dataset wikitq \
    --split test \
    --output results/qwen14b_no_grpo_wikitq_test.json

# 创建对比
python scripts/compare_results.py \
    --baseline results/qwen14b_no_grpo_wikitq_test.json \
    --grpo results/grpo_qwen14b_wikitq_test.json \
    --output results/grpo_comparison.json
```

**预期结果**:
```
Qwen-14B + Diagnosis (No GRPO): 68.5%
Qwen-14B + Diagnosis + GRPO:    70.2%
Improvement:                     +1.7%

Avg iterations:
  No GRPO: 1.95
  GRPO:    1.78

Error recovery:
  No GRPO: 84%
  GRPO:    87%
```

**Milestone检查点**:
- [ ] GRPO训练完成 (5 epochs)
- [ ] 在WikiTQ test上达到68-72%
- [ ] GRPO相比无GRPO版本提升1-2%
- [ ] 迭代效率提升 (~10% fewer iterations)

---

## Week 11: 全面评估

### Day 71-77: 所有数据集评估

#### Day 71-72: WikiTQ完整评估
```bash
# 1. 所有baselines
python scripts/batch_evaluate.py \
    --models gpt-4o,claude-3.5,qwen-14b-zero,qwen-14b-cot,qwen-14b-ails,llama-70b-ails \
    --dataset wikitq \
    --split test \
    --output results/wikitq_baselines.json

# 2. 我们的方法 (ablations)
python scripts/batch_evaluate.py \
    --models qwen-14b-no-diagnosis,qwen-14b-diagnosis-no-grpo,qwen-14b-full \
    --dataset wikitq \
    --split test \
    --output results/wikitq_ours.json

# 3. Qwen-32B
python scripts/evaluate.py \
    --model Qwen/Qwen2.5-32B-Instruct \
    --use_diagnosis \
    --use_grpo \
    --checkpoint checkpoints/grpo_qwen32b/best_model \
    --dataset wikitq \
    --split test \
    --output results/qwen32b_full_wikitq_test.json
```

#### Day 73-74: TabFact评估
```bash
# 同样的流程，在TabFact上评估
python scripts/batch_evaluate.py \
    --models all \
    --dataset tabfact \
    --split test \
    --output results/tabfact_all.json
```

#### Day 75: FeTaQA评估
```bash
python scripts/batch_evaluate.py \
    --models all \
    --dataset fetaqa \
    --split test \
    --output results/fetaqa_all.json
```

#### Day 76: SemEval-2025 Task 8评估
```bash
python scripts/batch_evaluate.py \
    --models all \
    --dataset semeval2025 \
    --split test \
    --output results/semeval2025_all.json
```

#### Day 77: 结果汇总
创建notebook `notebooks/04_final_results.ipynb`:
```python
import pandas as pd
import json

# 加载所有结果
wikitq_results = json.load(open('results/wikitq_all.json'))
tabfact_results = json.load(open('results/tabfact_all.json'))
fetaqa_results = json.load(open('results/fetaqa_all.json'))
semeval_results = json.load(open('results/semeval2025_all.json'))

# 创建主表格
main_table = pd.DataFrame({
    'Model': ['GPT-4o', 'Claude-3.5', 'Qwen-14B Zero', 'Qwen-14B CoT',
              'Qwen-14B AILS', 'Llama-70B AILS', 'Qwen-14B Ours (No Diag)',
              'Qwen-14B Ours (No GRPO)', 'Qwen-14B Ours (Full)',
              'Qwen-32B Ours (Full)'],
    'Size': ['?', '?', '14B', '14B', '14B', '70B', '14B', '14B', '14B', '32B'],
    'WikiTQ': [...],  # 从结果文件提取
    'TabFact': [...],
    'FeTaQA': [...],
    'SemEval': [...]
})

print(main_table.to_markdown(index=False))

# 保存为LaTeX
with open('results/main_table.tex', 'w') as f:
    f.write(main_table.to_latex(index=False))
```

---

## Week 12: 论文撰写

### Day 78-84: 写作与提交

#### Day 78-79: 方法部分
编写文件 `paper/sections/03_method.tex`:
```latex
\section{Methodology}

\subsection{System Overview}
Our system consists of three core components:
(1) Hierarchical Error Diagnosis,
(2) Iterative Code Refinement, and
(3) GRPO-based Policy Learning.

\subsection{Hierarchical Error Diagnosis}
We propose a 4-layer diagnostic framework...

[详细描述4层系统]

\subsection{GRPO for Iterative Refinement}
Unlike prior work that optimizes single-step generation,
we apply GRPO to optimize the entire repair trajectory...

[详细描述GRPO方法]
```

#### Day 80-81: 实验部分
编写文件 `paper/sections/04_experiments.tex`:
```latex
\section{Experiments}

\subsection{Experimental Setup}

\paragraph{Datasets}
We evaluate on four benchmarks:
WikiTQ, TabFact, FeTaQA, and SemEval-2025 Task 8.

\paragraph{Baselines}
We compare against 9 baselines...

\paragraph{Implementation Details}
All experiments use Qwen-2.5-14B/32B...

\subsection{Main Results}

\begin{table}[t]
\centering
\caption{Main results on WikiTQ and TabFact.}
\label{tab:main_results}
\input{tables/main_results.tex}
\end{table}

Our method achieves XX\% on WikiTQ...

\subsection{Ablation Studies}

[插入ablation表格和分析]

\subsection{Error Analysis}

[插入错误恢复分析]
```

#### Day 82: 相关工作
编写文件 `paper/sections/02_related_work.tex`:
```latex
\section{Related Work}

\subsection{Table Question Answering}

\subsection{Code Generation for Tables}

\subsection{Error Correction and Refinement}

\subsection{Reinforcement Learning for Reasoning}
```

#### Day 83: Introduction与Abstract
编写文件:
- `paper/sections/01_introduction.tex`
- `paper/sections/00_abstract.tex`

#### Day 84: 校对与提交
```bash
# 编译论文
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

# 检查
python scripts/check_paper.py \
    --file paper/main.pdf \
    --check_length \
    --check_citations \
    --check_figures

# 提交
# 按照ACL/EMNLP/NAACL的submission guidelines提交
```

---

## 关键文件清单

### 代码文件 (src/)
```
src/
├── data/
│   └── data_loader.py           # 数据加载
├── execution/
│   └── sandbox.py               # 代码执行沙盒
├── baselines/
│   ├── gpt4_baseline.py
│   ├── qwen_baseline.py
│   └── ails_ntua.py
├── diagnosis/
│   ├── error_taxonomy.py        # Taxonomy定义
│   ├── error_classifier.py      # Layer 1
│   ├── root_cause_analyzer.py   # Layer 2
│   ├── strategy_selector.py     # Layer 3
│   ├── prompt_generator.py      # Layer 4
│   ├── diagnostic_system.py     # 完整系统
│   └── strategies/              # 20个策略
│       ├── column_strategies.py
│       ├── type_strategies.py
│       └── ...
├── iteration/
│   ├── iteration_controller.py  # 迭代控制
│   └── dynamic_budget.py        # 动态预算
├── grpo/
│   ├── reward_function.py       # 奖励函数
│   ├── group_sampler.py         # Group采样
│   └── grpo_trainer.py          # GRPO训练
└── system/
    └── table_qa_system.py       # 完整系统
```

### 脚本文件 (scripts/)
```
scripts/
├── preprocess_wikitq.py
├── collect_errors.py            # 收集错误
├── annotate_errors.py           # 错误标注
├── compute_agreement.py         # 标注一致性
├── validate_taxonomy.py         # Taxonomy验证
├── eval_baseline.py             # 评估baseline
├── full_evaluation.py           # 完整评估
├── train_grpo.py                # GRPO训练
├── batch_evaluate.py            # 批量评估
└── compare_results.py           # 结果对比
```

### 数据文件 (data/)
```
data/
├── wikitq/
│   ├── train.jsonl (11,321)
│   ├── dev.jsonl (2,831)
│   └── test.jsonl (4,344)
├── tabfact/
├── fetaqa/
├── semeval2025/
└── errors/
    ├── wikitq_qwen14b_errors.jsonl
    ├── annotation_samples_500.jsonl
    └── annotated_500.jsonl
```

### 结果文件 (results/)
```
results/
├── gpt4o_wikitq_dev_100.json
├── qwen14b_wikitq_dev_100.json
├── qwen14b_ails_wikitq_dev_100.json
├── qwen14b_full_system_wikitq_dev.json
├── ablation_*.json
├── wikitq_all.json
├── tabfact_all.json
├── fetaqa_all.json
├── semeval2025_all.json
├── main_table.csv
└── main_table.tex
```

---

## 时间估算

### 人力需求
- **1名研究生 (主力)**: 负责代码实现、实验运行
- **1名合作者 (辅助)**: 负责错误标注、论文撰写
- **导师指导**: 每周1-2次meeting

### 计算资源需求
- **开发阶段 (Week 1-8)**: 1×RTX 3090 or A100 (24GB)
- **GRPO训练 (Week 9-10)**: 4×A100 (40GB), ~30-40小时
- **评估阶段 (Week 11)**: 2×A100, ~20小时

### 预算估算
- GPU租用 (如果没有本地GPU):
  - 开发: $200 (8周 × $25/周)
  - 训练: $400 (40小时 × $10/小时 × 4卡)
  - 评估: $200
  - **总计: ~$800**

- API费用 (GPT-4o, Claude baselines):
  - 每个数据集评估: ~$50
  - 总计: ~$200

- **总预算: ~$1,000**

---

## 关键Milestone

- [ ] **Week 2 End**: Baseline评估完成 (Qwen-14B在WikiTQ达到58-62%)
- [ ] **Week 4 End**: Error Taxonomy设计完成 (覆盖率>95%)
- [ ] **Week 6 End**: 诊断系统实现完成 (20个策略)
- [ ] **Week 8 End**: 完整系统评估 (WikiTQ达到68-72%)
- [ ] **Week 10 End**: GRPO训练完成 (+1-2% improvement)
- [ ] **Week 11 End**: 所有数据集评估完成
- [ ] **Week 12 End**: 论文提交

---

## 风险与应对

| 风险 | 概率 | 应对策略 |
|------|------|----------|
| Baseline复现困难 | 中 | 使用官方代码,联系作者 |
| GRPO训练不稳定 | 中 | 降低learning rate,增加warmup |
| 性能不达预期 | 中 | 强调error recovery和efficiency |
| GPU资源不足 | 低 | 使用云服务器或申请集群 |
| 论文被拒 | 中 | 投EMNLP Findings或Workshop |

---

**文档版本**: 1.0
**最后更新**: 2025-10-16
**状态**: 执行计划完成
