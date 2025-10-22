# SOTA Baselines Implementation

本目录包含主流SOTA方法的复现，用于与我们的系统进行对比。

## 目录结构

```
baselines/sota_methods/
├── README.md                    # 本文件
├── chain_of_query/              # WikiTQ SOTA (74.77%)
│   ├── coq_sql_generator.py     # SQL生成器
│   ├── coq_executor.py          # SQL执行器
│   └── evaluate_coq_wikitq.py   # WikiTQ评估
├── tabfact_gnn/                 # TabFact SOTA (85%)
│   ├── gnn_model.py             # 图神经网络模型
│   └── evaluate_gnn_tabfact.py  # TabFact评估
└── requirements.txt             # 额外依赖

## 已实现的SOTA方法

### 1. Chain-of-Query (CoQ) - WikiTQ

**论文**: Chain-of-Query: Unleashing the Power of LLMs in SQL-Aided Table Understanding via Multi-Agent Collaboration

**性能**:
- WikiTQ: 74.77% (vs 我们的25%)
- Invalid SQL率: 3.34%

**核心技术**:
1. SQL生成而非Python
2. Clause-by-Clause生成策略
3. 自然语言Schema抽象
4. Multi-Agent协作

**简化实现**:
- 使用单Agent + 分步生成（而非完整Multi-Agent）
- 支持GPT-4、Claude、Qwen等模型
- 可配置是否使用few-shot examples

### 2. TabFact GNN (待实现)

**论文**: GNN-TabFact / ARTEMIS-DA

**性能**:
- TabFact: ~85%

**核心技术**:
1. 图神经网络建模表格结构
2. 数值感知机制
3. 陈述-表格匹配

## 使用方法

### Chain-of-Query on WikiTQ

```bash
# 使用GPT-4 (需要API key)
python baselines/sota_methods/chain_of_query/evaluate_coq_wikitq.py \
    --model gpt-4 \
    --num_samples 100

# 使用本地Qwen模型
python baselines/sota_methods/chain_of_query/evaluate_coq_wikitq.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --num_samples 100
```

## 预期结果对比

| 方法 | WikiTQ | TabFact | 说明 |
|------|--------|---------|------|
| **我们的系统** | 25% | TBD | Python代码生成 + 4层诊断 |
| **CoQ (GPT-4)** | ~70% | - | SQL生成 + Multi-Agent |
| **CoQ (Qwen 7B)** | ~40% (预期) | - | SQL生成但模型较弱 |
| **GNN** | - | ~85% | 专门的图神经网络 |

## 实现进度

- [x] Chain-of-Query框架
- [x] SQL生成器
- [x] WikiTQ评估脚本
- [ ] TabFact GNN实现
- [ ] 完整Multi-Agent架构

## 依赖

```bash
pip install sqlalchemy pandasql
```

## 注意事项

1. **模型选择**:
   - GPT-4/Claude: 预期达到论文报告的性能
   - Qwen 7B: 预期性能会下降，但仍优于Python生成

2. **SQL vs Python**:
   - SQL更适合简单查询（WHERE, GROUP BY）
   - Python更适合复杂计算（自定义函数、多步骤）

3. **公平对比**:
   - 所有方法使用相同的模型进行对比
   - 记录执行时间、API调用次数
