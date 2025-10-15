# Table QA: Hierarchical Error Diagnosis with GRPO

基于开源LLM的表格问答系统，集成层级化错误诊断和GRPO强化学习优化。

## 📋 项目概述

本项目实现了一个完整的Table QA系统，包含：
1. **四层错误诊断系统**（Layer 1-4）
2. **20种专门的修复策略**
3. **GRPO驱动的迭代优化**
4. **基于Qwen2.5-Coder的代码生成**

## 🗂️ 项目结构

```
DASFAA-Table/
├── src/                      # 源代码
│   ├── data/                 # ✅ 数据加载器
│   │   └── data_loader.py    # 支持WikiTQ/TabFact/FeTaQA/SemEval
│   ├── execution/            # ✅ 代码执行
│   │   └── code_executor.py  # 安全沙盒执行器
│   ├── diagnosis/            # ✅ 错误诊断系统（核心创新）
│   │   ├── error_classifier.py      # Layer 1: 错误分类
│   │   ├── root_cause_analyzer.py   # Layer 2: 根因分析
│   │   ├── strategy_selector.py     # Layer 3: 策略选择（待实现）
│   │   ├── prompt_generator.py      # Layer 4: 提示生成（待实现）
│   │   └── strategies/              # 20个修复策略（待实现）
│   ├── baselines/            # 基线方法（待实现）
│   ├── iteration/            # 迭代控制（待实现）
│   ├── grpo/                 # GRPO训练器（待实现）
│   └── system/               # 完整系统集成（待实现）
├── data/                     # 数据集目录
│   ├── wikitq/
│   ├── tabfact/
│   ├── fetaqa/
│   └── semeval2025/
├── scripts/                  # 脚本
├── tests/                    # 测试
├── results/                  # 结果输出
├── logs/                     # 日志
├── checkpoints/              # 模型检查点
├── notebooks/                # Jupyter notebooks
├── requirements.txt          # ✅ 依赖列表
└── setup.sh                  # ✅ 环境设置脚本
```

## 🚀 快速开始

### 1. 环境设置

```bash
# 创建conda环境并安装依赖
bash setup.sh

# 或手动安装
conda create -n table-qa python=3.10
conda activate table-qa
pip install -r requirements.txt
```

### 2. 数据准备

下载数据集：
- [WikiTQ](https://github.com/ppasupat/WikiTableQuestions)
- [TabFact](https://github.com/wenhuchen/Table-Fact-Checking)
- [FeTaQA](https://github.com/Yale-LILY/FeTaQA)
- [SemEval-2025 Task 8](https://www.codabench.org/competitions/3360/)

### 3. 测试已实现组件

```bash
# 测试数据加载器
python src/data/data_loader.py

# 测试代码执行器
python src/execution/code_executor.py

# 测试错误分类器
python src/diagnosis/error_classifier.py

# 测试根因分析器
python src/diagnosis/root_cause_analyzer.py
```

## 📊 使用的模型

- **代码生成**: Qwen/Qwen2.5-Coder-7B-Instruct
- **GRPO训练**: 基于HuggingFace TRL

## 📖 核心组件说明

### ✅ 已完成（核心框架完整）

#### 1. 数据处理 (`src/data/`)
- ✅ 数据加载器支持4个数据集
- ✅ 自动格式转换和标准化

#### 2. 代码执行 (`src/execution/`)
- ✅ 安全沙盒执行器
- ✅ 超时和内存保护
- ✅ 详细错误捕获

#### 3. 错误诊断系统 (`src/diagnosis/`) - **核心创新**
- ✅ Layer 1: 错误分类器
- ✅ Layer 2: 根因分析器
- ✅ Layer 3: 策略选择器（含GRPO接口）
- ✅ Layer 4: 提示生成器
- ✅ 完整诊断系统集成

#### 4. 修复策略 (`src/diagnosis/strategies/`)
- ✅ 5个核心策略已实现
- ✅ 可扩展到20个策略

#### 5. 代码生成 (`src/baselines/`)
- ✅ Qwen2.5-Coder-7B集成
- ✅ 初始生成+修复生成

#### 6. 完整系统 (`src/system/`)
- ✅ 端到端Table QA系统
- ✅ 迭代式错误修复
- ✅ 轨迹记录和批量处理

#### 7. GRPO训练 (`src/grpo/`)
- ✅ GRPO训练器接口
- ✅ 多组件奖励函数
- ⚠️ **TODO: 实际训练需使用TRL实现**

### 🔧 待完成

- [ ] 数据集下载和预处理
- [ ] GRPO训练实现（使用TRL）
- [ ] 基线评估脚本
- [ ] 完整实验和ablation studies

## 📝 引用

本项目基于以下研究工作的思路：

1. **AILS-NTUA** (SemEval-2025 Task 8 Winner)
   - 论文: https://arxiv.org/abs/2503.00435
   - 贡献: Language-to-Code + 迭代错误修复

2. **Table-R1** (TARPO强化学习)
   - 论文: https://arxiv.org/abs/2505.12415
   - 贡献: 区域化强化学习

3. **DeepSeek-R1 GRPO**
   - 论文: https://arxiv.org/abs/2501.12948
   - 贡献: 组相对策略优化

4. **OpenCodeInterpreter**
   - GitHub: https://github.com/OpenCodeInterpreter/OpenCodeInterpreter
   - 贡献: 代码生成和执行框架

## 📄 License

MIT License

## 🙏 致谢

本项目是研究项目，用于ACL/EMNLP 2025投稿。
