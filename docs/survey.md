Table 问答数据集

反思

基于当前的报错
然后把任务重新叙述一下



通过LLM与 py的交互，然后 tools 

迭代 遇到错误分析错误然后 纠正



对比的话

Baseline


API LLM

Base LLM


https://arxiv.org/pdf/2503.00435


加点奖励函数，用RL  GRPO


https://arxiv.org/pdf/2407.09025


  1. Table QA任务 - 基于表格的问答数据集
  2. 核心思想:
    - 通过LLM与Python工具的交互
    - 迭代式错误分析和纠正(反思机制)
    - 遇到错误时重新叙述任务
  3. 对比实验:
    - Baseline: API LLM和Base LLM
  4. 可能的增强方向:
    - 使用强化学习(RL)的GRPO算法
    - 添加奖励函数

  相关论文和资源推荐
                                                                                                                                                                                                    

  思路

  1. 切合当前热点: SemEval 2025 Task 8专注于Tabular QA,多个顶级团队都在使用iterative error correction机制
  2. 实用性强: Chain-of-Table (ICLR 2024)已证明表格推理的有效性,你们的迭代纠错思路是自然延伸
  3. 技术路线清晰: LLM + Python tools + 反思机制正是SOTA方法的核心

  📚 核心相关论文推荐

  必读论文 (与你们思路最契合)

  1. AILS-NTUA at SemEval-2025 Task 8 (2025)
    - arXiv: https://arxiv.org/abs/2503.00435
    - 你们survey.md里提到的论文!
    - 核心贡献: Language-to-Code + Error Fixing for Tabular QA
    - 关键技术: 迭代式错误修正,最多2次迭代
    - 成绩: SemEval 2025 Task 8 第一名(专有模型类别)
    - ⭐ 强烈建议: 这篇论文的思路与你们高度一致!
  2. Table-R1 (2025)
    - arXiv: https://arxiv.org/abs/2505.12415
    - 核心贡献: Region-based RL for Table Understanding
    - 关键技术: GRPO + 规则based table RL
    - 性能: 平均提升14.36分,8B模型超过GPT-4o
    - ⭐ 你们可以参考的RL方法
  3. TabSQLify (NAACL 2024)
    - arXiv: https://arxiv.org/abs/2404.10150
    - GitHub: https://github.com/mahadi-nahid/TabSQLify
    - 核心贡献: Table Decomposition增强推理
    - WikiTQ: 64.7%, TabFact: 79.5%

  Chain-of-Table后续工作

  4. STRuCT-LLM (2025)
    - arXiv: https://arxiv.org/abs/2506.21575
    - 核心贡献: GRPO应用于Text-to-SQL
    - 关键技术: execution, structural, syntactical feedback作为reward
  5. NormTab (EMNLP 2024 Findings)
    - 核心贡献: 表格标准化预处理提升符号推理

  🔥 高质量GitHub资源

  可直接使用的代码库

  1. OpenCodeInterpreter ⭐⭐⭐
    - https://github.com/OpenCodeInterpreter/OpenCodeInterpreter
    - 功能: 开源代码生成系统,集成execution + iterative refinement
    - Code-Feedback (68K多轮交互)
    - 适合你们: 提供完整的代码执行+反馈循环框架
  2. Self-Refine ⭐⭐⭐
    - https://github.com/madaan/self-refine
    - 功能: LLM自我反思和迭代改进框架
  - 实现反思机制的参考实现
  1. Chain-of-Table官方实现
    - https://github.com/google-research/chain-of-table
    - Google Research官方代码
  2. SemEval 2025 Task 8相关
    - LyS团队: https://github.com/adrian-gude/Tabular_QA
    - 评估工具: https://github.com/jorses/databench_eval
  3. Awesome-Tabular-LLMs ⭐⭐⭐
    - https://github.com/SpursGoZmy/Awesome-Tabular-LLMs
    - 功能: Table+LLM相关论文整理
    - 快速了解领域全貌

  💡 具体建议

  实现方案

  你们的系统架构可以是:
  1. Table + Question → LLM生成Python代码
  2. 执行代码 → 捕获错误
  3. 如果错误:
     - 将错误信息反馈给LLM
     - LLM分析错误并重新生成代码
     - 迭代最多N次(建议2-3次)
  4. 使用GRPO优化整个流程

  Baseline对比建议

  - 弱Baseline: 直接API调用(GPT-4, Claude等)
  - 强Baseline: Chain-of-Table, Dater
  - 你们的方法: Iterative Error Correction + GRPO

  数据集选择

  根据Chain-of-Table论文:
  - WikiTQ: 短答案QA
  - TabFact: 事实验证
  - FeTaQA: 长文本生成
  - SemEval 2025 Task 8: 最新benchmark

  🎯 创新点建议

  1. 更智能的错误分类: 不同类型错误采用不同修复策略
  2. GRPO优化: 使用execution feedback作为reward signal
  3. 表格操作历史: 记录操作链,类似Chain-of-Table但加入错误恢复
  4. Few-shot error examples: 在prompt中加入典型错误案例

  ⚠️ 潜在挑战

  1. 计算成本: 迭代次数增加会提高API成本
  2. 错误累积: 需要防止错误修复过程中引入新错误
  3. 评估指标: 除了准确率,还需考虑执行效率


AILS-NTUA团队在SemEval 2025夺冠的方法。AILS-NTUA和Table-R1两篇论文,OpenCodeInterpreter的代码框架

