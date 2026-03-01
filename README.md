# BERT 新闻立场检测与意图分析

假新闻和标题党泛滥的时代，一篇新闻的标题和正文之间的关系（支持、反对、讨论、无关）是判断信息可信度的重要线索。本项目用 BERT 预训练模型对新闻标题与正文的立场关系进行自动分类——输入一对"标题+正文"，模型判断正文对标题观点的态度。

## 痛点与目的

- **问题**：社交媒体和新闻平台上标题党、假新闻横行，人工核实标题与正文的一致性效率极低，需要自动化的立场检测工具辅助事实核查
- **方案**：使用 BERT（bert-base-uncased）做文本对分类，将新闻标题和正文通过 `[SEP]` 拼接后输入模型，微调训练实现立场自动判断
- **效果**：输出每个立场类别的 Precision、Recall、F1-score 和整体 Accuracy，支持训练过程可视化

## 核心功能

- **BERT 微调分类**：基于 HuggingFace Transformers 的 BertForSequenceClassification
- **文本对输入**：标题 + 正文通过 `[SEP]` 拼接，BERT 自动学习两段文本的关系
- **HuggingFace Trainer**：标准化训练流程，每轮自动评估验证集
- **分类报告**：输出每个类别的 Precision / Recall / F1-score
- **训练可视化**：自动绘制 Loss 和 Accuracy 的训练曲线
- **立场分布分析**：可视化数据集中各立场标签的分布情况
- **模型保存**：训练完成后保存完整模型和 tokenizer，方便部署

## 使用方法

### 环境依赖

```bash
pip install torch transformers datasets scikit-learn pandas matplotlib
```

### 训练与评估

```bash
python test.py
```

程序会自动完成以下流程：
1. 读取 `test_merged.csv` 数据集
2. 可视化立场标签分布
3. 拼接标题和正文作为模型输入
4. 划分训练集（80%）和验证集（20%）
5. 加载 BERT 预训练模型并微调 3 个 epoch
6. 输出验证集分类报告
7. 保存训练好的模型到 `final_stance_model/`
8. 绘制训练曲线并保存

## 项目结构

```
.
├── test.py                   # 完整训练+评估流程
├── test_merged.csv           # 数据集（标题+正文+立场标签）
├── final_stance_model/       # 训练好的 BERT 模型
│   ├── model.safetensors     # 模型权重
│   ├── config.json           # 模型配置
│   ├── vocab.txt             # 词表
│   └── tokenizer_config.json # Tokenizer 配置
├── results/                  # 训练中间结果
├── logs/                     # 训练日志
├── stance_distribution.png   # 立场标签分布图
└── training_logs.png         # 训练曲线图
```

## 模型参数

| 参数 | 值 |
|------|-----|
| 预训练模型 | bert-base-uncased |
| 最大序列长度 | 128 |
| 学习率 | 2e-5 |
| Batch Size | 8 |
| 训练轮数 | 3 |
| 权重衰减 | 0.01 |
| 评估策略 | 每个 epoch |

## 适用场景

- 假新闻与标题党检测
- 新闻事实核查辅助
- 社交媒体舆情立场分析
- 观点挖掘与意图理解
- NLP 文本对分类实践

## 技术栈

- HuggingFace Transformers（BERT 微调）
- PyTorch（深度学习框架）
- scikit-learn（分类评估指标）
- Pandas（数据处理）
- Matplotlib（训练可视化）

## License

MIT License
