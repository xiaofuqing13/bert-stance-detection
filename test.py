# test.py

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
from datasets import Dataset

# ------------------ 可选：在 macOS 上显示中文（仅在需要可视化时启用） ------------------
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties

font_path = '/System/Library/Fonts/STHeiti Light.ttc'  # 你的字体路径
font_prop = FontProperties(fname=font_path)
rcParams['font.sans-serif'] = [font_prop.get_name()]  # 使用自定义字体
rcParams['axes.unicode_minus'] = False  # 正常显示负号

# ------------------ 自定义回调，用于在每个epoch结束后打印详细日志 ------------------
class LoggingCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"\nEpoch {state.epoch}结束：")
        # 如果log_history中有最新的 eval 结果，可以打印
        if len(state.log_history) > 0:
            last_log = state.log_history[-1]
            # 可能存在 'eval_loss'、'eval_accuracy'、'loss'、'learning_rate' 等键
            if 'eval_loss' in last_log:
                print(f"  eval_loss = {last_log['eval_loss']}")
            if 'eval_accuracy' in last_log:
                print(f"  eval_accuracy = {last_log['eval_accuracy']}")
            if 'loss' in last_log:
                print(f"  training_loss = {last_log['loss']}")
            if 'learning_rate' in last_log:
                print(f"  learning_rate = {last_log['learning_rate']}")

def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128  # 视需求可调整
    )

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

def main():
    # ========== 步骤1：读取 CSV 数据 ==========
    data_path = os.path.join(os.path.dirname(__file__), "test_merged.csv")
    df = pd.read_csv(data_path, encoding='latin-1')  # 保证使用latin-1编码

    print("数据前5行预览：")
    print(df.head())
    print("数据形状：", df.shape)

    # 如果没有空值问题可跳过，如有空值可先行删除
    df.dropna(subset=['articleBody', 'Headline', 'Stance'], inplace=True)

    # ========== 步骤2：可视化 - 立场标签分布 ==========
    stance_counts = df['Stance'].value_counts()
    print("\n各立场标签分布：\n", stance_counts)

    # 简单地做一个柱状图
    plt.figure(figsize=(6, 4))
    stance_counts.plot(kind='bar')
    plt.title("Stance 标签分布")
    plt.xlabel("Stance")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("stance_distribution.png")
    plt.close()
    print("已保存立场标签分布图 -> stance_distribution.png")

    # ========== 步骤3：合并文本作为模型输入 (可根据需要选用) ==========
    # 这里将 'Headline' 和 'articleBody' 用 [SEP] 拼接成一个文本
    df['text'] = df['Headline'] + " [SEP] " + df['articleBody']

    # ========== 步骤4：标签处理 ==========
    unique_labels = df['Stance'].unique().tolist()
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    df['label'] = df['Stance'].map(label2id)

    # ========== 步骤5：划分训练集和验证集 ==========
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df['label']
    )
    print(f"\n训练集大小: {train_df.shape[0]}, 验证集大小: {val_df.shape[0]}")

    # ========== 步骤6：准备 Tokenizer 和 Dataset ==========
    MODEL_NAME = "bert-base-uncased"  # 如果处理中文，可换成 "bert-base-chinese" 等
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
    val_dataset   = Dataset.from_pandas(val_df[['text', 'label']])

    train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    val_dataset   = val_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    train_dataset = train_dataset.rename_column("label", "labels")
    val_dataset   = val_dataset.rename_column("label", "labels")

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # ========== 步骤7：加载预训练 BERT 模型 ==========
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(unique_labels),
        id2label=id2label,
        label2id=label2id
    )

    # ========== 步骤8：设置训练参数和 Trainer ==========
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",   # 每个epoch做一次eval
        save_strategy="epoch",         # 每个epoch保存一次模型
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[LoggingCallback()]  # 注册自定义回调
    )

    # ========== 步骤9：训练并验证 ==========
    trainer.train()

    # 查看最终的验证结果
    eval_result = trainer.evaluate()
    print("\n最终验证集评估结果：", eval_result)

    # 更详细的分类报告
    predictions = trainer.predict(val_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    print("\n详细分类报告：")
    print(classification_report(predictions.label_ids, preds, target_names=unique_labels))

    # ========== 步骤10：保存模型 (可选) ==========
    trainer.save_model("./final_stance_model")
    tokenizer.save_pretrained("./final_stance_model")
    print("\n模型已保存至 ./final_stance_model")

    # ========== 步骤11：可视化训练日志 (Loss/Accuracy) ==========

    # trainer.state.log_history 将记录训练/验证过程中的日志
    log_history = trainer.state.log_history

    # 为了简单起见，我们提取 epoch级别 的 train_loss、eval_loss、eval_accuracy
    epochs = []
    train_loss_history = []
    eval_loss_history = []
    eval_acc_history  = []

    for record in log_history:
        if "epoch" in record:
            # 记录下 epoch
            epoch = int(record["epoch"])
            # 如果是train阶段，会包含 'loss'
            if "loss" in record:
                epochs.append(epoch)
                train_loss_history.append(record["loss"])
            # 如果是eval阶段，会包含 'eval_loss', 'eval_accuracy'
            if "eval_loss" in record:
                # 只更新eval
                # 注意：因为train和eval的log不一定在同一个record里
                if epoch not in epochs:
                    epochs.append(epoch)
                    train_loss_history.append(None)  # 表示此时刻没训练loss
                eval_loss_history.append(record["eval_loss"])
            if "eval_accuracy" in record:
                eval_acc_history.append(record["eval_accuracy"])

    # 进行可视化（需要先保证长度对齐）
    # 简单做个处理：数据可能不对齐时，用None补齐
    max_len = max(len(epochs), len(train_loss_history), len(eval_loss_history), len(eval_acc_history))
    while len(epochs) < max_len: epochs.append(None)
    while len(train_loss_history) < max_len: train_loss_history.append(None)
    while len(eval_loss_history) < max_len: eval_loss_history.append(None)
    while len(eval_acc_history) < max_len: eval_acc_history.append(None)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss_history, 'o-', label='Train Loss')
    plt.plot(epochs, eval_loss_history, 's-', label='Eval Loss')
    plt.plot(epochs, eval_acc_history, '^-', label='Eval Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training Logs')
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_logs.png")
    plt.close()
    print("已保存训练日志曲线 -> training_logs.png")

    print("\n全部流程完成！")

if __name__ == "__main__":
    main()