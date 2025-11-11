import os
import json
import wandb
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import (
    BertTokenizer, BertModel, RobertaTokenizer, RobertaModel,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score, classification_report
)
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class FeedbackDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, tokenizer, rule_func, max_len=128):
        self.data = pd.read_csv(csv_file)
        self.data['feedback_text'] = self.data['feedback_text'].fillna("").astype(str)
        self.tokenizer = tokenizer
        self.rule_func = rule_func
        self.max_len = max_len
        self.encoders = {
            col: LabelEncoder().fit(self.data[col])
            for col in ["sentiment", "urgency", "topic", "action"]
        }
        for col in self.encoders:
            self.data[col] = self.encoders[col].transform(self.data[col])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        tokens = self.tokenizer(
            row["feedback_text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in tokens.items()}
        item["rule_vec"] = self.rule_func(row["feedback_text"])
        item["sentiment"] = torch.tensor(row["sentiment"])
        item["urgency"] = torch.tensor(row["urgency"])
        item["topic"] = torch.tensor(row["topic"])
        item["action"] = torch.tensor(row["action"])
        return item


def rule_based_features(text):
    sentiment = ["excellent", "terrible", "good", "bad", "love", "hate"]
    urgency = ["immediately", "urgent", "asap", "delay"]
    topic = ["trainer", "venue", "content", "equipment"]
    vocab = sentiment + urgency + topic
    vec = torch.zeros(len(vocab))
    for i, word in enumerate(vocab):
        if word in text.lower():
            vec[i] = 1.0
    return vec


# =========================
# Updated Model with use_rule switch
# =========================
class MultiTaskModel(nn.Module):
    def __init__(self, model_name, base_model_class, rule_dim=14, unfreeze_layers=2, use_rule=True):
        super().__init__()
        self.encoder = base_model_class.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.use_rule = use_rule
        self.rule_dim = rule_dim if use_rule else 0

        total_layers = len(self.encoder.encoder.layer)
        freeze_until = max(0, total_layers - unfreeze_layers)
        for i, layer in enumerate(self.encoder.encoder.layer):
            if i < freeze_until:
                for p in layer.parameters():
                    p.requires_grad = False

        self.dropout = nn.Dropout(0.2)
        self.sentiment = nn.Linear(hidden_size + self.rule_dim, 3)
        self.urgency = nn.Linear(hidden_size + self.rule_dim, 3)
        self.topic = nn.Linear(hidden_size + self.rule_dim, 7)
        self.action = nn.Linear(hidden_size + self.rule_dim, 6)

    def forward(self, input_ids, attention_mask, rule_vec):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        x = torch.cat([cls, rule_vec], dim=1) if self.use_rule else cls
        x = self.dropout(x)
        return {
            "sentiment": self.sentiment(x),
            "urgency": self.urgency(x),
            "topic": self.topic(x),
            "action": self.action(x),
        }



# =========================
# Evaluation
# =========================


def evaluate(model, loader):
    model.eval()
    task_names = ["sentiment", "urgency", "topic", "action"]

    preds = {t: [] for t in task_names}
    labels = {t: [] for t in task_names}
    all_metrics = {}

    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            rule = batch["rule_vec"].to(DEVICE)

            # 真值
            gt = {t: batch[t].to(DEVICE) for t in task_names}

            # 模型输出
            out = model(ids, mask, rule)

            for t in task_names:
                preds[t] += torch.argmax(out[t], dim=1).cpu().tolist()
                labels[t] += gt[t].cpu().tolist()

    # ======== 每个任务计算多指标 ========
    for t in task_names:
        f1 = f1_score(labels[t], preds[t], average="macro")
        acc = accuracy_score(labels[t], preds[t])
        prec = precision_score(labels[t], preds[t], average="macro", zero_division=0)
        rec = recall_score(labels[t], preds[t], average="macro", zero_division=0)
        all_metrics[t] = {
            "F1": f1,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec
        }

    # ======== 汇总平均指标 ========
    avg_f1 = sum(v["F1"] for v in all_metrics.values()) / len(all_metrics)
    avg_acc = sum(v["Accuracy"] for v in all_metrics.values()) / len(all_metrics)
    avg_prec = sum(v["Precision"] for v in all_metrics.values()) / len(all_metrics)
    avg_rec = sum(v["Recall"] for v in all_metrics.values()) / len(all_metrics)

    all_metrics["Overall_Avg"] = {
        "F1": avg_f1,
        "Accuracy": avg_acc,
        "Precision": avg_prec,
        "Recall": avg_rec
    }

    # ======== 打印结果 ========
    print("\n📊 Validation Results (Per Task)")
    print("-" * 60)
    for t in task_names:
        print(f"{t:<10} | F1={all_metrics[t]['F1']:.3f} | "
              f"Acc={all_metrics[t]['Accuracy']:.3f} | "
              f"P={all_metrics[t]['Precision']:.3f} | "
              f"R={all_metrics[t]['Recall']:.3f}")
    print("-" * 60)
    print(f"✅ Overall Avg | F1={avg_f1:.3f} | Acc={avg_acc:.3f} | "
          f"P={avg_prec:.3f} | R={avg_rec:.3f}\n")

    return all_metrics



# =========================
# Training Loop
# =========================
def train_model(model, train_loader, val_loader, optimizer, scheduler, loss_fn, cfg):
    wandb.watch(model, log="all")
    train_losses, val_f1s = [], []

    for epoch in range(cfg["epochs"]):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"{cfg['exp_name']} Epoch {epoch+1}/{cfg['epochs']}"):
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            rule = batch["rule_vec"].to(DEVICE)
            s = batch["sentiment"].to(DEVICE)
            u = batch["urgency"].to(DEVICE)
            t = batch["topic"].to(DEVICE)
            a = batch["action"].to(DEVICE)

            out = model(ids, mask, rule)
            loss = (
                0.25 * loss_fn(out["sentiment"], s)
                + 0.25 * loss_fn(out["urgency"], u)
                + 0.35 * loss_fn(out["topic"], t)
                + 0.15 * loss_fn(out["action"], a)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        metrics = evaluate(model, val_loader)   # 调用新版 evaluate()
        train_losses.append(avg_loss)
        val_f1s.append(metrics["Overall_Avg"]["F1"])

        # === WandB Logging ===
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "val_F1_avg": metrics["Overall_Avg"]["F1"],
            "val_Acc_avg": metrics["Overall_Avg"]["Accuracy"],
            "f1_sentiment": metrics["sentiment"]["F1"],
            "f1_urgency": metrics["urgency"]["F1"],
            "f1_topic": metrics["topic"]["F1"],
            "f1_action": metrics["action"]["F1"],
            "acc_sentiment": metrics["sentiment"]["Accuracy"],
            "acc_urgency": metrics["urgency"]["Accuracy"],
            "acc_topic": metrics["topic"]["Accuracy"],
            "acc_action": metrics["action"]["Accuracy"],
        })

        # === 控制台输出 ===
        print(f"✅ Epoch {epoch+1} | Loss={avg_loss:.4f} | "
            f"F1(avg)={metrics['Overall_Avg']['F1']:.3f} | "
            f"Sent={metrics['sentiment']['F1']:.3f} | "
            f"Urg={metrics['urgency']['F1']:.3f} | "
            f"Top={metrics['topic']['F1']:.3f} | "
            f"Act={metrics['action']['F1']:.3f}")


    return train_losses, val_f1s


# =========================
# Experiment Runner
# =========================
def run_experiment(cfg):
    wandb.init(project="FeedbackAnalyzer", name=cfg["exp_name"], config=cfg)

    tokenizer_cls = RobertaTokenizer if "roberta" in cfg["model_name"] else BertTokenizer
    model_cls = RobertaModel if "roberta" in cfg["model_name"] else BertModel

    tokenizer = tokenizer_cls.from_pretrained(cfg["model_name"])
    train_ds = FeedbackDataset("train.csv", tokenizer, rule_based_features)
    val_ds = FeedbackDataset("val.csv", tokenizer, rule_based_features)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)

    # 传入 use_rule 参数 👇
    model = MultiTaskModel(
        cfg["model_name"], model_cls,
        unfreeze_layers=cfg["unfreeze_layers"],
        use_rule=cfg.get("use_rule", True)
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["lr"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * cfg["epochs"]
    )

    loss_fn = nn.CrossEntropyLoss()
    train_loss, val_f1 = train_model(model, train_loader, val_loader, optimizer, scheduler, loss_fn, cfg)

    os.makedirs("saved_models", exist_ok=True)
    torch.save(model.state_dict(), f"saved_models/{cfg['exp_name']}.pt")

    plt.figure(figsize=(7,4))
    plt.plot(train_loss, "o-", label="Train Loss", linewidth=2)
    plt.plot(val_f1, "x--", label="Validation F1", linewidth=2)
    plt.title(f"{cfg['exp_name']} Training Curve", fontsize=13)
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"saved_models/{cfg['exp_name']}_curve.png", dpi=150)
    wandb.log({"curve": wandb.Image(f"saved_models/{cfg['exp_name']}_curve.png")})

    wandb.finish()
    return {"name": cfg["exp_name"], "train": train_loss, "val": val_f1}

import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

# ===============================
# 全局绘图样式设置
# ===============================
mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['axes.grid'] = True
sns.set_style("whitegrid")

# 调色板
palette_model = sns.color_palette("Set2", 5)
palette_ablation = sns.color_palette("Spectral", 6)
palette_rule = sns.color_palette("coolwarm", 2)


def smooth_curve(values, weight=0.6):
    """指数滑动平均平滑曲线"""
    if len(values) < 3:
        return values
    smoothed, last = [], values[0]
    for v in values:
        last = last * weight + (1 - weight) * v
        smoothed.append(last)
    return smoothed


def plot_curves(results, metric="val", title="", ylabel="", filename="plot.png", palette=None):
    """绘制单张图"""
    plt.figure(figsize=(7.5, 4.5))
    palette = palette or sns.color_palette("husl", len(results))

    for i, r in enumerate(results):
        if metric not in r:
            continue
        curve = smooth_curve(r[metric])
        name = r["name"]
        plt.plot(
            curve,
            lw=2.5,
            label=name,
            color=palette[i % len(palette)],
            alpha=0.88,
        )

    plt.title(title, fontsize=14, weight="bold", pad=10)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(
        loc="best",
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=9,
        title="Experiments",
        title_fontsize=10,
    )
    plt.grid(alpha=0.3)
    sns.despine()
    plt.tight_layout()

    os.makedirs("saved_models", exist_ok=True)
    save_path = os.path.join("saved_models", filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📊 Saved: {save_path}")


def summarize_results(all_results):
    """汇总实验结果写入txt"""
    summary_lines = []
    header = f"{'Experiment':<35} | {'Final F1':<8} | {'Best F1':<8} | {'Final Loss':<10} | {'Epochs':<6}"
    summary_lines.append("=" * len(header))
    summary_lines.append(header)
    summary_lines.append("=" * len(header))

    for r in all_results:
        val_curve = r.get("val", [])
        train_curve = r.get("train", [])
        if not val_curve:
            continue
        final_f1 = val_curve[-1]
        best_f1 = max(val_curve)
        final_loss = train_curve[-1] if train_curve else None
        epochs = len(val_curve)
        summary_lines.append(
            f"{r['name']:<35} | {final_f1:<8.4f} | {best_f1:<8.4f} | {final_loss if final_loss else 0:<10.4f} | {epochs:<6}"
        )

    summary_lines.append("=" * len(header))
    summary_text = "\n".join(summary_lines)

    os.makedirs("saved_models", exist_ok=True)
    txt_path = "saved_models/experiment_summary.txt"
    with open(txt_path, "w") as f:
        f.write(summary_text)

    print("\n📄 Experiment Summary:")
    print(summary_text)
    print(f"\n✅ Summary saved to: {txt_path}\n")


def plot_all_experiments(all_results):
    """绘制所有对比图 + 输出结果汇总"""

    # ====== 1️⃣ 全局 Loss 和 F1 总览 ======
    plot_curves(
        all_results, metric="train", title="Training Loss Comparison",
        ylabel="Loss", filename="comparison_loss.png", palette=palette_model
    )
    plot_curves(
        all_results, metric="val", title="Validation F1 Comparison",
        ylabel="Validation F1", filename="comparison_f1.png", palette=palette_model
    )

    # ====== 2️⃣ 模型对比 ======
    model_results = [
        r for r in all_results
        if any(k in r["name"].lower() for k in ["bert_base", "roberta", "small"])
    ]
    if model_results:
        plot_curves(
            model_results, metric="val",
            title="Model Comparison (BERT, RoBERTa, Small)",
            ylabel="Validation F1", filename="compare_models.png", palette=palette_model
        )

    # ====== 3️⃣ 层数消融 ======
    ablation_results = [r for r in all_results if "unfreeze" in r["name"].lower()]
    ablation_results.sort(key=lambda x: int(x["name"].split("unfreeze")[-1]) if "unfreeze" in x["name"] else 0)
    if ablation_results:
        plot_curves(
            ablation_results, metric="val",
            title="Ablation: Unfrozen Transformer Layers (BERT-base)",
            ylabel="Validation F1", filename="compare_unfreeze.png", palette=palette_ablation
        )

    # ====== 4️⃣ Rule-based 对比 ======
    rule_results = [r for r in all_results if "rule" in r["name"].lower()]
    if rule_results:
        plot_curves(
            rule_results, metric="val",
            title="Effect of Rule-based Features (BERT-base)",
            ylabel="Validation F1", filename="compare_rule.png", palette=palette_rule
        )

    # ====== 5️⃣ 汇总结果输出 ======
    summarize_results(all_results)
    print("✅ All comparison plots and summary saved in 'saved_models/'.")


# 调用函数：
# plot_all_experiments(all_results)




# =========================
# Main: Multi-Model + Ablation
# =========================
if __name__ == "__main__":
    # if __name__ == "__main__":
    experiments = [
        # 原多模型实验
        {"model_name": "bert-base-uncased", "exp_name": "bert_base_lr2e-5_uf2", "lr": 2e-5, "epochs": 50, "unfreeze_layers": 2, "use_rule": True},
        {"model_name": "prajjwal1/bert-small", "exp_name": "bert_small_lr3e-5_uf2", "lr": 3e-5, "epochs": 50, "unfreeze_layers": 2, "use_rule": True},
        {"model_name": "roberta-base", "exp_name": "roberta_base_lr2e-5_uf2", "lr": 2e-5, "epochs": 50, "unfreeze_layers": 2, "use_rule": True},

        # ✨ 消融实验：BERT-base 不同解冻层数
        {"model_name": "bert-base-uncased", "exp_name": "bert_base_unfreeze0", "lr": 2e-5, "epochs": 50, "unfreeze_layers": 0, "use_rule": True},
        {"model_name": "bert-base-uncased", "exp_name": "bert_base_unfreeze2", "lr": 2e-5, "epochs": 50, "unfreeze_layers": 2, "use_rule": True},
        {"model_name": "bert-base-uncased", "exp_name": "bert_base_unfreeze4", "lr": 2e-5, "epochs": 50, "unfreeze_layers": 4, "use_rule": True},
        # {"model_name": "bert-base-uncased", "exp_name": "bert_base_unfreeze6", "lr": 2e-5, "epochs": 50, "unfreeze_layers": 6, "use_rule": True},

        # ✨ Rule-based 对比
        {"model_name": "bert-base-uncased", "exp_name": "bert_base_rule_ON", "lr": 2e-5, "epochs": 50, "unfreeze_layers": 4, "use_rule": True},
        {"model_name": "bert-base-uncased", "exp_name": "bert_base_rule_OFF", "lr": 2e-5, "epochs": 50, "unfreeze_layers": 4, "use_rule": False},
    ]



    all_results = []
    for cfg in experiments:
        result = run_experiment(cfg)
        all_results.append(result)

    # # 统一比较曲线
    # plt.figure(figsize=(8,5))
    # for r in all_results:
    #     plt.plot(r["val"], label=r["name"], linewidth=2)
    # plt.title("Model Comparison - Validation F1", fontsize=14)
    # plt.xlabel("Epoch", fontsize=12)
    # plt.ylabel("Validation F1", fontsize=12)
    # plt.legend()
    # plt.grid(True, linestyle="--", alpha=0.7)
    # plt.tight_layout()
    # plt.savefig("saved_models/overall_comparison.png", dpi=200)
    # plt.show()
    # # 提取有/无 rule-based 对比结果
    # rule_on = [r for r in all_results if "rule_ON" in r["name"]][0]
    # rule_off = [r for r in all_results if "rule_OFF" in r["name"]][0]

    # plt.figure(figsize=(7,4))
    # plt.plot(rule_on["val"], 'o-', label='With Rule-based', linewidth=2)
    # plt.plot(rule_off["val"], 'x--', label='Without Rule-based', linewidth=2)
    # plt.title("Effect of Rule-based Features (BERT-Base)", fontsize=14)
    # plt.xlabel("Epoch", fontsize=12)
    # plt.ylabel("Validation F1", fontsize=12)
    # plt.legend()
    # plt.grid(True, linestyle="--", alpha=0.7)
    # plt.tight_layout()
    # plt.savefig("saved_models/rule_based_ablation.png", dpi=200)
    # plt.show()
    plot_all_experiments(all_results)