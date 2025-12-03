import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
import re
import gc
import json
import random
import inspect
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

from transformers import (
    AutoModel, AutoTokenizer, DebertaV2Tokenizer,
    get_linear_schedule_with_warmup,
    PretrainedConfig, PreTrainedModel,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# os.environ["TRANSFORMERS_NO_TF"] = "1"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["TRANSFORMERS_NO_FLAX"] = "1"
# os.environ["USE_TF"] = "0"
# os.environ["USE_FLAX"] = "0"

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True

if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=1.0, reduction="mean", weight=None):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.reduction = reduction
        if weight is not None:
            self.register_buffer("weight", weight.clone().detach())
        else:
            self.weight = None

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none", weight=self.weight)
        pt = torch.exp(-ce)
        loss = self.alpha * (1.0 - pt) ** self.gamma * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class CustomBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout, num_layers=2, activation="relu", layer_norm=False):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        act_fn = nn.ReLU if activation.lower() == "relu" else nn.GELU if activation.lower() == "gelu" else None
        if act_fn is None:
            raise ValueError(f"Unsupported activation: {activation}")

        layers, input_dim = [], in_dim
        hidden_dim = hidden_dim if hidden_dim > 0 else in_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(input_dim, hidden_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(act_fn())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class AttentionNeck(nn.Module):
    def __init__(self, hidden_size, num_heads=1, dropout=0.1):
        super().__init__()
        if num_heads < 1:
            raise ValueError("num_heads 必須 >= 1")
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        out = self.dropout(attn_out) + x
        return self.norm(out)

class SentimentConfig(PretrainedConfig):
    model_type = "sentiment-transformer"
    def __init__(
        self,
        model_name="bert-base-multilingual-cased",
        num_labels=3,
        head="mlp",
        dropout=0.1,
        pooling="cls",
        classifier_hidden=0,
        classifier_layers=2,
        classifier_activation="relu",
        classifier_layer_norm=False,
        label_smoothing=0.0,
        class_weights=None,
        neck="none",
        neck_dropout=0.15,
        neck_heads=2,
        use_focal_loss=True,
        focal_gamma=2.0,
        focal_alpha=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.num_labels = num_labels
        self.head = head
        self.dropout = float(dropout)
        self.pooling = pooling
        self.classifier_hidden = int(classifier_hidden)
        self.classifier_layers = int(classifier_layers)
        self.classifier_activation = classifier_activation
        self.classifier_layer_norm = bool(classifier_layer_norm)
        self.label_smoothing = float(label_smoothing)
        self.class_weights = class_weights if class_weights is None else list(class_weights)
        self.neck = str(neck)
        self.neck_dropout = float(neck_dropout)
        self.neck_heads = int(neck_heads)
        self.use_focal_loss = bool(use_focal_loss)
        self.focal_gamma = float(focal_gamma)
        self.focal_alpha = float(focal_alpha)

class SentimentClassifier(PreTrainedModel):
    config_class = SentimentConfig

    def __init__(self, config: SentimentConfig):
        super().__init__(config)
        encoder = AutoModel.from_pretrained(config.model_name)

        self.encoder = encoder
        self.hidden_size = int(self.encoder.config.hidden_size)
        self.norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.pooling = config.pooling
        self._enc_forward_params = set(inspect.signature(self.encoder.forward).parameters.keys())
        self._supports_token_type_ids = "token_type_ids" in self._enc_forward_params

        self.neck = None
        if getattr(config, "neck", "none") and config.neck.lower() not in ("none", "", "null"):
            neck_key = config.neck.lower()
            if neck_key in ("attn", "attention"):
                heads = max(1, getattr(config, "neck_heads", 1))
                self.neck = AttentionNeck(
                    self.hidden_size,
                    num_heads=heads,
                    dropout=getattr(config, "neck_dropout", config.dropout),
                )
            else:
                raise ValueError(f"Unknown neck type: {config.neck}")

        h = config.classifier_hidden if config.classifier_hidden > 0 else self.hidden_size
        self.classifier = CustomBlock(
            in_dim=self.hidden_size,
            hidden_dim=h,
            out_dim=config.num_labels,
            dropout=config.dropout,
            num_layers=max(1, getattr(config, "classifier_layers", 2)),
            activation=getattr(config, "classifier_activation", "relu"),
            layer_norm=bool(getattr(config, "classifier_layer_norm", False)),
        )

        self.label_smoothing = float(config.label_smoothing) if hasattr(config, "label_smoothing") else 0.0
        self.class_weights = (
            torch.as_tensor(config.class_weights, dtype=torch.float)
            if getattr(config, "class_weights", None) is not None else None
        )
        self.loss_fn = None
        self.post_init()

    def _pool(self, last_hidden_state, attention_mask):
        if self.pooling == "cls":
            return last_hidden_state[:, 0, :]
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
        summed = (last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1e-6)
        return summed / denom

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        encoder_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if self._supports_token_type_ids and token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = token_type_ids
        outputs = self.encoder(**encoder_kwargs)
        feat = outputs.last_hidden_state
        if self.neck is not None:
            feat = self.neck(feat, attention_mask)
        pooled = self._pool(feat, attention_mask)
        pooled = self.dropout(self.norm(pooled))
        logits = self.classifier(pooled)

        out = {"logits": logits}
        if labels is not None:
            if self.loss_fn is None:
                weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
                if getattr(self.config, "use_focal_loss", False):
                    self.loss_fn = FocalLoss(
                        gamma=getattr(self.config, "focal_gamma", 2.0),
                        alpha=getattr(self.config, "focal_alpha", 1.0),
                        weight=weight,
                    )
                else:
                    self.loss_fn = nn.CrossEntropyLoss(weight=weight, label_smoothing=self.label_smoothing)
            out["loss"] = self.loss_fn(logits, labels)
        return out

def _basic_clean(text: str) -> str:
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#\w+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

class SentimentDataset(Dataset):
    def __init__(self, csv_path: str, tokenizer: AutoTokenizer, max_length: int):
        df = pd.read_csv(csv_path)
        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError(f"CSV {csv_path} 必須包含 'text' 與 'label' 欄位")
        self.texts = [_basic_clean(str(t)) for t in df["text"].tolist()]
        self.labels = df["label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = int(self.labels[idx])
        enc = self.tokenizer(
            text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        token_type_ids = enc["token_type_ids"].squeeze(0) if "token_type_ids" in enc else torch.zeros_like(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": torch.tensor(label, dtype=torch.long),
        }

def estimate_flops(hidden_size: int, num_layers: int, seq_len: int, batch_size: int) -> float:
    attn_proj = 4 * hidden_size * hidden_size * seq_len
    attn_scores = 2 * seq_len * seq_len * hidden_size
    ffn = 8 * hidden_size * hidden_size * seq_len
    per_layer = attn_proj + attn_scores + ffn
    total = per_layer * num_layers * batch_size
    return total / 1e9

@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    all_y, all_pred = [], []
    with torch.inference_mode():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
            token_type_ids = batch["token_type_ids"].to(DEVICE, non_blocking=True)
            labels = batch["labels"].to(DEVICE, non_blocking=True)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            preds = outputs["logits"].argmax(dim=-1)
            all_y.extend(labels.detach().cpu().tolist())
            all_pred.extend(preds.detach().cpu().tolist())
    acc = accuracy_score(all_y, all_pred)
    return acc, np.array(all_y), np.array(all_pred)

def predict_sentences(ckpt_dir: str, texts, max_length: int = 128, batch_size: int = 32):
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, use_fast=True)
    model = SentimentClassifier.from_pretrained(ckpt_dir).to(DEVICE).eval()
    results = []
    iterator = range(0, len(texts), batch_size)
    if len(texts) >= batch_size:
        iterator = tqdm(iterator, total=(len(texts) + batch_size - 1) // batch_size, desc="Collecting predictions", leave=False)
    for i in iterator:
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
        input_ids = enc["input_ids"].to(DEVICE)
        attention_mask = enc["attention_mask"].to(DEVICE)
        token_type_ids = enc.get("token_type_ids", torch.zeros_like(input_ids)).to(DEVICE)
        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)["logits"]
            probs = torch.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)
        for t, p, pr in zip(batch, preds.detach().cpu().tolist(), probs.detach().cpu().tolist()):
            results.append({"text": t, "pred": int(p), "prob": pr})
    return results

def plot_training_history(train_losses, train_accs, val_accs, out_dir):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2)
    plt.title('Training Loss per Epoch', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'g-o', label='Training Accuracy', linewidth=2)
    plt.plot(epochs, val_accs, 'r-s', label='Validation Accuracy', linewidth=2)
    plt.title('Training & Validation Accuracy', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()
    save_path = os.path.join(out_dir, "training_history.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"訓練趨勢圖已儲存至: {save_path}")

def plot_confusion_matrix_heatmap(cm, classes, out_dir, filename="confusion_matrix.png", title="Confusion Matrix"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, cbar=False)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(title, fontsize=14)
    plt.tight_layout()

    save_path = os.path.join(out_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"混淆矩陣圖已儲存至: {save_path}")

def train(
    model_name,
    train_csv,
    val_csv,
    test_csv,
    out_dir,
    epochs,
    batch_size,
    max_length,
    head="mlp",
    classifier_hidden=0,
    classifier_layers=2,
    classifier_activation="gelu",
    classifier_layer_norm=False,
    neck="none",
    neck_heads=1,
    neck_dropout=0.2,
    pooling="cls",
    dropout=0.2,
    lr_encoder=3e-5,
    lr_head=2e-4,
    warmup_ratio=0.1,
    seed=42,
    param_limit=500_000_000,
    label_smoothing=0.05,
    class_weight="none",
    use_focal_loss=False,
    focal_gamma=2.0,
    focal_alpha=1.0,
):
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception as e_fast:
        print(f"Fast tokenizer failed ({e_fast}). Trying slow tokenizer.")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        except Exception as e_slow:
            if "deberta-v3" in model_name.lower():
                print(f"AutoTokenizer slow also failed ({e_slow}). Using DebertaV2Tokenizer slow as fallback.")
                tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
            else:
                raise
    ds_train = SentimentDataset(train_csv, tokenizer, max_length)
    ds_val = SentimentDataset(val_csv, tokenizer, max_length)
    ds_test = SentimentDataset(test_csv, tokenizer, max_length)
    num_workers = 0 if DEVICE.type == "mps" else min(4, os.cpu_count() or 1)
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())

    class_weights = None
    if class_weight == "balanced":
        from collections import Counter
        label_counts = Counter(ds_train.labels)
        classes = sorted(label_counts.keys())
        num_classes = len(classes)
        N = len(ds_train.labels)
        weights = {c: (N / (num_classes * max(1, label_counts[c]))) for c in classes}
        max_label = max(classes)
        class_weights = [weights.get(i, 0.0) for i in range(max_label + 1)]

    config = SentimentConfig(
        model_name=model_name,
        num_labels=3,
        head=head,
        dropout=dropout,
        pooling=pooling,
        classifier_hidden=classifier_hidden,
        classifier_layers=classifier_layers,
        classifier_activation=classifier_activation,
        classifier_layer_norm=classifier_layer_norm,
        label_smoothing=label_smoothing,
        class_weights=class_weights,
        neck=neck,
        neck_heads=neck_heads,
        neck_dropout=neck_dropout,
        use_focal_loss=use_focal_loss,
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha,
    )
    model = SentimentClassifier(config).to(DEVICE)

    params_total = int(sum(p.numel() for p in model.parameters()))
    params_trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    if params_total > param_limit:
        raise ValueError(
            f"Model parameters ({params_total}) exceed limit ({param_limit}). "
            f"請選擇較小的 backbone 或降低 head 規模。"
        )

    no_decay = ["bias", "LayerNorm.weight"]
    enc_params = list(model.encoder.named_parameters())
    head_params = list(model.classifier.named_parameters())
    neck_params = list(model.neck.named_parameters()) if getattr(model, "neck", None) is not None else []

    def group(params, lr):
        return [
            {"params": [p for n, p in params if not any(nd in n for nd in no_decay) and p.requires_grad], "weight_decay": 0.01, "lr": lr},
            {"params": [p for n, p in params if any(nd in n for nd in no_decay) and p.requires_grad], "weight_decay": 0.0, "lr": lr},
        ]
    optimizer_groups = group(enc_params, lr_encoder)
    if neck_params:
        optimizer_groups += group(neck_params, lr_head)
    optimizer_groups += group(head_params, lr_head)
    optimizer = optim.AdamW(optimizer_groups, betas=(0.9, 0.999), eps=1e-8)

    total_steps = max(1, len(dl_train) * epochs)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    ckpt_dir = os.path.join(out_dir, "checkpoint")
    os.makedirs(ckpt_dir, exist_ok=True)
    tokenizer.save_pretrained(ckpt_dir)
    num_layers = getattr(model.encoder.config, "num_hidden_layers", 12)
    print(f"Estimated FLOPs/step (GFLOPs): {estimate_flops(model.hidden_size, num_layers, max_length, batch_size):.2f}")
    print(f"Total params: {params_total}, trainable params: {params_trainable}")

    best_val = -1.0
    epoch_train_loss, epoch_val_acc, epoch_train_acc = [], [], []

    for epoch in range(1, epochs + 1):
        model.train()
        running, train_correct, train_seen = 0.0, 0, 0
        pbar = tqdm(dl_train, desc=f"Epoch {epoch}/{epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
            token_type_ids = batch["token_type_ids"].to(DEVICE, non_blocking=True)
            labels = batch["labels"].to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            try:
                autocast_dtype = torch.float16 if DEVICE.type != "cpu" else torch.bfloat16
                with torch.autocast(device_type=DEVICE.type, dtype=autocast_dtype):
                    out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
                    loss = out["loss"]
            except Exception:
                out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
                loss = out["loss"]

            preds = out["logits"].argmax(dim=-1)
            train_correct += (preds == labels).sum().item()
            train_seen += labels.size(0)
            train_acc_running = train_correct / max(1, train_seen)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += float(loss.item())
            pbar.set_postfix(loss=f"{running/(pbar.n or 1):.4f}", acc=f"{train_acc_running:.4f}")

        epoch_loss = running / max(1, len(dl_train))
        epoch_train_loss.append(epoch_loss)
        epoch_train_acc.append(train_correct / max(1, train_seen))
        val_acc, _, _ = evaluate(model, dl_val)
        epoch_val_acc.append(val_acc)
        print(f"Epoch {epoch}: Train Acc = {epoch_train_acc[-1]:.4f} | Val Acc = {val_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            model.save_pretrained(ckpt_dir, safe_serialization=True)

    plot_training_history(epoch_train_loss, epoch_train_acc, epoch_val_acc, out_dir)

    best = SentimentClassifier.from_pretrained(ckpt_dir).to(DEVICE)

    def eval_and_save(split, dl, title_suffix):
        acc, y, yhat = evaluate(best, dl)
        cm = confusion_matrix(y, yhat, labels=[0, 1, 2])

        pd.DataFrame(cm, index=[0, 1, 2], columns=[0, 1, 2]).to_csv(os.path.join(ckpt_dir, f"{split}_cm.csv"))

        rpt = classification_report(y, yhat, digits=4, labels=[0, 1, 2])
        with open(os.path.join(ckpt_dir, f"{split}_report.txt"), "w", encoding="utf-8") as f:
            f.write(rpt)

        plot_confusion_matrix_heatmap(
            cm,
            classes=[0, 1, 2],
            out_dir=out_dir,
            filename=f"{split}_cm.png",
            title=f"{title_suffix} Confusion Matrix"
        )
        return float(acc), cm

    train_acc, train_cm = eval_and_save("train", dl_train, "Training")
    val_acc,   val_cm   = eval_and_save("val",   dl_val,   "Validation")
    test_acc,  test_cm  = eval_and_save("test",  dl_test,  "Test")

    summary = {
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "params_trainable": int(sum(p.numel() for p in best.parameters() if p.requires_grad)),
        "params_total": int(sum(p.numel() for p in best.parameters())),
        "param_limit": int(param_limit),
        "model_name": model_name,
        "neck": neck,
        "head": head,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    try:
        test_df = pd.read_csv(test_csv)
        samples = test_df["text"].head(5).tolist()
        preds = predict_sentences(ckpt_dir, samples, max_length=max_length)
        for r in preds:
            print(f"[pred={r['pred']}] {r['text'][:80]} ... prob={np.round(r['prob'],3)}")
    except Exception:
        pass

    try:
        best.to("cpu"); model.to("cpu")
    except Exception:
        pass
    del best, model, tokenizer, optimizer, scheduler, dl_train, dl_val, dl_test
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main():
    config = {
        "train_csv": "./dataset/train.csv",
        "test_csv": "./dataset/test.csv",
        "out_dir": "./saved_models/",
        "model_name": "microsoft/deberta-v3-large",
        "max_length": 90,
        "batch_size": 32,
        "epochs": 10,
        "classifier_hidden": 128,
        "classifier_layers": 2,
        "classifier_activation": "relu",
        "classifier_layer_norm": True,
        "neck": "none",
        "neck_heads": 0,
        "neck_dropout": 0.0,
        "pooling": "cls",
        "dropout": 0.3,
        "label_smoothing": 0.07,
        "class_weight": "none",
        "lr_encoder": 3e-5,
        "lr_head": 2e-4,
        "warmup_ratio": 0.13,
        "seed": 42,
        "param_limit": 500_000_000,
        "val_csv": "",
        "val_ratio": 0.05,
    }

    dataset_dir = os.path.dirname(config["train_csv"]) if os.path.dirname(config["train_csv"]) else "./dataset"
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(config["out_dir"], exist_ok=True)
    set_seed(config["seed"])
    train_source = config["train_csv"]
    test_source = config["test_csv"]

    if not os.path.exists(train_source):
        candidate = os.path.join(dataset_dir, "dataset.csv")
        if os.path.exists(candidate):
            full = pd.read_csv(candidate)
            if "text" not in full.columns or "label" not in full.columns:
                raise ValueError(f"{candidate} 必須包含 'text' 與 'label' 欄位")
            train_df, test_df = train_test_split(
                full, test_size=0.2, random_state=config["seed"], stratify=full["label"],
            )
            train_df.to_csv(train_source, index=False)
            test_df.to_csv(test_source, index=False)
            print(f"從 dataset.csv 產生 train/test: train={len(train_df)}, test={len(test_df)}")
        else:
            raise FileNotFoundError(f"找不到訓練資料 {train_source}，亦無 dataset.csv 可切分。")
    if not os.path.exists(test_source):
        raise FileNotFoundError(f"找不到測試資料 {test_source}")

    val_path = config["val_csv"].strip()
    auto_split_dir = os.path.join(config["out_dir"], "auto_val_split")
    os.makedirs(auto_split_dir, exist_ok=True)
    if val_path:
        if not os.path.exists(val_path):
            raise FileNotFoundError(f"指定的驗證檔案不存在: {val_path}")
        train_used = train_source
        val_used = val_path
    else:
        df_train = pd.read_csv(train_source)
        if "text" not in df_train.columns or "label" not in df_train.columns:
            raise ValueError(f"{train_source} 必須包含 'text' 與 'label' 欄位")
        if not (0 < config["val_ratio"] < 1):
            raise ValueError("val_ratio 必須介於 0 與 1 之間")
        stratify_col = df_train["label"] if df_train["label"].nunique() > 1 else None
        train_df, val_df = train_test_split(
            df_train, test_size=config["val_ratio"], random_state=config["seed"], stratify=stratify_col,
        )
        train_used = os.path.join(auto_split_dir, "train.csv")
        val_used = os.path.join(auto_split_dir, "val.csv")
        train_df.to_csv(train_used, index=False)
        val_df.to_csv(val_used, index=False)
        print(f"自動從 {train_source} 切分訓練/驗證: train={len(train_df)}, val={len(val_df)} (ratio={config['val_ratio']})")

    print("開始訓練單一模型，設定如下：")
    for k, v in config.items():
        if k not in {"train_csv", "test_csv", "out_dir", "val_csv"}:
            print(f"  {k}: {v}")

    train(
        model_name=config["model_name"],
        train_csv=train_used,
        val_csv=val_used,
        test_csv=test_source,
        out_dir=config["out_dir"],
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        max_length=config["max_length"],
        classifier_hidden=config["classifier_hidden"],
        classifier_layers=config["classifier_layers"],
        classifier_activation=config["classifier_activation"],
        classifier_layer_norm=config["classifier_layer_norm"],
        neck=config["neck"],
        neck_heads=config["neck_heads"],
        neck_dropout=config["neck_dropout"],
        pooling=config["pooling"],
        dropout=config["dropout"],
        lr_encoder=config["lr_encoder"],
        lr_head=config["lr_head"],
        warmup_ratio=config["warmup_ratio"],
        seed=config["seed"],
        param_limit=config["param_limit"],
        label_smoothing=config["label_smoothing"],
        class_weight=config["class_weight"],
    )

if __name__ == "__main__":
    main()
