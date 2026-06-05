

import os
import csv
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME   = "microsoft/deberta-v3-small"
MAX_LEN      = 128         
BATCH_SIZE   = 16
EPOCHS       = 12
LR           = 2e-5
WARMUP_RATIO = 0.1
PATIENCE     = 3            #
SAVE_DIR     = Path("model/best")
DATA_DIR     = Path("data")

ID2LABEL = {0: "free_gift", 1: "buy_x_get_y", 2: "tiered_discount", 3: "unsupported"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

# ── Dataset ───────────────────────────────────────────────────────────────────
def load_csv(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append((row["text"], int(row["label"])))
    return rows


class PromotionDataset(Dataset):
    def __init__(self, examples, tokenizer):
        self.examples  = examples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text, label = self.examples[idx]
        enc = self.tokenizer(
            text,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(label, dtype=torch.long),
        }


def accuracy(preds, labels):
    return (np.array(preds) == np.array(labels)).mean()


def run_epoch(model, loader, optimizer=None, scheduler=None, class_weights=None, device="cpu"):
    """Single train or eval pass. Pass optimizer=None for eval mode."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    loss_fn = CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)

    total_loss, all_preds, all_labels = 0.0, [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits  = outputs.logits

            loss = loss_fn(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=-1).cpu().numpy().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader)
    acc      = accuracy(all_preds, all_labels)
    return avg_loss, acc


def main():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    print("Loading data")
    train_data = load_csv(DATA_DIR / "train.csv")
    val_data   = load_csv(DATA_DIR / "val.csv")
    print(f"   Train: {len(train_data)}  Val: {len(val_data)}")
    print(f"   Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = PromotionDataset(train_data, tokenizer)
    val_ds   = PromotionDataset(val_data,   tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    # ── Class weights (handle imbalance) ───────────────────────────────────
    all_train_labels = [label for _, label in train_data]
    classes          = sorted(set(all_train_labels))
    weights          = compute_class_weight("balanced", classes=np.array(classes), y=all_train_labels)
    class_weights    = torch.tensor(weights, dtype=torch.float)
    print(f"Class weights: { {ID2LABEL[i]: round(w, 3) for i, w in enumerate(weights)} }")

    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=4,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_val_loss = float("inf")
    patience_counter = 0

    print("\n Starting training \n")
    print(f"{'Epoch':<7} {'Train Loss':<13} {'Train Acc':<12} {'Val Loss':<11} {'Val Acc':<10} {'Status'}")
    print("─" * 68)

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(
            model, train_loader, optimizer, scheduler, class_weights, device
        )
        vl_loss, vl_acc = run_epoch(
            model, val_loader, device=device
        )

        status = ""
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            patience_counter = 0
            SAVE_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(SAVE_DIR)
            tokenizer.save_pretrained(SAVE_DIR)
            status = "saved"
        else:
            patience_counter += 1
            status = f"patience {patience_counter}/{PATIENCE}"

        print(
            f"{epoch:<7} {tr_loss:<13.4f} {tr_acc:<12.4f} {vl_loss:<11.4f} {vl_acc:<10.4f} {status}"
        )

        if patience_counter >= PATIENCE:
            print(f"\n Early stopping at epoch {epoch}")
            break

    print(f"\n Training complete.  val loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {SAVE_DIR}/")


if __name__ == "__main__":
    main()
