"""
03_evaluate.py
────────────────────────────────────────────────────────────────────────────
Loads the best saved model and evaluates it on the held-out test set.
Prints a full sklearn classification report + confusion matrix.
"""

import csv
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix

MODEL_DIR  = Path("model/best")
TEST_PATH  = Path("data/test.csv")
MAX_LEN    = 128
BATCH_SIZE = 32

LABEL_NAMES = ["free_gift", "buy_x_get_y", "tiered_discount", "unsupported"]


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


def main():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"📂 Loading test data from {TEST_PATH}...")
    test_data = load_csv(TEST_PATH)
    print(f"   {len(test_data)} examples")

    print(f"🤖 Loading model from {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()

    dataset = PromotionDataset(test_data, tokenizer)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds   = outputs.logits.argmax(dim=-1).cpu().numpy().tolist()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy().tolist())

    print("\n" + "═" * 60)
    print("  CLASSIFICATION REPORT")
    print("═" * 60)
    print(classification_report(all_labels, all_preds, target_names=LABEL_NAMES))

    print("CONFUSION MATRIX")
    print("─" * 60)
    cm = confusion_matrix(all_labels, all_preds)
    # Pretty print with labels
    col_w = 16
    header = "".join(f"{n:>{col_w}}" for n in LABEL_NAMES)
    print(f"{'':>20}{header}")
    for i, row in enumerate(cm):
        row_str = "".join(f"{v:>{col_w}}" for v in row)
        print(f"{LABEL_NAMES[i]:>20}{row_str}")

    # Per-example failures
    failures = [
        (test_data[i][0], LABEL_NAMES[test_data[i][1]], LABEL_NAMES[all_preds[i]])
        for i in range(len(test_data))
        if test_data[i][1] != all_preds[i]
    ]
    if failures:
        print(f"\n⚠️  Misclassified ({len(failures)} examples):")
        for text, true, pred in failures:
            print(f"  [{true}] → [{pred}]  \"{text}\"")
    else:
        print("\n🎉 Perfect score on test set!")


if __name__ == "__main__":
    main()
