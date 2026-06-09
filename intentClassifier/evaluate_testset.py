import csv
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = Path("intentClassifier/model/best")
CSV_PATH = Path("intentClassifier/TestSetLabeledAndValidated.csv")
MAX_LEN = 128

ID2LABEL = {0: "free_gift", 1: "buy_x_get_y", 2: "tiered_discount", 3: "unsupported", 4: "clarification"}

def main():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # Find where the model files actually exist
    possible_dirs = [Path("model/best"), Path("intentClassifier/model/best")]
    selected_dir = None
    for d in possible_dirs:
        if d.exists() and (d / "model.safetensors").exists() or (d / "pytorch_model.bin").exists():
            selected_dir = d
            break
            
    if selected_dir is None:
        # Fallback to the first existing directory or raise error
        for d in possible_dirs:
            if d.exists():
                selected_dir = d
                break
                
    if selected_dir is None or not (selected_dir / "config.json").exists():
        print(f"Error: Model directory not found. Ensure train.py ran successfully.")
        return

    print(f"Loading model and tokenizer from {selected_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(selected_dir)
    model = AutoModelForSequenceClassification.from_pretrained(selected_dir)
    model.to(device)
    model.eval()

    # Find where the CSV file actually exists
    global CSV_PATH
    if not CSV_PATH.exists():
        CSV_PATH = Path("TestSetLabeledAndValidated.csv")

    print(f"Loading test cases from {CSV_PATH}...")
    test_cases = []
    with open(CSV_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_cases.append({
                "id": row["id"],
                "requirement": row["requirement"],
                "rule_label": row["label"],
                "validated_label": row["validated_label"],
            })

    print(f"Evaluating {len(test_cases)} cases...")
    rule_correct = 0
    model_correct = 0
    
    print("\n" + "="*110)
    print(f"{'ID':<4} {'Requirement':<45} {'Rule Label':<16} {'Model Pred':<16} {'Validated (True)':<16} {'Status'}")
    print("="*110)

    for tc in test_cases:
        req = tc["requirement"]
        rule_lbl = tc["rule_label"]
        true_lbl = tc["validated_label"]

        # Run model prediction
        enc = tokenizer(
            req,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            ).logits
        
        pred_idx = int(logits.argmax(dim=-1).cpu().item())
        pred_lbl = ID2LABEL[pred_idx]

        # Check correctness
        is_rule_ok = (rule_lbl == true_lbl)
        is_model_ok = (pred_lbl == true_lbl)

        if is_rule_ok:
            rule_correct += 1
        if is_model_ok:
            model_correct += 1

        # Truncate requirement for display
        req_disp = req[:42] + "..." if len(req) > 42 else req
        
        status_str = "✓ Model OK" if is_model_ok else "✗ Model MISMATCH"
        if not is_rule_ok:
            status_str += " (Rule was wrong)"
            
        print(f"{tc['id']:<4} {req_disp:<45} {rule_lbl:<16} {pred_lbl:<16} {true_lbl:<16} {status_str}")

    total = len(test_cases)
    rule_acc = (rule_correct / total) * 100 if total else 0
    model_acc = (model_correct / total) * 100 if total else 0

    print("="*110)
    print(f"Rule-Based Method Accuracy:  {rule_acc:.2f}% ({rule_correct}/{total})")
    print(f"Model-Based Method Accuracy: {model_acc:.2f}% ({model_correct}/{total})")
    print("="*110)

if __name__ == "__main__":
    main()
