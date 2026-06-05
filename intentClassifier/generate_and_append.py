"""
generate_and_append.py
======================
Full pipeline:
  1. Generate fresh training examples for each class until every class has
     >= TARGET_PER_CLASS examples (existing + newly generated).
  2. Validate all newly generated examples using the same Azure OpenAI
     validation logic as validate_generated_data.py.
  3. Append only the freshly validated examples into ALL_EXAMPLES inside
     generated_data_validated.py (no duplicate write of the existing rows).

Usage:
    python intentClassifier/generate_and_append.py
"""

import json
import os
import sys
import time
from collections import Counter

from dotenv import load_dotenv
from openai import AzureOpenAI

# ── path setup ──────────────────────────────────────────────────────────────
load_dotenv()
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from intentClassifier.generated_data_validated import ALL_EXAMPLES as EXISTING_EXAMPLES
from app.pormpts import (
    DATA_GENERATION_SYSTEM_PROMPT,
    DATA_GENERATION_USER_PROMPT_TEMPLATE,
    CLUSTER_INSTRUCTIONS,
    VALIDATION_CLASSIFICATION_PROMPT,
)

# ── Azure OpenAI client (same credentials as validate_generated_data.py) ────
client = AzureOpenAI(
    api_key=os.getenv("OPENAI_API_4_KEY"),
    api_version=os.getenv("OPENAI_API_4_VERSION"),
    azure_endpoint=os.getenv("OPENAI_4_BASE_URL"),
)
DEPLOYMENT = os.getenv("OPEN_API_4_ENGINE")

# ── constants ────────────────────────────────────────────────────────────────
TARGET_PER_CLASS = 400
BATCH_GENERATE = 120   # examples requested per API generation call
VALID_LABELS = {"free_gift", "buy_x_get_y", "tiered_discount", "unsupported"}
VALIDATION_BATCH_SIZE = 50
LABELS = ["free_gift", "buy_x_get_y", "tiered_discount", "unsupported"]


# ════════════════════════════════════════════════════════════════════════════
# GENERATION
# ════════════════════════════════════════════════════════════════════════════

def build_user_prompt(label: str, count: int) -> str:
    short_count = max(15, count // 5)
    medium_count = max(40, count // 3)
    return DATA_GENERATION_USER_PROMPT_TEMPLATE.format(
        label=label,
        count=count,
        cluster_instructions=CLUSTER_INSTRUCTIONS[label],
        short_count=short_count,
        medium_count=medium_count,
    )


def generate_batch(label: str, count: int, call_num: int) -> list[tuple[str, str]]:
    """Call Azure OpenAI to generate `count` examples for `label`."""
    prompt = build_user_prompt(label, count)
    print(f"  [Gen call #{call_num}] Requesting {count} × '{label}' ...", end=" ", flush=True)
    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": DATA_GENERATION_SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.9,
        max_tokens=4096,
    )
    raw = response.choices[0].message.content
    parsed = json.loads(raw)
    examples = parsed.get("examples", [])
    print(f"received {len(examples)} raw.")

    clean = []
    for ex in examples:
        text   = ex.get("text", "").strip()
        intent = ex.get("intent", "").strip()
        if text and intent == label:          # strict label match
            clean.append((text, intent))
    return clean


def generate_for_all_classes(existing_counts: Counter) -> list[tuple[str, str]]:
    """
    Generate batches until every class reaches TARGET_PER_CLASS.
    Returns the combined raw (unvalidated) generated examples.
    """
    all_generated: list[tuple[str, str]] = []
    running_counts = dict(existing_counts)     # track total (existing + generated so far)
    call_num = 0

    print("\n=== PHASE 1 – Generation ===")
    for label in LABELS:
        still_needed = TARGET_PER_CLASS - running_counts.get(label, 0)
        if still_needed <= 0:
            print(f"  '{label}' already at {running_counts[label]} — skipping generation.")
            continue

        label_generated: list[tuple[str, str]] = []
        while (TARGET_PER_CLASS - running_counts.get(label, 0) - len(label_generated)) > 0:
            remaining = TARGET_PER_CLASS - running_counts.get(label, 0) - len(label_generated)
            ask_for = min(BATCH_GENERATE, remaining + 30)  # ask a bit extra to account for refusals
            call_num += 1
            try:
                batch = generate_batch(label, ask_for, call_num)
                label_generated.extend(batch)
            except Exception as exc:
                print(f"  ERROR generating '{label}': {exc}")
                time.sleep(2)
                break
            time.sleep(0.5)

        print(
            f"  Generated {len(label_generated)} raw examples for '{label}' "
            f"(need {still_needed}, have {running_counts.get(label, 0)} existing)."
        )
        all_generated.extend(label_generated)

    raw_counts = Counter(lbl for _, lbl in all_generated)
    print("\n=== Raw generated counts ===")
    for lbl in LABELS:
        print(f"  {lbl:<22} {raw_counts.get(lbl, 0)}")
    print(f"  {'TOTAL':<22} {len(all_generated)}")
    return all_generated


# ════════════════════════════════════════════════════════════════════════════
# VALIDATION  (mirrors validate_generated_data.py logic)
# ════════════════════════════════════════════════════════════════════════════

def validate_batch(batch: list[tuple[str, str]], offset: int) -> list[dict]:
    payload = [
        {"id": offset + i, "text": text, "label": label}
        for i, (text, label) in enumerate(batch)
    ]
    user_msg = json.dumps(payload, ensure_ascii=False)
    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": VALIDATION_CLASSIFICATION_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    raw = response.choices[0].message.content
    return json.loads(raw)["results"]


def run_validation(examples: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """
    Validate all examples in batches.
    Returns a list of (text, correct_label) keeping only items whose
    correct_label is a valid intent (i.e. the validator did not reject them).
    """
    print(f"\n=== PHASE 2 – Validation ({len(examples)} examples) ===")
    offset = 0
    corrected: list[tuple[str, str]] = []
    mismatches = 0

    for i in range(0, len(examples), VALIDATION_BATCH_SIZE):
        batch = examples[i: i + VALIDATION_BATCH_SIZE]
        print(f"  Validating batch {offset}–{offset + len(batch) - 1} ...", end=" ", flush=True)
        try:
            results = validate_batch(batch, offset)
            for (text, orig_label), result in zip(batch, results):
                correct_label = result.get("correct_label", orig_label)
                if correct_label not in VALID_LABELS:
                    correct_label = orig_label
                if not result.get("is_correct", True):
                    mismatches += 1
                corrected.append((text, correct_label))
            print(f"done ({len(results)} results).")
        except Exception as exc:
            print(f"ERROR: {exc}. Keeping original labels for this batch.")
            corrected.extend(batch)   # keep originals on API error
        offset += len(batch)
        time.sleep(0.3)

    val_counts = Counter(lbl for _, lbl in corrected)
    print(f"\n  Validator corrected {mismatches} label(s).")
    print("=== Validated counts ===")
    for lbl in LABELS:
        print(f"  {lbl:<22} {val_counts.get(lbl, 0)}")
    print(f"  {'TOTAL':<22} {len(corrected)}")
    return corrected


# ════════════════════════════════════════════════════════════════════════════
# APPEND to generated_data_validated.py
# ════════════════════════════════════════════════════════════════════════════

def escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def append_to_validated_file(new_examples: list[tuple[str, str]], out_path: str):
    """
    Reads the existing file, removes the closing `]` of ALL_EXAMPLES,
    appends the new rows, then re-closes the list.
    """
    print(f"\n=== PHASE 3 – Appending {len(new_examples)} validated examples ===")

    with open(out_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Find the end of ALL_EXAMPLES list
    # The list ends with `]\n\nGENERATED_CLEAN_VALIDATED` or `]\n` at top-level
    # We look for the closing bracket of ALL_EXAMPLES specifically.
    marker = "\n]\n\nGENERATED_CLEAN_VALIDATED"
    alt_marker = "\n]\n"  # fallback if GENERATED_CLEAN_VALIDATED section not present

    if marker in content:
        split_point = content.index(marker)
        before = content[:split_point]
        after  = content[split_point:]
    elif alt_marker in content:
        split_point = content.index(alt_marker)
        before = content[:split_point]
        after  = content[split_point:]
    else:
        # File ends with just `]`
        before = content.rstrip().rstrip("]").rstrip()
        after  = "\n]\n"

    new_rows = "\n".join(
        f'    ("{escape(text)}", "{label}"),'
        for text, label in new_examples
    )

    updated = before + "\n" + new_rows + after

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(updated)

    print(f"  Appended {len(new_examples)} rows to ALL_EXAMPLES in {out_path}")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    existing_counts = Counter(lbl for _, lbl in EXISTING_EXAMPLES)
    print("=== Existing ALL_EXAMPLES counts ===")
    for lbl in LABELS:
        print(f"  {lbl:<22} {existing_counts.get(lbl, 0)}")
    print(f"  {'TOTAL':<22} {len(EXISTING_EXAMPLES)}")
    print(f"\nTarget per class: {TARGET_PER_CLASS}")

    # Phase 1 – Generate
    raw_generated = generate_for_all_classes(existing_counts)

    if not raw_generated:
        print("\nNo new examples generated. Exiting.")
        return

    # Phase 2 – Validate
    validated = run_validation(raw_generated)

    # Final class counts summary
    final_counts = Counter(lbl for _, lbl in EXISTING_EXAMPLES)
    for lbl, cnt in Counter(lbl for _, lbl in validated).items():
        final_counts[lbl] += cnt

    print("\n=== Projected final counts after append ===")
    for lbl in LABELS:
        status = "✓" if final_counts.get(lbl, 0) >= TARGET_PER_CLASS else "✗ (below target)"
        print(f"  {lbl:<22} {final_counts.get(lbl, 0)}  {status}")

    # Phase 3 – Append
    out_path = os.path.join(os.path.dirname(__file__), "generated_data_validated.py")
    append_to_validated_file(validated, out_path)

    print("\n✅ Done! generated_data_validated.py has been updated.")
    print(f"   Total new examples appended: {len(validated)}")


if __name__ == "__main__":
    main()
