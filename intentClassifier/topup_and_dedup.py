"""
topup_and_dedup.py
==================
1. Checks GENERATED_CLEAN class counts against targets
2. Generates only for under-target classes (buy_x_get_y, clarification)
3. Validates new examples with VALIDATION_CLASSIFICATION_PROMPT
4. Appends validated examples to GENERATED_CLEAN in classificationData.py
5. Re-runs semantic dedup on the full updated GENERATED_CLEAN
6. Prints final 5-label balance report

Usage:
    conda run -n skailama python intentClassifier/topup_and_dedup.py
"""

import json
import os
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from openai import AzureOpenAI
from intentClassifier.dedup_and_validate import deduplicate, validate
from app.pormpts import (
    DATA_GENERATION_SYSTEM_PROMPT,
    DATA_GENERATION_USER_PROMPT_TEMPLATE,
    CLUSTER_INSTRUCTIONS,
    VALIDATION_CLASSIFICATION_PROMPT,
)

# ── Azure OpenAI ─────────────────────────────────────────────────────────────
client = AzureOpenAI(
    api_key=os.getenv("OPENAI_API_4_KEY"),
    api_version=os.getenv("OPENAI_API_4_VERSION"),
    azure_endpoint=os.getenv("OPENAI_4_BASE_URL"),
)
DEPLOYMENT = os.getenv("OPEN_API_4_ENGINE")

# Targets for GENERATED_CLEAN total count.
# 70% goes to training → need ceil(300 / 0.70) ≈ 430 per class to guarantee
# at least 300 examples in the training split.
TARGETS = {
    "free_gift":       430,
    "buy_x_get_y":     430,
    "tiered_discount": 430,
    "unsupported":     430,
    "clarification":   430,   # bumped to match all other classes → 300 in training
}
VALID_LABELS   = set(TARGETS.keys())
BATCH_GEN      = 60   # examples requested per generation call (ask extra for buffer)
BATCH_VALIDATE = 50
TARGET_FILE    = Path(__file__).parent / "classificationData.py"

# ── Token tracker ─────────────────────────────────────────────────────────────
class TokenTracker:
    def __init__(self):
        self.input = self.output = self.calls = 0

    def record(self, usage):
        self.input  += usage.prompt_tokens
        self.output += usage.completion_tokens
        self.calls  += 1

    def print_summary(self):
        total = self.input + self.output
        cost  = (self.input / 1000) * 0.005 + (self.output / 1000) * 0.015
        print()
        print("=" * 50)
        print("  💰  TOKEN USAGE & COST ESTIMATE")
        print("=" * 50)
        print(f"  API calls        : {self.calls}")
        print(f"  Input tokens     : {self.input:,}")
        print(f"  Output tokens    : {self.output:,}")
        print(f"  Total tokens     : {total:,}")
        print(f"  Estimated cost   : ${cost:.4f} USD  (GPT-4o Azure)")
        print("=" * 50)

tracker = TokenTracker()


# ════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Generation
# ════════════════════════════════════════════════════════════════════════════

def build_user_prompt(label: str, count: int) -> str:
    short_count  = max(5, count // 4)
    medium_count = max(10, count // 2)
    return DATA_GENERATION_USER_PROMPT_TEMPLATE.format(
        label=label,
        count=count,
        cluster_instructions=CLUSTER_INSTRUCTIONS[label],
        short_count=short_count,
        medium_count=medium_count,
    )


def generate_for_label(label: str, needed: int) -> list[tuple[str, str]]:
    """Generate examples via multiple API calls until `needed` count is met."""
    collected: list[tuple[str, str]] = []
    call_num = 0

    while len(collected) < needed:
        remaining = needed - len(collected)
        ask_for   = min(remaining + 20, BATCH_GEN)
        call_num += 1
        print(f"\n  [call #{call_num}] '{label}': have {len(collected)}/{needed}, "
              f"requesting {ask_for} ...", end=" ", flush=True)
        response = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=[
                {"role": "system", "content": DATA_GENERATION_SYSTEM_PROMPT},
                {"role": "user",   "content": build_user_prompt(label, ask_for)},
            ],
            response_format={"type": "json_object"},
            temperature=0.9,
            max_tokens=4096,
        )
        tracker.record(response.usage)
        raw      = response.choices[0].message.content
        examples = json.loads(raw).get("examples", [])
        clean = [
            (ex["text"].strip(), ex["intent"].strip())
            for ex in examples
            if ex.get("text", "").strip() and ex.get("intent", "").strip() == label
        ]
        collected.extend(clean)
        print(f"got {len(clean)}  (total: {len(collected)})")
        time.sleep(0.5)

    print(f"  ✓ Done — {len(collected)} raw examples for '{label}'.")
    return collected


# ════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Validation
# ════════════════════════════════════════════════════════════════════════════

def validate_new_examples(examples: list[tuple[str, str]]) -> list[tuple[str, str]]:
    if not examples:
        return []
    print(f"\n  Validating {len(examples)} newly generated examples ...")
    validated = []
    for i in range(0, len(examples), BATCH_VALIDATE):
        batch = examples[i : i + BATCH_VALIDATE]
        payload = [{"id": i + j, "text": t, "label": l} for j, (t, l) in enumerate(batch)]
        response = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=[
                {"role": "system", "content": VALIDATION_CLASSIFICATION_PROMPT},
                {"role": "user",   "content": json.dumps(payload, ensure_ascii=False)},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        tracker.record(response.usage)
        results = json.loads(response.choices[0].message.content)["results"]

        for (text, orig_label), result in zip(batch, results):
            new_label = result.get("correct_label", orig_label)
            if new_label not in VALID_LABELS:
                new_label = orig_label
            validated.append((text, new_label))

        time.sleep(0.3)

    # Only keep examples that passed validation with the intended label
    passed = [(t, l) for (t, l), (_, vl) in zip(examples, validated) if l == vl]
    corrections = len(examples) - len(passed)
    print(f"  Validation done — {len(passed)} passed, {corrections} rejected/corrected.")
    return passed


# ════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Append to GENERATED_CLEAN
# ════════════════════════════════════════════════════════════════════════════

def escape_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def append_to_generated_clean(new_examples: list[tuple[str, str]]) -> None:
    if not new_examples:
        print("  Nothing to append.")
        return

    shutil.copy2(TARGET_FILE, TARGET_FILE.with_suffix(".py.topup_bak"))
    print(f"\n  📦  Backup saved → classificationData.py.topup_bak")

    source = TARGET_FILE.read_text(encoding="utf-8")

    # Find closing ] of GENERATED_CLEAN and insert new rows before it
    new_rows = "\n".join(
        f'    ("{escape_text(text)}", "{label}"),'
        for text, label in new_examples
    )

    pattern = re.compile(
        r"(^GENERATED_CLEAN\s*=\s*\[.*?)(^\])",
        re.MULTILINE | re.DOTALL,
    )

    def replacer(m):
        return m.group(1) + new_rows + "\n" + m.group(2)

    updated = pattern.sub(replacer, source, count=1)
    TARGET_FILE.write_text(updated, encoding="utf-8")
    print(f"  ✅  Appended {len(new_examples)} new examples to GENERATED_CLEAN.")


# ════════════════════════════════════════════════════════════════════════════
# PHASE 4 — Re-dedup
# ════════════════════════════════════════════════════════════════════════════

def run_dedup_pass() -> None:
    # Reload after write
    if "intentClassifier.classificationData" in sys.modules:
        del sys.modules["intentClassifier.classificationData"]

    from intentClassifier.classificationData import GENERATED_CLEAN
    print(f"\n=== PHASE 4 — Semantic Dedup on {len(GENERATED_CLEAN)} entries ===")

    deduped = deduplicate(GENERATED_CLEAN, threshold=0.90)
    removed = len(GENERATED_CLEAN) - len(deduped)
    print(f"  Removed {removed} near-duplicate(s). {len(deduped)} entries remain.")

    # Validate + print report
    validate(deduped)

    # Rewrite file
    shutil.copy2(TARGET_FILE, TARGET_FILE.with_suffix(".py.postdedup_bak"))
    source = TARGET_FILE.read_text(encoding="utf-8")

    lines = ["GENERATED_CLEAN = [\n"]
    for text, label in deduped:
        lines.append(f'    ("{escape_text(text)}", "{label}"),\n')
    lines.append("]\n")
    new_block = "".join(lines)

    pat = re.compile(r"^GENERATED_CLEAN\s*=\s*\[.*?^\]", re.MULTILINE | re.DOTALL)
    updated = pat.sub(new_block.rstrip("\n"), source, count=1)
    TARGET_FILE.write_text(updated, encoding="utf-8")
    print(f"\n  ✅  classificationData.py rewritten with {len(deduped)} final entries.")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    from intentClassifier.classificationData import GENERATED_CLEAN

    current_dist = Counter(lbl for _, lbl in GENERATED_CLEAN)
    print("=" * 55)
    print("  topup_and_dedup.py")
    print(f"  Current GENERATED_CLEAN: {len(GENERATED_CLEAN)} entries")
    print("=" * 55)
    print("\n  Current class counts vs targets:")
    for label, target in TARGETS.items():
        count  = current_dist.get(label, 0)
        needed = max(0, target - count)
        status = f"✅ OK" if needed == 0 else f"⚠️  short by {needed}"
        print(f"  {label:<22} {count:>4} / {target}  {status}")

    # ── PHASE 1: Generate ────────────────────────────────────────────────────
    all_generated: list[tuple[str, str]] = []
    print("\n=== PHASE 1 — Generation ===")
    for label, target in TARGETS.items():
        needed = target - current_dist.get(label, 0)
        if needed <= 0:
            print(f"  '{label}' already at target — skipping.")
            continue
        raw = generate_for_label(label, needed)
        all_generated.extend(raw)
        time.sleep(0.5)

    if not all_generated:
        print("\nAll classes already at target. Nothing to generate.")
        return

    gen_dist = Counter(lbl for _, lbl in all_generated)
    print(f"\n  Raw generated: {len(all_generated)} total")
    for lbl, cnt in gen_dist.items():
        print(f"    {lbl:<22} {cnt}")

    # ── PHASE 2: Validate ────────────────────────────────────────────────────
    print("\n=== PHASE 2 — Validation ===")
    validated = validate_new_examples(all_generated)

    val_dist = Counter(lbl for _, lbl in validated)
    print(f"\n  Validated examples: {len(validated)} total")
    for lbl, cnt in val_dist.items():
        print(f"    {lbl:<22} {cnt}")

    # ── PHASE 3: Append ──────────────────────────────────────────────────────
    print("\n=== PHASE 3 — Appending to GENERATED_CLEAN ===")
    append_to_generated_clean(validated)

    # ── PHASE 4: Re-dedup ────────────────────────────────────────────────────
    run_dedup_pass()

    # ── Token / cost summary ──────────────────────────────────────────────────
    tracker.print_summary()

    print("\n✅  All done!")
    print("   Next step: python intentClassifier/classificationData.py")
    print("       → regenerates data/train.csv, data/val.csv, data/test.csv\n")


if __name__ == "__main__":
    main()
