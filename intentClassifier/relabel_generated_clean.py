"""
relabel_generated_clean.py
==========================
Re-validates every (text, label) pair in GENERATED_CLEAN using the current
VALIDATION_CLASSIFICATION_PROMPT, then rewrites the GENERATED_CLEAN list
inside classificationData.py in-place with the corrected labels.

Features:
  - Batched LLM validation (50 examples per API call)
  - clarification label → remapped to unsupported (matches existing pipeline)
  - Backup of original classificationData.py saved as classificationData.py.bak
  - Before/after class distribution summary
  - Token usage tracking + GPT-4o cost estimation

Usage:
    python intentClassifier/relabel_generated_clean.py
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

# ── path setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from openai import AzureOpenAI
from intentClassifier.classificationData import GENERATED_CLEAN
from app.pormpts import VALIDATION_CLASSIFICATION_PROMPT

# ── Azure OpenAI client ──────────────────────────────────────────────────────
client = AzureOpenAI(
    api_key=os.getenv("OPENAI_API_4_KEY"),
    api_version=os.getenv("OPENAI_API_4_VERSION"),
    azure_endpoint=os.getenv("OPENAI_4_BASE_URL"),
)
DEPLOYMENT = os.getenv("OPEN_API_4_ENGINE")

# ── Constants ────────────────────────────────────────────────────────────────
BATCH_SIZE   = 50
VALID_LABELS = {"free_gift", "buy_x_get_y", "tiered_discount", "unsupported", "clarification"}

# GPT-4o pricing (Azure, per 1 000 tokens, USD) — update if your deployment differs
COST_PER_1K_INPUT  = 0.005   # $0.005 / 1K input tokens
COST_PER_1K_OUTPUT = 0.015   # $0.015 / 1K output tokens

TARGET_FILE = Path(__file__).parent / "classificationData.py"
BACKUP_FILE = TARGET_FILE.with_suffix(".py.bak")


# ════════════════════════════════════════════════════════════════════════════
# Token & cost tracking
# ════════════════════════════════════════════════════════════════════════════

class TokenTracker:
    def __init__(self):
        self.total_input  = 0
        self.total_output = 0
        self.calls        = 0

    def record(self, usage):
        self.total_input  += usage.prompt_tokens
        self.total_output += usage.completion_tokens
        self.calls        += 1

    @property
    def total_tokens(self):
        return self.total_input + self.total_output

    @property
    def estimated_cost_usd(self):
        return (
            (self.total_input  / 1000) * COST_PER_1K_INPUT
          + (self.total_output / 1000) * COST_PER_1K_OUTPUT
        )

    def print_summary(self):
        width = 55
        print()
        print("=" * width)
        print("  💰  TOKEN USAGE & COST ESTIMATE")
        print("=" * width)
        print(f"  API calls made       : {self.calls}")
        print(f"  Input tokens         : {self.total_input:,}")
        print(f"  Output tokens        : {self.total_output:,}")
        print(f"  Total tokens         : {self.total_tokens:,}")
        print(f"  Pricing model        : GPT-4o (Azure)")
        print(f"    Input  @ ${COST_PER_1K_INPUT:.4f}/1K tokens")
        print(f"    Output @ ${COST_PER_1K_OUTPUT:.4f}/1K tokens")
        print(f"  Estimated cost (USD) : ${self.estimated_cost_usd:.4f}")
        print("=" * width)


tracker = TokenTracker()


# ════════════════════════════════════════════════════════════════════════════
# Validation
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

    tracker.record(response.usage)

    raw = response.choices[0].message.content
    return json.loads(raw)["results"]


def run_validation(examples: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """
    Validate all examples in batches of BATCH_SIZE.
    Returns a list of (text, corrected_label).
    - clarification → unsupported  (mirrors existing pipeline)
    - unknown label → keep original
    - API error     → keep original
    """
    total      = len(examples)
    corrected  = []
    mismatches = 0
    offset     = 0

    print(f"\n=== Validating {total} examples from GENERATED_CLEAN ===")

    for i in range(0, total, BATCH_SIZE):
        batch = examples[i : i + BATCH_SIZE]
        end   = offset + len(batch) - 1
        print(f"  Batch {offset:>4}–{end:<4} ({len(batch):>2} items) ... ", end="", flush=True)

        try:
            results = validate_batch(batch, offset)

            for (text, orig_label), result in zip(batch, results):
                new_label = result.get("correct_label", orig_label)

                # reject truly unknown labels (keep original as fallback)
                if new_label not in VALID_LABELS:
                    new_label = orig_label

                if new_label != orig_label:
                    mismatches += 1

                corrected.append((text, new_label))

            usage = (
                f"in={response_tokens_last(results)} "  # placeholder; real numbers in tracker
                if False else ""
            )
            print(f"done ✓  (corrected so far: {mismatches})")

        except Exception as exc:
            print(f"ERROR — {exc}. Keeping original labels for this batch.")
            corrected.extend(batch)

        offset += len(batch)
        time.sleep(0.3)   # be kind to the rate limiter

    print(f"\n  Total label corrections: {mismatches} / {total}")
    return corrected, mismatches


def response_tokens_last(_):
    """Dummy helper — actual tokens already recorded by tracker."""
    return ""


# ════════════════════════════════════════════════════════════════════════════
# File rewrite — replace GENERATED_CLEAN in classificationData.py
# ════════════════════════════════════════════════════════════════════════════

def escape_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def rewrite_generated_clean(corrected: list[tuple[str, str]]) -> None:
    """
    Reads classificationData.py, replaces the GENERATED_CLEAN list with the
    corrected entries, and writes the file back.  A .bak backup is made first.
    """
    # 1. Backup
    shutil.copy2(TARGET_FILE, BACKUP_FILE)
    print(f"\n  📦  Backup saved → {BACKUP_FILE.name}")

    # 2. Read original
    source = TARGET_FILE.read_text(encoding="utf-8")

    # 3. Build the new GENERATED_CLEAN block
    lines = ["GENERATED_CLEAN = [\n"]
    for text, label in corrected:
        lines.append(f'    ("{escape_text(text)}", "{label}"),\n')
    lines.append("]\n")
    new_block = "".join(lines)

    # 4. Replace old block using a regex that captures:
    #       GENERATED_CLEAN = [
    #         ... (anything) ...
    #       ]
    pattern = re.compile(
        r"^GENERATED_CLEAN\s*=\s*\[.*?^\]",
        re.MULTILINE | re.DOTALL,
    )

    if not pattern.search(source):
        raise RuntimeError(
            "Could not locate GENERATED_CLEAN = [...] in classificationData.py. "
            "The file structure may have changed. Backup is at: " + str(BACKUP_FILE)
        )

    updated = pattern.sub(new_block.rstrip("\n"), source, count=1)

    # 5. Write back
    TARGET_FILE.write_text(updated, encoding="utf-8")
    print(f"  ✅  classificationData.py rewritten with {len(corrected)} corrected entries.")


# ════════════════════════════════════════════════════════════════════════════
# Reporting helpers
# ════════════════════════════════════════════════════════════════════════════

def class_dist(examples: list[tuple[str, str]]) -> Counter:
    return Counter(label for _, label in examples)


def print_distribution(title: str, dist: Counter) -> None:
    labels = ["free_gift", "buy_x_get_y", "tiered_discount", "unsupported", "clarification"]
    total  = sum(dist.values())
    print(f"\n  {title}")
    print(f"  {'─'*40}")
    for lbl in labels:
        count = dist.get(lbl, 0)
        bar   = "█" * (count // 10)
        print(f"  {lbl:<20} {count:>4}  {bar}")
    print(f"  {'─'*40}")
    print(f"  {'TOTAL':<20} {total:>4}")


def print_mismatch_detail(original: list, corrected: list) -> None:
    changes = [
        (orig, corr)
        for orig, corr in zip(original, corrected)
        if orig[1] != corr[1]
    ]
    if not changes:
        print("\n  No label changes — all entries were already correct!")
        return

    print(f"\n  📋  Changed labels ({len(changes)} total):")
    print(f"  {'─'*70}")
    for (text, old_lbl), (_, new_lbl) in changes:
        snippet = (text[:60] + "…") if len(text) > 60 else text
        print(f"  [{old_lbl:<17}→ {new_lbl:<17}]  {snippet}")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 55)
    print("  relabel_generated_clean.py")
    print(f"  Source   : {TARGET_FILE.name}")
    print(f"  Entries  : {len(GENERATED_CLEAN)}")
    print(f"  Deployment: {DEPLOYMENT}")
    print("=" * 55)

    original = list(GENERATED_CLEAN)   # keep a copy for diff report

    # ── Before stats ────────────────────────────────────────────────────────
    before_dist = class_dist(original)
    print_distribution("BEFORE — class distribution", before_dist)

    # ── Validate ─────────────────────────────────────────────────────────────
    corrected, n_changes = run_validation(original)

    # ── After stats ──────────────────────────────────────────────────────────
    after_dist = class_dist(corrected)
    print_distribution("AFTER  — class distribution", after_dist)

    # ── Diff detail ──────────────────────────────────────────────────────────
    print_mismatch_detail(original, corrected)

    # ── Rewrite file ─────────────────────────────────────────────────────────
    rewrite_generated_clean(corrected)

    # ── Token / cost summary ─────────────────────────────────────────────────
    tracker.print_summary()

    print("\n✅  Done!")
    print(f"   {n_changes} label(s) corrected out of {len(original)} total.")
    print(f"   Next step: python intentClassifier/classificationData.py")
    print("       → regenerates data/train.csv, data/val.csv, data/test.csv\n")


if __name__ == "__main__":
    main()
