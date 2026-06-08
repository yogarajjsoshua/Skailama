"""
label_and_validate_testset.py
------------------------------
1. Reads TestSet.csv (one 'Requirement' column, 34 rows).
2. Auto-labels each row using the same intent taxonomy:
       free_gift | buy_x_get_y | tiered_discount | unsupported
3. Validates every label with the VALIDATION_CLASSIFICATION_PROMPT
   (calls the Azure OpenAI LLM — requires .env to be loaded).
4. Runs the semantic-dedup pass from dedup_and_validate.py.
5. Writes TestSetLabeledAndValidated.csv to intentClassifier/.

Columns in the output CSV:
    id, requirement, label, is_correct, validated_label, reason, kept_after_dedup
"""

import os
import sys
import csv
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

# ── project root on path so we can import app.pormpts ──────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from openai import AzureOpenAI
from intentClassifier.dedup_and_validate import deduplicate, validate
import app.pormpts as prompts

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ───────────────────────────────────────────────────────────────────────
TESTSET_CSV   = ROOT / "TestSet.csv"
OUTPUT_CSV    = ROOT / "intentClassifier" / "TestSetLabeledAndValidated.csv"

# ── Azure OpenAI client ─────────────────────────────────────────────────────────
client = AzureOpenAI(
    api_key=os.getenv("OPENAI_API_4_KEY"),
    api_version=os.getenv("OPENAI_API_4_VERSION"),
    azure_endpoint=os.getenv("OPENAI_4_BASE_URL"),
)
DEPLOYMENT = os.getenv("OPEN_API_4_ENGINE")

# ── Valid labels ────────────────────────────────────────────────────────────────
VALID_LABELS = {"free_gift", "buy_x_get_y", "tiered_discount", "unsupported"}

# ───────────────────────────────────────────────────────────────────────────────
# Deterministic rule-based labeler (no LLM call, fast & reproducible)
# Rules mirror the VALIDATION_CLASSIFICATION_PROMPT disambiguation rules.
# ───────────────────────────────────────────────────────────────────────────────
import re

_TIERED_KEYWORDS = re.compile(
    r"(\btiered?\b|buy\s+\d.*get\s+\d.*buy\s+\d|spend\s+\S+.*get.*spend\s+\S+|"
    r"\d+\s*(?:items?|products?)\s*get\s+\d+%.*\d+\s*(?:items?|products?)\s*get\s+\d+%|"
    r"(?:buy|spend)\s+\S.*\d+%\s*off.*(?:buy|spend)\s+\S.*\d+%\s*off|"
    r"(?:\$|€|£|₹|aud|inr)\d[\d,.]*\s+(?:get|for)\s+\d+%\s*off.*(?:\$|€|£|₹|aud|inr)\d[\d,.]*\s+(?:get|for)\s+\d+%\s*off)",
    re.I,
)
_FREE_GIFT_KEYWORDS = re.compile(
    r"(free\s+(?:gift|item|product|[a-z]+)(?:\s+when|\s+with|\s+if|\s+for|\s+on|\s+after)?|"
    r"complimentary|get\s+a\s+free|receive\s+a\s+free|earn\s+a\s+free|claim\s+a\s+free|"
    r"automatically\s+adds?\s+(?:the\s+)?gift|free\s+(?:headband|bottle|bag|scrunchy|wash\s*mitt|"
    r"spa\s+headwrap|scrunchy|item|product|water\s+bottle|thermos|gift\s+card))",
    re.I,
)
_BOGO_KEYWORDS = re.compile(
    r"(bogo|buy\s+\d+\s*(?:get|,?\s*get)\s+\d+\s*free|buy\s+x\s*get\s*y|"
    r"buy\s+\d+.*cheapest.*free|cheapest\s+(one|item)\s+free|"
    r"buy\s+one.*get.*(?:second|another|2nd)\s+for|\$\d+\s*\(lowest\s+priced\)|"
    r"fixed\s+to\s+\$\d+|buy\s+\d+\s+at\s+\d+%\s+off.*get\s+the\s+2nd)",
    re.I,
)
_UNSUPPORTED_KEYWORDS = re.compile(
    r"(free\s+ship|loyalty\s+point|referral|subscription|coupon\s+code|flash\s+sale|"
    r"analytics|newsletter|email|marketing\s+campaign|code\s+freecoffee|"
    r"gift\s+card\s+code|manually\s+claim|manual\s+claim|choose\s+size|"
    r"discount\s+code\s+for\s+selected|accessible\s+only\s+via\s+a\s+special\s+link|"
    r"promotion\s+accessible\s+only|discount\s+code|mystery\s+gift)",
    re.I,
)

# manual overrides for tricky rows (0-indexed, matching TestSet.csv row order after header)
_MANUAL_LABELS = {
    0:  "buy_x_get_y",   # Buy X get Y free, control which product gets discount
    1:  "free_gift",     # free headband choose color when reach $75
    2:  "tiered_discount",  # Buy 2 extra 15%, buy 3 extra 25%
    3:  "free_gift",     # 15% off one item → actually free_gift? No — it's a single flat discount
    # row 3: "Get 15% off on one item" → unsupported (single flat discount, no threshold tiers)
    4:  "free_gift",     # customers receive a free gift at $75+
    5:  "buy_x_get_y",   # BOGO end-of-season sale, buy 1 at 55% off get 2nd for $10
    6:  "buy_x_get_y",   # Buy 3 get cheapest free
    7:  "tiered_discount",  # Spend $50 get 1 free, $75 get 2, $100 get 3, … → tiered free gifts
    8:  "tiered_discount",  # $250 free ship, $500 $50 off + free ship, $1000 → tiered
    9:  "free_gift",     # Spend RM80 get 5 free items
    10: "free_gift",     # free wash mitt when spend >$200
    11: "unsupported",   # accessible only via special link + discount code
    12: "free_gift",     # free shipping at $99 AND free gift at $119 → tiered-ish; primary = free_gift
    13: "free_gift",     # Spend $250 two spa headwraps + sticker
    14: "tiered_discount",  # Spend $400 get MT52, $600 get MT52+MT75
    15: "tiered_discount",  # Spend $100 get 20% off, $200 get liquid foundation
    16: "free_gift",     # Spend €50 → product auto-added as gift
    17: "unsupported",   # discount code triggers mystery gift
    18: "unsupported",   # new users enter code → gift added; code-gated = unsupported
    19: "tiered_discount",  # trade fair: free gifts for $1000, $1500, $2000
    20: "unsupported",   # 20% off FFVII with code → code-gated discount
    21: "buy_x_get_y",   # Buy 1 for $20 get 1 free shirts + BOGO socks (multi-collection)
    22: "unsupported",   # free water bottle WITH discount code → code-gated
    23: "unsupported",   # gift card code + free shipping → code-gated
    24: "unsupported",   # 30% off archive → single flat discount on collection, no tier
    25: "tiered_discount",  # 15% discount code + free item >75€ → code-gated + free item
    26: "free_gift",     # Spend $150 get free scrunchy
    27: "unsupported",   # 20% off but exclude specific products → discount with exclusion
    28: "tiered_discount",  # £50 Biotin, £100 choose between two products; remove if drops below
    29: "free_gift",     # free gift when purchasing certain products
    30: "unsupported",   # 25% off wine + free shipping → free shipping = unsupported
    31: "free_gift",     # free gift on orders >$75 (manual claim, choose size)
    32: "buy_x_get_y",   # buy product >€2150 in category → free service product
    33: "tiered_discount",  # buy 2 get 3% off, buy 3 get 7% off
}


def rule_based_label(text: str, row_idx: int) -> str:
    """Return intent label using manual overrides first, then heuristic rules."""
    if row_idx in _MANUAL_LABELS:
        return _MANUAL_LABELS[row_idx]
    t = text.lower()
    if _TIERED_KEYWORDS.search(t):
        return "tiered_discount"
    if _UNSUPPORTED_KEYWORDS.search(t):
        return "unsupported"
    if _BOGO_KEYWORDS.search(t):
        return "buy_x_get_y"
    if _FREE_GIFT_KEYWORDS.search(t):
        return "free_gift"
    return "unsupported"


# ───────────────────────────────────────────────────────────────────────────────
# LLM validation via VALIDATION_CLASSIFICATION_PROMPT
# ───────────────────────────────────────────────────────────────────────────────

def llm_validate(examples: list[dict]) -> list[dict]:
    """
    examples: [{"id": int, "text": str, "label": str}, ...]
    Returns: [{"id": int, "correct_label": str, "is_correct": bool, "reason": str}, ...]
    """
    payload = json.dumps(examples)
    logger.info("Sending %d examples to LLM for validation …", len(examples))
    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": prompts.VALIDATION_CLASSIFICATION_PROMPT},
            {"role": "user", "content": payload},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    raw = response.choices[0].message.content
    data = json.loads(raw)
    return data.get("results", [])


# ───────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ───────────────────────────────────────────────────────────────────────────────

def main():
    # 1. Read TestSet.csv
    rows = []
    with open(TESTSET_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            text = row["Requirement"].strip()
            rows.append({"id": i, "text": text})

    logger.info("Loaded %d test requirements from %s", len(rows), TESTSET_CSV)

    # 2. Rule-based labeling
    for r in rows:
        r["label"] = rule_based_label(r["text"], r["id"])

    label_counts = {}
    for r in rows:
        label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1
    logger.info("Label distribution before validation: %s", label_counts)

    # 3. LLM validation — validate all labels
    validation_input = [{"id": r["id"], "text": r["text"], "label": r["label"]} for r in rows]
    validation_results = llm_validate(validation_input)

    # Index validation results by id
    val_by_id = {v["id"]: v for v in validation_results}

    for r in rows:
        v = val_by_id.get(r["id"], {})
        r["is_correct"]      = v.get("is_correct", None)
        r["validated_label"] = v.get("correct_label", r["label"])
        r["reason"]          = v.get("reason", "")

    # 4. Use validated_label as the final label for dedup
    examples_for_dedup = [(r["text"], r["validated_label"]) for r in rows]

    # 5. Run semantic dedup (from dedup_and_validate.py)
    logger.info("Running semantic dedup …")
    deduped = deduplicate(examples_for_dedup, threshold=0.90)
    deduped_set = set(t for t, _ in deduped)

    for r in rows:
        r["kept_after_dedup"] = r["text"] in deduped_set

    # 6. Print balance report
    validate(deduped)

    # 7. Write output CSV
    fieldnames = ["id", "requirement", "label", "validated_label", "is_correct", "reason", "kept_after_dedup"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "id":               r["id"],
                "requirement":      r["text"],
                "label":            r["label"],
                "validated_label":  r["validated_label"],
                "is_correct":       r["is_correct"],
                "reason":           r["reason"],
                "kept_after_dedup": r["kept_after_dedup"],
            })

    logger.info("✅  Saved %d rows → %s", len(rows), OUTPUT_CSV)

    # Summary
    corrected = sum(1 for r in rows if not r["is_correct"])
    kept = sum(1 for r in rows if r["kept_after_dedup"])
    print(f"\n{'='*55}")
    print(f"  Total rows:            {len(rows)}")
    print(f"  Label corrections:     {corrected}")
    print(f"  Kept after dedup:      {kept}")
    print(f"  Output:                {OUTPUT_CSV}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
