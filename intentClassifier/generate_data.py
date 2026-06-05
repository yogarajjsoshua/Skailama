import os
import json
import ast
import sys
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("OPENAI_API_4_KEY"),
    api_version=os.getenv("OPENAI_API_4_VERSION"),
    azure_endpoint=os.getenv("OPENAI_4_BASE_URL"),
)
DEPLOYMENT = os.getenv("OPEN_API_4_ENGINE")

SYSTEM_PROMPT = """You are a training-data engineer for an e-commerce intent classifier.
The classifier has four intent classes:

  free_gift      — a spend OR quantity threshold triggers a FREE PHYSICAL GIFT ITEM (not a discount).
                   Example: "Spend $75 and get a free tote bag"
  buy_x_get_y    — buying X units/products gives a discount or free item on Y (different product or same BOGO).
                   Example: "Buy 2 shirts, get 1 cap free" | "Buy 3 items get the cheapest one free"
  tiered_discount — MULTIPLE spend or quantity thresholds each unlock an increasing % or $ discount.
                   Example: "Spend $100 get 10% off, spend $200 get 20% off"
  unsupported    — vague, out-of-scope, impossible, noise, or ambiguous requests.
                   Example: "Run a sale this weekend" | "Give everyone a discount"

STRICT RULES:
- Do NOT mix intents. Every example must unambiguously belong to its labeled class.
- Do NOT include examples that could belong to two classes (exclude them entirely).
- Vary: sentence length (short/medium/long), tone (formal/marketing/casual/terse),
  currency (USD/EUR/GBP/INR/AUD), trigger type, and wording style.
- Output ONLY a JSON object with one key "examples" whose value is a list of
  {"text": "...", "intent": "..."} objects. No markdown, no explanation."""

CALL_SPECS = [
    {
        "label": "free_gift",
        "prompt": (
            "Generate exactly 120 training examples for the intent class: free_gift.\n"
            "A free_gift is triggered by a spend threshold OR a quantity threshold and rewards the customer "
            "with a FREE PHYSICAL GIFT ITEM (not a percentage off, not a fixed-amount discount).\n"
            "Cover these cluster types evenly (about 15 examples each):\n"
            "1. Cart subtotal triggers (USD amounts $40-$250)\n"
            "2. Cart subtotal triggers (EUR/GBP/INR/AUD amounts)\n"
            "3. Cart quantity triggers (buy N items get gift)\n"
            "4. Collection-specific quantity triggers (buy N from collection X)\n"
            "5. Collection-specific spend triggers (spend $X in collection Y)\n"
            "6. Product-specific triggers (buy N of product X get gift Y)\n"
            "7. Customer-tag gated (VIP/Gold/Wholesale + spend or quantity)\n"
            "8. Market/login gated (US/UK/AU/logged-in + spend or quantity)\n"
            "Vary sentence length: at least 30 short (3-7 words), 50 medium (8-15 words), 40 long (16+ words).\n"
            "Return JSON: {\"examples\": [{\"text\": \"...\", \"intent\": \"free_gift\"}, ...]}"
        ),
    },
    {
        "label": "buy_x_get_y",
        "prompt": (
            "Generate exactly 120 training examples for the intent class: buy_x_get_y.\n"
            "buy_x_get_y means: purchasing X quantity/products gives a benefit on Y "
            "(free item, % off, fixed $ off, or BOGO). The reward is on a SPECIFIC PRODUCT OR SET, "
            "NOT a percentage off the whole cart, and NOT a free gift unrelated to the purchase.\n"
            "Cover these cluster types evenly:\n"
            "1. BOGO / buy-one-get-one-free (same product)\n"
            "2. Buy N get cheapest free\n"
            "3. Buy product X get product Y free\n"
            "4. Buy product X get % off product Y\n"
            "5. Buy from collection A get % off collection B\n"
            "6. Cart quantity triggers (buy N items get % off or free item)\n"
            "7. Spend threshold triggers (spend $X get % off specific item)\n"
            "8. Customer-tag or market gated variants\n"
            "Vary currencies (USD/EUR/GBP/INR/AUD) and sentence length (short/medium/long).\n"
            "Return JSON: {\"examples\": [{\"text\": \"...\", \"intent\": \"buy_x_get_y\"}, ...]}"
        ),
    },
    {
        "label": "tiered_discount",
        "prompt": (
            "Generate exactly 120 training examples for the intent class: tiered_discount.\n"
            "tiered_discount means: TWO OR MORE spend or quantity thresholds each unlock an increasing "
            "discount (% off or fixed $ off the cart/collection). There must be multiple tiers — "
            "a single threshold is NOT tiered.\n"
            "Cover these cluster types evenly:\n"
            "1. Cart quantity tiers — percentage off (2 tiers)\n"
            "2. Cart quantity tiers — percentage off (3+ tiers)\n"
            "3. Cart subtotal tiers — percentage off (USD)\n"
            "4. Cart subtotal tiers — percentage off (EUR/GBP/INR/AUD)\n"
            "5. Cart subtotal tiers — fixed amount off\n"
            "6. Cart quantity tiers — fixed amount off\n"
            "7. Collection-scoped tiered discounts\n"
            "8. Customer-tag or market gated tiered discounts\n"
            "Vary sentence style: structured (2 items: X%, 4 items: Y%), prose, bullet-style shorthand.\n"
            "Return JSON: {\"examples\": [{\"text\": \"...\", \"intent\": \"tiered_discount\"}, ...]}"
        ),
    },
    {
        "label": "unsupported",
        "prompt": (
            "Generate exactly 120 training examples for the intent class: unsupported.\n"
            "unsupported means requests that cannot be handled by the classifier because they are: "
            "vague, out-of-scope, impossible, contradictory, noise, or ambiguous across two classes.\n"
            "Cover these cluster types evenly:\n"
            "1. Vague discount requests ('give everyone a discount', 'run a sale')\n"
            "2. Out-of-scope features (loyalty points, referrals, email campaigns, free shipping, coupons)\n"
            "3. Time-based / flash-sale requests (not in classifier spec)\n"
            "4. First-time customer or subscription discounts (not in spec)\n"
            "5. Impossible / contradictory rules ($0 spend, -1 items, 200% off)\n"
            "6. Pure noise / random text / questions / greetings\n"
            "7. Partially formed ambiguous statements ('buy stuff and get a reward')\n"
            "8. Analytics / store management requests unrelated to promotions\n"
            "Return JSON: {\"examples\": [{\"text\": \"...\", \"intent\": \"unsupported\"}, ...]}"
        ),
    },
    {
        "label": "mixed_topup",
        "prompt": (
            "Generate 30 additional training examples — 10 for each of these three classes ONLY: "
            "free_gift, buy_x_get_y, tiered_discount.\n"
            "Focus on edge cases and unusual phrasings not commonly seen:\n"
            "- free_gift: very short terse examples (under 6 words) and very long formal ones (20+ words)\n"
            "- buy_x_get_y: INR and AUD currency variants, and wholesale/bulk quantity patterns\n"
            "- tiered_discount: 4-tier and 5-tier discount ladders, and collection-scoped INR/AUD variants\n"
            "Return JSON: {\"examples\": [{\"text\": \"...\", \"intent\": \"<class>\"}, ...]}"
        ),
    },
]


def call_azure(prompt: str, call_index: int) -> list:
    print(f"[Call {call_index}/5] Sending request to Azure OpenAI...")
    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.85,
        max_tokens=4096,
    )
    raw = response.choices[0].message.content
    parsed = json.loads(raw)
    examples = parsed.get("examples", [])
    print(f"         Received {len(examples)} examples.")
    return examples


def main():
    all_examples = []
    api_call_count = 0

    for i, spec in enumerate(CALL_SPECS, start=1):
        if api_call_count >= 5:
            print("Guardrail hit: 5 API calls reached. Stopping.")
            break
        examples = call_azure(spec["prompt"], i)
        api_call_count += 1
        for ex in examples:
            text = ex.get("text", "").strip()
            intent = ex.get("intent", "").strip()
            if text and intent in ("free_gift", "buy_x_get_y", "tiered_discount", "unsupported"):
                all_examples.append((text, intent))

    from collections import Counter
    dist = Counter(label for _, label in all_examples)
    print("\n=== Raw generated counts ===")
    for label, count in sorted(dist.items()):
        print(f"  {label:<20} {count}")
    print(f"  {'TOTAL':<20} {len(all_examples)}")

    out_path = os.path.join(os.path.dirname(__file__), "generated_data.py")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("ALL_EXAMPLES = [\n")
        for text, label in all_examples:
            escaped = text.replace("\\", "\\\\").replace('"', '\\"')
            f.write(f'    ("{escaped}", "{label}"),\n')
        f.write("]\n")
    print(f"\nSaved {len(all_examples)} examples to {out_path}")
    print(f"Total Azure API calls made: {api_call_count}")


if __name__ == "__main__":
    main()
