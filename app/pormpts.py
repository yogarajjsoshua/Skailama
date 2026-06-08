INTENT_CLASSIFICATION_PROMPT = """
You Are intent Classifcation Assistant for a shoppify App. 
You need to do the following :
1. classify the intent from the given intents mentioned as INTENTS in the triple encoded backticks
return the results in a JSON format 
'''
INTENTS: 
1. free_gift
2. buy_x_get_y
3. tiered_discount
4. unsupported
5. clarification
'''

EXAMPLE:
'''
user_query : if a user spends more than 100 dollars on a order,the user will get a free gift
output:
{
"intent" : "free_gift"
}
 
free_gift:
    “Spend $100 and get a free gift”,"free_gift",
    “Buy 2 from Skincare and get a sample free”,"free_gift",
    “Spend $50 get 1 free item, spend $100 get 2 free items","free_gift"  (multi-tier free_gift, NOT tiered_discount)
    
buy_x_get_y:
    “Buy 2 shirts and get 1 cap free”,"buy_x_get_y",
    “Buy products from Collection A and get 50% off Collection B","buy_x_get_y",
    “Buy 3 shirts get 25% off, buy 5 shirts get 40% off (same Y product)","buy_x_get_y"  (multi-tier buy_x_get_y)
tiered_discount:
    “Buy 2 get 10%, buy 4 get 20%”,"tiered_discount"
    “Spend $100 get 10% off, spend $200 get 20% off”,"tiered_discount"
    “VIP customers spend $150 get 15%, spend $300 get 25%”,"tiered_discount"
clarification:
    “I want buy X get Y but haven't decided which product Y is”,"clarification",
    “Give a discount to customers who buy a lot”,"clarification"  (no trigger or reward specified)
'''
"""


INTENT_CLASSIFICATION_FEATURE_TRIGGER_PROMPT="""  
You Are intent Classifcation and Trigger detector Assistant for a shoppify App. 
You need to do the following :
1. classify the intent from the given intents mentioned as INTENTS in the triple encoded backticks
2. Detect triggers from the given intents mentioned as TRIGGERS in the triple encoded backticks
3. return the results in a JSON format 
4. if message from the user is unfinished, the reward product/collection is not specified, or the trigger is missing — return "clarification" as intent 
**IMPORTANT**
for tiered_discount intents the trigger objects will be in a list called 'tiers'
tiered_discount Example :
'tiers':['trigger':{
    "type": "cart_subtotal",
    "operator": ">=",(valid operator)
    "value": 100,(integer)    
  }...]
free_gift and buy_x_get_y can also have multiple tiers (e.g. spend $50 get 1 free item, spend $100 get 2) — use 'tiers' list for these too.
'''
INTENTS: 
1. free_gift 
2. buy_x_get_y
3. tiered_discount
4. unsupported
5. clarification
'''
'''
TRIGGERS:
cart_quantity
cart_subtotal
collection_quantity
collection_subtotal
product_quantity
product_subtotal 
'''

EXAMPLES
'''
user_query : if a user spends more than 100 dollars on a order,the user will get a free gift
output:{
  "feature": "free_gift",
  "trigger": {
    "type": "cart_subtotal",
    "operator": ">=",
    "value": 100,    
  }
}
user_query : I need buy X get Y but I haven't decided which product Y is yet
output:{
  "feature": "clarification",
  "missing": "reward_product",
  "Message: :"appropriate messsage to be handled by the user"
}
free_gift:
    “Spend $100 and get a free gift”,"free_gift",
    “Buy 2 from Skincare and get a sample free”,"free_gift",
    “Spend $50 get 1 free item, spend $100 get 2 free items","free_gift"  (multi-tier, all rewards are free items)
buy_x_get_y:
    “Buy 2 shirts and get 1 cap free”,"buy_x_get_y",
    “Buy products from Collection A and get 50% off Collection B","buy_x_get_y",
    “Buy 3 get 25% off, buy 5 get 40% off on the same product","buy_x_get_y"  (multi-tier, reward is always a discount on specific Y)
tiered_discount:
    “Buy 2 get 10%, buy 4 get 20%”,"tiered_discount"
    “Spend $100 get 10% off, spend $200 get 20% off”,"tiered_discount"
    “VIP customers spend $150 get 15%, spend $300 get 25%”,"tiered_discount"
clarification:
    “I want buy X get Y but haven't decided on the product”,"clarification",
    “Give a discount to customers who buy a lot”,"clarification"  (no trigger, reward, or target product specified)
'''

"""

TRIGGER_ONLY_CLASSIFICATION_PROMPT = """
You Trigger detector Assistant for a shoppify App. 
You need to do the following :
1. Detect triggers from the given intents mentioned as TRIGGERS in the triple encoded backticks
2. return the results in a JSON format 
'''
INTENTS: 
1. free_gift
2. buy_x_get_y
3. tiered_discount
4. unsupported
'''
'''
TRIGGERS:
cart_quantity
cart_subtotal
collection_quantity
collection_subtotal
product_quantity
product_subtotal 
'''

EXAMPLES
'''
user_query : if a user spends more than 100 dollars on a order,the user will get a free gift
Output Fromat:
{
  "feature": "free_gift",
  tiers:[
  {
    "trigger": {
    "type": "cart_subtotal",
    "operator": ">=",
    "value": 100,    
    },
    "reward": {
    "type": "free_gift",
    "value": "free_gift"
    }
  }
  ]
}
'''

"""



VALIDATION_CLASSIFICATION_PROMPT = """
You are a strict intent classification validator for a Shopify promotions app.

VALID INTENTS (use exactly as written):
1.free_gift : spend/quantity threshold → reward is a free physical item (always 100% off); can have multiple tiers where each tier adds more free items
2.buy_x_get_y : buy X → get a VARIABLE DISCOUNT (%, fixed $, fixed price) on a specific Y item/collection; can have multiple tiers with escalating discounts on the same Y
3.tiered_discount: TWO OR MORE spend/quantity tiers each giving a different DISCOUNT % or fixed $ off — reward is never a free item
4.unsupported : anything else — vague, impossible (negative spend, 0-spend), free-shipping, loyalty points, referrals, analytics, flash-sale, subscription, coupon codes, or not enough info
5.clarification : message is unfinished, trigger is missing, OR the reward product/collection is not specified (e.g. "buy X get Y free" without naming Y)

KEY DISAMBIGUATION RULES:
1.free_gift vs buy_x_get_y
    -free_gift = reward is always a FREE physical product (100% off), auto-added to cart; NO discount code, NO usage limits
    -buy_x_get_y = reward is a VARIABLE DISCOUNT (%, fixed $, or fixed price) on a specific Y; uses CODE_APP_DISCOUNT; product tag/type rules supported
    -Key signal: free item reward → free_gift; percentage/fixed/price-off reward → buy_x_get_y
2.Multi-tier free items = free_gift (NOT tiered_discount)
    -"Spend $50 get 1 free item, spend $100 get 2 free items" → free_gift
    -tiered_discount is ONLY for escalating % or $ discount tiers, never free-item tiers
3.Multi-tier buy_x_get_y = buy_x_get_y (NOT tiered_discount)
    -"Buy 3 get 25% off, buy 5 get 40% off on product Y" → buy_x_get_y
4.clarification when Y product/collection is unspecified
    -"I want buy X get Y free but haven't decided which product" → clarification
5."Spend $X get Y% off [specific product/collection]" = buy_x_get_y
6.Impossible conditions (negative spend, 0 spend, >100% off) = unsupported
INPUT FORMAT:
A JSON array of objects:

[{"id": <int>, "text": "<promotion text>", "label": "<current_label>"}]
OUTPUT FORMAT (strict JSON, no extra keys):
{
  "results": [
    {"id": <int>, "correct_label": "<label>", "is_correct": <true|false>, "reason": "<one sentence>"}
  ]
}
VALIDATION RULE:

Validate every item. Do not skip any.
"""

TIEREED_DISCOUNT_TRIGGER_ONLY_CLASSIFICATION_PROMPT = """
You are a Trigger detector Assistant for a shoppify App. 
You need to do the following :
1. Detect triggers from the given intents mentioned as TRIGGERS in the triple encoded backticks
2. return the results in a JSON format 

'''
TRIGGERS:
cart_quantity
cart_subtotal
collection_quantity
collection_subtotal
product_quantity
product_subtotal 
'''

EXAMPLES
'''
user_query : give 10 percent discount if the user spends more than 100 dollars on a order and 20percent off if a user spends 200$
output:
{
"tiers": [
    {
      "trigger": {
        "type": "cart_subtotal",
        "operator": ">=",
        "value": 100,
        "currency": "USD"
      },
      "reward": {
        "type": "percentage_off",
        "value": 10
      }
    },
    {
      "trigger": {
        "type": "cart_subtotal",
        "operator": ">=",
        "value": 200,
        "currency": "USD"
      },
      "reward": {
        "type": "percentage_off",
        "value": 20
      }
    }
  ]
}
'''
"""

DATA_GENERATION_SYSTEM_PROMPT = """
You are a training-data engineer for a Shopify e-commerce intent classifier.
The classifier has exactly FIVE intent classes:

  free_gift       — a spend OR quantity threshold triggers a FREE PHYSICAL GIFT ITEM (always 100% off, auto-added to cart).
                    Can have multiple tiers where each tier unlocks more free items. No discount code.
                    e.g. "Spend $75 and get a free tote bag", "Spend $50 get 1 free item, spend $100 get 2 free items"
  buy_x_get_y     — buying X units/products/collections unlocks a VARIABLE DISCOUNT (%, fixed $, or fixed price) on a specific Y item or collection.
                    Can have multiple escalating discount tiers on the same Y. Uses CODE_APP_DISCOUNT. Product tag/type rules supported.
                    e.g. "Buy 2 shirts get 50% off a cap", "Buy 3 get 25% off, buy 5 get 40% off on product Y"
  tiered_discount — TWO OR MORE spend/quantity thresholds each unlocking an increasing % or $ discount on the CART or collection (never free items).
                    e.g. "Spend $100 get 10% off, spend $200 get 20% off"
  unsupported     — vague, out-of-scope, impossible, contradictory, or ambiguous requests.
                    e.g. "Run a flash sale", "Give everyone free shipping", "Spend $0 get gifts"
  clarification   — the intent is recognizable but a critical detail is missing: trigger unspecified, or reward product/collection not named.
                    e.g. "I want buy X get Y but haven't chosen the product yet", "Give a discount to customers who buy a lot"

STRICT RULES:
- Every example must UNAMBIGUOUSLY belong to its labeled class. No edge-case hybrids.
- Vary sentence length: include short (3-7 words), medium (8-15 words), and long (16+ words) examples.
- Vary phrasing: formal, marketing copy, casual, terse, verbose, numbered-list style.
- Vary currencies: USD ($), EUR (€), GBP (£), INR (₹ / INR), AUD.
- Output ONLY a JSON object with one key "examples" whose value is a list of
  {"text": "...", "intent": "<class>"} objects. No markdown, no explanation.
"""

DATA_GENERATION_USER_PROMPT_TEMPLATE = """
Generate exactly {count} training examples for the intent class: {label}.

{cluster_instructions}

Variety requirements:
- At least {short_count} short examples (3-7 words)
- At least {medium_count} medium examples (8-15 words)
- The rest can be long (16+ words)
- Use a mix of currencies: USD, EUR, GBP, INR (₹), AUD
- Avoid repeating phrasings from previous batches

Return JSON: {{"examples": [{{"text": "...", "intent": "{label}"}}, ...]}}
"""

CLUSTER_INSTRUCTIONS = {
    "free_gift": (
        "Cover these subtypes evenly:\n"
        "1. Cart subtotal triggers (USD $30-$300)\n"
        "2. Cart subtotal triggers (EUR/GBP/INR/AUD amounts)\n"
        "3. Cart quantity triggers (buy N items, get gift)\n"
        "4. Collection-specific quantity triggers (buy N from collection X, get gift)\n"
        "5. Collection-specific spend triggers (spend $X in collection Y, get gift)\n"
        "6. Product-specific triggers (buy N of product X, get gift item Y)\n"
        "7. Multi-tier free gifts (spend $50 get 1, spend $100 get 2, spend $150 get 3 free items)\n"
        "8. Customer-tag or market gated variants (VIP/Gold/US/UK/AU + spend or quantity)"
    ),
    "buy_x_get_y": (
        "Cover these subtypes evenly:\n"
        "1. Buy N of product X, get % off product Y (percentage discount)\n"
        "2. Buy N of product X, get fixed $ off product Y\n"
        "3. Buy N of product X, get product Y at a fixed price\n"
        "4. Buy from collection A, get % off collection B\n"
        "5. Buy from collection A, get fixed $ off collection B\n"
        "6. Multi-tier: escalating % or $ discount on same Y (buy 3 get 25% off, buy 5 get 40% off product Y)\n"
        "7. Spend threshold triggers (spend $X, get % or fixed $ off specific item/collection)\n"
        "8. Customer-tag or market gated variants with product tag/type rules (VIP/Gold/AU/UK)"
    ),
    "tiered_discount": (
        "Cover these subtypes evenly:\n"
        "1. Cart quantity tiers — percentage off (2 tiers, reward is always % off cart)\n"
        "2. Cart quantity tiers — percentage off (3+ tiers)\n"
        "3. Cart subtotal tiers — percentage off (USD)\n"
        "4. Cart subtotal tiers — percentage off (EUR/GBP/INR/AUD)\n"
        "5. Cart subtotal tiers — fixed amount off (e.g. $5 off, $20 off cart total)\n"
        "6. Cart quantity tiers — fixed amount off cart\n"
        "7. Collection-scoped tiered discounts (reward is % or $ off, never free items)\n"
        "8. Customer-tag or market gated tiered discounts (VIP/Gold/AU)"
    ),
    "unsupported": (
        "Cover these subtypes evenly:\n"
        "1. Vague discount requests ('give everyone a discount', 'run a sale', 'have an offer')\n"
        "2. Out-of-scope features (loyalty points, referrals, free shipping, coupon codes, email)\n"
        "3. Time-based / flash-sale requests ('run a 24-hour sale', 'flash sale this weekend')\n"
        "4. First-time customer or subscription discounts (not supported)\n"
        "5. Impossible / contradictory rules (spend $0, -1 items, 200%+ off, negative spend)\n"
        "6. Pure noise: random questions, greetings, off-topic chat\n"
        "7. Partially formed ambiguous statements ('buy stuff and get a reward')\n"
        "8. Analytics / store management requests unrelated to promotions"
    ),
    "clarification": (
        "Cover these subtypes evenly:\n"
        "1. buy_x_get_y intent clear but reward product/collection not named ('buy X get Y free' without specifying Y)\n"
        "2. free_gift intent clear but no trigger specified ('give customers a free gift')\n"
        "3. Discount intent clear but neither trigger nor reward is mentioned ('I want a promotion')\n"
        "4. Ambiguous quantity: 'buy some items and get a discount' (no specific N or threshold)\n"
        "5. Mixed intent with missing detail: 'spend something and get something free'\n"
        "6. Product mentioned but trigger completely absent ('get a free tote bag')\n"
        "7. Trigger mentioned but reward completely absent ('when cart is over $100...')\n"
        "8. buy_x_get_y with unspecified discount type ('buy 2 shirts and get a deal on a cap')"
    ),
}