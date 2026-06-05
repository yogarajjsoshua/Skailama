#!/usr/bin/env python3
"""
Patch ModelTrain.ipynb to add call_model_then_trigger function and update all charts.
"""
import json, copy

NB_PATH = "ModelTrain.ipynb"

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

# ── helpers ────────────────────────────────────────────────────────────────────
def find_cell(cell_id):
    for i, c in enumerate(cells):
        if c.get("id") == cell_id:
            return i, c
    return None, None

def src(lines):
    """Turn a list of raw Python lines into a notebook source list."""
    result = []
    for line in lines:
        result.append(line + "\n" if not line.endswith("\n") else line)
    # Remove trailing newline from last item
    if result:
        result[-1] = result[-1].rstrip("\n")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# CELL 4  –  add call_model_then_trigger
# ══════════════════════════════════════════════════════════════════════════════
idx4, cell4 = find_cell("c9d0e1f2")
assert idx4 is not None, "Cell 4 not found"

cell4_source = [
    "from app.pormpts import (\n",
    "    INTENT_CLASSIFICATION_FEATURE_TRIGGER_PROMPT,\n",
    "    TRIGGER_ONLY_CLASSIFICATION_PROMPT,\n",
    "    TIEREED_DISCOUNT_TRIGGER_ONLY_CLASSIFICATION_PROMPT,\n",
    "    INTENT_CLASSIFICATION_PROMPT,\n",
    ")\n",
    "\n",
    "# ── import the local trained model classifier ─────────────────────────────\n",
    "from intentClassifier.classify import classify as model_classify\n",
    "\n",
    "\n",
    "def call_combined(query: str) -> tuple[dict, float, dict]:\n",
    "    \"\"\"Combined intent + trigger via INTENT_CLASSIFICATION_FEATURE_TRIGGER_PROMPT.\"\"\"\n",
    "    t0 = time.perf_counter()\n",
    "    response = client.chat.completions.create(\n",
    "        model=DEPLOYMENT,\n",
    "        messages=[\n",
    "            {\"role\": \"system\", \"content\": INTENT_CLASSIFICATION_FEATURE_TRIGGER_PROMPT},\n",
    "            {\"role\": \"user\",   \"content\": query},\n",
    "        ],\n",
    "        response_format={\"type\": \"json_object\"},\n",
    "        top_p=0.0,\n",
    "        temperature=1,\n",
    "    )\n",
    "    elapsed = time.perf_counter() - t0\n",
    "    reply = json.loads(response.choices[0].message.content)\n",
    "    usage = {\n",
    "        \"prompt_tokens\":     response.usage.prompt_tokens,\n",
    "        \"completion_tokens\": response.usage.completion_tokens,\n",
    "        \"total_tokens\":      response.usage.total_tokens,\n",
    "    }\n",
    "    return reply, elapsed, usage\n",
    "\n",
    "\n",
    "def call_trigger_only(query: str, intent: str) -> tuple[dict, float, dict | None]:\n",
    "    \"\"\"Trigger-only prompt with pre-known intent.\n",
    "\n",
    "    - 'unsupported' → skip API, return ({}, 0.0, None).\n",
    "    - 'tiered_discount' → TIEREED_DISCOUNT_TRIGGER_ONLY_CLASSIFICATION_PROMPT.\n",
    "    - otherwise      → TRIGGER_ONLY_CLASSIFICATION_PROMPT.\n",
    "    \"\"\"\n",
    "    if intent == \"unsupported\":\n",
    "        return {}, 0.0, None\n",
    "\n",
    "    system_prompt = (\n",
    "        TIEREED_DISCOUNT_TRIGGER_ONLY_CLASSIFICATION_PROMPT\n",
    "        if intent == \"tiered_discount\"\n",
    "        else TRIGGER_ONLY_CLASSIFICATION_PROMPT\n",
    "    )\n",
    "\n",
    "    user_msg = f\"intent: {intent}\\nuser_query: {query}\"\n",
    "    t0 = time.perf_counter()\n",
    "    response = client.chat.completions.create(\n",
    "        model=DEPLOYMENT,\n",
    "        messages=[\n",
    "            {\"role\": \"system\", \"content\": system_prompt},\n",
    "            {\"role\": \"user\",   \"content\": user_msg},\n",
    "        ],\n",
    "        response_format={\"type\": \"json_object\"},\n",
    "        top_p=0.0,\n",
    "        temperature=1,\n",
    "    )\n",
    "    elapsed = time.perf_counter() - t0\n",
    "    reply = json.loads(response.choices[0].message.content)\n",
    "    usage = {\n",
    "        \"prompt_tokens\":     response.usage.prompt_tokens,\n",
    "        \"completion_tokens\": response.usage.completion_tokens,\n",
    "        \"total_tokens\":      response.usage.total_tokens,\n",
    "    }\n",
    "    return reply, elapsed, usage\n",
    "\n",
    "\n",
    "def call_intent_then_trigger(query: str) -> tuple[dict, float, dict, float, str]:\n",
    "    \"\"\"Two-step pipeline: INTENT_CLASSIFICATION_PROMPT → trigger prompt.\n",
    "\n",
    "    Returns (trigger_reply, total_latency, total_usage, intent_latency, classified_intent).\n",
    "    - 'unsupported' → only intent call, trigger step skipped.\n",
    "    - 'tiered_discount' → TIEREED_DISCOUNT_TRIGGER_ONLY_CLASSIFICATION_PROMPT.\n",
    "    - otherwise → TRIGGER_ONLY_CLASSIFICATION_PROMPT.\n",
    "    \"\"\"\n",
    "    # Step 1: intent classification\n",
    "    t0 = time.perf_counter()\n",
    "    resp_intent = client.chat.completions.create(\n",
    "        model=DEPLOYMENT,\n",
    "        messages=[\n",
    "            {\"role\": \"system\", \"content\": INTENT_CLASSIFICATION_PROMPT},\n",
    "            {\"role\": \"user\",   \"content\": query},\n",
    "        ],\n",
    "        response_format={\"type\": \"json_object\"},\n",
    "        top_p=0.0,\n",
    "        temperature=1,\n",
    "    )\n",
    "    intent_latency = time.perf_counter() - t0\n",
    "    classified_intent = json.loads(resp_intent.choices[0].message.content).get(\"intent\", \"unsupported\")\n",
    "    intent_usage = {\n",
    "        \"prompt_tokens\":     resp_intent.usage.prompt_tokens,\n",
    "        \"completion_tokens\": resp_intent.usage.completion_tokens,\n",
    "        \"total_tokens\":      resp_intent.usage.total_tokens,\n",
    "    }\n",
    "\n",
    "    # Step 2: trigger detection (skip for unsupported)\n",
    "    if classified_intent == \"unsupported\":\n",
    "        return {}, intent_latency, intent_usage, intent_latency, classified_intent\n",
    "\n",
    "    system_prompt = (\n",
    "        TIEREED_DISCOUNT_TRIGGER_ONLY_CLASSIFICATION_PROMPT\n",
    "        if classified_intent == \"tiered_discount\"\n",
    "        else TRIGGER_ONLY_CLASSIFICATION_PROMPT\n",
    "    )\n",
    "    user_msg = f\"intent: {classified_intent}\\nuser_query: {query}\"\n",
    "    t1 = time.perf_counter()\n",
    "    resp_trigger = client.chat.completions.create(\n",
    "        model=DEPLOYMENT,\n",
    "        messages=[\n",
    "            {\"role\": \"system\", \"content\": system_prompt},\n",
    "            {\"role\": \"user\",   \"content\": user_msg},\n",
    "        ],\n",
    "        response_format={\"type\": \"json_object\"},\n",
    "        top_p=0.0,\n",
    "        temperature=1,\n",
    "    )\n",
    "    trigger_latency = time.perf_counter() - t1\n",
    "    trigger_reply = json.loads(resp_trigger.choices[0].message.content)\n",
    "    trigger_usage = {\n",
    "        \"prompt_tokens\":     resp_trigger.usage.prompt_tokens,\n",
    "        \"completion_tokens\": resp_trigger.usage.completion_tokens,\n",
    "        \"total_tokens\":      resp_trigger.usage.total_tokens,\n",
    "    }\n",
    "    total_latency = intent_latency + trigger_latency\n",
    "    total_usage = {\n",
    "        \"prompt_tokens\":     intent_usage[\"prompt_tokens\"]     + trigger_usage[\"prompt_tokens\"],\n",
    "        \"completion_tokens\": intent_usage[\"completion_tokens\"] + trigger_usage[\"completion_tokens\"],\n",
    "        \"total_tokens\":      intent_usage[\"total_tokens\"]      + trigger_usage[\"total_tokens\"],\n",
    "    }\n",
    "    return trigger_reply, total_latency, total_usage, intent_latency, classified_intent\n",
    "\n",
    "\n",
    "def call_model_then_trigger(query: str) -> tuple[dict, float, dict | None, float, str]:\n",
    "    \"\"\"Two-step pipeline: Local trained model → trigger prompt (AzureOpenAI).\n",
    "\n",
    "    Step 1: Use the fine-tuned local DistilBERT model (classify.py) to predict intent.\n",
    "    Step 2: Call AzureOpenAI with the appropriate trigger-extraction prompt:\n",
    "        - 'free_gift' or 'buy_x_get_y' → TRIGGER_ONLY_CLASSIFICATION_PROMPT\n",
    "        - 'tiered_discount'             → TIEREED_DISCOUNT_TRIGGER_ONLY_CLASSIFICATION_PROMPT\n",
    "        - 'unsupported'                 → skip API, return ({}, total_latency, None, model_latency, 'unsupported')\n",
    "\n",
    "    Returns (trigger_reply, total_latency, trigger_usage, model_latency, classified_intent).\n",
    "    trigger_usage is None for 'unsupported'.\n",
    "    \"\"\"\n",
    "    # Step 1: local model intent classification\n",
    "    t0 = time.perf_counter()\n",
    "    model_result = model_classify(query)   # returns {label, confidence, scores, latency_ms, text}\n",
    "    model_latency = time.perf_counter() - t0\n",
    "    classified_intent = model_result[\"label\"]\n",
    "\n",
    "    # Step 2: trigger detection (skip for unsupported)\n",
    "    if classified_intent == \"unsupported\":\n",
    "        return {}, model_latency, None, model_latency, classified_intent\n",
    "\n",
    "    system_prompt = (\n",
    "        TIEREED_DISCOUNT_TRIGGER_ONLY_CLASSIFICATION_PROMPT\n",
    "        if classified_intent == \"tiered_discount\"\n",
    "        else TRIGGER_ONLY_CLASSIFICATION_PROMPT\n",
    "    )\n",
    "    user_msg = f\"intent: {classified_intent}\\nuser_query: {query}\"\n",
    "    t1 = time.perf_counter()\n",
    "    resp_trigger = client.chat.completions.create(\n",
    "        model=DEPLOYMENT,\n",
    "        messages=[\n",
    "            {\"role\": \"system\", \"content\": system_prompt},\n",
    "            {\"role\": \"user\",   \"content\": user_msg},\n",
    "        ],\n",
    "        response_format={\"type\": \"json_object\"},\n",
    "        top_p=0.0,\n",
    "        temperature=1,\n",
    "    )\n",
    "    trigger_latency = time.perf_counter() - t1\n",
    "    trigger_reply = json.loads(resp_trigger.choices[0].message.content)\n",
    "    trigger_usage = {\n",
    "        \"prompt_tokens\":     resp_trigger.usage.prompt_tokens,\n",
    "        \"completion_tokens\": resp_trigger.usage.completion_tokens,\n",
    "        \"total_tokens\":      resp_trigger.usage.total_tokens,\n",
    "    }\n",
    "    total_latency = model_latency + trigger_latency\n",
    "    return trigger_reply, total_latency, trigger_usage, model_latency, classified_intent\n",
    "\n",
    "\n",
    "print(\"✅ Helper functions defined: call_combined, call_trigger_only, call_intent_then_trigger, call_model_then_trigger.\")\n",
]

cell4["source"] = cell4_source
cell4["outputs"] = [
    {
        "name": "stdout",
        "output_type": "stream",
        "text": [
            "✅ Helper functions defined: call_combined, call_trigger_only, call_intent_then_trigger, call_model_then_trigger.\n"
        ]
    }
]
cell4["execution_count"] = None


# ══════════════════════════════════════════════════════════════════════════════
# CELL 5  –  benchmark runner: add 4th approach
# ══════════════════════════════════════════════════════════════════════════════
idx5, cell5 = find_cell("e1f2a3b4")
assert idx5 is not None, "Cell 5 not found"

cell5_source = [
    "results: list[dict] = []\n",
    "\n",
    "total = len(selected)\n",
    "for i, example in enumerate(selected, 1):\n",
    "    text   = example[\"text\"]\n",
    "    intent = example[\"intent\"]\n",
    "\n",
    "    label = (f\"[{i:02d}/{total}] intent={intent:<20} | query={text[:55]}...\"\n",
    "             if len(text) > 55\n",
    "             else f\"[{i:02d}/{total}] intent={intent:<20} | query={text}\")\n",
    "    print(label)\n",
    "\n",
    "    # ── 1. Combined prompt ──────────────────────────────────────────\n",
    "    try:\n",
    "        combined_reply, combined_latency, combined_usage = call_combined(text)\n",
    "        print(f\"  combined            → {combined_latency:.3f}s | {combined_reply}\")\n",
    "    except Exception as e:\n",
    "        combined_reply, combined_latency, combined_usage = {}, None, None\n",
    "        print(f\"  combined            → ERROR: {e}\")\n",
    "\n",
    "    # ── 2. Trigger-only (pre-known intent) ─────────────────────────\n",
    "    try:\n",
    "        trigger_reply, trigger_latency, trigger_usage = call_trigger_only(text, intent)\n",
    "        print(f\"  trigger-only        → {trigger_latency:.3f}s | {trigger_reply}\")\n",
    "    except Exception as e:\n",
    "        trigger_reply, trigger_latency, trigger_usage = {}, None, None\n",
    "        print(f\"  trigger-only        → ERROR: {e}\")\n",
    "\n",
    "    # ── 3. Intent-then-trigger (two-step pipeline, LLM intent) ──────\n",
    "    try:\n",
    "        itt_reply, itt_latency, itt_usage, itt_intent_lat, itt_intent = call_intent_then_trigger(text)\n",
    "        print(f\"  intent+trigger (2step) → {itt_latency:.3f}s (intent={itt_intent_lat:.3f}s, intent='{itt_intent}') | {itt_reply}\")\n",
    "    except Exception as e:\n",
    "        itt_reply, itt_latency, itt_usage, itt_intent_lat, itt_intent = {}, None, None, None, None\n",
    "        print(f\"  intent+trigger (2step) → ERROR: {e}\")\n",
    "\n",
    "    # ── 4. Model-then-trigger (local model intent + LLM trigger) ────\n",
    "    try:\n",
    "        mtt_reply, mtt_latency, mtt_usage, mtt_model_lat, mtt_intent = call_model_then_trigger(text)\n",
    "        print(f\"  model+trigger  (2step) → {mtt_latency:.3f}s (model={mtt_model_lat:.3f}s, intent='{mtt_intent}') | {mtt_reply}\")\n",
    "    except Exception as e:\n",
    "        mtt_reply, mtt_latency, mtt_usage, mtt_model_lat, mtt_intent = {}, None, None, None, None\n",
    "        print(f\"  model+trigger  (2step) → ERROR: {e}\")\n",
    "\n",
    "    results.append({\n",
    "        \"intent\":          intent,\n",
    "        \"text\":            text,\n",
    "        # function 1 – combined\n",
    "        \"combined_latency\": combined_latency,\n",
    "        \"combined_reply\":   combined_reply,\n",
    "        \"combined_usage\":   combined_usage,\n",
    "        # function 2 – trigger-only\n",
    "        \"trigger_latency\":  trigger_latency,\n",
    "        \"trigger_reply\":    trigger_reply,\n",
    "        \"trigger_usage\":    trigger_usage,\n",
    "        # function 3 – intent-then-trigger (LLM intent)\n",
    "        \"itt_latency\":      itt_latency,\n",
    "        \"itt_reply\":        itt_reply,\n",
    "        \"itt_usage\":        itt_usage,\n",
    "        \"itt_intent_lat\":   itt_intent_lat,\n",
    "        \"itt_intent\":       itt_intent,\n",
    "        # function 4 – model-then-trigger (local model intent)\n",
    "        \"mtt_latency\":      mtt_latency,\n",
    "        \"mtt_reply\":        mtt_reply,\n",
    "        \"mtt_usage\":        mtt_usage,\n",
    "        \"mtt_model_lat\":    mtt_model_lat,\n",
    "        \"mtt_intent\":       mtt_intent,\n",
    "    })\n",
    "\n",
    "print(f\"\\n✅ Benchmark complete — {len(results)} examples processed.\")\n",
]

cell5["source"] = cell5_source
cell5["outputs"] = []
cell5["execution_count"] = None


# ══════════════════════════════════════════════════════════════════════════════
# CELL 6  –  summary table: add model+trigger column
# ══════════════════════════════════════════════════════════════════════════════
idx6, cell6 = find_cell("a3b4c5d6")
assert idx6 is not None, "Cell 6 not found"

cell6_source = [
    "from collections import defaultdict\n",
    "\n",
    "INTENTS = [\"free_gift\", \"buy_x_get_y\", \"tiered_discount\", \"unsupported\"]\n",
    "\n",
    "combined_by_intent = defaultdict(list)\n",
    "trigger_by_intent  = defaultdict(list)\n",
    "itt_by_intent      = defaultdict(list)\n",
    "mtt_by_intent      = defaultdict(list)\n",
    "\n",
    "for r in results:\n",
    "    if r[\"combined_latency\"] is not None:\n",
    "        combined_by_intent[r[\"intent\"]].append(r[\"combined_latency\"])\n",
    "    if r[\"trigger_latency\"] is not None:\n",
    "        trigger_by_intent[r[\"intent\"]].append(r[\"trigger_latency\"])\n",
    "    if r[\"itt_latency\"] is not None:\n",
    "        itt_by_intent[r[\"intent\"]].append(r[\"itt_latency\"])\n",
    "    if r[\"mtt_latency\"] is not None:\n",
    "        mtt_by_intent[r[\"intent\"]].append(r[\"mtt_latency\"])\n",
    "\n",
    "# ── Accuracy: how often does each function's classified intent match ground truth? ──\n",
    "def intent_from_reply(reply, key=\"feature\"):\n",
    "    \"\"\"Extract the intent label from a reply dict.\"\"\"\n",
    "    return reply.get(key) if isinstance(reply, dict) else None\n",
    "\n",
    "combined_correct = sum(\n",
    "    1 for r in results\n",
    "    if intent_from_reply(r[\"combined_reply\"]) == r[\"intent\"]\n",
    ")\n",
    "trigger_correct = sum(\n",
    "    1 for r in results\n",
    "    if r[\"intent\"] == \"unsupported\"   # trigger-only skips unsupported → counts as correct\n",
    "    or intent_from_reply(r[\"trigger_reply\"]) is not None\n",
    ")\n",
    "itt_correct = sum(\n",
    "    1 for r in results\n",
    "    if r.get(\"itt_intent\") == r[\"intent\"]\n",
    ")\n",
    "mtt_correct = sum(\n",
    "    1 for r in results\n",
    "    if r.get(\"mtt_intent\") == r[\"intent\"]\n",
    ")\n",
    "\n",
    "n = len(results)\n",
    "print(f\"\\n{'='*76}\")\n",
    "print(\"  ACCURACY SUMMARY\")\n",
    "print(f\"{'='*76}\")\n",
    "print(f\"  combined              : {combined_correct}/{n} = {combined_correct/n*100:.1f}%\")\n",
    "print(f\"  trigger-only          : {trigger_correct}/{n} = {trigger_correct/n*100:.1f}%  (intent pre-known)\")\n",
    "print(f\"  intent-then-trigger   : {itt_correct}/{n} = {itt_correct/n*100:.1f}%\")\n",
    "print(f\"  model-then-trigger    : {mtt_correct}/{n} = {mtt_correct/n*100:.1f}%  (local model intent)\")\n",
    "print(f\"{'='*76}\")\n",
    "\n",
    "# ── Latency table ──────────────────────────────────────────────────────────────\n",
    "hdr = (f\"{'Intent':<22} {'Combined(s)':>12} {'TrigOnly(s)':>12} {'IttTotal(s)':>12} {'MttTotal(s)':>12}\")\n",
    "print(\"\\n\" + hdr)\n",
    "print(\"-\" * len(hdr))\n",
    "for intent in INTENTS:\n",
    "    c = np.mean(combined_by_intent.get(intent, [0]))\n",
    "    t = np.mean(trigger_by_intent.get(intent,  [0]))\n",
    "    i = np.mean(itt_by_intent.get(intent,      [0]))\n",
    "    m = np.mean(mtt_by_intent.get(intent,      [0]))\n",
    "    print(f\"{intent:<22} {c:>12.3f} {t:>12.3f} {i:>12.3f} {m:>12.3f}\")\n",
    "print(\"-\" * len(hdr))\n",
    "all_c = [r[\"combined_latency\"] for r in results if r[\"combined_latency\"] is not None]\n",
    "all_t = [r[\"trigger_latency\"]  for r in results if r[\"trigger_latency\"]  is not None]\n",
    "all_i = [r[\"itt_latency\"]      for r in results if r[\"itt_latency\"]      is not None]\n",
    "all_m = [r[\"mtt_latency\"]      for r in results if r[\"mtt_latency\"]      is not None]\n",
    "print(f\"{'OVERALL MEAN':<22} {np.mean(all_c):>12.3f} {np.mean(all_t):>12.3f} {np.mean(all_i):>12.3f} {np.mean(all_m):>12.3f}\")\n",
]

cell6["source"] = cell6_source
cell6["outputs"] = []
cell6["execution_count"] = None


# ══════════════════════════════════════════════════════════════════════════════
# CELL 9  –  cost analysis: add model+trigger (trigger API only)
# ══════════════════════════════════════════════════════════════════════════════
idx9, cell9 = find_cell("v5u6t7s8")
assert idx9 is not None, "Cell 9 not found"

cell9_source = [
    "PRICE_PER_1M_INPUT  = 5.00\n",
    "PRICE_PER_1M_OUTPUT = 15.00\n",
    "\n",
    "def compute_cost(usage: dict | None) -> float:\n",
    "    if not usage:\n",
    "        return 0.0\n",
    "    return (\n",
    "        (usage[\"prompt_tokens\"]     / 1_000_000) * PRICE_PER_1M_INPUT\n",
    "        + (usage[\"completion_tokens\"] / 1_000_000) * PRICE_PER_1M_OUTPUT\n",
    "    )\n",
    "\n",
    "tot_c   = {\"p\": 0, \"c\": 0, \"t\": 0, \"cost\": 0.0}\n",
    "tot_tr  = {\"p\": 0, \"c\": 0, \"t\": 0, \"cost\": 0.0}\n",
    "tot_itt = {\"p\": 0, \"c\": 0, \"t\": 0, \"cost\": 0.0}\n",
    "tot_mtt = {\"p\": 0, \"c\": 0, \"t\": 0, \"cost\": 0.0}\n",
    "rows = []\n",
    "\n",
    "for r in results:\n",
    "    cu  = r.get(\"combined_usage\") or {}\n",
    "    tu  = r.get(\"trigger_usage\")  or {}\n",
    "    iu  = r.get(\"itt_usage\")      or {}\n",
    "    mu  = r.get(\"mtt_usage\")      or {}   # only the trigger API call; model is free\n",
    "    c_cost   = compute_cost(r.get(\"combined_usage\"))\n",
    "    t_cost   = compute_cost(r.get(\"trigger_usage\"))\n",
    "    itt_cost = compute_cost(r.get(\"itt_usage\"))\n",
    "    mtt_cost = compute_cost(r.get(\"mtt_usage\"))  # local model has no token cost\n",
    "\n",
    "    for tot, u, cost in [(tot_c, cu, c_cost), (tot_tr, tu, t_cost), (tot_itt, iu, itt_cost), (tot_mtt, mu, mtt_cost)]:\n",
    "        tot[\"p\"]    += u.get(\"prompt_tokens\", 0)\n",
    "        tot[\"c\"]    += u.get(\"completion_tokens\", 0)\n",
    "        tot[\"t\"]    += u.get(\"total_tokens\", 0)\n",
    "        tot[\"cost\"] += cost\n",
    "\n",
    "    rows.append({\n",
    "        \"intent\": r[\"intent\"],\n",
    "        \"query\":  (r[\"text\"][:38] + \"…\") if len(r[\"text\"]) > 39 else r[\"text\"],\n",
    "        \"cp\": cu.get(\"prompt_tokens\",0),  \"cc\": cu.get(\"completion_tokens\",0),\n",
    "        \"ct\": cu.get(\"total_tokens\",0),   \"c$\": c_cost,\n",
    "        \"tp\": tu.get(\"prompt_tokens\",0),  \"tc\": tu.get(\"completion_tokens\",0),\n",
    "        \"tt\": tu.get(\"total_tokens\",0),   \"t$\": t_cost,\n",
    "        \"ip\": iu.get(\"prompt_tokens\",0),  \"ic\": iu.get(\"completion_tokens\",0),\n",
    "        \"it\": iu.get(\"total_tokens\",0),   \"i$\": itt_cost,\n",
    "        \"mp\": mu.get(\"prompt_tokens\",0),  \"mc\": mu.get(\"completion_tokens\",0),\n",
    "        \"mt\": mu.get(\"total_tokens\",0),   \"m$\": mtt_cost,\n",
    "    })\n",
    "\n",
    "HDR = (f\"{'#':<3} {'Intent':<18} {'Query':<40} \"\n",
    "       f\"{'C-In':>6}{'C-Out':>6}{'C-$':>9} \"\n",
    "       f\"{'T-In':>6}{'T-Out':>6}{'T-$':>9} \"\n",
    "       f\"{'I-In':>6}{'I-Out':>6}{'I-$':>9} \"\n",
    "       f\"{'M-In':>6}{'M-Out':>6}{'M-$':>9}\")\n",
    "print(HDR)\n",
    "print(\"-\" * len(HDR))\n",
    "for i, row in enumerate(rows, 1):\n",
    "    print(f\"{i:<3} {row['intent']:<18} {row['query']:<40} \"\n",
    "          f\"{row['cp']:>6}{row['cc']:>6}{row['c$']:>9.6f} \"\n",
    "          f\"{row['tp']:>6}{row['tc']:>6}{row['t$']:>9.6f} \"\n",
    "          f\"{row['ip']:>6}{row['ic']:>6}{row['i$']:>9.6f} \"\n",
    "          f\"{row['mp']:>6}{row['mc']:>6}{row['m$']:>9.6f}\")\n",
    "print(\"-\" * len(HDR))\n",
    "print(f\"{'TOT':<3} {'':<18} {'':<40} \"\n",
    "      f\"{tot_c['p']:>6}{tot_c['c']:>6}{tot_c['cost']:>9.6f} \"\n",
    "      f\"{tot_tr['p']:>6}{tot_tr['c']:>6}{tot_tr['cost']:>9.6f} \"\n",
    "      f\"{tot_itt['p']:>6}{tot_itt['c']:>6}{tot_itt['cost']:>9.6f} \"\n",
    "      f\"{tot_mtt['p']:>6}{tot_mtt['c']:>6}{tot_mtt['cost']:>9.6f}\")\n",
    "\n",
    "print(f\"\\n{'='*66}\")\n",
    "print(\"  COST SUMMARY\")\n",
    "print(f\"{'='*66}\")\n",
    "print(f\"  combined              — tokens: {tot_c['t']:>7} | cost: ${tot_c['cost']:.6f}\")\n",
    "print(f\"  trigger-only          — tokens: {tot_tr['t']:>7} | cost: ${tot_tr['cost']:.6f}\")\n",
    "print(f\"  intent-then-trigger   — tokens: {tot_itt['t']:>7} | cost: ${tot_itt['cost']:.6f}\")\n",
    "print(f\"  model-then-trigger    — tokens: {tot_mtt['t']:>7} | cost: ${tot_mtt['cost']:.6f}  (model inference is free)\")\n",
    "print(f\"{'='*66}\")\n",
]

cell9["source"] = cell9_source
cell9["outputs"] = []
cell9["execution_count"] = None


# ══════════════════════════════════════════════════════════════════════════════
# CELL 7  –  grouped bar chart: add 4th bar (model+trigger)
# ══════════════════════════════════════════════════════════════════════════════
idx7, cell7 = find_cell("c5d6e7f8")
assert idx7 is not None, "Cell 7 not found"

cell7_source = [
    "fig, ax = plt.subplots(figsize=(14, 6))\n",
    "\n",
    "x         = np.arange(len(INTENTS))\n",
    "bw        = 0.20\n",
    "COLOR_C   = \"#4C6EF5\"   # indigo  – combined\n",
    "COLOR_T   = \"#2EC4B6\"   # teal    – trigger-only\n",
    "COLOR_I   = \"#FF6B6B\"   # coral   – intent-then-trigger\n",
    "COLOR_M   = \"#F7B731\"   # amber   – model-then-trigger\n",
    "\n",
    "mc = [np.mean(combined_by_intent.get(i, [0])) for i in INTENTS]\n",
    "mt = [np.mean(trigger_by_intent.get(i,  [0])) for i in INTENTS]\n",
    "mi = [np.mean(itt_by_intent.get(i,      [0])) for i in INTENTS]\n",
    "mm = [np.mean(mtt_by_intent.get(i,      [0])) for i in INTENTS]\n",
    "\n",
    "offset = 1.5 * bw\n",
    "b1 = ax.bar(x - offset,        mc, bw, label=\"Combined (intent+trigger)\",    color=COLOR_C, alpha=0.88, edgecolor=\"white\")\n",
    "b2 = ax.bar(x - offset + bw,   mt, bw, label=\"Trigger-only (pre-known)\",     color=COLOR_T, alpha=0.88, edgecolor=\"white\")\n",
    "b3 = ax.bar(x - offset + 2*bw, mi, bw, label=\"Intent+trigger (LLM 2-step)\",  color=COLOR_I, alpha=0.88, edgecolor=\"white\")\n",
    "b4 = ax.bar(x - offset + 3*bw, mm, bw, label=\"Model+trigger (model 2-step)\", color=COLOR_M, alpha=0.88, edgecolor=\"white\")\n",
    "\n",
    "for bars, color in [(b1, COLOR_C), (b2, COLOR_T), (b3, COLOR_I), (b4, COLOR_M)]:\n",
    "    for bar in bars:\n",
    "        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,\n",
    "                f\"{bar.get_height():.2f}s\", ha=\"center\", va=\"bottom\",\n",
    "                fontsize=7, color=color, fontweight=\"bold\")\n",
    "\n",
    "ax.set_xticks(x)\n",
    "ax.set_xticklabels(INTENTS, fontsize=11)\n",
    "ax.set_ylabel(\"Mean Latency (s)\", fontsize=12)\n",
    "ax.set_title(\"Mean LLM Latency per Intent\\n(4 Approaches Compared)\", fontsize=14, fontweight=\"bold\", pad=14)\n",
    "ax.legend(fontsize=9, framealpha=0.8)\n",
    "ax.set_ylim(0, max(max(mc), max(mt), max(mi), max(mm)) * 1.4)\n",
    "ax.spines[[\"top\",\"right\"]].set_visible(False)\n",
    "ax.yaxis.grid(True, linestyle=\"--\", alpha=0.4)\n",
    "ax.set_axisbelow(True)\n",
    "plt.tight_layout()\n",
    "plt.savefig(\"latency_bar_chart_4way.png\", dpi=150, bbox_inches=\"tight\")\n",
    "plt.show()\n",
    "print(\"📊 4-way bar chart saved.\")\n",
]

cell7["source"] = cell7_source
cell7["outputs"] = []
cell7["execution_count"] = None


# ══════════════════════════════════════════════════════════════════════════════
# CELL 8  –  box plot: add 4th group (model+trigger)
# ══════════════════════════════════════════════════════════════════════════════
idx8, cell8 = find_cell("e7f8a9b0")
assert idx8 is not None, "Cell 8 not found"

cell8_source = [
    "fig, axes = plt.subplots(1, len(INTENTS), figsize=(16, 6), sharey=True)\n",
    "\n",
    "COLOR_C = \"#4C6EF5\"\n",
    "COLOR_T = \"#2EC4B6\"\n",
    "COLOR_I = \"#FF6B6B\"\n",
    "COLOR_M = \"#F7B731\"\n",
    "\n",
    "LABELS = [\"Combined\", \"Trig-only\", \"ITT\", \"MTT\"]\n",
    "COLORS = [COLOR_C, COLOR_T, COLOR_I, COLOR_M]\n",
    "\n",
    "for ax, intent in zip(axes, INTENTS):\n",
    "    data = [\n",
    "        combined_by_intent.get(intent, []),\n",
    "        trigger_by_intent.get(intent,  []),\n",
    "        itt_by_intent.get(intent,      []),\n",
    "        mtt_by_intent.get(intent,      []),\n",
    "    ]\n",
    "    bp = ax.boxplot(\n",
    "        [d if d else [0] for d in data],\n",
    "        patch_artist=True,\n",
    "        widths=0.5,\n",
    "        medianprops={\"color\": \"white\", \"linewidth\": 2},\n",
    "    )\n",
    "    for patch, color in zip(bp[\"boxes\"], COLORS):\n",
    "        patch.set_facecolor(color)\n",
    "        patch.set_alpha(0.85)\n",
    "    for whisker in bp[\"whiskers\"]:\n",
    "        whisker.set(color=\"#555\", linewidth=1.2)\n",
    "    for cap in bp[\"caps\"]:\n",
    "        cap.set(color=\"#555\", linewidth=1.2)\n",
    "    for flier in bp[\"fliers\"]:\n",
    "        flier.set(marker=\"o\", color=\"#555\", alpha=0.5)\n",
    "\n",
    "    ax.set_title(intent.replace(\"_\", \"\\n\"), fontsize=10, fontweight=\"bold\")\n",
    "    ax.set_xticks([1, 2, 3, 4])\n",
    "    ax.set_xticklabels(LABELS, fontsize=8)\n",
    "    ax.yaxis.grid(True, linestyle=\"--\", alpha=0.4)\n",
    "    ax.set_axisbelow(True)\n",
    "    ax.spines[[\"top\", \"right\"]].set_visible(False)\n",
    "\n",
    "axes[0].set_ylabel(\"Latency (s)\", fontsize=12)\n",
    "fig.suptitle(\"Latency Distribution per Intent\\n(4 Approaches: Combined | Trig-only | LLM 2-step | Model 2-step)\",\n",
    "             fontsize=13, fontweight=\"bold\", y=1.02)\n",
    "\n",
    "legend_patches = [\n",
    "    plt.matplotlib.patches.Patch(color=COLOR_C, label=\"Combined (intent+trigger)\"),\n",
    "    plt.matplotlib.patches.Patch(color=COLOR_T, label=\"Trigger-only (pre-known)\"),\n",
    "    plt.matplotlib.patches.Patch(color=COLOR_I, label=\"Intent+trigger (LLM 2-step)\"),\n",
    "    plt.matplotlib.patches.Patch(color=COLOR_M, label=\"Model+trigger (model 2-step)\"),\n",
    "]\n",
    "fig.legend(handles=legend_patches, loc=\"lower center\", ncol=4, fontsize=9,\n",
    "           framealpha=0.8, bbox_to_anchor=(0.5, -0.08))\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.savefig(\"latency_box_plot_4way.png\", dpi=150, bbox_inches=\"tight\")\n",
    "plt.show()\n",
    "print(\"📊 4-way box plot saved.\")\n",
]

cell8["source"] = cell8_source
cell8["outputs"] = []
cell8["execution_count"] = None


# ══════════════════════════════════════════════════════════════════════════════
# UPDATE markdown headers
# ══════════════════════════════════════════════════════════════════════════════
for c in cells:
    if c.get("cell_type") == "markdown":
        src_text = "".join(c.get("source", []))
        if "Cell 4 — Benchmark Runner" in src_text or "Cell 4 \u2014 Benchmark Runner" in src_text:
            c["source"] = [
                "## Cell 4 — Benchmark Runner\n",
                "\n",
                "Defines four helper functions:\n",
                "1. **`call_combined`** — single LLM call for intent + trigger.\n",
                "2. **`call_trigger_only`** — trigger detection with pre-known intent.\n",
                "3. **`call_intent_then_trigger`** — LLM intent → LLM trigger (2-step).\n",
                "4. **`call_model_then_trigger`** — **local trained model** intent → LLM trigger (2-step).\n",
                "   - `free_gift` / `buy_x_get_y` → `TRIGGER_ONLY_CLASSIFICATION_PROMPT`\n",
                "   - `tiered_discount` → `TIEREED_DISCOUNT_TRIGGER_ONLY_CLASSIFICATION_PROMPT`\n",
                "   - `unsupported` → returns `{}` immediately (no API call)."
            ]
        if "Cell 7 — Visualisation: Grouped Bar Chart" in src_text:
            c["source"] = ["## Cell 7 — Visualisation: Grouped Bar Chart (Mean Latency per Intent, 4 Approaches)"]
        if "Cell 8 — Visualisation: Box Plot" in src_text:
            c["source"] = ["## Cell 8 — Visualisation: Box Plot (Latency Distribution per Intent, 4 Approaches)"]


# ══════════════════════════════════════════════════════════════════════════════
# Save
# ══════════════════════════════════════════════════════════════════════════════
with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("✅ Notebook patched successfully.")
