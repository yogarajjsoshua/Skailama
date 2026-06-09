# 🛍️ Skailama — Shopify Mini Promotion Agent

A **LangGraph-powered FastAPI service** that parses natural-language merchant promotion requests, classifies their intent, extracts structured trigger/reward data, validates the result, and returns a machine-readable promotion object ready for a Shopify app.

The repo also contains a standalone **Intent Classifier** — a fine-tuned DeBERTa-v3-small model — that was built as a research baseline to explore whether a lightweight ML model could replace LLM calls for the classification step.
--- NOTE  : the model that had beeen trained and used for intent_classificatio WAS good but after i realized the my data definition propmpt itself was wrong, and was clarified only after the 1st call i was asked to set this approach to the side and go ahead only with LLM and hence the approach is promising and will defintely increase speed and be cost benifical and accurate, hence why i even resorted to even try this approach --- 

---

## 📐 System Architecture

Detailed system layout and dataflow diagrams can be found in the [Architecture](file:///Users/yogarajkomati/Documents/Interview/Skailama/Architecture/) folder:
* 🗺️ [High-Level Design (HLD)](file:///Users/yogarajkomati/Documents/Interview/Skailama/Architecture/HLD.md): High-level visual model of components, data flow, and third-party API bindings.
* 📝 [Low-Level Design (LLD)](file:///Users/yogarajkomati/Documents/Interview/Skailama/Architecture/LLD.md): Structural details of classes, models, schemas, and MongoDB collection details.
* 🔁 [LangGraph Flowchart](file:///Users/yogarajkomati/Documents/Interview/Skailama/Architecture/LangGraph_Flow.md): Detailed state diagram illustrating LangGraph nodes, conditional edges, loopbacks, and interrupts.

---

## 📁 Project Structure

```
Skailama/
├── app/                        # FastAPI + LangGraph service
│   ├── main.py                 # App entry-point, startup health-checks, API routes
│   ├── graph.py                # LangGraph pipeline definition
│   ├── nodes.py                # Individual pipeline node functions
│   ├── state.py                # PromotionState TypedDict
│   ├── llm.py                  # Azure OpenAI wrapper (LangSmith-traceable)
│   ├── pormpts.py              # All system prompts used by the pipeline
│   └── models.py               # Pydantic request/response models
│
├── Architecture/               # System architecture and workflow diagrams
│   ├── HLD.md                  # High-Level Design document
│   ├── LLD.md                  # Low-Level Design document
│   └── LangGraph_Flow.md       # LangGraph state machine workflow details
│
├── intentClassifier/           # Standalone ML intent classifier (research/baseline)
│   ├── ModelTrain.ipynb        # Interactive training notebook
│   ├── train.py                # Fine-tune DeBERTa-v3-small on promotion data
│   ├── evaluvate.py            # Evaluate trained model on held-out test set
│   ├── classify.py             # CLI tool to classify a single promotion text
│   ├── generate_data.py        # Synthetic training data generation via GPT-4o
│   ├── generate_and_append.py  # Incremental data generation & deduplication
│   ├── validate_generated_data.py  # LLM-based quality validation of generated data
│   ├── dedup_and_validate.py   # Remove duplicates and bad examples
│   ├── classificationData.py   # Full raw training corpus (hardcoded)
│   ├── generated_data.py       # GPT-4o generated examples (raw)
│   ├── generated_data_validated.py # GPT-4o generated examples (validated)
│   └── model/best/             # Saved fine-tuned model weights (after training)
│
├── data/                       # Train / Val / Test CSV splits
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
│
├── chat.py                     # CLI chatbot client that talks to the FastAPI backend
├── requirements.txt            # All Python dependencies
└── .env                        # Environment secrets (not committed)
```

---

## ⚙️ Requirements

### Python

Python **3.10 or 3.11** is recommended. Python 3.12+ may have minor compatibility issues with some pinned packages (e.g., `torch==2.2.2`).

### Hardware (for the Intent Classifier only)

| Device | Supported? | Notes |
|--------|------------|-------|
| NVIDIA GPU (CUDA) | ✅ Best | Fastest training & inference |
| Apple Silicon (MPS) | ✅ Good | Supported via PyTorch MPS backend |
| CPU | ✅ Works | Slow for training, fine for inference |

> The FastAPI service itself has **no GPU requirement** — it relies entirely on Azure OpenAI API calls.

---

## 🔧 Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd Skailama
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # macOS / Linux
# venv\Scripts\activate       # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `torch==2.2.2` is pinned. If you are on a newer Mac (Apple Silicon), PyTorch will automatically use the MPS backend. No extra steps needed.

### 4. Configure environment variables

Create a `.env` file in the project root:

```ini
# ── Azure OpenAI ──────────────────────────────────────────────────────────────
OPENAI_API_4_KEY=<your-azure-openai-api-key>
OPENAI_4_BASE_URL=<your-azure-openai-endpoint>
OPENAI_API_4_VERSION=2025-01-01-preview
OPEN_API_4_ENGINE=gpt-4o

# ── LangSmith Tracing (optional but recommended) ─────────────────────────────
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=<your-langsmith-api-key>
LANGCHAIN_PROJECT=Skailama
```

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_4_KEY` | ✅ Yes | Azure OpenAI API key |
| `OPENAI_4_BASE_URL` | ✅ Yes | Full Azure endpoint URL |
| `OPENAI_API_4_VERSION` | ✅ Yes | API version string |
| `OPEN_API_4_ENGINE` | ✅ Yes | Deployment / model name (e.g. `gpt-4o`) |
| `LANGCHAIN_TRACING_V2` | ⬜ Optional | Set to `true` to enable LangSmith tracing |
| `LANGCHAIN_ENDPOINT` | ⬜ Optional | LangSmith API endpoint |
| `LANGCHAIN_API_KEY` | ⬜ Optional | Your LangSmith API key (`ls__...`) |
| `LANGCHAIN_PROJECT` | ⬜ Optional | LangSmith project name for grouping traces |

> LangSmith is **non-blocking** — if tracing fails, the server continues running normally.

---

## 🚀 Running the Service

### Start the FastAPI server

```bash
uvicorn app.main:app --reload
```

The server starts at **http://127.0.0.1:8000**.

On startup, the server:
1. Pings Azure OpenAI to verify the API key and deployment name.
2. Verifies LangSmith connectivity (non-fatal if it fails).

### Start the CLI chatbot client

In a separate terminal:

```bash
python chat.py
```

Type a promotion description in natural language. Type `exit` to quit.

---

## 🌐 API Endpoints

All endpoints accept `POST` with a JSON body: `{"message": "<promotion text>"}`

| Endpoint | Description |
|----------|-------------|
| `GET /` | Health check — returns `{"message": "Hello World"}` |
| `POST /chat` | Raw echo — passes message directly to GPT-4o |
| `POST /chat/intent-classifier/llm` | LLM-only intent classification. Returns `{"reply": "<intent>"}` |
| `POST /chat/intent-classifier/triggers/llm` | LLM intent + trigger extraction. Returns full structured JSON |
| `POST /chat/mini-promotion-agent` | **Full pipeline** — runs the LangGraph graph and returns a complete structured promotion object |

### Example: Mini Promotion Agent

**Request**

```bash
curl -X POST http://127.0.0.1:8000/chat/mini-promotion-agent \
  -H "Content-Type: application/json" \
  -d '{"message": "Spend $100 get 10% off, spend $200 get 20% off"}'
```

**Response**

```json
{
  "reply": {
    "feature": "tiered_discount",
    "tiers": [
      { "trigger": { "type": "cart_subtotal", "operator": ">=", "value": 100 }, "reward": { "type": "percentage_off", "value": 10 } },
      { "trigger": { "type": "cart_subtotal", "operator": ">=", "value": 200 }, "reward": { "type": "percentage_off", "value": 20 } }
    ],
    "tier_behavior": "best_tier_only",
    "customer_eligibility": [],
    "status": "supported",
    "blockers": []
  }
}
```

---

## 🔁 LangGraph Pipeline (Mini Promotion Agent)

The `/chat/mini-promotion-agent` endpoint runs a 4-node LangGraph pipeline:

```
[intent_classification] → [trigger_detection] → [state_assembly] → [validation]
```

| Node | What it does |
|------|-------------|
| `intent_classification` | Classifies the promotion into `free_gift`, `buy_x_get_y`, `tiered_discount`, or `unsupported` |
| `trigger_detection` | Extracts structured trigger/reward tiers (e.g. spend thresholds, quantity thresholds) |
| `state_assembly` | Assembles metadata fields: `tier_behavior`, `customer_eligibility`, `status`, `blockers` |
| `validation` | Sends the classified intent back to the LLM as a strict validator; sets `status = unsupported` if classification is wrong |

All nodes are individually traced in LangSmith via the `@traceable` decorator.

---

## 🤖 Why is the Intent Classifier Here?

The `intentClassifier/` directory contains a **research baseline** built to answer a specific engineering question:

> *Can a small, locally-running fine-tuned model (DeBERTa-v3-small) replace or augment the LLM for the intent classification step — with lower latency and cost?*

### What was built

A **DeBERTa-v3-small** model was fine-tuned (via `train.py`) on ~3,000 synthetic promotion examples generated by GPT-4o (`generate_data.py`). The model classifies any promotion text into one of four classes:

| Class | Meaning |
|-------|---------|
| `free_gift` | Spend/quantity threshold → free physical item |
| `buy_x_get_y` | Buy X product/quantity → get Y item discounted or free |
| `tiered_discount` | 2+ spend/quantity tiers each unlocking a bigger discount |
| `unsupported` | Vague, impossible, out-of-scope, or ambiguous requests |

### What was evaluated

Four classification strategies were benchmarked against a labelled test set:

| Strategy | Description |
|----------|-------------|
| `combined` | One LLM call classifies intent AND extracts triggers simultaneously |
| `trigger-only` | LLM extracts triggers only (intent is pre-supplied) |
| `intent-then-trigger` | Two sequential LLM calls: first classify, then extract triggers |
| `model-then-trigger` | Fine-tuned DeBERTa classifies intent, then LLM extracts triggers |

### Findings

- The fine-tuned model achieved competitive accuracy for well-formed promotion texts.
- `trigger-only` did **not** reach 100% despite knowing the intent in advance — confirming that trigger extraction is non-trivial even with a known class label.
- The `model-then-trigger` approach had the **lowest latency** since no LLM call is needed for the classification step.
- The production pipeline currently uses `intent-then-trigger` (two LLM calls) for maximum accuracy, with the fine-tuned model available as a lower-latency fallback.

### How to use the classifier CLI

After training (see below), you can classify any text from the command line:

```bash
cd intentClassifier

# Basic classification
python classify.py "Spend $100 and get a free tote bag"

# Verbose (show all class scores)
python classify.py "Buy 2 shirts get 1 cap free" --verbose

# JSON output
python classify.py "Buy 2 get 10%, buy 4 get 20%" --json
```

---

## 🏋️ Training the Intent Classifier (Optional)

> Skip this section if you only want to run the FastAPI service.

### 1. Generate synthetic training data

```bash
cd intentClassifier
python generate_data.py
```

This calls GPT-4o to produce labelled promotion examples for all four classes.

### 2. Validate and deduplicate

```bash
python validate_generated_data.py
python dedup_and_validate.py
```

### 3. Prepare CSV splits

Ensure `data/train.csv`, `data/val.csv`, and `data/test.csv` exist with columns `text` and `label`.

### 4. Train the model

```bash
python train.py
```

Training config (in `train.py`):
- **Base model:** `microsoft/deberta-v3-small`
- **Epochs:** 12 (with early stopping, patience = 3)
- **Batch size:** 16
- **Learning rate:** 2e-5
- **Max token length:** 128

The best checkpoint is saved to `intentClassifier/model/best/`.

### 5. Evaluate on test set

```bash
python evaluvate.py
```

Prints a full classification report and confusion matrix.

---

## 🔍 LangSmith Tracing

When `LANGCHAIN_TRACING_V2=true` is set, every LLM call and every LangGraph node is automatically traced to your LangSmith project. This lets you:

- See the exact system prompt and user message sent to the LLM at each step.
- Inspect intermediate state between nodes.
- Measure latency per node and per LLM call.
- Debug misclassifications by replaying individual traces.

Navigate to [smith.langchain.com](https://smith.langchain.com) → your project to view traces.

---

## 🤝 Contributing

1. Fork the repo and create a feature branch.
2. Make your changes with descriptive commit messages.
3. Open a pull request describing what you changed and why.

---

## 📄 License

This project is for demonstration and interview purposes.
