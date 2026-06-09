"""
Generate all Architecture diagrams:
  1. HLD.png  - High-Level Design
  2. LLD.png  - Low-Level Design
  3. LangGraph_Flow.png - LangGraph execution flow
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────────────────────
# Shared theme
# ─────────────────────────────────────────────────────────────────────────────
BG        = "#0D1117"
CARD_DARK = "#161B22"
CARD_MED  = "#21262D"
CARD_LITE = "#30363D"
BORDER    = "#3D4450"
ACCENT1   = "#58A6FF"  # blue
ACCENT2   = "#3FB950"  # green
ACCENT3   = "#D29922"  # yellow
ACCENT4   = "#F78166"  # red/orange
ACCENT5   = "#BC8CFF"  # purple
ACCENT6   = "#39D0D8"  # cyan
WHITE     = "#E6EDF3"
MUTED     = "#8B949E"
ARROW_CLR = "#58A6FF"


def _fig(w, h):
    fig = plt.figure(figsize=(w, h), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    ax.set_facecolor(BG)
    return fig, ax


def rbox(ax, x, y, w, h, facecolor, edgecolor=BORDER, radius=0.3, alpha=1.0, lw=1.5, zorder=2):
    """Rounded rectangle."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=lw,
        edgecolor=edgecolor,
        facecolor=facecolor,
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_patch(box)
    return box


def txt(ax, x, y, s, color=WHITE, size=9, bold=False, ha="center", va="center", zorder=5, wrap=False, alpha=1.0):
    weight = "bold" if bold else "normal"
    ax.text(x, y, s, color=color, fontsize=size, fontweight=weight,
            ha=ha, va=va, zorder=zorder, alpha=alpha,
            wrap=wrap, family="DejaVu Sans")


def arrow(ax, x1, y1, x2, y2, color=ARROW_CLR, lw=1.5, style="-|>", zorder=3, label="", ls="-"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, linestyle=ls),
                zorder=zorder)
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        txt(ax, mx + 0.1, my, label, color=MUTED, size=7.5)


def divider(ax, x, y, w, color=BORDER, lw=0.8):
    ax.plot([x, x + w], [y, y], color=color, lw=lw, zorder=4)


# ═════════════════════════════════════════════════════════════════════════════
# 1. HIGH-LEVEL DESIGN
# ═════════════════════════════════════════════════════════════════════════════

def make_hld():
    W, H = 22, 14
    fig, ax = _fig(W, H)

    # ── Title bar ──────────────────────────────────────────────────────────
    rbox(ax, 0, H - 1.1, W, 1.1, CARD_DARK, BORDER, radius=0, lw=0)
    txt(ax, W / 2, H - 0.55, "🏗  High-Level Design — Skailama Promotion Agent",
        color=WHITE, size=16, bold=True)

    # ── Legend ─────────────────────────────────────────────────────────────
    legend_items = [
        (ACCENT1, "Client / External"),
        (ACCENT2, "API Layer"),
        (ACCENT5, "AI / ML"),
        (ACCENT3, "Persistence"),
        (ACCENT6, "Observability"),
    ]
    for i, (c, label) in enumerate(legend_items):
        lx = 0.8 + i * 4.2
        rbox(ax, lx, 0.25, 0.35, 0.35, c, c, radius=0.08, lw=0)
        txt(ax, lx + 0.55, 0.43, label, color=MUTED, size=8, ha="left")

    # ── Layer background bands ──────────────────────────────────────────────
    layers = [
        (11.5, 2.0, "#1C3A54", "CLIENT LAYER"),
        (9.0,  2.2, "#162E20", "API GATEWAY LAYER"),
        (6.1,  2.6, "#2A1D4A", "AI AGENT LAYER (LangGraph)"),
        (3.3,  2.5, "#2E2008", "PERSISTENCE LAYER"),
        (0.9,  2.1, "#0E2929", "OBSERVABILITY LAYER"),
    ]
    for ly, lh, lc, label in layers:
        rbox(ax, 0.3, ly, W - 0.6, lh, lc, BORDER, radius=0.2, alpha=0.55, lw=1.0, zorder=1)
        txt(ax, 1.2, ly + lh - 0.22, label, color=MUTED, size=7.5, bold=True, ha="left")

    # ── CLIENT LAYER ────────────────────────────────────────────────────────
    clients = [
        ("🖥  HTTP Client\n(REST Consumer)", 2.5),
        ("🔁  Chat Client\n(chat.py)", 8.5),
        ("🧪  Test Suite\n(test_mini_promotion_agent.py)", 14.5),
    ]
    for label, cx in clients:
        rbox(ax, cx - 1.8, 12.15, 3.5, 1.15, CARD_MED, ACCENT1, radius=0.2, lw=1.5)
        txt(ax, cx, 12.72, label, color=ACCENT1, size=9, bold=True)

    # ── API GATEWAY LAYER ───────────────────────────────────────────────────
    routes = [
        ("POST /chat", 3.0, ACCENT2),
        ("POST /chat/intent-classifier/llm", 8.0, ACCENT2),
        ("POST /chat/mini-promotion-agent", 13.0, ACCENT2),
        ("POST /chat/.../clarify", 17.5, ACCENT2),
        ("GET  /.../history", 20.5, ACCENT2),
    ]
    for label, rx, c in routes:
        w = 3.2 if "mini" in label or "classifier" in label else 2.6
        rbox(ax, rx - w / 2, 9.6, w, 1.1, CARD_MED, c, radius=0.18, lw=1.4)
        txt(ax, rx, 10.15, label, color=c, size=8, bold=True)

    rbox(ax, 0.6, 9.3, W - 1.2, 1.85, CARD_DARK, ACCENT2, radius=0.22, alpha=0.2, lw=1.2, zorder=1)
    txt(ax, 1.6, 11.0, "FastAPI  ·  uvicorn  ·  /app/main.py", color=MUTED, size=8, ha="left")

    # ── AI AGENT LAYER ──────────────────────────────────────────────────────
    nodes = [
        ("🎯 Intent\nClassification", 2.5),
        ("🔍 Trigger\nDetection", 5.8),
        ("✅ Schema\nValidation", 9.1),
        ("🔨 State\nAssembly", 12.4),
        ("🔏 Final\nValidation", 15.7),
        ("❓ Clarification\n(interrupt)", 18.9),
        ("🚫 Unsupported\n(terminal)", 21.4),
    ]
    for label, nx in nodes:
        rbox(ax, nx - 1.55, 6.7, 3.0, 1.35, CARD_MED, ACCENT5, radius=0.2, lw=1.4)
        txt(ax, nx - 0.05, 7.38, label, color=ACCENT5, size=8.5, bold=True)

    # mini arrows between nodes
    arrowdata = [(2.5, 5.8), (5.8, 9.1), (9.1, 12.4), (12.4, 15.7)]
    for x1, x2 in arrowdata:
        arrow(ax, x1 + 1.45, 7.37, x2 - 1.55, 7.37, color=ACCENT5, lw=1.2)

    # LLM box
    rbox(ax, 0.6, 6.4, W - 1.2, 2.3, CARD_DARK, ACCENT5, radius=0.22, alpha=0.15, lw=1.2, zorder=1)
    txt(ax, 1.8, 8.55, "LangGraph StateGraph  ·  /app/graph.py  ·  MongoEngineCheckpointer", color=MUTED, size=8, ha="left")

    # Azure OpenAI box
    rbox(ax, 6.5, 5.8, 4.5, 0.55, "#1A2540", ACCENT1, radius=0.1, lw=1.2, zorder=3)
    txt(ax, 8.75, 6.07, "⚡ Azure OpenAI  GPT-4  (/app/llm.py)", color=ACCENT1, size=8.5, bold=True)

    # ── PERSISTENCE LAYER ───────────────────────────────────────────────────
    stores = [
        ("🗄  MongoDB Atlas\nskailama db", 3.5, ACCENT3),
        ("📋 chats collection\nconversation context", 8.5, ACCENT3),
        ("💬 messages collection\nper-turn history", 13.5, ACCENT3),
        ("🔖 checkpoints collection\nLangGraph state snapshots", 18.5, ACCENT3),
    ]
    for label, sx, c in stores:
        rbox(ax, sx - 2.2, 3.95, 4.2, 1.2, CARD_MED, c, radius=0.2, lw=1.4)
        txt(ax, sx - 0.1, 4.55, label, color=c, size=8.5, bold=True)

    # ── OBSERVABILITY LAYER ─────────────────────────────────────────────────
    obs = [
        ("📊 LangSmith\nTracing", 4.5, ACCENT6),
        ("📝 Python Logging\n(structlog)", 11.5, ACCENT6),
        ("🤖 DeBERTa-v3\nIntent Classifier", 18.5, ACCENT6),
    ]
    for label, ox, c in obs:
        rbox(ax, ox - 2.3, 1.5, 4.4, 1.15, CARD_MED, c, radius=0.2, lw=1.4)
        txt(ax, ox, 2.08, label, color=c, size=9, bold=True)

    # ── Cross-layer arrows ──────────────────────────────────────────────────
    # Clients → API
    for cx in [2.5, 8.5, 14.5]:
        arrow(ax, cx, 12.15, cx, 10.7, color=ACCENT2, lw=1.5)

    # API → Graph
    arrow(ax, 11.0, 9.6, 11.0, 8.7, color=ACCENT5, lw=1.8, label="invoke()")

    # Graph → Azure OpenAI
    arrow(ax, 8.75, 6.7, 8.75, 6.35, color=ACCENT1, lw=1.5)

    # Graph → Persistence
    arrow(ax, 10.0, 6.4, 10.0, 5.15, color=ACCENT3, lw=1.5, label="save state")

    # Graph → LangSmith
    arrow(ax, 5.0, 6.4, 4.5, 2.65, color=ACCENT6, lw=1.2, ls="--")

    # ── Watermark ───────────────────────────────────────────────────────────
    txt(ax, W - 0.4, 0.2, "Skailama · Architecture · HLD", color=MUTED, size=7, ha="right", alpha=0.6)

    fig.savefig(os.path.join(OUT_DIR, "HLD.png"), dpi=180, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    print("✅  HLD.png saved")


# ═════════════════════════════════════════════════════════════════════════════
# 2. LOW-LEVEL DESIGN
# ═════════════════════════════════════════════════════════════════════════════

def make_lld():
    W, H = 26, 20
    fig, ax = _fig(W, H)

    # ── Title ──────────────────────────────────────────────────────────────
    rbox(ax, 0, H - 1.05, W, 1.05, CARD_DARK, BORDER, radius=0, lw=0)
    txt(ax, W / 2, H - 0.52, "🔬  Low-Level Design — Skailama Promotion Agent",
        color=WHITE, size=16, bold=True)

    # ──────────────────────────────────────────────────────────────────────
    # Column setup  (4 columns)
    # ──────────────────────────────────────────────────────────────────────
    COL = [1.2, 7.4, 14.2, 20.6]
    CW = 5.8

    col_headers = [
        (ACCENT2,  "API Layer\n/app/main.py"),
        (ACCENT5,  "Graph & Nodes\n/app/graph.py · /app/nodes.py"),
        (ACCENT3,  "Data Models\n/app/models.py"),
        (ACCENT6,  "Intent Classifier\n/intentClassifier/"),
    ]
    for ci, (c, label) in enumerate(col_headers):
        cx = COL[ci]
        rbox(ax, cx, 17.5, CW, 1.2, CARD_DARK, c, radius=0.2, lw=2.0)
        txt(ax, cx + CW / 2, 18.1, label, color=c, size=10, bold=True)

    # ──────────────────────────────────────────────────────────────────────
    # COLUMN 0 — API Layer
    # ──────────────────────────────────────────────────────────────────────
    api_blocks = [
        ("FastAPI App", "lifespan: check_azure_openai_connection()\ncheck_langsmith_connection()\ncheck_mongodb_connection()", ACCENT2, 14.8),
        ("ChatRequest / ClarifyRequest", "message: str\nthread_id: str (clarify only)", ACCENT2, 12.0),
        ("POST /chat/mini-promotion-agent", "1. gen thread_id = uuid4()\n2. save user msg → MongoDB\n3. graph.invoke(initial_state)\n4. poll for interrupt()\n5. _save_graph_result_to_db()", ACCENT2, 8.8),
        ("POST /chat/.../clarify", "1. validate thread paused\n2. graph.invoke(Command(resume=...))\n3. poll for interrupt()\n4. return final_reply", ACCENT2, 6.2),
        ("GET /.../history", "DB lookup messages → fallback\nto checkpoint state.history", ACCENT2, 4.3),
        ("_save_graph_result_to_db()", "saves chats context\n+ messages collection", ACCENT2, 2.6),
    ]

    cx = COL[0]
    for title, body, c, top in api_blocks:
        bh = 0.35 + body.count("\n") * 0.38
        rbox(ax, cx, top - bh, CW, bh + 0.45, CARD_MED, c, radius=0.18, lw=1.3)
        txt(ax, cx + CW / 2, top - 0.1, title, color=c, size=8.5, bold=True)
        divider(ax, cx + 0.1, top - 0.28, CW - 0.2, color=c)
        for li, line in enumerate(body.split("\n")):
            txt(ax, cx + 0.25, top - 0.52 - li * 0.38, line, color=MUTED, size=7.5, ha="left")

    # ──────────────────────────────────────────────────────────────────────
    # COLUMN 1 — Graph & Nodes
    # ──────────────────────────────────────────────────────────────────────
    cx = COL[1]
    nodes_data = [
        ("build_graph()", "StateGraph(PromotionState)\nMongoEngineCheckpointer\ngraph.compile(checkpointer=cp)", ACCENT5, 16.8),
        ("intent_classification_node", "→ llm(INTENT_CLASSIFICATION_PROMPT)\nParse JSON → state['feature']\nInit: tiers/tier_behavior/status/blockers", ACCENT5, 14.2),
        ("trigger_detection_node", "→ Choose prompt by feature:\n  tiered_discount → TIERED prompt\n  else → TRIGGER_ONLY prompt\n→ llm() → state['tiers']", ACCENT5, 11.5),
        ("schema_validation_node", "For each tier:\n  TierModel.model_validate(tier)\n  RewardModel.model_validate(reward)\nOn fail → status='schema_error'\n          blockers / missing_fields set", ACCENT5, 8.8),
        ("state_assembly_node", "tier_behavior = 'best_tier_only'\ncustomer_eligibility = []\nstatus = 'pending_validation'", ACCENT5, 6.4),
        ("validation_node", "→ llm(VALIDATION_CLASSIFICATION_PROMPT)\nIf is_correct=False:\n  status='unsupported' + blockers\nElse: status='supported'", ACCENT5, 4.4),
        ("clarification_node", "interrupt(question_payload)\nResumes with clarification_text\nclarification_attempts += 1\n→ loops back to intent_classification", ACCENT5, 2.3),
    ]

    for title, body, c, top in nodes_data:
        bh = 0.35 + body.count("\n") * 0.38
        rbox(ax, cx, top - bh, CW, bh + 0.45, CARD_MED, c, radius=0.18, lw=1.3)
        txt(ax, cx + CW / 2, top - 0.1, title, color=c, size=8.5, bold=True)
        divider(ax, cx + 0.1, top - 0.28, CW - 0.2, color=c)
        for li, line in enumerate(body.split("\n")):
            txt(ax, cx + 0.25, top - 0.52 - li * 0.38, line, color=MUTED, size=7.5, ha="left")

    # Router annotations
    rbox(ax, cx, 9.5, CW, 0.35, "#1A1D2A", ACCENT4, radius=0.08, lw=1.0, zorder=4)
    txt(ax, cx + CW / 2, 9.67, "_route_after_intent: clarification | unsupported | trigger_detection",
        color=ACCENT4, size=7.2, bold=True)
    rbox(ax, cx, 7.25, CW, 0.35, "#1A1D2A", ACCENT4, radius=0.08, lw=1.0, zorder=4)
    txt(ax, cx + CW / 2, 7.42, "_route_after_schema_validation: schema_error → END | state_assembly",
        color=ACCENT4, size=7.2, bold=True)

    # ──────────────────────────────────────────────────────────────────────
    # COLUMN 2 — Data Models
    # ──────────────────────────────────────────────────────────────────────
    cx = COL[2]
    model_blocks = [
        ("PromotionState (TypedDict)", "message: str\nhistory: List[Dict]\nfeature: Optional[str]\ntiers: List[Any]\ntier_behavior: Optional[str]\nstatus: Optional[str]\nblockers: List[Any]\nmissing_fields: List[Any]\nclarification_attempts: int\nthread_id: Optional[str]", ACCENT3, 16.9),
        ("TierModel", "trigger: TriggerModel\nreward: Optional[RewardModel]", ACCENT3, 12.85),
        ("TriggerModel", "type: TriggerType (enum)\noperator: TriggerOperator (enum)\nvalue: Union[int,float] > 0\ncurrency: Optional[str]\nscope: Optional[TriggerScope]\n@validator: currency req for subtotal\n@validator: scope req for scoped", ACCENT3, 11.6),
        ("RewardModel", "type: RewardType (enum)\nvalue: Optional[float]\ny_target: Optional[ProductTarget]\nquantity: Optional[int]\ngift_product: Optional[GiftProductTarget]\n@validator: enforce per-type fields", ACCENT3, 8.4),
        ("TriggerType (Enum)", "cart_quantity | cart_subtotal\ncollection_quantity | collection_subtotal\nproduct_quantity | product_subtotal", ACCENT3, 5.5),
        ("RewardType (Enum)", "percentage_off | fixed_amount_off\npercentage_off_y | fixed_amount_off_y\nfree_gift | free_gift_product", ACCENT3, 3.9),
        ("MongoDBClient (/app/db/mongo.py)", "chats: Collection\nmessages: Collection\ncreate_chat_if_missing()\nsave_message()  get_messages()\nsave_chat_context()", ACCENT3, 2.45),
    ]

    for title, body, c, top in model_blocks:
        bh = 0.35 + body.count("\n") * 0.38
        rbox(ax, cx, top - bh, CW, bh + 0.45, CARD_MED, c, radius=0.18, lw=1.3)
        txt(ax, cx + CW / 2, top - 0.1, title, color=c, size=8.5, bold=True)
        divider(ax, cx + 0.1, top - 0.28, CW - 0.2, color=c)
        for li, line in enumerate(body.split("\n")):
            txt(ax, cx + 0.25, top - 0.52 - li * 0.38, line, color=MUTED, size=7.5, ha="left")

    # ──────────────────────────────────────────────────────────────────────
    # COLUMN 3 — Intent Classifier
    # ──────────────────────────────────────────────────────────────────────
    cx = COL[3]
    ic_blocks = [
        ("Model: DeBERTa-v3-small", "Backbone: microsoft/deberta-v3-small\nTask: Sequence Classification\nLabels: 5 (free_gift, buy_x_get_y,\n  tiered_discount, unsupported, clarification)\nMAX_LEN: 128 tokens", ACCENT6, 16.9),
        ("Training Pipeline (train.py)", "Data: intentClassifier/data/\n  train.csv + val.csv\nBATCH_SIZE: 16  EPOCHS: 12\nLR: 2e-5  WARMUP_RATIO: 0.1\nOptimizer: AdamW + weight_decay\nLoss: CrossEntropyLoss(class_weights)\nEarly stopping: PATIENCE=3\nSave: model/best/", ACCENT6, 13.8),
        ("Inference (classify.py)", "get_model() → singleton load\nclassify(text) → label + confidence\n  + scores (all 5 classes)\n  + latency_ms\nDevice: CUDA > MPS > CPU", ACCENT6, 9.8),
        ("Data Pipeline", "classificationData.py: raw samples\ngenerate_data.py: LLM generation\ndedup_and_validate.py: dedup\ntopup_and_dedup.py: augment\nrelabel_generated_clean.py: fix labels\nlabel_and_validate_testset.py: test set", ACCENT6, 7.3),
        ("Evaluation", "evaluate_testset.py\nTestSetLabeledAndValidated.csv\nMetrics: accuracy, per-class F1\nLatency benchmarks", ACCENT6, 4.3),
        ("MongoEngineCheckpointer", "BaseCheckpointSaver (LangGraph)\nget_tuple() / list() / put()\nput_writes()\nSerialization: JsonPlusSerializer\nCollection: checkpoints", ACCENT6, 2.4),
    ]

    for title, body, c, top in ic_blocks:
        bh = 0.35 + body.count("\n") * 0.38
        rbox(ax, cx, top - bh, CW, bh + 0.45, CARD_MED, c, radius=0.18, lw=1.3)
        txt(ax, cx + CW / 2, top - 0.1, title, color=c, size=8.5, bold=True)
        divider(ax, cx + 0.1, top - 0.28, CW - 0.2, color=c)
        for li, line in enumerate(body.split("\n")):
            txt(ax, cx + 0.25, top - 0.52 - li * 0.38, line, color=MUTED, size=7.5, ha="left")

    # ── Vertical dividers ───────────────────────────────────────────────────
    for x_div in [7.2, 14.0, 20.8]:
        ax.plot([x_div, x_div], [1.1, 18.5], color=BORDER, lw=0.8, alpha=0.5, zorder=1)

    # ── Watermark ───────────────────────────────────────────────────────────
    txt(ax, W - 0.4, 0.2, "Skailama · Architecture · LLD", color=MUTED, size=7, ha="right", alpha=0.6)

    fig.savefig(os.path.join(OUT_DIR, "LLD.png"), dpi=180, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    print("✅  LLD.png saved")


# ═════════════════════════════════════════════════════════════════════════════
# 3. LANGGRAPH FLOW DIAGRAM
# ═════════════════════════════════════════════════════════════════════════════

def make_langgraph():
    W, H = 20, 26
    fig, ax = _fig(W, H)

    # ── Title ──────────────────────────────────────────────────────────────
    rbox(ax, 0, H - 1.1, W, 1.1, CARD_DARK, BORDER, radius=0, lw=0)
    txt(ax, W / 2, H - 0.55, "🕸  LangGraph Flow — Promotion Agent State Machine",
        color=WHITE, size=15, bold=True)

    CX = W / 2  # center X

    # ────────────────────────────────────────────────────────────────────
    # NODE definitions  (label, y, color, width, height, body)
    # ────────────────────────────────────────────────────────────────────
    NW = 8.5
    NH = 1.5

    def node(label, y, c, body="", extra_h=0):
        h = NH + extra_h
        rbox(ax, CX - NW / 2, y, NW, h, CARD_MED, c, radius=0.25, lw=2.0, zorder=3)
        txt(ax, CX, y + h - 0.42, label, color=c, size=11, bold=True)
        if body:
            divider(ax, CX - NW / 2 + 0.15, y + h - 0.62, NW - 0.3, color=c)
            for li, line in enumerate(body.split("\n")):
                txt(ax, CX - NW / 2 + 0.25, y + h - 0.95 - li * 0.4, line,
                    color=MUTED, size=8, ha="left")
        return y + h / 2  # mid Y

    def side_node(label, x, y, c, body="", w=5.5, h=1.6):
        rbox(ax, x - w / 2, y - h / 2, w, h, CARD_MED, c, radius=0.22, lw=1.8, zorder=3)
        txt(ax, x, y + h / 2 - 0.38, label, color=c, size=9.5, bold=True)
        if body:
            divider(ax, x - w / 2 + 0.12, y + h / 2 - 0.57, w - 0.24, color=c)
            for li, line in enumerate(body.split("\n")):
                txt(ax, x - w / 2 + 0.2, y + h / 2 - 0.88 - li * 0.38, line,
                    color=MUTED, size=7.5, ha="left")

    def diamond(label, y, c):
        """Draw a diamond (decision) shape."""
        # Diamond using a rotated square matplotlib patch
        hw = 2.0
        hh = 0.7
        pts = np.array([
            [CX,      y + hh],
            [CX + hw, y],
            [CX,      y - hh],
            [CX - hw, y],
        ])
        polygon = plt.Polygon(pts, closed=True, facecolor=CARD_MED, edgecolor=c, lw=2.0, zorder=3)
        ax.add_patch(polygon)
        txt(ax, CX, y, label, color=c, size=9, bold=True)
        return y  # mid

    # ── START bullet ───────────────────────────────────────────────────────
    circle_start = plt.Circle((CX, 24.3), 0.32, color=ACCENT2, zorder=5)
    ax.add_patch(circle_start)
    txt(ax, CX, 24.3, "▶", color=BG, size=10, bold=True)
    txt(ax, CX + 0.55, 24.3, "START  (new message)", color=MUTED, size=8.5, ha="left")

    arrow(ax, CX, 23.98, CX, 23.35, color=ACCENT2, lw=2.0)

    # ── Node 1 — intent_classification ────────────────────────────────────
    node("🎯  intent_classification_node",
         21.6, ACCENT5,
         body="llm(INTENT_CLASSIFICATION_PROMPT, message, history)\n"
              "→ state.feature = intent\n"
              "→ init: tiers=[], status='', blockers=[]",
         extra_h=0.7)
    txt(ax, CX + NW / 2 + 0.2, 22.2 + 0.35, "/app/nodes.py", color=MUTED, size=7.5, ha="left")

    arrow(ax, CX, 21.6, CX, 21.0, color=ACCENT4, lw=2.5, label="")
    diamond("Route by feature", 20.55, ACCENT4)

    # ── Branch: clarification ──────────────────────────────────────────────
    SIDE_L = 3.5
    SIDE_R = 16.5

    # Left branch — clarification
    ax.annotate("", xy=(SIDE_L, 20.55), xytext=(CX - 2.0, 20.55),
                arrowprops=dict(arrowstyle="-|>", color=ACCENT3, lw=1.8))
    txt(ax, (SIDE_L + CX - 2.0) / 2, 20.75, "clarification", color=ACCENT3, size=8.5, bold=True)

    side_node("❓ clarification_node",
              SIDE_L, 19.0, ACCENT3,
              body="interrupt(question_payload)\n→ pauses graph\nresumes on /clarify\nattempts += 1 (max 3)")

    # Loop-back arrow from clarification → top of intent
    ax.annotate("", xy=(CX - NW / 2, 22.5), xytext=(SIDE_L - 2.75, 22.5),
                arrowprops=dict(arrowstyle="-|>", color=ACCENT3, lw=1.6, linestyle="dashed"))
    ax.plot([SIDE_L - 2.75, SIDE_L - 2.75], [17.35, 22.5], color=ACCENT3, lw=1.6, ls="--", zorder=3)
    ax.plot([SIDE_L - 2.75, SIDE_L + 2.75], [17.35, 17.35], color=ACCENT3, lw=1.6, ls="--", zorder=3)
    ax.plot([SIDE_L + 2.75, SIDE_L + 2.75], [17.35, 18.2], color=ACCENT3, lw=1.6, ls="--", zorder=3)
    txt(ax, 1.0, 20.0, "loop\nback", color=ACCENT3, size=7.5, ha="center")

    # Right branch — unsupported
    ax.annotate("", xy=(SIDE_R, 20.55), xytext=(CX + 2.0, 20.55),
                arrowprops=dict(arrowstyle="-|>", color=ACCENT4, lw=1.8))
    txt(ax, (SIDE_R + CX + 2.0) / 2, 20.75, "unsupported", color=ACCENT4, size=8.5, bold=True)

    side_node("🚫 unsupported_node",
              SIDE_R, 19.0, ACCENT4,
              body="status = 'unsupported'\nblockers = [intent reason]\n→ END (terminal)")

    # Unsupported → END
    end_circle_r = plt.Circle((SIDE_R, 17.45), 0.32, color=ACCENT4, zorder=5)
    ax.add_patch(end_circle_r)
    txt(ax, SIDE_R, 17.45, "■", color=BG, size=10, bold=True)
    txt(ax, SIDE_R + 0.45, 17.45, "END", color=ACCENT4, size=8.5, ha="left", bold=True)
    arrow(ax, SIDE_R, 18.2, SIDE_R, 17.77, color=ACCENT4, lw=1.8)

    # Happy path down from diamond
    arrow(ax, CX, 19.85, CX, 19.2, color=ACCENT2, lw=2.2)
    txt(ax, CX + 0.2, 19.55, "supported intent", color=ACCENT2, size=8.5, ha="left", bold=True)

    # ── Node 2 — trigger_detection ─────────────────────────────────────────
    node("🔍  trigger_detection_node",
         17.5, ACCENT1,
         body="feature=='tiered_discount' → TIERED_DISCOUNT prompt\n"
              "else → TRIGGER_ONLY_CLASSIFICATION prompt\n"
              "llm(prompt, message) → state.tiers",
         extra_h=0.6)
    txt(ax, CX + NW / 2 + 0.2, 18.3, "/app/nodes.py", color=MUTED, size=7.5, ha="left")

    arrow(ax, CX, 17.5, CX, 16.9, color=ACCENT1, lw=2.0)

    # ── Node 3 — schema_validation ─────────────────────────────────────────
    node("✅  schema_validation_node",
         15.0, ACCENT2,
         body="For each tier in state.tiers:\n"
              "  TierModel.model_validate(tier)  ← trigger check\n"
              "  RewardModel.model_validate(reward) ← reward check\n"
              "On fail → status='schema_error', blockers appended",
         extra_h=0.8)
    txt(ax, CX + NW / 2 + 0.2, 15.9, "/app/nodes.py", color=MUTED, size=7.5, ha="left")

    arrow(ax, CX, 15.0, CX, 14.4, color=ACCENT4, lw=2.5)
    diamond("Schema valid?", 13.95, ACCENT4)

    # Schema error branch (right)
    ax.annotate("", xy=(SIDE_R, 13.95), xytext=(CX + 2.0, 13.95),
                arrowprops=dict(arrowstyle="-|>", color=ACCENT4, lw=1.8))
    txt(ax, (SIDE_R + CX + 2) / 2, 14.15, "schema_error", color=ACCENT4, size=8.5, bold=True)
    end_circle_r2 = plt.Circle((SIDE_R, 13.95), 0.32, color=ACCENT4, zorder=5)
    ax.add_patch(end_circle_r2)
    txt(ax, SIDE_R, 13.95, "■", color=BG, size=10, bold=True)
    txt(ax, SIDE_R + 0.45, 13.95, "END\n(blockers returned)", color=ACCENT4, size=8, ha="left", bold=True)

    # Success path down
    arrow(ax, CX, 13.25, CX, 12.65, color=ACCENT2, lw=2.2)
    txt(ax, CX + 0.2, 12.95, "valid schema", color=ACCENT2, size=8.5, ha="left", bold=True)

    # ── Node 4 — state_assembly ────────────────────────────────────────────
    node("🔨  state_assembly_node",
         11.0, ACCENT3,
         body="tier_behavior = 'best_tier_only'\n"
              "customer_eligibility = []\n"
              "status = 'pending_validation'",
         extra_h=0.55)
    txt(ax, CX + NW / 2 + 0.2, 11.7, "/app/nodes.py", color=MUTED, size=7.5, ha="left")

    arrow(ax, CX, 11.0, CX, 10.4, color=ACCENT3, lw=2.0)

    # ── Node 5 — validation ────────────────────────────────────────────────
    node("🔏  validation_node",
         8.5, ACCENT6,
         body="llm(VALIDATION_CLASSIFICATION_PROMPT,\n"
              "    JSON({id, text, label=feature}))\n"
              "If is_correct=False → status='unsupported'\n"
              "Else → status='supported'",
         extra_h=0.7)
    txt(ax, CX + NW / 2 + 0.2, 9.2, "/app/nodes.py", color=MUTED, size=7.5, ha="left")

    arrow(ax, CX, 8.5, CX, 7.9, color=ACCENT2, lw=2.5)

    # ── END (success) ──────────────────────────────────────────────────────
    end_circle_ok = plt.Circle((CX, 7.45), 0.32, color=ACCENT2, zorder=5)
    ax.add_patch(end_circle_ok)
    txt(ax, CX, 7.45, "■", color=BG, size=10, bold=True)
    txt(ax, CX + 0.5, 7.45, "END  (final reply)", color=ACCENT2, size=9, ha="left", bold=True)

    # ── MongoDB Checkpointer callout ────────────────────────────────────────
    rbox(ax, 0.4, 3.0, 7.5, 3.8, CARD_DARK, ACCENT3, radius=0.22, lw=1.5, alpha=0.85)
    txt(ax, 4.15, 6.55, "🗄  MongoEngineCheckpointer", color=ACCENT3, size=10, bold=True)
    divider(ax, 0.55, 6.3, 7.2, color=ACCENT3)
    cp_lines = [
        "• put()   — persist snapshot after every node",
        "• get_tuple() — restore state on graph.invoke()",
        "• put_writes() — accumulate pending writes",
        "• Serializer: JsonPlusSerializer",
        "• Collection: MongoDB 'checkpoints'",
        "• Enables interrupt() / resume across HTTP calls",
    ]
    for li, line in enumerate(cp_lines):
        txt(ax, 0.7, 5.95 - li * 0.48, line, color=MUTED, size=8, ha="left")

    # ── Azure OpenAI callout ────────────────────────────────────────────────
    rbox(ax, 11.8, 3.0, 7.8, 3.8, CARD_DARK, ACCENT1, radius=0.22, lw=1.5, alpha=0.85)
    txt(ax, 15.7, 6.55, "⚡  Azure OpenAI  (llm.py)", color=ACCENT1, size=10, bold=True)
    divider(ax, 11.95, 6.3, 7.5, color=ACCENT1)
    llm_lines = [
        "Deployment: GPT-4 (OPEN_API_4_ENGINE)",
        "API Version: OPENAI_API_4_VERSION",
        "response_format: json_object",
        "temperature: 0  (deterministic)",
        "Prompts injected as system messages",
        "@traceable → LangSmith traces",
    ]
    for li, line in enumerate(llm_lines):
        txt(ax, 12.0, 5.95 - li * 0.48, line, color=MUTED, size=8, ha="left")

    # ── State shape legend ──────────────────────────────────────────────────
    rbox(ax, 0.4, 0.4, 19.2, 2.35, CARD_DARK, BORDER, radius=0.2, lw=1.0, alpha=0.7)
    txt(ax, 10.0, 2.5, "PromotionState  Fields", color=WHITE, size=9.5, bold=True)
    divider(ax, 0.55, 2.28, 19.0, color=BORDER)
    fields = [
        "message: str", "history: List[Dict]", "feature: Optional[str]",
        "tiers: List[Any]", "tier_behavior: str", "status: str",
        "blockers: List", "missing_fields: List", "clarification_attempts: int", "thread_id: str",
    ]
    for fi, f in enumerate(fields):
        col_idx = fi % 5
        row_idx = fi // 5
        fx = 0.8 + col_idx * 3.8
        fy = 1.92 - row_idx * 0.5
        rbox(ax, fx - 0.1, fy - 0.15, 3.5, 0.38, CARD_MED, BORDER, radius=0.08, lw=0.8, zorder=4)
        txt(ax, fx + 1.65, fy + 0.05, f, color=MUTED, size=8)

    # ── Watermark ───────────────────────────────────────────────────────────
    txt(ax, W - 0.4, 0.12, "Skailama · Architecture · LangGraph Flow", color=MUTED, size=7, ha="right", alpha=0.6)

    fig.savefig(os.path.join(OUT_DIR, "LangGraph_Flow.png"), dpi=180, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    print("✅  LangGraph_Flow.png saved")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Saving diagrams to: {OUT_DIR}")
    make_hld()
    make_lld()
    make_langgraph()
    print("\n🎉  All 3 PNG diagrams generated successfully!")
