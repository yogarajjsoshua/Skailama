import sys
import json
import time
import torch
import typer
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_DIR   = Path("model/best")
MAX_LEN     = 128
LABEL_NAMES = ["free_gift", "buy_x_get_y", "tiered_discount", "unsupported"]

LABEL_COLORS = {
    "free_gift":       "green",
    "buy_x_get_y":    "cyan",
    "tiered_discount": "yellow",
    "unsupported":     "red",
}

LABEL_DESCRIPTIONS = {
    "free_gift":       "Customer satisfies a trigger condition and receives a free product",
    "buy_x_get_y":    "Buy qualifying X items; Y items receive a discount or are free",
    "tiered_discount": "Multiple spend/quantity thresholds, each unlocking a bigger discount",
    "unsupported":     "Does not match any supported promotion family",
}

console = Console()
app     = typer.Typer(add_completion=False)

# ── Model singleton (loaded once per process) ─────────────────────────────────
_tokenizer = None
_model     = None
_device    = None


def get_model():
    global _tokenizer, _model, _device
    if _model is None:
        if not MODEL_DIR.exists():
            console.print(
                f"Model not found at '{MODEL_DIR}'.[/bold red]\n"
                "   Run [bold]python 02_train.py[/bold] first."
            )
            raise SystemExit(1)

        if torch.cuda.is_available():
            _device = "cuda"
        elif torch.backends.mps.is_available():
            _device = "mps"
        else:
            _device = "cpu"
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        _model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        _model.to(_device)
        _model.eval()

    return _tokenizer, _model, _device


# ── Core inference ────────────────────────────────────────────────────────────
def classify(text: str) -> dict:
    """
    Returns:
        {
            "text":        str,
            "label":       str,
            "confidence":  float,          # 0–1
            "scores":      {label: float}, # softmax over all 4 classes
            "latency_ms":  float,
        }
    """
    tokenizer, model, device = get_model()

    t0  = time.perf_counter()
    enc = tokenizer(
        text,
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

    probs      = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    label_idx  = int(np.argmax(probs))
    latency_ms = (time.perf_counter() - t0) * 1000

    return {
        "text":       text,
        "label":      LABEL_NAMES[label_idx],
        "confidence": float(probs[label_idx]),
        "scores":     {LABEL_NAMES[i]: float(probs[i]) for i in range(len(LABEL_NAMES))},
        "latency_ms": latency_ms,
    }


# ── Rich output helpers ───────────────────────────────────────────────────────
def print_result(result: dict, verbose: bool = False):
    label      = result["label"]
    confidence = result["confidence"]
    color      = LABEL_COLORS[label]

    # Confidence bar (20 chars wide)
    filled = int(confidence * 20)
    bar    = "█" * filled + "░" * (20 - filled)

    panel_content = (
        f'[dim]Input:[/dim]  [italic]"{result["text"]}[/italic]"\n\n'
        f"[dim]Intent:[/dim] [{color} bold]{label}[/{color} bold]\n"
        f"[dim]Desc:  [/dim] [dim]{LABEL_DESCRIPTIONS[label]}[/dim]\n\n"
        f"[dim]Conf:  [/dim] [{color}]{bar}[/{color}]  [{color} bold]{confidence:.1%}[/{color} bold]\n"
        f"[dim]Speed: [/dim] {result['latency_ms']:.1f} ms"
    )

    console.print(Panel(panel_content, title="🏷  Promotion Intent", border_style=color, expand=False))

    if verbose:
        table = Table(title="All class scores", box=box.SIMPLE, show_header=True)
        table.add_column("Label",      style="bold")
        table.add_column("Score",      justify="right")
        table.add_column("Bar",        min_width=22)

        sorted_scores = sorted(result["scores"].items(), key=lambda x: x[1], reverse=True)
        for lbl, score in sorted_scores:
            c    = LABEL_COLORS[lbl]
            f    = int(score * 20)
            sbar = f"[{c}]{'█' * f}{'░' * (20 - f)}[/{c}]"
            table.add_row(f"[{c}]{lbl}[/{c}]", f"{score:.4f}", sbar)

        console.print(table)


# ── CLI commands ──────────────────────────────────────────────────────────────
@app.command()
def main(
    query: str = typer.Argument(
        ...,
        help='Promotion text to classify. Use "-" to read from stdin.',
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v",
        help="Show all class scores alongside the top prediction.",
    ),
    json_output: bool = typer.Option(
        False, "--json", "-j",
        help="Output raw JSON instead of the formatted display.",
    ),
):
    """
    Classify a merchant promotion prompt into one of:
    free_gift | buy_x_get_y | tiered_discount | unsupported
    """
    # Read from stdin if query is "-"
    if query == "-":
        query = sys.stdinpip.read().strip()

    if not query:
        console.print("Empty input.[/red]")
        raise typer.Exit(1)

    result = classify(query)

    if json_output:
        print(json.dumps(result, indent=2))
    else:
        print_result(result, verbose=verbose)


if __name__ == "__main__":
    app()
