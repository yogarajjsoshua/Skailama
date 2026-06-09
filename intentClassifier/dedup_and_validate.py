import os
import sys
from collections import Counter
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL = SentenceTransformer("all-MiniLM-L6-v2")
THRESHOLD = 0.90
MIN_PER_CLASS = 430
MIN_PER_CLASS_CLARIFICATION = 430   # matches all other classes
OUTPUT_VAR = "GENERATED_CLEAN"
DATA_FILE = os.path.join(os.path.dirname(__file__), "classificationData.py")


def deduplicate(examples: list, threshold: float = THRESHOLD) -> list:
    texts = [t for t, _ in examples]
    labels = [l for _, l in examples]
    embeddings = MODEL.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    kept_indices = []
    kept_embeddings = []
    for i, emb in enumerate(embeddings):
        if not kept_embeddings:
            kept_indices.append(i)
            kept_embeddings.append(emb)
            continue
        sims = cosine_similarity([emb], kept_embeddings)[0]
        if sims.max() <= threshold:
            kept_indices.append(i)
            kept_embeddings.append(emb)
    return [(texts[i], labels[i]) for i in kept_indices]


def validate(examples: list) -> bool:
    dist = Counter(label for _, label in examples)
    print("\n=== Post-Dedup Balance Report ===")
    print(f"Total examples: {len(examples)}\n")
    all_ok = True
    checks = [
        ("free_gift",      MIN_PER_CLASS),
        ("buy_x_get_y",    MIN_PER_CLASS),
        ("tiered_discount", MIN_PER_CLASS),
        ("unsupported",    MIN_PER_CLASS),
        ("clarification",  MIN_PER_CLASS_CLARIFICATION),
    ]
    for label, min_count in checks:
        count = dist.get(label, 0)
        status = "OK" if count >= min_count else f"BELOW TARGET (need {min_count})"
        print(f"  {label:<22} {count:>4}  [{status}]")
        if count < min_count:
            all_ok = False
    return all_ok


def load_existing_texts() -> set:
    sys.path.insert(0, os.path.dirname(__file__))
    from classificationData import GENERATED_CLEAN
    return set(t.strip() for t, _ in GENERATED_CLEAN)


def append_to_classification_data(examples: list) -> None:
    lines = [f"\n\n{OUTPUT_VAR} = [\n"]
    for text, label in examples:
        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'    ("{escaped}", "{label}"),\n')
    lines.append("]\n")
    with open(DATA_FILE, "a", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"\nAppended {len(examples)} new examples to classificationData.py as {OUTPUT_VAR}")


def main():
    sys.path.insert(0, os.path.dirname(__file__))
    try:
        from generated_data_validated import ALL_EXAMPLES
    except ImportError:
        print("ERROR: generated_data_validated.py not found. Run validate_generated_data.py first.")
        sys.exit(1)

    print(f"Loaded {len(ALL_EXAMPLES)} raw generated examples.")

    existing_texts = load_existing_texts()
    novel = [(t, l) for t, l in ALL_EXAMPLES if t.strip() not in existing_texts]
    print(f"After removing exact matches with existing data: {len(novel)} examples remain.")

    deduped = deduplicate(novel)
    print(f"After semantic dedup (threshold={THRESHOLD}): {len(deduped)} examples remain.")

    passed = validate(deduped)
    if not passed:
        print("\nWARNING: One or more classes are below the 300-example target.")
        print("Consider re-running generate_data.py and repeating dedup.")
    else:
        print("\nAll classes meet the 300-example minimum.")

    append_to_classification_data(deduped)


if __name__ == "__main__":
    main()
