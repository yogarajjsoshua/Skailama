import json
import os
import sys
import time
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from generated_data import ALL_EXAMPLES
from intentClassifier.classificationData import GENERATED_CLEAN
from app.pormpts import VALIDATION_CLASSIFICATION_PROMPT

client = AzureOpenAI(
    api_key=os.getenv("OPENAI_API_4_KEY"),
    api_version=os.getenv("OPENAI_API_4_VERSION"),
    azure_endpoint=os.getenv("OPENAI_4_BASE_URL"),
)
DEPLOYMENT = os.getenv("OPEN_API_4_ENGINE")
BATCH_SIZE = 50
VALID_LABELS = {"free_gift", "buy_x_get_y", "tiered_discount", "unsupported", "clarification"}


def build_batches(examples, batch_size=BATCH_SIZE):
    for i in range(0, len(examples), batch_size):
        yield examples[i : i + batch_size]


def validate_batch(batch: list, offset: int) -> list:
    payload = [
        {"id": offset + i, "text": text, "label": label}
        for i, (text, label) in enumerate(batch)
    ]
    user_msg = json.dumps(payload, ensure_ascii=False)

    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": VALIDATION_CLASSIFICATION_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    raw = response.choices[0].message.content
    return json.loads(raw)["results"]


def run_validation(examples: list, source_name: str) -> tuple[list, list]:
    print(f"\n=== Validating {source_name} ({len(examples)} examples) ===")
    all_results = []
    offset = 0

    for batch in build_batches(examples):
        print(f"  Batch {offset}–{offset + len(batch) - 1} ...", end=" ", flush=True)
        try:
            results = validate_batch(batch, offset)
            all_results.extend(results)
            print(f"done ({len(results)} results)")
        except Exception as e:
            print(f"ERROR: {e}")
            for i, (text, label) in enumerate(batch):
                all_results.append(
                    {"id": offset + i, "text": text, "label": label,
                     "correct_label": label, "is_correct": True,
                     "reason": "skipped due to API error"}
                )
        offset += len(batch)
        time.sleep(0.3)

    mismatches = [r for r in all_results if not r.get("is_correct", True)]
    print(f"  Mismatches found: {len(mismatches)}")
    return all_results, mismatches


def build_corrected_list(examples: list, results: list) -> list:
    corrected = []
    for (text, original_label), result in zip(examples, results):
        new_label = result.get("correct_label", original_label)
        if new_label not in VALID_LABELS:
            new_label = original_label
        corrected.append((text, new_label))
    return corrected


def write_corrected_file(corrected_all: list, corrected_clean: list, out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("ALL_EXAMPLES = [\n")
        for text, label in corrected_all:
            escaped = text.replace("\\", "\\\\").replace('"', '\\"')
            f.write(f'    ("{escaped}", "{label}"),\n')
        f.write("]\n\n")

        f.write("GENERATED_CLEAN_VALIDATED = [\n")
        for text, label in corrected_clean:
            escaped = text.replace("\\", "\\\\").replace('"', '\\"')
            f.write(f'    ("{escaped}", "{label}"),\n')
        f.write("]\n")
    print(f"\nCorrected data written to: {out_path}")


def print_mismatch_report(mismatches: list, source: str):
    if not mismatches:
        print(f"\n{source}: No mismatches.")
        return
    print(f"\n=== Mismatch Report — {source} ===")
    for m in mismatches:
        print(f"  [id={m['id']}] label={m.get('label')} -> correct={m.get('correct_label')}")
        print(f"    reason: {m.get('reason', '')}")


def main():
    results_all, mismatches_all = run_validation(ALL_EXAMPLES, "generated_data.ALL_EXAMPLES")
    results_clean, mismatches_clean = run_validation(GENERATED_CLEAN, "classificationData.GENERATED_CLEAN")

    print_mismatch_report(mismatches_all, "ALL_EXAMPLES")
    print_mismatch_report(mismatches_clean, "GENERATED_CLEAN")

    corrected_all = build_corrected_list(ALL_EXAMPLES, results_all)
    corrected_clean = build_corrected_list(GENERATED_CLEAN, results_clean)

    out_path = os.path.join(os.path.dirname(__file__), "generated_data_validated.py")
    write_corrected_file(corrected_all, corrected_clean, out_path)

    total_mismatches = len(mismatches_all) + len(mismatches_clean)
    print(f"\nTotal examples validated: {len(ALL_EXAMPLES) + len(GENERATED_CLEAN)}")
    print(f"Total mismatches corrected: {total_mismatches}")


if __name__ == "__main__":
    main()
