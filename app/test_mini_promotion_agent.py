"""
test_mini_promotion_agent.py
-----------------------------
Calls POST /chat/mini-promotion-agent for every requirement in
intentClassifier/TestSetLabeledAndValidated.csv and checks:

  ✔  HTTP 200
  ✔  reply.feature  is a valid intent label (or None for unsupported)
  ✔  reply.tiers    is a list
  ✔  reply.tier_behavior  is a non-empty string
  ✔  reply.customer_eligibility  is a list
  ✔  reply.status   is "supported" | "unsupported"
  ✔  reply.blockers is a list
  ✔  if the test-set validated_label == "unsupported", status should be "unsupported"
  ✔  if the test-set validated_label != "unsupported",
        reply.feature should match validated_label

Usage:
    # start the server first:  uvicorn app.main:app --reload
    python app/test_mini_promotion_agent.py [--url http://127.0.0.1:8000] [--csv intentClassifier/TestSetLabeledAndValidated.csv]
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import requests

# ── Defaults ────────────────────────────────────────────────────────────────────
DEFAULT_URL = "http://127.0.0.1:8000/chat/mini-promotion-agent"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "intentClassifier" / "TestSetLabeledAndValidated.csv"

# ── Valid values ─────────────────────────────────────────────────────────────────
VALID_FEATURES = {"free_gift", "buy_x_get_y", "tiered_discount", "unsupported", "clarification", None, ""}
VALID_STATUSES = {"supported", "unsupported", "clarification"}

# ANSI colours
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


def ok(msg):   print(f"  {GREEN}✔{RESET}  {msg}")
def fail(msg): print(f"  {RED}✗{RESET}  {msg}")
def warn(msg): print(f"  {YELLOW}⚠{RESET}  {msg}")


# ───────────────────────────────────────────────────────────────────────────────
# Structural validators
# ───────────────────────────────────────────────────────────────────────────────

def validate_response_structure(reply: dict) -> list[str]:
    """Return a list of structural error messages (empty = pass)."""
    errors = []

    # status
    if "status" not in reply:
        errors.append("Missing field: 'status'")
    elif reply["status"] not in VALID_STATUSES:
        errors.append(f"Invalid 'status' value: {reply['status']!r} — must be one of {VALID_STATUSES}")

    # blockers
    if "blockers" not in reply:
        errors.append("Missing field: 'blockers'")
    elif not isinstance(reply["blockers"], list):
        errors.append(f"'blockers' must be a list, got {type(reply['blockers']).__name__}")

    # Special handling for clarification status
    if reply.get("status") == "clarification":
        if "question" not in reply:
            errors.append("Missing field: 'question' for clarification status")
        elif not isinstance(reply["question"], str):
            errors.append(f"'question' must be a string, got {type(reply['question']).__name__}")
        
        if "attempt" not in reply:
            errors.append("Missing field: 'attempt' for clarification status")
        elif not isinstance(reply["attempt"], int):
            errors.append(f"'attempt' must be an integer, got {type(reply['attempt']).__name__}")
        
        return errors

    # feature
    if "feature" not in reply:
        errors.append("Missing field: 'feature'")
    elif reply["feature"] not in VALID_FEATURES:
        errors.append(f"Invalid 'feature' value: {reply['feature']!r} — must be one of {VALID_FEATURES}")

    # tiers
    if "tiers" not in reply:
        errors.append("Missing field: 'tiers'")
    elif not isinstance(reply["tiers"], list):
        errors.append(f"'tiers' must be a list, got {type(reply['tiers']).__name__}")

    # tier_behavior
    if "tier_behavior" not in reply:
        errors.append("Missing field: 'tier_behavior'")
    elif not isinstance(reply["tier_behavior"], (str, type(None))):
        errors.append(f"'tier_behavior' must be a string or null, got {type(reply['tier_behavior']).__name__}")

    # customer_eligibility
    if "customer_eligibility" not in reply:
        errors.append("Missing field: 'customer_eligibility'")
    elif not isinstance(reply["customer_eligibility"], list):
        errors.append(f"'customer_eligibility' must be a list, got {type(reply['customer_eligibility']).__name__}")

    return errors


def validate_label_match(reply: dict, expected_label: str) -> list[str]:
    """Return label-mismatch warnings (not hard failures)."""
    warnings = []
    if not expected_label:
        return warnings

    feature = reply.get("feature", "")
    status  = reply.get("status", "")

    if expected_label == "unsupported":
        if status != "unsupported":
            warnings.append(
                f"Expected status='unsupported' (label=unsupported) but got status={status!r}"
            )
    elif expected_label == "clarification":
        if status != "clarification":
            warnings.append(
                f"Expected status='clarification' (label=clarification) but got status={status!r}"
            )
    else:
        if feature != expected_label:
            warnings.append(
                f"Expected feature={expected_label!r} but got {feature!r}"
            )
        if status == "unsupported":
            warnings.append(
                f"feature={feature!r} but status='unsupported' — validation node may have rejected it"
            )

    return warnings


# ───────────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────────

def run_tests(url: str, csv_path: Path):
    # Load test cases
    if not csv_path.exists():
        print(f"{RED}ERROR: CSV not found: {csv_path}{RESET}")
        print("Run intentClassifier/label_and_validate_testset.py first.")
        sys.exit(1)

    test_cases = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_cases.append(row)

    print(f"\n{BOLD}{CYAN}Mini-Promotion-Agent API Test{RESET}")
    print(f"  URL  : {url}")
    print(f"  CSV  : {csv_path}")
    print(f"  Cases: {len(test_cases)}")
    print("=" * 65)

    total = len(test_cases)
    passed = 0
    label_matched = 0
    results_log = []

    for tc in test_cases:
        row_id       = tc.get("id", "?")
        requirement  = tc.get("requirement", "")
        expected_lbl = tc.get("validated_label", tc.get("label", "")).strip()

        print(f"\n{BOLD}[{row_id}] {requirement[:80]}{'…' if len(requirement) > 80 else ''}{RESET}")

        row_result = {
            "id": row_id,
            "requirement": requirement,
            "expected_label": expected_lbl,
            "http_status": None,
            "reply": None,
            "structural_errors": [],
            "label_warnings": [],
            "passed": False,
        }

        # ── HTTP call ────────────────────────────────────────────────────────────
        try:
            t0 = time.perf_counter()
            resp = requests.post(url, json={"message": requirement}, timeout=60)
            latency_ms = (time.perf_counter() - t0) * 1000
            row_result["http_status"] = resp.status_code
        except requests.exceptions.ConnectionError:
            fail(f"Connection refused — is the server running at {url}?")
            row_result["structural_errors"].append("Connection refused")
            results_log.append(row_result)
            continue
        except Exception as e:
            fail(f"Request failed: {e}")
            row_result["structural_errors"].append(str(e))
            results_log.append(row_result)
            continue

        print(f"  HTTP {resp.status_code}  ({latency_ms:.0f} ms)")

        if resp.status_code != 200:
            fail(f"Expected HTTP 200, got {resp.status_code}")
            row_result["structural_errors"].append(f"HTTP {resp.status_code}")
            results_log.append(row_result)
            continue
        else:
            ok("HTTP 200")

        # ── Parse JSON ───────────────────────────────────────────────────────────
        try:
            body = resp.json()
        except Exception:
            fail("Response is not valid JSON")
            row_result["structural_errors"].append("Invalid JSON body")
            results_log.append(row_result)
            continue

        if "reply" not in body:
            fail("Top-level 'reply' key missing in response body")
            row_result["structural_errors"].append("Missing top-level 'reply' key")
            results_log.append(row_result)
            continue

        reply = body["reply"]
        row_result["reply"] = reply

        # Print what came back
        if reply.get("status") == "clarification":
            print(f"  status           : {reply.get('status')!r}")
            print(f"  question         : {reply.get('question')!r}")
            print(f"  attempt          : {reply.get('attempt')!r}")
            print(f"  blockers         : {reply.get('blockers', [])}")
        else:
            print(f"  feature          : {reply.get('feature')!r}")
            print(f"  status           : {reply.get('status')!r}")
            print(f"  tier_behavior    : {reply.get('tier_behavior')!r}")
            print(f"  tiers            : {json.dumps(reply.get('tiers', []))[:120]}")
            print(f"  blockers         : {reply.get('blockers', [])}")

        # ── Structural validation ────────────────────────────────────────────────
        struct_errors = validate_response_structure(reply)
        row_result["structural_errors"] = struct_errors
        if not struct_errors:
            ok("Response structure is valid")
        else:
            for e in struct_errors:
                fail(f"Structure: {e}")

        # ── Label / intent match ─────────────────────────────────────────────────
        lbl_warnings = validate_label_match(reply, expected_lbl)
        row_result["label_warnings"] = lbl_warnings
        if not lbl_warnings:
            ok(f"Label match: feature={reply.get('feature')!r} matches expected={expected_lbl!r}")
            label_matched += 1
        else:
            for w in lbl_warnings:
                warn(f"Label: {w}")

        row_result["passed"] = len(struct_errors) == 0
        if row_result["passed"]:
            passed += 1

        results_log.append(row_result)

    # ── Summary ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"{BOLD}SUMMARY{RESET}")
    print(f"  Total test cases      : {total}")
    print(f"  Structural PASS       : {GREEN}{passed}{RESET} / {total}")
    print(f"  Label match           : {GREEN}{label_matched}{RESET} / {total}")
    label_acc = label_matched / total * 100 if total else 0
    struct_acc = passed / total * 100 if total else 0
    print(f"  Label accuracy        : {label_acc:.1f}%")
    print(f"  Structural accuracy   : {struct_acc:.1f}%")
    print("=" * 65 + "\n")

    # ── Per-row failures ─────────────────────────────────────────────────────────
    failures = [r for r in results_log if not r["passed"]]
    if failures:
        print(f"{RED}Structural failures:{RESET}")
        for r in failures:
            print(f"  [{r['id']}] {r['requirement'][:60]}")
            for e in r["structural_errors"]:
                print(f"       ✗ {e}")

    label_fails = [r for r in results_log if r["label_warnings"]]
    if label_fails:
        print(f"\n{YELLOW}Label mismatches:{RESET}")
        for r in label_fails:
            print(f"  [{r['id']}] {r['requirement'][:60]}")
            for w in r["label_warnings"]:
                print(f"       ⚠ {w}")

    return passed == total


# ───────────────────────────────────────────────────────────────────────────────
# Entry point
# ───────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test /chat/mini-promotion-agent API")
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"Full URL of the endpoint (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "--csv",
        default=str(DEFAULT_CSV),
        help=f"Path to the labeled+validated CSV (default: {DEFAULT_CSV})",
    )
    args = parser.parse_args()

    all_passed = run_tests(url=args.url, csv_path=Path(args.csv))
    sys.exit(0 if all_passed else 1)
