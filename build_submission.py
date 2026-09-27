#!/usr/bin/env python3
"""
build_submission.py — regenerates submission.jsonl from the canonical 30
(merchant, trigger[, customer]) test pairs, by calling the exact same
`composer.compose()` the live bot uses.

Usage:
    python dataset/generate_dataset.py --seed-dir dataset --out dataset/expanded
    python build_submission.py

This makes submission.jsonl fully reproducible: nothing in it is hand-typed
separately from composer.py. If ANTHROPIC_API_KEY / OPENAI_API_KEY is set,
it will use the LLM path (see llm_client.py); otherwise it uses the
deterministic heuristic composer, which is what ships by default here.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from composer import compose  # noqa: E402

ROOT = Path(__file__).parent
DATASET = ROOT / "dataset"
EXPANDED = DATASET / "expanded"


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    if not EXPANDED.exists():
        print(f"Expected expanded dataset at {EXPANDED} — run:\n"
              f"  python {DATASET/'generate_dataset.py'} --seed-dir {DATASET} --out {EXPANDED}")
        sys.exit(1)

    test_pairs = load_json(EXPANDED / "test_pairs.json")["pairs"]

    categories = {f.stem: load_json(f) for f in (EXPANDED / "categories").glob("*.json")}
    merchants = {}
    for f in (EXPANDED / "merchants").glob("*.json"):
        m = load_json(f)
        merchants[m["merchant_id"]] = m
    customers = {}
    for f in (EXPANDED / "customers").glob("*.json"):
        c = load_json(f)
        customers[c["customer_id"]] = c
    triggers = {}
    for f in (EXPANDED / "triggers").glob("*.json"):
        t = load_json(f)
        triggers[t["id"]] = t

    out_lines = []
    for pair in test_pairs:
        trigger = triggers[pair["trigger_id"]]
        merchant = merchants[pair["merchant_id"]]
        category = categories[merchant["category_slug"]]
        customer = customers.get(pair["customer_id"]) if pair.get("customer_id") else None

        result = compose(category, merchant, trigger, customer=customer, is_first_touch=False)
        out_lines.append({
            "test_id": pair["test_id"],
            "body": result["body"],
            "cta": result["cta"],
            "send_as": result["send_as"],
            "suppression_key": result["suppression_key"],
            "rationale": result["rationale"],
        })

    out_path = ROOT / "submission.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for line in out_lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    print(f"Wrote {len(out_lines)} lines to {out_path}")


if __name__ == "__main__":
    main()
