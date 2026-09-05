#!/usr/bin/env python3
import argparse
import json
import sys


def load_results(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {
        item["request_index"]: item for item in data["individual_results"]
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare deterministic LLM benchmark outputs"
    )
    parser.add_argument("baseline")
    parser.add_argument("offload")
    args = parser.parse_args()

    baseline = load_results(args.baseline)
    offload = load_results(args.offload)
    all_ids = sorted(set(baseline) | set(offload))
    input_mismatches = []
    output_mismatches = []
    status_mismatches = []

    for request_id in all_ids:
        left = baseline.get(request_id)
        right = offload.get(request_id)
        if left is None or right is None:
            status_mismatches.append(request_id)
            continue
        if left["prompt_sha256"] != right["prompt_sha256"]:
            input_mismatches.append(request_id)
        if (left["success"], left["finish_reason"]) != (
            right["success"],
            right["finish_reason"],
        ):
            status_mismatches.append(request_id)
        if left["generated_text"] != right["generated_text"]:
            output_mismatches.append(request_id)

    print(f"requests: {len(all_ids)}")
    print(f"input mismatches: {len(input_mismatches)} {input_mismatches[:20]}")
    print(f"status mismatches: {len(status_mismatches)} {status_mismatches[:20]}")
    print(f"output mismatches: {len(output_mismatches)} {output_mismatches[:20]}")
    return 1 if input_mismatches or status_mismatches or output_mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
