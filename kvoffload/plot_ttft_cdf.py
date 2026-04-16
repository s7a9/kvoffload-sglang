from __future__ import annotations

import argparse
import fnmatch
import json
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterator, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def iter_jsonl(path: Path) -> Iterator[Tuple[int, Dict]]:
    """Stream JSONL records line by line."""

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc


def _format_label(tag: str, record: Dict) -> str:
    """Build a compact legend label from the experiment tag."""

    tag_lower = tag.lower()
    if tag_lower.endswith("-local") or "-local-" in tag_lower or "local" in tag_lower:
        return "Tokenflow"
    if tag_lower.endswith("-default") or "-default-" in tag_lower or "default" in tag_lower:
        return "SGLang"
    return str(tag)


def load_ttfts_by_tag(
    jsonl_path: Path,
    tag_glob: str = "",
    tags: List[str] | None = None,
) -> Tuple[Dict[str, List[float]], Dict[str, str]]:
    """Read the large JSONL file once and collect TTFT samples per experiment tag."""

    ttfts_by_tag: DefaultDict[str, List[float]] = defaultdict(list)
    label_by_tag: Dict[str, str] = {}

    tag_set = set(tags or [])
    use_tag_set = bool(tag_set)
    tag_glob = tag_glob.strip()

    matched = 0
    for line_no, record in iter_jsonl(jsonl_path):
        tag = str(record.get("tag") or "")
        if not tag:
            continue

        if use_tag_set and tag not in tag_set:
            continue
        if tag_glob and not fnmatch.fnmatchcase(tag, tag_glob):
            continue

        ttfts = record.get("ttfts", [])
        if not isinstance(ttfts, list) or not ttfts:
            continue

        # Convert seconds to milliseconds once while streaming the file.
        values_ms = [float(value) * 1000.0 for value in ttfts if value is not None]
        if not values_ms:
            continue

        ttfts_by_tag[tag].extend(values_ms)
        if tag not in label_by_tag:
            label_by_tag[tag] = _format_label(tag, record)
        matched += 1

        print(f"Loaded line {line_no}: tag={tag}, ttft_samples={len(values_ms)}")

    if not ttfts_by_tag:
        filters = []
        if use_tag_set:
            filters.append(f"tags={sorted(tag_set)}")
        if tag_glob:
            filters.append(f"tag_glob={tag_glob}")
        detail = ", ".join(filters) if filters else "no filters"
        raise ValueError(f"No TTFT samples found in {jsonl_path} ({detail})")

    print(f"Matched experiment records: {matched}")
    return dict(ttfts_by_tag), label_by_tag


def _cdf_curve(values: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    sorted_values = np.sort(np.asarray(values, dtype=float))
    y_values = np.arange(1, len(sorted_values) + 1, dtype=float) / float(len(sorted_values))
    return sorted_values, y_values


def plot_ttft_cdf(
    ttfts_by_tag: Dict[str, List[float]],
    label_by_tag: Dict[str, str],
    output_path: Path,
    title: str,
) -> None:
    plt.figure(figsize=(5, 3))
    seen_labels: set[str] = set()

    ordered_tags = sorted(
        ttfts_by_tag.keys(),
        key=lambda tag: (np.median(ttfts_by_tag[tag]), tag),
    )

    for tag in ordered_tags:
        values = ttfts_by_tag[tag]
        if not values:
            continue
        x_values, y_values = _cdf_curve(values)
        label = label_by_tag.get(tag, tag)
        if label in seen_labels:
            label = "_nolegend_"
        else:
            seen_labels.add(label)
        plt.plot(x_values, y_values, linewidth=1.8, label=label)

    plt.title(title, fontsize=13, fontweight="bold")
    plt.xlabel("TTFT (ms)")
    plt.ylabel("CDF")
    plt.xscale("log")
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8, ncol=1)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()
    print(f"Saved: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream a large benchmark JSONL file and plot TTFT CDF curves for multiple experiments."
    )
    parser.add_argument(
        "--jsonl-path",
        type=Path,
        default=Path("kvoffload/bench_results.jsonl"),
        help="Path to the benchmark JSONL file.",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=Path("kvoffload/ttft_cdf/ttft_cdf.png"),
        help="Output image path for the CDF plot.",
    )
    parser.add_argument(
        "--tags",
        nargs="*",
        default=None,
        help="Optional explicit experiment tags to include. If omitted, all matching records are used.",
    )
    parser.add_argument(
        "--tag-glob",
        type=str,
        default="",
        help="Optional glob pattern to filter experiment tags, for example '*default*'.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="TTFT CDF Comparison",
        help="Plot title.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.jsonl_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {args.jsonl_path}")

    if args.tags and args.tag_glob:
        print("Both --tags and --tag-glob are set; records must satisfy both filters.")

    ttfts_by_tag, label_by_tag = load_ttfts_by_tag(
        jsonl_path=args.jsonl_path,
        tag_glob=args.tag_glob,
        tags=args.tags,
    )

    for tag, values in sorted(ttfts_by_tag.items(), key=lambda item: (np.median(item[1]), item[0])):
        print(
            f"tag={tag}, samples={len(values)}, median_ttft_ms={np.median(values):.3f}, "
            f"p99_ttft_ms={np.percentile(values, 99):.3f}"
        )

    plot_ttft_cdf(
        ttfts_by_tag=ttfts_by_tag,
        label_by_tag=label_by_tag,
        output_path=args.out_path,
        title=args.title,
    )


if __name__ == "__main__":
    main()
