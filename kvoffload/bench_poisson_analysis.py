from __future__ import annotations

import argparse
import fnmatch
import json
import re
from pathlib import Path
from typing import Dict, Iterator

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def iter_jsonl(path: Path) -> Iterator[Dict]:
    """Stream records from a JSONL file line by line."""
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {i}: {e}") from e


def summarize_experiment(rec: Dict, exp_idx: int) -> pd.Series:
    def _mean_or_nan(values):
        if not isinstance(values, list) or len(values) == 0:
            return float("nan")
        return float(np.mean(values))

    input_lens = rec.get("input_lens", [])
    output_lens = rec.get("output_lens", [])
    ttfts = rec.get("ttfts", [])
    itls = rec.get("itls", [])

    flat_itls = []
    if isinstance(itls, list):
        for x in itls:
            if isinstance(x, list):
                flat_itls.extend(x)

    # e2e reconstruction per request: ttft + sum(itls)
    e2e_latencies = []
    if isinstance(ttfts, list) and isinstance(itls, list):
        n = min(len(ttfts), len(itls))
        for i in range(n):
            if isinstance(itls[i], list):
                e2e_latencies.append(float(ttfts[i]) + float(np.sum(itls[i])))

    return pd.Series(
        {
            "experiment_index": exp_idx,
            "tag": rec.get("tag"),
            "backend": rec.get("backend"),
            "dataset_name": rec.get("dataset_name"),
            "request_rate": rec.get("request_rate"),
            "num_requests": len(ttfts) if isinstance(ttfts, list) else np.nan,
            "avg_input_len": _mean_or_nan(input_lens),
            "avg_output_len": _mean_or_nan(output_lens),
            "mean_ttft_ms(top_level)": rec.get("mean_ttft_ms"),
            "median_ttft_ms(top_level)": rec.get("median_ttft_ms"),
            "p99_ttft_ms(top_level)": rec.get("p99_ttft_ms"),
            "std_ttft_ms(top_level)": rec.get("std_ttft_ms"),
            "mean_itl_ms(top_level)": rec.get("mean_itl_ms"),
            "median_itl_ms(top_level)": rec.get("median_itl_ms"),
            "p95_itl_ms(top_level)": rec.get("p95_itl_ms"),
            "p99_itl_ms(top_level)": rec.get("p99_itl_ms"),
            "mean_e2e_latency_ms(top_level)": rec.get("mean_e2e_latency_ms"),
            "median_e2e_latency_ms(top_level)": rec.get("median_e2e_latency_ms"),
            "p90_e2e_latency_ms(top_level)": rec.get("p90_e2e_latency_ms"),
            "p99_e2e_latency_ms(top_level)": rec.get("p99_e2e_latency_ms"),
            "std_e2e_latency_ms(top_level)": rec.get("std_e2e_latency_ms"),
            "mean_ttft_ms(from_ttfts)": float(np.mean(ttfts) * 1000) if ttfts else np.nan,
            "mean_ibt_ms(from_itls_flat)": float(np.mean(flat_itls) * 1000) if flat_itls else np.nan,
            "mean_e2e_latency_ms(from_ttft_itl)": float(np.mean(e2e_latencies) * 1000) if e2e_latencies else np.nan,
            "median_e2e_latency_ms(from_ttft_itl)": float(np.median(e2e_latencies) * 1000) if e2e_latencies else np.nan,
            "p90_e2e_latency_ms(from_ttft_itl)": float(np.percentile(e2e_latencies, 90) * 1000)
            if e2e_latencies
            else np.nan,
            "p99_e2e_latency_ms(from_ttft_itl)": float(np.percentile(e2e_latencies, 99) * 1000)
            if e2e_latencies
            else np.nan,
            "request_throughput": rec.get("request_throughput"),
            "output_throughput": rec.get("output_throughput"),
            "total_throughput": rec.get("total_throughput"),
            "duration": rec.get("duration"),
            "completed": rec.get("completed"),
        }
    )


def build_token_arrival_df(rec: Dict) -> pd.DataFrame:
    """Build per-token arrival timestamps for each request in one experiment."""
    ttfts = rec.get("ttfts", [])
    itls = rec.get("itls", [])
    output_lens = rec.get("output_lens", [])

    n = min(len(ttfts), len(itls), len(output_lens))
    rows = []

    for req_id in range(n):
        ttft = float(ttfts[req_id])
        itl_list = itls[req_id] if isinstance(itls[req_id], list) else []
        out_len = int(output_lens[req_id]) if output_lens[req_id] is not None else 0

        available_tokens = len(itl_list) + 1
        token_count = out_len if out_len > 0 else available_tokens
        token_count = min(token_count, available_tokens)
        if token_count <= 0:
            continue

        if token_count == 1:
            abs_times = np.array([ttft], dtype=float)
        else:
            partial_itls = np.array(itl_list[: token_count - 1], dtype=float)
            abs_times = np.concatenate(([ttft], ttft + np.cumsum(partial_itls)))

        rel_times = abs_times - abs_times[0]
        token_idx = np.arange(1, len(abs_times) + 1)

        for t_idx, t_abs, t_rel in zip(token_idx, abs_times, rel_times):
            rows.append(
                {
                    "request_id": req_id,
                    "token_idx": int(t_idx),
                    "abs_time_s": float(t_abs),
                    "rel_time_s": float(t_rel),
                }
            )

    return pd.DataFrame(rows)


def _safe_name(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", text).strip("_") or "untitled"


def plot_experiment(rec: Dict, df_tokens: pd.DataFrame, plot_dir: Path, exp_idx: int) -> tuple[str, str]:
    tag = str(rec.get("tag") or f"exp_{exp_idx}")
    prefix = f"exp{exp_idx:03d}_{_safe_name(tag)}"

    abs_path = plot_dir / f"{prefix}_token_arrival_abs.png"
    rel_path = plot_dir / f"{prefix}_token_arrival_rel0.png"

    # Plot A: absolute time since request start.
    plt.figure(figsize=(12, 7))
    for _, g in df_tokens.groupby("request_id"):
        plt.plot(g["abs_time_s"].to_numpy(), g["token_idx"].to_numpy(), alpha=0.25, linewidth=1.0)
    plt.title(f"Token Arrival Timeline (Absolute) | tag={tag}")
    plt.xlabel("Time (s)")
    plt.ylabel("Token Index")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(abs_path, dpi=150)
    plt.close()

    # Plot B: first-token aligned to zero.
    plt.figure(figsize=(12, 7))
    for _, g in df_tokens.groupby("request_id"):
        plt.plot(g["rel_time_s"].to_numpy(), g["token_idx"].to_numpy(), alpha=0.25, linewidth=1.0)
    plt.title(f"Token Arrival Timeline (First token = 0) | tag={tag}")
    plt.xlabel("Time Since First Token (s)")
    plt.ylabel("Token Index")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(rel_path, dpi=150)
    plt.close()

    return str(abs_path), str(rel_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read bench JSONL, export metrics CSV, and optionally save token-arrival plots for all experiments."
    )
    parser.add_argument(
        "--jsonl-path",
        type=Path,
        default=Path("bench_poisson.jsonl"),
        help="Input benchmark JSONL path (each line is one experiment).",
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=Path("bench_metrics_summary.csv"),
        help="Output CSV path for key metrics.",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default="",
        help="If non-empty, save two timeline plots per experiment into this directory.",
    )
    parser.add_argument(
        "--tag-glob",
        type=str,
        default="",
        help="If non-empty, only include experiments whose tag matches this glob pattern (e.g. '*default*').",
    )
    parser.add_argument(
        "--list-tags",
        action="store_true",
        help="List tags in the JSONL (one per line, like ls) and exit without exporting CSV/plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.jsonl_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {args.jsonl_path}")

    all_experiments = list(iter_jsonl(args.jsonl_path))
    if not all_experiments:
        raise ValueError(f"No experiment records found in {args.jsonl_path}")

    tag_glob = args.tag_glob.strip()
    if tag_glob:
        experiments = [
            rec
            for rec in all_experiments
            if fnmatch.fnmatchcase(str(rec.get("tag") or ""), tag_glob)
        ]
    else:
        experiments = all_experiments

    if not experiments:
        raise ValueError(f"No experiments matched --tag-glob='{tag_glob}' in {args.jsonl_path}")

    if args.list_tags:
        tags = sorted({str(rec.get("tag") or "") for rec in experiments})
        for tag in tags:
            print(tag)
        return

    rows = []
    enable_plot = bool(args.plot_dir and args.plot_dir.strip())
    plot_dir = Path(args.plot_dir) if enable_plot else None
    if plot_dir is not None:
        plot_dir.mkdir(parents=True, exist_ok=True)

    for exp_idx, rec in enumerate(experiments):
        summary = summarize_experiment(rec, exp_idx)
        if plot_dir is not None:
            df_tokens = build_token_arrival_df(rec)
            if not df_tokens.empty:
                plot_experiment(rec, df_tokens, plot_dir, exp_idx)
        rows.append(summary)

    df_metrics = pd.DataFrame(rows)
    args.metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    df_metrics.to_csv(args.metrics_csv, index=False)

    print(f"Loaded experiments: {len(experiments)} / {len(all_experiments)}")
    if tag_glob:
        print(f"Tag glob filter: {tag_glob}")
    print(f"Metrics CSV saved to: {args.metrics_csv.resolve()}")
    if plot_dir is not None:
        print(f"Plot directory: {plot_dir.resolve()}")
    else:
        print("Plot directory: disabled (pass --plot-dir to enable saving figures)")


if __name__ == "__main__":
    main()
