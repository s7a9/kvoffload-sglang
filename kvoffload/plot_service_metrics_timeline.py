#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def iter_jsonl(path: Path) -> Iterator[Tuple[int, Dict]]:
    """Stream JSONL records line by line to avoid loading large files into memory."""
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_no}: {e}") from e


def infer_consume_speed(tag: str, rec: Dict, cli_speed: float | None) -> float:
    """Resolve per-request token consume speed (tokens/s)."""

    if cli_speed is not None and cli_speed > 0:
        return float(cli_speed)

    output_speed = rec.get("output_speed")
    if isinstance(output_speed, (int, float)) and output_speed > 0:
        return float(output_speed)

    m = re.search(r"-speed(\d+(?:\.\d+)?)-", tag)
    if m:
        return float(m.group(1))

    raise ValueError(
        f"Cannot infer consume speed for tag '{tag}'. "
        "Please provide --consume-speed (tokens/s)."
    )


def build_token_events(rec: Dict) -> Tuple[List[Tuple[float, int]], np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct token arrival events: (arrival_time_s, request_id).

    Returns:
    - token_events: sorted list[(time_s, req_id)]
    - send_times: np.ndarray shape [num_requests]
    - token_targets: np.ndarray shape [num_requests], expected token count per request
    - first_token_times: np.ndarray shape [num_requests], absolute time of first token
      (inf if the request has no tokens)
    """
    ttfts = rec.get("ttfts", [])
    itls = rec.get("itls", [])
    output_lens = rec.get("output_lens", [])
    send_times_raw = rec.get("send_times", [])

    n = min(len(ttfts), len(itls), len(output_lens))
    send_times = np.zeros(n, dtype=float)
    token_targets = np.zeros(n, dtype=int)
    first_token_times = np.full(n, np.inf, dtype=float)
    events: List[Tuple[float, int]] = []

    for req_id in range(n):
        send_t = 0.0
        if req_id < len(send_times_raw):
            send_t = float(send_times_raw[req_id])
        send_times[req_id] = send_t

        ttft = float(ttfts[req_id])
        itl_list = itls[req_id] if isinstance(itls[req_id], list) else []

        out_len_raw = output_lens[req_id]
        out_len = int(out_len_raw) if out_len_raw is not None else 0
        available_tokens = len(itl_list) + 1
        token_count = out_len if out_len > 0 else available_tokens
        token_count = max(0, min(token_count, available_tokens))
        token_targets[req_id] = token_count

        if token_count <= 0:
            continue

        t = send_t + ttft
        first_token_times[req_id] = t
        events.append((t, req_id))

        for delta in itl_list[: token_count - 1]:
            t += float(delta)
            events.append((t, req_id))

    events.sort(key=lambda x: x[0])
    return events, send_times, token_targets, first_token_times


def simulate_timeline(
    rec: Dict,
    consume_speed: float,
    num_steps: int,
) -> Dict[str, List[float]]:
    """Simulate request buffer evolution and collect per-step service metrics."""
    events, send_times, token_targets, first_token_times = build_token_events(rec)
    num_requests = len(token_targets)

    if num_requests == 0:
        return {
            "time_s": [],
            "waiting_requests_num": [],
            "running_requests_num": [],
            "active_requests_num": [],
            "queued_requests_num": [],
            "valid_throughput_tps": [],
        }

    if not events:
        start_time = float(np.min(send_times))
        end_time = float(rec.get("duration", start_time))
        if end_time <= start_time:
            end_time = start_time + 1.0
        return {
            "time_s": list(np.linspace(start_time, end_time, num_steps + 1)[1:]),
            "waiting_requests_num": [0.0] * num_steps,
            "running_requests_num": [0.0] * num_steps,
            "active_requests_num": [0.0] * num_steps,
            "queued_requests_num": [0.0] * num_steps,
            "valid_throughput_tps": [0.0] * num_steps,
        }

    start_time = float(min(np.min(send_times), events[0][0]))
    duration = rec.get("duration")
    if isinstance(duration, (int, float)) and duration > start_time:
        end_time = float(max(duration, events[-1][0]))
    else:
        end_time = float(events[-1][0])

    if end_time <= start_time:
        end_time = start_time + 1e-6

    edges = np.linspace(start_time, end_time, num_steps + 1)
    dt = float(edges[1] - edges[0])
    consume_per_step = consume_speed * dt

    buffers = np.zeros(num_requests, dtype=float)
    completed = np.zeros(num_requests, dtype=bool)
    arrived_tokens = np.zeros(num_requests, dtype=int)

    timeline: Dict[str, List[float]] = {
        "time_s": [],
        "waiting_requests_num": [],
        "running_requests_num": [],
        "active_requests_num": [],
        "queued_requests_num": [],
        "valid_throughput_tps": [],
    }

    event_idx = 0
    total_events = len(events)
    eps = 1e-12

    for step in range(num_steps):
        left = float(edges[step])
        right = float(edges[step + 1])

        while event_idx < total_events:
            t, req_id = events[event_idx]
            in_this_step = (t < right) or (step == num_steps - 1 and t <= right)
            if not in_this_step:
                break
            if t >= left - eps:
                buffers[req_id] += 1.0
                arrived_tokens[req_id] += 1
            event_idx += 1

        eligible = (
            (send_times <= right + eps)
            & (token_targets > 0)
            & (~completed)
        )

        running_mask = eligible & (buffers > consume_per_step + eps)
        waiting_mask = eligible & (~running_mask)

        eligible_idx = np.where(eligible)[0]
        if eligible_idx.size > 0:
            actual_consumption = np.minimum(buffers[eligible_idx], consume_per_step)
            buffers[eligible_idx] -= actual_consumption
            valid_throughput = float(np.sum(actual_consumption) / dt)
        else:
            valid_throughput = 0.0

        np.maximum(buffers, 0.0, out=buffers)

        newly_completed = (
            eligible
            & (arrived_tokens >= token_targets)
            & (buffers <= eps)
        )
        completed[newly_completed] = True

        queued_mask = (
            (send_times <= right + eps)
            & (token_targets > 0)
            & (~completed)
            & (first_token_times > right + eps)
        )

        timeline["time_s"].append(right)
        timeline["waiting_requests_num"].append(float(np.sum(waiting_mask)))
        timeline["running_requests_num"].append(float(np.sum(running_mask)))
        timeline["active_requests_num"].append(float(np.sum(eligible)))
        timeline["queued_requests_num"].append(float(np.sum(queued_mask)))
        timeline["valid_throughput_tps"].append(valid_throughput)

    return timeline


def slice_timeline(
    timeline: Dict[str, List[float]],
    time_start: float,
    time_end: float,
) -> Dict[str, List[float]]:
    """Keep only the data points whose time falls within [time_start, time_end]."""
    times = timeline["time_s"]
    indices = [i for i, t in enumerate(times) if time_start <= t <= time_end]
    return {key: [values[i] for i in indices] for key, values in timeline.items()}


def format_method_label(tag: str) -> str:
    """Map a benchmark tag to a compact method label for plots."""

    tag_lower = str(tag).lower()
    if tag_lower.endswith("-local") or "-local-" in tag_lower or "local" in tag_lower:
        return "Tokenflow"
    if tag_lower.endswith("-default") or "-default-" in tag_lower or "default" in tag_lower:
        return "SGLang"
    return str(tag)


def load_records_by_tags(jsonl_path: Path, tags: List[str]) -> Dict[str, Dict]:
    """Load required experiment records by tag with one-pass streaming."""
    required = list(dict.fromkeys(tags))
    required_set = set(required)
    found: Dict[str, Dict] = {}
    found_line: Dict[str, int] = {}

    for line_no, rec in iter_jsonl(jsonl_path):
        tag = rec.get("tag")
        if tag in required_set and tag not in found:
            found[tag] = rec
            found_line[tag] = line_no
            if len(found) == len(required_set):
                break

    missing = [t for t in required if t not in found]
    if missing:
        raise ValueError(f"Tags not found in {jsonl_path}: {missing}")

    print("Loaded tags:")
    for tag in required:
        print(f"  - {tag} (line {found_line[tag]})")

    return {tag: found[tag] for tag in required}


def plot_metric(
    timelines_by_tag: Dict[str, Dict[str, List[float]]],
    metric_key: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(6, 3))
    seen_labels: set[str] = set()
    for tag, timeline in timelines_by_tag.items():
        label = format_method_label(tag)
        if label in seen_labels:
            label = "_nolegend_"
        else:
            seen_labels.add(label)
        plt.plot(timeline["time_s"], timeline[metric_key], linewidth=1.8, label=label)

    plt.title(title, fontsize=13, fontweight="bold")
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    print(f"Saved: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate waiting requests and valid throughput over time from benchmark JSONL, "
            "then plot curves for selected experiment tags."
        )
    )
    parser.add_argument(
        "--jsonl-path",
        type=Path,
        default=Path("kvoffload/bench_results.jsonl"),
        help="Path to benchmark JSONL file.",
    )
    parser.add_argument(
        "--tags",
        nargs="+",
        required=True,
        help="Experiment tags to compare. Example: --tags tag1 tag2",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        required=True,
        help="Number of time steps used to split the full experiment timeline.",
    )
    parser.add_argument(
        "--consume-speed",
        type=float,
        default=None,
        help=(
            "Fallback token consume speed (tokens/s) when record has no output_speed "
            "and tag has no -speedXX-."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("kvoffload/timeline_figures"),
        help="Directory to save output figures and optional JSON metrics.",
    )
    parser.add_argument(
        "--time-range",
        nargs=2,
        type=float,
        metavar=("START", "END"),
        default=None,
        help=(
            "Only keep data within [START, END] seconds and plot that slice. "
            "Example: --time-range 0 100"
        ),
    )
    parser.add_argument(
        "--dump-json",
        action="store_true",
        help="Dump timeline dict for each tag to out_dir/timeline_metrics.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.num_steps <= 0:
        raise ValueError("--num-steps must be > 0")
    if args.consume_speed is not None and args.consume_speed <= 0:
        raise ValueError("--consume-speed must be > 0")

    records_by_tag = load_records_by_tags(args.jsonl_path, args.tags)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.time_range is not None:
        time_start, time_end = args.time_range
        if time_start >= time_end:
            raise ValueError("--time-range START must be less than END")
        print(f"Time range filter: [{time_start}, {time_end}] s")

    timelines_by_tag: Dict[str, Dict[str, List[float]]] = {}
    for tag, rec in records_by_tag.items():
        speed = infer_consume_speed(tag, rec, args.consume_speed)
        timeline = simulate_timeline(rec=rec, consume_speed=speed, num_steps=args.num_steps)
        if args.time_range is not None:
            timeline = slice_timeline(timeline, time_start, time_end)
        timelines_by_tag[tag] = timeline
        print(f"Simulated tag={tag}, consume_speed={speed:.4f} tok/s")

    plot_metric(
        timelines_by_tag=timelines_by_tag,
        metric_key="waiting_requests_num",
        ylabel="Waiting Requests Num",
        title="Waiting Requests Num Over Time",
        output_path=args.out_dir / "waiting_requests_num.png",
    )
    plot_metric(
        timelines_by_tag=timelines_by_tag,
        metric_key="queued_requests_num",
        ylabel="Queued Requests Num",
        title="Queued Requests Num Over Time",
        output_path=args.out_dir / "queued_requests_num.png",
    )
    plot_metric(
        timelines_by_tag=timelines_by_tag,
        metric_key="valid_throughput_tps",
        ylabel="Valid Throughput (tokens/s)",
        title="Valid Throughput Over Time",
        output_path=args.out_dir / "valid_throughput.png",
    )

    if args.dump_json:
        output_json = args.out_dir / "timeline_metrics.json"
        with output_json.open("w", encoding="utf-8") as f:
            json.dump(timelines_by_tag, f, ensure_ascii=False)
        print(f"Saved: {output_json}")


if __name__ == "__main__":
    main()
