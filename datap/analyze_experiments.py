#!/usr/bin/env python3
"""Create objective summaries and figures for physical-robot experiment logs.

The script is deliberately read-only with respect to the raw experiment folder.
It accepts the copied ``new`` folder on the USB disk and writes only CSV/PNG/MD
outputs under ``datap/outputs`` (or a supplied output directory).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_DATA_ROOT = Path("/media/crystal/KINGSTON/qwer/new")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

RUN_LABELS = {
    "20260813_163910_xkx1": "xkx1",
    "20260813_164327_xkx2": "xkx2",
    "20260814_154822_hn": "hn",
    "20260814_171114_ht": "ht",
    "20260817_193646_ds": "ds",
}
STYLE_ORDER = ("neutral", "positive", "care", "sad", "angry")
SCHEDULE_ORDER = ("base", "after_user", "motion_event", "motion_event_priority")


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def normalise_schedule(event: dict[str, Any]) -> str:
    result = event.get("vlm_result") or {}
    schedule = result.get("schedule") or event.get("schedule") or "unknown"
    schedule = str(schedule)
    if schedule == "meaningful_action":
        return "motion_event"
    return schedule


def analyse_run(run_dir: Path) -> tuple[dict[str, Any], Counter[str], Counter[str]]:
    data_dir = run_dir / "camera_asr_dialogue"
    turns = read_json(data_dir / "dialogue_turns.json")
    visual_events = read_json(data_dir / "vlm_frame_evidence.json")
    if not isinstance(turns, list) or not isinstance(visual_events, list):
        raise ValueError(f"Unexpected JSON structure in {run_dir}")

    dialogue_turns = [turn for turn in turns if turn.get("turn_type") == "dialogue"]
    proactive_turns = [turn for turn in turns if turn.get("turn_type") == "proactive"]
    dialogue_latencies = [value for turn in dialogue_turns if (value := number(turn.get("dialogue_latency_s"))) is not None]
    speech_durations = [value for turn in dialogue_turns if (value := number(turn.get("speech_duration_s"))) is not None]
    duration = max((number(turn.get("time_s")) or 0.0 for turn in turns), default=0.0)
    styles = Counter(str(turn.get("response_style") or "unknown") for turn in turns)

    confirmed = [event for event in visual_events if event.get("status") == "confirmed"]
    vlm_latencies = [
        value
        for event in confirmed
        if isinstance(event.get("vlm_result"), dict)
        and (value := number(event["vlm_result"].get("latency_s"))) is not None
    ]
    schedules = Counter(normalise_schedule(event) for event in confirmed)

    run_id = run_dir.name
    row = {
        "run_id": run_id,
        "participant": RUN_LABELS.get(run_id, run_id),
        "duration_s": round(duration, 3),
        "duration_min": round(duration / 60, 3),
        "total_turns": len(turns),
        "dialogue_turns": len(dialogue_turns),
        "proactive_turns": len(proactive_turns),
        "proactive_ratio": round(len(proactive_turns) / len(turns), 4) if turns else 0.0,
        "dialogue_turns_per_min": round(len(dialogue_turns) / (duration / 60), 3) if duration else 0.0,
        "mean_dialogue_latency_s": round(mean(dialogue_latencies) or 0.0, 3),
        "total_user_speech_s": round(sum(speech_durations), 3),
        "visual_requests": len(visual_events),
        "confirmed_visual_requests": len(confirmed),
        "visual_success_rate": round(len(confirmed) / len(visual_events), 4) if visual_events else 0.0,
        "mean_vlm_latency_s": round(mean(vlm_latencies) or 0.0, 3),
        "keyframe_count": len(list((data_dir / "keyframes").glob("*.jpg"))),
        "trigger_frame_count": len(list((data_dir / "trigger_frames").glob("*.jpg"))),
    }
    row.update({f"style_{style}": styles[style] for style in STYLE_ORDER})
    row.update({f"schedule_{schedule}": schedules[schedule] for schedule in SCHEDULE_ORDER})
    row["schedule_unknown"] = schedules["unknown"]
    return row, styles, schedules


def save_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def setup_plot() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "figure.dpi": 150})


def plot_turns(rows: list[dict[str, Any]], output: Path) -> None:
    labels = [row["participant"] for row in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    dialogue = np.array([row["dialogue_turns"] for row in rows])
    proactive = np.array([row["proactive_turns"] for row in rows])
    ax.bar(x, dialogue, label="User-initiated dialogue", color="#4C78A8")
    ax.bar(x, proactive, bottom=dialogue, label="Robot proactive turn", color="#F58518")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Turns")
    ax.set_title("Interaction volume by experiment")
    ax.legend(frameon=False)
    for index, total in enumerate(dialogue + proactive):
        ax.text(index, total + 0.25, str(int(total)), ha="center")
    fig.tight_layout()
    fig.savefig(output / "01_interaction_volume.png", bbox_inches="tight")
    plt.close(fig)


def plot_latency(rows: list[dict[str, Any]], output: Path) -> None:
    labels = [row["participant"] for row in rows]
    x = np.arange(len(rows))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x - width / 2, [row["mean_dialogue_latency_s"] for row in rows], width, label="Dialogue response", color="#54A24B")
    ax.bar(x + width / 2, [row["mean_vlm_latency_s"] for row in rows], width, label="VLM request", color="#E45756")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Mean latency (s)")
    ax.set_title("Mean response latency by experiment")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "02_latency.png", bbox_inches="tight")
    plt.close(fig)


def plot_schedules(rows: list[dict[str, Any]], output: Path) -> None:
    labels = [row["participant"] for row in rows]
    colors = {"base": "#4C78A8", "after_user": "#72B7B2", "motion_event": "#F58518", "motion_event_priority": "#E45756"}
    x = np.arange(len(rows))
    bottom = np.zeros(len(rows))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for schedule in SCHEDULE_ORDER:
        values = np.array([row[f"schedule_{schedule}"] for row in rows])
        ax.bar(x, values, bottom=bottom, label=schedule, color=colors[schedule])
        bottom += values
    ax.set_xticks(x, labels)
    ax.set_ylabel("Confirmed VLM requests")
    ax.set_title("Visual scheduling composition")
    ax.legend(frameon=False, title="Schedule")
    fig.tight_layout()
    fig.savefig(output / "03_visual_scheduling.png", bbox_inches="tight")
    plt.close(fig)


def plot_styles(rows: list[dict[str, Any]], output: Path) -> None:
    labels = [row["participant"] for row in rows]
    colors = {"neutral": "#A0A0A0", "positive": "#54A24B", "care": "#B279A2", "sad": "#4C78A8", "angry": "#E45756"}
    x = np.arange(len(rows))
    bottom = np.zeros(len(rows))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for style in STYLE_ORDER:
        values = np.array([row[f"style_{style}"] for row in rows])
        if values.sum() == 0:
            continue
        ax.bar(x, values, bottom=bottom, label=style, color=colors[style])
        bottom += values
    ax.set_xticks(x, labels)
    ax.set_ylabel("Turns")
    ax.set_title("Robot response styles")
    ax.legend(frameon=False, title="Style")
    fig.tight_layout()
    fig.savefig(output / "04_response_styles.png", bbox_inches="tight")
    plt.close(fig)


def write_report(rows: list[dict[str, Any]], output: Path) -> None:
    total_turns = sum(row["total_turns"] for row in rows)
    total_dialogue = sum(row["dialogue_turns"] for row in rows)
    total_proactive = sum(row["proactive_turns"] for row in rows)
    total_visual = sum(row["visual_requests"] for row in rows)
    confirmed = sum(row["confirmed_visual_requests"] for row in rows)
    weighted_dialogue_latency = sum(row["mean_dialogue_latency_s"] * row["dialogue_turns"] for row in rows) / total_dialogue
    weighted_vlm_latency = sum(row["mean_vlm_latency_s"] * row["visual_requests"] for row in rows) / total_visual
    lines = [
        "# Objective log summary",
        "",
        f"- Experiments: {len(rows)}",
        f"- Interaction turns: {total_turns} ({total_dialogue} dialogue, {total_proactive} proactive)",
        f"- Confirmed VLM requests: {confirmed}/{total_visual} ({confirmed / total_visual:.1%})",
        f"- Weighted mean dialogue latency: {weighted_dialogue_latency:.2f} s",
        f"- Weighted mean VLM latency: {weighted_vlm_latency:.2f} s",
        "",
        "## Interpretation limits",
        "",
        "These are system-log measures, not ground-truth correctness or user-experience scores. "
        "VLM accuracy, ASR accuracy, response relevance, and appropriateness of proactive turns require later human annotation.",
        "",
        "`robot_timeline.jsonl` is intentionally not used for turn counts because at least one run contains replayed TTS requests. "
        "`dialogue_turns.json` is the canonical turn-level source.",
    ]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    run_dirs = sorted(path for path in args.data_root.iterdir() if (path / "camera_asr_dialogue" / "dialogue_turns.json").exists())
    if not run_dirs:
        raise SystemExit(f"No experiment runs found under {args.data_root}")
    rows = [analyse_run(run_dir)[0] for run_dir in run_dirs]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_plot()
    save_csv(rows, args.output_dir / "run_summary.csv")
    plot_turns(rows, args.output_dir)
    plot_latency(rows, args.output_dir)
    plot_schedules(rows, args.output_dir)
    plot_styles(rows, args.output_dir)
    write_report(rows, args.output_dir)
    print(f"Wrote {len(rows)} run summaries and 4 figures to {args.output_dir}")


if __name__ == "__main__":
    main()
