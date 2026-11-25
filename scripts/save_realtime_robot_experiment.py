#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_DEST = Path("/media/crystal/KINGSTON/qwer/robot_experiments")
DEFAULT_OUTPUT = Path("streaming_camera_outputs/camera_asr_dialogue")
DEFAULT_FRAME_BRIDGE = Path("streaming_camera_outputs/frame_bridge")
DEFAULT_CONFIG = Path("runtime_configs/realtime_robot_dialogue.json")


def load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def safe_name(raw: str) -> str:
    text = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in raw.strip())
    return text.strip("_") or "experiment"


def run_git(args: list[str], root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=8,
            check=False,
        )
        return result.stdout.strip()
    except Exception as exc:
        return f"git command failed: {exc}"


def copy_file_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists() or not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def copy_tree_if_exists(src: Path, dst: Path, ignore_archive: bool = False) -> bool:
    if not src.exists() or not src.is_dir():
        return False
    ignore = shutil.ignore_patterns("archive") if ignore_archive else None
    shutil.copytree(src, dst, dirs_exist_ok=True, ignore=ignore)
    return True


def count_log_lines(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as file:
            return sum(1 for _ in file)
    except Exception:
        return 0


def summarize(output_dir: Path) -> dict[str, Any]:
    dialogue_turns = load_json(output_dir / "dialogue_turns.json")
    vlm_evidence = load_json(output_dir / "vlm_frame_evidence.json")
    keyframes_dir = output_dir / "keyframes"
    trigger_frames_dir = output_dir / "trigger_frames"
    turns = dialogue_turns if isinstance(dialogue_turns, list) else []
    vlm_records = vlm_evidence if isinstance(vlm_evidence, list) else []
    proactive_turns = [
        turn for turn in turns
        if isinstance(turn, dict) and turn.get("turn_type") == "proactive"
    ]
    user_turns = [
        turn for turn in turns
        if isinstance(turn, dict) and str(turn.get("user_text", "")).strip()
    ]
    confirmed_actions = [event for event in vlm_records if isinstance(event, dict) and event.get("status") == "confirmed"]
    salience_counts: dict[str, int] = {}
    style_counts: dict[str, int] = {}
    vlm_errors = 0
    latencies: list[float] = []
    for event in vlm_records:
        if not isinstance(event, dict):
            continue
        result = event.get("vlm_result") or {}
        if result.get("error"):
            vlm_errors += 1
        salience = str(result.get("salience", ""))
        salience_counts[salience] = salience_counts.get(salience, 0) + 1
        style = str(result.get("comment_style", ""))
        style_counts[style] = style_counts.get(style, 0) + 1
        latency = result.get("latency_s")
        if isinstance(latency, (int, float)):
            latencies.append(float(latency))
    return {
        "dialogue_turn_count": len(turns),
        "user_turn_count": len(user_turns),
        "proactive_turn_count": len(proactive_turns),
        "visual_event_count": len(vlm_records),
        "confirmed_visual_event_count": len(confirmed_actions),
        "vlm_error_count": vlm_errors,
        "keyframe_count": len(list(keyframes_dir.glob("*.jpg"))) if keyframes_dir.exists() else 0,
        "trigger_frame_count": len(list(trigger_frames_dir.glob("*.jpg"))) if trigger_frames_dir.exists() else 0,
        "vlm_evidence_count": len(vlm_records),
        "avg_vlm_latency_s": round(sum(latencies) / len(latencies), 3) if latencies else None,
        "salience_counts": salience_counts,
        "comment_style_counts": style_counts,
        "dialogue_log_lines": count_log_lines(output_dir / "dialogue.log"),
        "vision_log_lines": count_log_lines(output_dir / "vision.log"),
        "system_log_lines": count_log_lines(output_dir / "system.log"),
    }


def write_index(dest_root: Path, metadata: dict[str, Any]) -> None:
    dest_root.mkdir(parents=True, exist_ok=True)
    jsonl_path = dest_root / "experiment_index.jsonl"
    with jsonl_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(metadata, ensure_ascii=False) + "\n")
    csv_path = dest_root / "experiment_index.csv"
    fields = [
        "run_id",
        "label",
        "created_at",
        "experiment_dir",
        "dialogue_turn_count",
        "user_turn_count",
        "proactive_turn_count",
        "visual_event_count",
        "confirmed_visual_event_count",
        "vlm_error_count",
        "keyframe_count",
        "trigger_frame_count",
        "vlm_evidence_count",
        "avg_vlm_latency_s",
    ]
    exists = csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if not exists:
            writer.writeheader()
        row = {field: metadata.get(field, "") for field in fields}
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Save one realtime robot experiment run for later analysis.")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--experiment-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frame-bridge-dir", type=Path, default=DEFAULT_FRAME_BRIDGE)
    parser.add_argument("--config-json", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--label", default="")
    parser.add_argument("--notes", default="")
    parser.add_argument("--include-frame-bridge", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path.cwd().resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    config_json = args.config_json if args.config_json.is_absolute() else root / args.config_json
    frame_bridge_dir = args.frame_bridge_dir if args.frame_bridge_dir.is_absolute() else root / args.frame_bridge_dir
    if args.experiment_dir is not None:
        experiment_dir = args.experiment_dir
        dest_root = experiment_dir.parent
        run_id = experiment_dir.name
    else:
        dest_root = args.dest
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        label = safe_name(args.label or "realtime_robot")
        run_id = f"{timestamp}_{label}"
        experiment_dir = dest_root / run_id

    summary = summarize(output_dir)
    metadata: dict[str, Any] = {
        "run_id": run_id,
        "label": args.label or "realtime_robot",
        "notes": args.notes,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "experiment_dir": str(experiment_dir),
        "source_root": str(root),
        "source_output_dir": str(output_dir),
        "frame_bridge_dir": str(frame_bridge_dir),
        "config_json": str(config_json),
        "runtime_config": load_json(config_json),
        "git_head": run_git(["rev-parse", "HEAD"], root),
        **summary,
    }

    if args.dry_run:
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return

    experiment_dir.mkdir(parents=True, exist_ok=args.experiment_dir is not None)
    copied: list[str] = []

    target_output_dir = experiment_dir / "camera_asr_dialogue"
    if output_dir.resolve() == target_output_dir.resolve():
        copied.append("camera_asr_dialogue:direct")
    elif copy_tree_if_exists(output_dir, target_output_dir, ignore_archive=True):
        copied.append("camera_asr_dialogue")
    target_frame_bridge_dir = experiment_dir / "frame_bridge"
    if frame_bridge_dir.resolve() == target_frame_bridge_dir.resolve():
        copied.append("frame_bridge:direct")
    elif args.include_frame_bridge and copy_tree_if_exists(frame_bridge_dir, target_frame_bridge_dir):
        copied.append("frame_bridge")

    code_dir = experiment_dir / "code_state"
    code_dir.mkdir(parents=True, exist_ok=True)
    (code_dir / "git_head.txt").write_text(metadata["git_head"] + "\n", encoding="utf-8")
    (code_dir / "git_status.txt").write_text(run_git(["status", "--short"], root) + "\n", encoding="utf-8")
    (code_dir / "git_diff.patch").write_text(run_git(["diff", "--"], root) + "\n", encoding="utf-8")

    metadata["copied"] = copied
    (experiment_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (experiment_dir / "summary.md").write_text(
        "\n".join(
            [
                f"# Realtime Robot Experiment {run_id}",
                "",
                f"- Label: {metadata['label']}",
                f"- Created: {metadata['created_at']}",
                f"- Dialogue turns: {metadata['dialogue_turn_count']}",
                f"- User turns: {metadata['user_turn_count']}",
                f"- Proactive turns: {metadata['proactive_turn_count']}",
                f"- Visual events: {metadata['visual_event_count']}",
                f"- VLM errors: {metadata['vlm_error_count']}",
                f"- Keyframes: {metadata['keyframe_count']}",
                f"- Trigger frames: {metadata['trigger_frame_count']}",
                f"- VLM evidence records: {metadata['vlm_evidence_count']}",
                f"- Avg VLM latency: {metadata['avg_vlm_latency_s']}",
                f"- Notes: {metadata['notes']}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    write_index(dest_root, metadata)
    print(f"[experiment_save] saved: {experiment_dir}")


if __name__ == "__main__":
    main()
