#!/usr/bin/env python3
"""Audit proactive trigger outcomes in the four measured runs."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"
DATA_ROOT = Path(r"E:\qwer\new")
EVENTS_CSV = OUTPUT / "proactive_trigger_outcomes_actual.csv"
EPISODES_CSV = OUTPUT / "proactive_trigger_episodes_actual.csv"
SUMMARY_CSV = OUTPUT / "proactive_trigger_episode_summary_actual.csv"
RUNS = {
    "xkx2": "20260813_164327_xkx2",
    "hn": "20260814_154822_hn",
    "ht": "20260814_171114_ht",
    "ds": "20260817_193646_ds",
}
ACTIVITY_EVENTS = {
    "ASR_SPEECH", "ASR_CONNECTED", "ASR_PARTIAL", "ASR_TEXT",
    "USER_ASR_BUFFER", "USER_ASR", "ROBOT",
}


def strategy(message: str) -> str:
    return "动作触发" if message.startswith("salient_action_comment") else "沉默触发"


def reason_group(message: str) -> str:
    lower = message.lower()
    if "deadline exceeded" in lower or "late cloud result" in lower:
        return "云端生成超时"
    if "user/asr/tts activity" in lower:
        return "用户、ASR或TTS活动冲突/请求失败"
    if "stale" in lower:
        return "视觉事件过期"
    if message:
        return message
    return "未记录明确原因"


def read_events() -> tuple[list[dict[str, object]], dict[str, list[dict[str, object]]]]:
    rows: list[dict[str, object]] = []
    timelines: dict[str, list[dict[str, object]]] = {}
    for user, run_name in RUNS.items():
        path = DATA_ROOT / run_name / "camera_asr_dialogue" / "main_timeline.jsonl"
        events = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()]
        timelines[user] = events
        trigger_number = 0
        for index, event in enumerate(events):
            if event.get("event") != "PROACTIVE_TRIGGER":
                continue
            trigger_number += 1
            terminal = None
            for later in events[index + 1:]:
                if later.get("event") in {"PROACTIVE_ROBOT", "PROACTIVE_DROP", "PROACTIVE_TRIGGER"}:
                    terminal = later
                    break
            terminal_event = terminal.get("event") if terminal else "NO_TERMINAL_EVENT"
            terminal_message = terminal.get("message", "") if terminal else ""
            successful = terminal_event == "PROACTIVE_ROBOT"
            terminal_index = events.index(terminal) if terminal else index
            terminal_window = events[index + 1:terminal_index]
            has_asr_activity = any(item.get("event") in ACTIVITY_EVENTS for item in terminal_window)
            has_tts_activity = any(
                item.get("event") == "TTS_STATUS" and "playing=True" in item.get("message", "")
                for item in terminal_window
            )
            failure_reason = ""
            if not successful:
                failure_reason = ("用户、ASR或TTS活动冲突/请求失败"
                                  if has_asr_activity or has_tts_activity
                                  else reason_group(terminal_message))
            rows.append({
                "用户": user,
                "触发序号": trigger_number,
                "触发时间_s": round(float(event.get("demo_time_s", 0)), 3),
                "策略": strategy(event.get("message", "")),
                "结果": "成功输出" if successful else "未成功输出",
                "终止事件": terminal_event,
                "失败原因": failure_reason,
                "日志原文": terminal_message,
                "_event_index": index,
            })
    return rows, timelines


def build_episodes(rows: list[dict[str, object]], timelines: dict[str, list[dict[str, object]]]) -> list[dict[str, object]]:
    episodes: list[dict[str, object]] = []
    for user in RUNS:
        attempts = [row for row in rows if row["用户"] == user]
        current: list[dict[str, object]] = []

        def flush() -> None:
            nonlocal current
            if not current:
                return
            success = any(row["结果"] == "成功输出" for row in current)
            episodes.append({
                "用户": user,
                "事件序号": sum(row["用户"] == user for row in episodes) + 1,
                "策略": current[0]["策略"],
                "首次触发时间_s": current[0]["触发时间_s"],
                "末次触发时间_s": current[-1]["触发时间_s"],
                "原始请求次数": len(current),
                "结果": "成功输出" if success else "未成功输出",
                "合并说明": "超时重试后成功" if success and len(current) > 1 else ("连续超时重试" if len(current) > 1 else "单次请求"),
            })
            current = []

        for attempt in attempts:
            # Activity conflicts are not valid proactive opportunities.
            if attempt["失败原因"] == "用户、ASR或TTS活动冲突/请求失败":
                flush()
                continue
            if current:
                previous = current[-1]
                between = timelines[user][int(previous["_event_index"]) + 1:int(attempt["_event_index"])]
                context_reset = any(event.get("event") == "ASR_TEXT" for event in between)
                if attempt["策略"] != current[0]["策略"] or context_reset:
                    flush()
            current.append(attempt)
            if attempt["结果"] == "成功输出":
                flush()
        flush()
    return episodes


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows, timelines = read_events()
    public_rows = [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]
    write_csv(EVENTS_CSV, public_rows)
    episodes = build_episodes(rows, timelines)
    write_csv(EPISODES_CSV, episodes)
    summary_rows: list[dict[str, object]] = []
    for strategy_name in ("动作触发", "沉默触发", "全部"):
        selected = episodes if strategy_name == "全部" else [row for row in episodes if row["策略"] == strategy_name]
        success = sum(row["结果"] == "成功输出" for row in selected)
        summary_rows.append({
            "策略": strategy_name,
            "有效触发事件数": len(selected),
            "成功输出次数": success,
            "成功率": success / len(selected) if selected else 0,
            "超时后仍失败的事件数": len(selected) - success,
            "合并的重试请求数": sum(int(row["原始请求次数"]) - 1 for row in selected),
        })
    write_csv(SUMMARY_CSV, summary_rows)
    excluded_conflicts = sum(row["失败原因"] == "用户、ASR或TTS活动冲突/请求失败" for row in rows)
    print(json.dumps({"raw_requests": len(rows), "excluded_conflicts": excluded_conflicts,
                      "episodes": len(episodes), "summary": summary_rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
