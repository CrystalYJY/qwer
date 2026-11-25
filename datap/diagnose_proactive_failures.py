#!/usr/bin/env python3
"""Inspect timeout attempts for hidden ASR/TTS activity and concurrent work."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"
DATA_ROOT = Path(r"E:\qwer\new")
OUT = OUTPUT / "proactive_timeout_diagnosis_actual.csv"
FAILED_OUT = OUTPUT / "proactive_failed_episode_diagnosis_actual.csv"
EPISODES = OUTPUT / "proactive_trigger_episodes_actual.csv"
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


def main() -> None:
    rows: list[dict[str, object]] = []
    for user, run_name in RUNS.items():
        path = DATA_ROOT / run_name / "camera_asr_dialogue" / "main_timeline.jsonl"
        events = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()]
        session_end = max(float(event.get("demo_time_s", 0)) for event in events)
        for index, event in enumerate(events):
            if event.get("event") != "PROACTIVE_TRIGGER":
                continue
            terminal_index = next((j for j in range(index + 1, len(events))
                                   if events[j].get("event") in {"PROACTIVE_ROBOT", "PROACTIVE_DROP", "PROACTIVE_TRIGGER"}), None)
            if terminal_index is None or events[terminal_index].get("event") != "PROACTIVE_DROP":
                continue
            terminal = events[terminal_index]
            if "deadline exceeded" not in terminal.get("message", "").lower():
                continue
            window = events[index + 1:terminal_index]
            activity = [item.get("event") for item in window if item.get("event") in ACTIVITY_EVENTS]
            tts_playing = [item for item in window if item.get("event") == "TTS_STATUS" and "playing=True" in item.get("message", "")]
            vlm = [item.get("event") for item in window if item.get("event") in {"VLM_SUBMIT", "VLM_RETURN"}]
            rows.append({
                "用户": user,
                "触发时间_s": round(float(event.get("demo_time_s", 0)), 3),
                "策略": "动作触发" if event.get("message", "").startswith("salient_action_comment") else "沉默触发",
                "超时等待_s": round(float(terminal.get("demo_time_s", 0)) - float(event.get("demo_time_s", 0)), 3),
                "期间存在用户或ASR活动": "是" if activity else "否",
                "期间存在TTS播放": "是" if tts_playing else "否",
                "活动事件": ";".join(activity),
                "并发VLM事件数": len(vlm),
                "距离日志结束_s": round(session_end - float(terminal.get("demo_time_s", 0)), 3),
            })
    with OUT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with EPISODES.open(encoding="utf-8-sig", newline="") as handle:
        failed_episodes = [row for row in csv.DictReader(handle) if row["结果"] == "未成功输出"]
    failed_rows: list[dict[str, object]] = []
    for episode in failed_episodes:
        match = next(row for row in rows
                     if row["用户"] == episode["用户"]
                     and abs(float(row["触发时间_s"]) - float(episode["首次触发时间_s"])) < 0.01)
        near_end = float(match["距离日志结束_s"]) <= 15
        concurrent_vlm = int(match["并发VLM事件数"]) > 0
        if near_end:
            interpretation = "7秒云端超时；位于实验尾段，没有再次重试机会"
        elif concurrent_vlm:
            interpretation = "7秒云端超时；同期存在VLM请求，可能有云端或网络资源竞争"
        else:
            interpretation = "7秒云端超时；无用户、ASR、TTS或VLM并发"
        failed_rows.append({
            "用户": episode["用户"],
            "事件序号": episode["事件序号"],
            "策略": episode["策略"],
            "触发时间_s": match["触发时间_s"],
            "超时等待_s": match["超时等待_s"],
            "并发VLM事件数": match["并发VLM事件数"],
            "距离日志结束_s": match["距离日志结束_s"],
            "分析结论": interpretation,
        })
    with FAILED_OUT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(failed_rows[0]))
        writer.writeheader()
        writer.writerows(failed_rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
