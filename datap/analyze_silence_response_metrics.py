#!/usr/bin/env python3
"""Calculate user-response metrics after eligible 15-second silence opportunities."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, median


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"
DATA_ROOT = Path(r"E:\qwer\new")
EPISODE_SOURCE = OUTPUT / "proactive_trigger_episodes_actual.csv"
DETAIL_OUTPUT = OUTPUT / "silence_response_opportunities_actual.csv"
USER_OUTPUT = OUTPUT / "silence_response_metrics_actual.csv"
NOTES_OUTPUT = OUTPUT / "SILENCE_RESPONSE_METRICS.md"

RUNS = {
    "xkx2": "20260813_164327_xkx2",
    "hn": "20260814_154822_hn",
    "ht": "20260814_171114_ht",
    "ds": "20260817_193646_ds",
}
WINDOW_S = 30.0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def event_time(event: dict[str, object]) -> float:
    return float(event.get("demo_time_s", 0.0))


def first_event_after(
    events: list[dict[str, object]], start_s: float, event_name: str, message_contains: str = ""
) -> dict[str, object] | None:
    return next(
        (
            event for event in events
            if event_time(event) >= start_s
            and event.get("event") == event_name
            and message_contains in str(event.get("message", ""))
        ),
        None,
    )


def build_detail() -> tuple[list[dict[str, object]], dict[str, dict[str, int]]]:
    episodes = [
        row for row in read_csv(EPISODE_SOURCE)
        if row["策略"] == "沉默触发"
    ]
    detail: list[dict[str, object]] = []
    totals: dict[str, dict[str, int]] = {}

    for participant, run_name in RUNS.items():
        run_dir = DATA_ROOT / run_name / "camera_asr_dialogue"
        timeline = [
            json.loads(line)
            for line in (run_dir / "main_timeline.jsonl").read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
        turns = json.loads((run_dir / "dialogue_turns.json").read_text(encoding="utf-8-sig"))
        user_asr_times = sorted(
            event_time(event) for event in timeline if event.get("event") == "USER_ASR"
        )
        completed_user_turns = sum(
            turn.get("turn_type") == "dialogue" and bool(str(turn.get("user_text", "")).strip())
            for turn in turns
        )
        totals[participant] = {
            "valid_user_asr": len(user_asr_times),
            "completed_user_turns": completed_user_turns,
        }
        session_end = max(event_time(event) for event in timeline)

        selected = [row for row in episodes if row["用户"] == participant]
        for episode in selected:
            first_trigger = float(episode["首次触发时间_s"])
            last_trigger = float(episode["末次触发时间_s"])
            e2e_users = [
                value for value in user_asr_times
                if first_trigger < value <= first_trigger + WINDOW_S
            ]
            e2e_response = bool(e2e_users)
            e2e_full_window = session_end >= first_trigger + WINDOW_S
            e2e_analyzable = e2e_response or e2e_full_window

            proactive_robot = None
            tts_stop = None
            if episode["结果"] == "成功输出":
                proactive_robot = first_event_after(timeline, last_trigger, "PROACTIVE_ROBOT")
                if proactive_robot is not None:
                    tts_stop = first_event_after(
                        timeline, event_time(proactive_robot), "TTS_STATUS", "STOPPED"
                    )

            robot_time = event_time(proactive_robot) if proactive_robot else None
            output_end = event_time(tts_stop) if tts_stop else robot_time
            post_users: list[float] = []
            post_full_window = False
            post_analyzable = False
            if output_end is not None:
                post_users = [
                    value for value in user_asr_times
                    if output_end < value <= output_end + WINDOW_S
                ]
                post_full_window = session_end >= output_end + WINDOW_S
                post_analyzable = bool(post_users) or post_full_window

            detail.append({
                "用户": participant,
                "沉默机会序号": sum(row["用户"] == participant for row in detail) + 1,
                "首次触发时间_s": round(first_trigger, 3),
                "末次触发时间_s": round(last_trigger, 3),
                "请求次数_重试已合并": int(episode["原始请求次数"]),
                "主动输出结果": episode["结果"],
                "机器人主动输出时间_s": "" if robot_time is None else round(robot_time, 3),
                "主动TTS结束时间_s": "" if output_end is None else round(output_end, 3),
                "日志结束时间_s": round(session_end, 3),
                "端到端30秒观察完整": "是" if e2e_full_window else "否",
                "端到端结局可判定": "是" if e2e_analyzable else "否",
                "触发后30秒内用户重新说话": "是" if e2e_response else ("否" if e2e_full_window else "删失"),
                "触发后30秒内用户发话轮次": len(e2e_users) if e2e_full_window else (f">={len(e2e_users)}" if e2e_users else "删失"),
                "触发到首次用户发话_s": "" if not e2e_users else round(e2e_users[0] - first_trigger, 3),
                "输出后30秒结局可判定": "不适用" if output_end is None else ("是" if post_analyzable else "否"),
                "输出后30秒内用户重新说话": "不适用" if output_end is None else ("是" if post_users else ("否" if post_full_window else "删失")),
                "输出后30秒内用户发话轮次": "不适用" if output_end is None else (len(post_users) if post_full_window else (f">={len(post_users)}" if post_users else "删失")),
                "TTS结束到首次用户发话_s": "" if not post_users else round(post_users[0] - output_end, 3),
            })
    return detail, totals


def avg(values: list[float]) -> str | float:
    return "" if not values else round(mean(values), 3)


def med(values: list[float]) -> str | float:
    return "" if not values else round(median(values), 3)


def summarize(detail: list[dict[str, object]], totals: dict[str, dict[str, int]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for participant in [*RUNS, "合计"]:
        selected = detail if participant == "合计" else [row for row in detail if row["用户"] == participant]
        participants = list(RUNS) if participant == "合计" else [participant]
        e2e_analyzable = [row for row in selected if row["端到端结局可判定"] == "是"]
        e2e_responded = [row for row in e2e_analyzable if row["触发后30秒内用户重新说话"] == "是"]
        complete_count_windows = [row for row in selected if row["端到端30秒观察完整"] == "是"]
        e2e_waits = [float(row["触发到首次用户发话_s"]) for row in e2e_responded]
        post_analyzable = [row for row in selected if row["输出后30秒结局可判定"] == "是"]
        post_responded = [row for row in post_analyzable if row["输出后30秒内用户重新说话"] == "是"]
        post_waits = [float(row["TTS结束到首次用户发话_s"]) for row in post_responded]

        rows.append({
            "用户": participant,
            "15秒沉默机会数": len(selected),
            "端到端30秒结局可判定机会数": len(e2e_analyzable),
            "30秒内用户重新说话机会数": len(e2e_responded),
            "30秒内用户重新说话率": "" if not e2e_analyzable else round(len(e2e_responded) / len(e2e_analyzable), 4),
            "完整30秒窗口内用户发话总轮次": sum(int(row["触发后30秒内用户发话轮次"]) for row in complete_count_windows),
            "每次完整沉默机会的用户发话轮次": "" if not complete_count_windows else round(
                sum(int(row["触发后30秒内用户发话轮次"]) for row in complete_count_windows) / len(complete_count_windows), 3
            ),
            "首次用户发话平均等待_s": avg(e2e_waits),
            "首次用户发话中位等待_s": med(e2e_waits),
            "成功输出后30秒结局可判定数": len(post_analyzable),
            "成功输出后用户回复数": len(post_responded),
            "成功输出后用户回复率": "" if not post_analyzable else round(len(post_responded) / len(post_analyzable), 4),
            "TTS结束到用户回复平均等待_s": avg(post_waits),
            "全场有效用户ASR发话数": sum(totals[name]["valid_user_asr"] for name in participants),
            "全场完成的用户对话轮次": sum(totals[name]["completed_user_turns"] for name in participants),
        })
    return rows


def write_notes() -> None:
    NOTES_OUTPUT.write_text(
        """# 沉默机会后的用户响应指标

## 统计范围

- 当前只包含启用主动策略且日志完整的4次实体实验：xkx2、hn、ht、ds。
- xkx1未纳入，因为此前已确定不进入主动策略分析。
- 当前结果只能描述条件Z，尚不能形成X、Y、Z组间比较。

## 指标定义

1. 15秒沉默机会：采用已清除ASR/TTS活动冲突、并合并超时重试后的“沉默触发事件”。
2. 30秒内用户重新说话：从沉默事件首次触发时刻开始，30秒内出现有效`USER_ASR`。
3. 每次沉默机会的用户发话轮次：同一30秒窗口内的`USER_ASR`数量。窗口不足30秒时标为删失，不用于轮次均值。
4. 首次用户发话等待时间：首次触发到窗口内第一个`USER_ASR`的时间差。没有在30秒内回复的不计入等待时间均值，需同时报告未回复和删失数量。
5. 全场用户发话回合：同时报告全部有效`USER_ASR`和写入`dialogue_turns.json`的完成对话轮次。两者差异来自实验结束前尚未完成机器人回复的用户发话。

## 补充诊断口径

端到端30秒窗口会包含LLM生成和TTS播放时间。为区分系统延迟与用户是否愿意回应，表中额外统计“TTS结束后30秒内用户回复率”和“TTS结束到首次用户回复等待时间”。该口径适合解释条件Z内部机制，但不能直接用于X、Y、Z组间比较，因为X和Y没有主动TTS。
""",
        encoding="utf-8",
    )


def main() -> None:
    detail, totals = build_detail()
    summary = summarize(detail, totals)
    write_csv(DETAIL_OUTPUT, detail)
    write_csv(USER_OUTPUT, summary)
    write_notes()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
