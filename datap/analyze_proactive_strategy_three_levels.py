#!/usr/bin/env python3
"""Summarize silence-triggered, action-triggered, and overall proactive outcomes."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, median


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"
DATA_ROOT = Path(r"E:\qwer\new")
EPISODES_PATH = OUTPUT / "proactive_trigger_episodes_actual.csv"
VLM_REVIEW_PATH = OUTPUT / "vlm_manual_review.csv"
DETAIL_PATH = OUTPUT / "proactive_strategy_outcomes_actual.csv"
SUMMARY_PATH = OUTPUT / "proactive_strategy_three_levels_actual.csv"
BY_USER_PATH = OUTPUT / "proactive_strategy_three_levels_by_user_actual.csv"
NOTES_PATH = OUTPUT / "PROACTIVE_STRATEGY_THREE_LEVELS.md"
WINDOW_S = 30.0

RUNS = {
    "xkx2": "20260813_164327_xkx2",
    "hn": "20260814_154822_hn",
    "ht": "20260814_171114_ht",
    "ds": "20260817_193646_ds",
}


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


def build_detail() -> tuple[list[dict[str, object]], dict[str, dict[str, int]]]:
    episodes = read_csv(EPISODES_PATH)
    reviews = {
        (row["用户"], row["事件ID"]): row["人工判定"]
        for row in read_csv(VLM_REVIEW_PATH)
    }
    detail: list[dict[str, object]] = []
    session_totals: dict[str, dict[str, int]] = {}

    for participant, run_name in RUNS.items():
        run_dir = DATA_ROOT / run_name / "camera_asr_dialogue"
        timeline = [
            json.loads(line)
            for line in (run_dir / "main_timeline.jsonl").read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
        turns = json.loads((run_dir / "dialogue_turns.json").read_text(encoding="utf-8-sig"))
        user_asr = sorted(
            event_time(event) for event in timeline if event.get("event") == "USER_ASR"
        )
        session_end = max(event_time(event) for event in timeline)
        session_totals[participant] = {
            "valid_user_asr": len(user_asr),
            "completed_user_turns": sum(
                turn.get("turn_type") == "dialogue" and bool(str(turn.get("user_text", "")).strip())
                for turn in turns
            ),
        }

        selected = [row for row in episodes if row["用户"] == participant]
        successful: list[dict[str, object]] = []
        for episode in selected:
            if episode["结果"] != "成功输出":
                continue
            last_trigger = float(episode["末次触发时间_s"])
            robot_event = next(
                event for event in timeline
                if event.get("event") == "PROACTIVE_ROBOT" and event_time(event) >= last_trigger
            )
            robot_time = event_time(robot_event)
            tts_stop = next(
                (
                    event for event in timeline
                    if event.get("event") == "TTS_STATUS"
                    and "STOPPED" in str(event.get("message", ""))
                    and event_time(event) >= robot_time
                ),
                None,
            )
            output_end = event_time(tts_stop) if tts_stop else robot_time
            proactive_turn = min(
                (turn for turn in turns if turn.get("turn_type") == "proactive"),
                key=lambda turn: abs(float(turn["time_s"]) - last_trigger),
            )
            if abs(float(proactive_turn["time_s"]) - last_trigger) > 0.2:
                raise RuntimeError(f"Cannot match proactive turn for {participant} at {last_trigger}")
            successful.append({
                "episode": episode,
                "robot_time": robot_time,
                "output_end": output_end,
                "reply": str(proactive_turn.get("reply", "")),
                "visual_event_id": str(proactive_turn.get("primary_visual_event_id", "")),
            })

        successful.sort(key=lambda item: float(item["robot_time"]))
        success_by_event = {
            int(item["episode"]["事件序号"]): (index, item)
            for index, item in enumerate(successful)
        }

        for episode in selected:
            event_number = int(episode["事件序号"])
            strategy = episode["策略"]
            output_success = episode["结果"] == "成功输出"
            row: dict[str, object] = {
                "用户": participant,
                "事件序号": event_number,
                "策略": strategy,
                "首次触发时间_s": float(episode["首次触发时间_s"]),
                "末次触发时间_s": float(episode["末次触发时间_s"]),
                "请求次数_重试已合并": int(episode["原始请求次数"]),
                "主动输出结果": episode["结果"],
                "机器人主动输出时间_s": "",
                "主动TTS结束时间_s": "",
                "30秒窗口完整": "不适用",
                "用户回复结局": "不适用",
                "TTS结束到首次用户回复_s": "",
                "窗口内用户发话轮次": "",
                "主动回复文本": "",
                "关联视觉事件ID": "",
                "动作视觉依据人工判定": "不适用" if strategy == "沉默触发" else "无输出",
                "动作评论恰当性人工判定": "未标注" if strategy == "动作触发" and output_success else "不适用",
            }
            if output_success:
                index, item = success_by_event[event_number]
                output_end = float(item["output_end"])
                next_robot_time = (
                    float(successful[index + 1]["robot_time"])
                    if index + 1 < len(successful) else float("inf")
                )
                window_limit = min(output_end + WINDOW_S, next_robot_time, session_end)
                responses = [time for time in user_asr if output_end < time <= window_limit]
                window_complete = output_end + WINDOW_S <= min(next_robot_time, session_end)
                if responses:
                    response_outcome = "已回复"
                elif window_complete:
                    response_outcome = "30秒内未回复"
                elif next_robot_time < session_end:
                    response_outcome = "删失_被下一次主动输出截断"
                else:
                    response_outcome = "删失_实验结束"

                visual_event_id = str(item["visual_event_id"])
                visual_judgement = (
                    reviews.get((participant, visual_event_id), "缺少人工标注")
                    if strategy == "动作触发" else "不适用"
                )
                row.update({
                    "机器人主动输出时间_s": round(float(item["robot_time"]), 3),
                    "主动TTS结束时间_s": round(output_end, 3),
                    "30秒窗口完整": "是" if window_complete else "否",
                    "用户回复结局": response_outcome,
                    "TTS结束到首次用户回复_s": "" if not responses else round(responses[0] - output_end, 3),
                    "窗口内用户发话轮次": len(responses) if window_complete else (f">={len(responses)}" if responses else "删失"),
                    "主动回复文本": item["reply"],
                    "关联视觉事件ID": visual_event_id,
                    "动作视觉依据人工判定": visual_judgement,
                })
            detail.append(row)
    return detail, session_totals


def numeric_observed_turns(value: object) -> int:
    text = str(value)
    if text.startswith(">="):
        return int(text[2:])
    return int(text) if text.isdigit() else 0


def summarize(
    detail: list[dict[str, object]], session_totals: dict[str, dict[str, int]]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label, strategy in (("沉默策略", "沉默触发"), ("动作策略", "动作触发"), ("主动策略总体", None)):
        selected = detail if strategy is None else [row for row in detail if row["策略"] == strategy]
        successful = [row for row in selected if row["主动输出结果"] == "成功输出"]
        evaluable = [
            row for row in successful
            if row["用户回复结局"] in {"已回复", "30秒内未回复"}
        ]
        replied = [row for row in successful if row["用户回复结局"] == "已回复"]
        waits = [float(row["TTS结束到首次用户回复_s"]) for row in replied]
        observed_turns = sum(numeric_observed_turns(row["窗口内用户发话轮次"]) for row in successful)
        action_success = [row for row in successful if row["策略"] == "动作触发"]
        correct_action_visual = sum(row["动作视觉依据人工判定"] == "对" for row in action_success)
        total_asr = sum(values["valid_user_asr"] for values in session_totals.values())
        completed_turns = sum(values["completed_user_turns"] for values in session_totals.values())
        rows.append({
            "分析层次": label,
            "有效触发事件数": len(selected),
            "成功输出数": len(successful),
            "成功输出率": round(len(successful) / len(selected), 4) if selected else "",
            "用户回复结局可判定数": len(evaluable),
            "用户已回复数": len(replied),
            "可判定输出的用户回复率": round(len(replied) / len(evaluable), 4) if evaluable else "",
            "被截断或实验结束删失数": len(successful) - len(evaluable),
            "TTS结束到回复平均_s": round(mean(waits), 3) if waits else "",
            "TTS结束到回复中位_s": round(median(waits), 3) if waits else "",
            "输出后窗口内观察到的用户发话轮次": observed_turns,
            "每个已回复输出的用户发话轮次": round(observed_turns / len(replied), 3) if replied else "",
            "动作输出的视觉依据正确数": correct_action_visual if action_success else "",
            "动作输出的视觉依据已标注数": len(action_success) if action_success else "",
            "动作评论恰当性": "缺少人工标注" if action_success else "不适用",
            "四次实验有效用户ASR发话总数": total_asr if strategy is None else "",
            "四次实验完成用户对话轮次": completed_turns if strategy is None else "",
            "主动输出后窗口用户发话占有效ASR比例": round(observed_turns / total_asr, 4) if strategy is None else "",
        })
    return rows


def summarize_by_user(detail: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for participant in RUNS:
        participant_rows = [row for row in detail if row["用户"] == participant]
        for label, strategy in (("沉默策略", "沉默触发"), ("动作策略", "动作触发"), ("主动策略总体", None)):
            selected = participant_rows if strategy is None else [row for row in participant_rows if row["策略"] == strategy]
            successful = [row for row in selected if row["主动输出结果"] == "成功输出"]
            evaluable = [row for row in successful if row["用户回复结局"] in {"已回复", "30秒内未回复"}]
            replied = [row for row in successful if row["用户回复结局"] == "已回复"]
            waits = [float(row["TTS结束到首次用户回复_s"]) for row in replied]
            rows.append({
                "用户": participant,
                "分析层次": label,
                "有效触发事件数": len(selected),
                "成功输出数": len(successful),
                "成功输出率": round(len(successful) / len(selected), 4) if selected else "",
                "回复结局可判定数": len(evaluable),
                "用户已回复数": len(replied),
                "可判定输出的用户回复率": round(len(replied) / len(evaluable), 4) if evaluable else "",
                "观察到的用户发话轮次": sum(numeric_observed_turns(row["窗口内用户发话轮次"]) for row in successful),
                "TTS结束到回复平均_s": round(mean(waits), 3) if waits else "",
            })
    return rows


def write_notes() -> None:
    NOTES_PATH.write_text(
        """# 条件Z主动说话策略的三个分析层次

## 数据范围

- 使用xkx2、hn、ht、ds四次启用主动策略且日志完整的实体实验。
- xkx1不进入主动策略分析。
- 活动冲突事件已经排除，超时后重试并最终成功的请求合并为一个事件。

## 用户回复归属规则

- 从主动TTS结束开始观察30秒。
- 如果下一次成功主动输出先发生，则上一事件的窗口在下一次主动输出开始时截止，避免同一用户发话重复归入两个主动事件。
- 窗口内出现`USER_ASR`即记为用户回复。
- 没有回复且窗口被下一次主动输出或实验结束截断时记为删失，不当作明确未回复。
- “窗口内用户发话轮次”是时间关联指标，只能表述为主动输出后的观察结果，不能单凭日志证明因果关系。

## 三个层次

1. 沉默策略：沉默触发事件的输出成功率、输出后用户回复率、回复等待和后续用户发话轮次。
2. 动作策略：动作触发事件的相同指标，并关联VLM人工标注判断动作视觉依据是否正确。
3. 主动策略总体：合并两类策略，并报告其后时间窗口覆盖的用户发话占全部有效用户ASR的比例。

## 当前缺失

- 日志能够证明动作评论采用了哪条视觉证据，VLM人工标注能够判断该视觉证据是否正确。
- 日志不能自动判断机器人评论是否自然、合适或打扰，需要新增“动作评论恰当性”人工标注。
- 当前只有条件Z，不能由这些结果单独证明主动策略优于X或Y。
""",
        encoding="utf-8",
    )


def main() -> None:
    detail, session_totals = build_detail()
    summary = summarize(detail, session_totals)
    by_user = summarize_by_user(detail)
    write_csv(DETAIL_PATH, detail)
    write_csv(SUMMARY_PATH, summary)
    write_csv(BY_USER_PATH, by_user)
    write_notes()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
