#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional


def _format_age(age: Optional[float]) -> str:
    if age is None:
        return "recently"
    if age < 1:
        return "just now"
    return f"about {age:.0f} seconds ago"


def _clean_text(text: str, max_len: int = 90) -> str:
    text = " ".join(str(text or "").split())
    if len(text) > max_len:
        text = text[: max_len - 3].rstrip() + "..."
    return text


def build_dialogue_prompt(
    dialogue_context: Dict,
    user_utterance: str,
    current_time_s: Optional[float] = None,
    dialogue_history: Optional[List[Dict]] = None,
    max_visual_events: int = 3,
    max_dialogue_turns: int = 3,
    output_mode: str = "english",
) -> str:
    lines = [
        "You are a social desktop robot. Reply naturally in English.",
        "Naturally combine the user's latest utterance with the recent VLM visual memory to form one coherent response.",
        'Return JSON only: {"reply":"","response_style":"neutral"}',
        "response_style options: neutral, positive, care, sad, angry. Use neutral when unsure or when emotion is not needed.",
        "",
    ]

    visual_events = dialogue_context.get("recent_visual_events") or []
    if not visual_events and dialogue_context.get("current_visual_observation"):
        visual_events = [dialogue_context["current_visual_observation"]]
    visual_events = visual_events[-max_visual_events:]

    if visual_events:
        lines.append("Visual memory:")
        for event in visual_events:
            # This is the age of the sampled scene, rather than the age of the
            # VLM result in memory.  That distinction prevents stale images
            # from being described as "just now" after a slow request.
            age = event.get("scene_age_at_prompt_s", event.get("age_s"))
            if age is None and current_time_s is not None:
                received_at = float(event.get("received_at") or event.get("timestamp") or 0.0)
                age = max(0.0, current_time_s - received_at)
            summary = (
                event.get("observation_raw")
                or event.get("description")
                or event.get("activity_summary")
                or event.get("observation_short")
                or ""
            )
            lines.append(f"- {_format_age(age)}: {_clean_text(summary, max_len=120)}")
        lines.append("")
    else:
        lines.extend(["Visual memory:", "- None", ""])

    history = (dialogue_history or [])[-max_dialogue_turns:]
    if history:
        lines.append("Recent dialogue:")
        for turn in history:
            user = _clean_text(str(turn.get("user_text") or ""), max_len=50)
            robot = _clean_text(str(turn.get("reply") or ""), max_len=70)
            if user:
                lines.append(f"User: {user}")
            if robot:
                lines.append(f"Robot: {robot}")
        lines.append("")
    else:
        lines.extend(["Recent dialogue:", "- None", ""])

    lines.extend(
        [
            "Rules: respond to the user's latest utterance first.",
            "Combine the user's words with the recent visual context in a conversational way rather than mechanically listing observations.",
            "Prefer a short, warm response that makes sense in both the spoken and visual context.",
            "If Visual memory is None, say you do not have a current visual observation; never invent visual details.",
            "Do not infer hidden emotions from vision.",
            "",
            "User said:",
            user_utterance,
            "",
            "Robot JSON:",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Build a dialogue LLM prompt from action memory context.")
    p.add_argument("--context-json", required=True)
    p.add_argument("--user-utterance", required=True)
    p.add_argument("--current-time-s", type=float, default=None)
    p.add_argument("--output", default="")
    args = p.parse_args()

    context = json.loads(Path(args.context_json).read_text(encoding="utf-8"))
    prompt = build_dialogue_prompt(context, args.user_utterance, args.current_time_s)
    if args.output:
        Path(args.output).write_text(prompt, encoding="utf-8")
        print(f"Dialogue prompt: {args.output}")
    else:
        print(prompt)


if __name__ == "__main__":
    main()
