#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ProactiveDecision:
    should_speak: bool
    proactive_type: str = ""
    score: int = 0
    reason: str = ""
    utterance_hint: str = ""
    skip_reason: str = ""
    candidates: List[Dict] = field(default_factory=list)


@dataclass
class ProactiveCandidate:
    proactive_type: str
    score: int
    reason: str
    utterance_hint: str

    def to_dict(self) -> Dict:
        return {
            "proactive_type": self.proactive_type,
            "score": self.score,
            "reason": self.reason,
            "utterance_hint": self.utterance_hint,
        }


@dataclass
class ProactivePolicyState:
    last_user_turn_time: float = 0.0
    last_robot_reply_time: float = -1e9
    last_general_checkin_time: float = -1e9
    proactive_count: int = 0
    asr_active: bool = False
    tts_playing: bool = False
    last_proactive_visual_event_id: str = ""
    action_lock_turns_remaining: int = 0


class ProactivePolicy:
    def __init__(
        self,
        max_visual_age_s: float = 30.0,
        visual_event_quiet_s: float = 2.0,
        general_quiet_s: float = 20.0,
        action_lock_user_turns: int = 2,
        action_salience_threshold: int = 2,
        trigger_threshold: int = 5,
    ) -> None:
        self.max_visual_age_s = max_visual_age_s
        self.visual_event_quiet_s = visual_event_quiet_s
        self.general_quiet_s = general_quiet_s
        self.action_lock_user_turns = max(0, int(action_lock_user_turns))
        self.action_salience_threshold = int(action_salience_threshold)
        self.trigger_threshold = trigger_threshold
        self.state = ProactivePolicyState()

    def on_asr_status(self, kind: str, now: float) -> None:
        if kind == "ASR_SPEECH":
            self.state.asr_active = True
        elif kind in {"ASR_TEXT", "USER_ASR", "ASR_EMPTY", "ASR_TIMEOUT", "ASR_ERROR", "ASR_CLOSED"}:
            self.state.asr_active = False

    def on_user_turn(self, text: str, now: float) -> None:
        self.state.last_user_turn_time = now

    def on_user_dialogue_turn(self, text: str, now: float) -> None:
        self.on_user_turn(text, now)
        if self.state.action_lock_turns_remaining > 0:
            self.state.action_lock_turns_remaining -= 1

    def on_robot_reply(self, now: float, proactive_type: str = "", visual_event_id: str = "") -> None:
        self.state.last_robot_reply_time = now
        if proactive_type:
            self.state.proactive_count += 1
            if proactive_type == "salient_action_comment":
                self.state.action_lock_turns_remaining = self.action_lock_user_turns
            elif proactive_type == "general_checkin":
                self.state.last_general_checkin_time = now
        if visual_event_id:
            self.state.last_proactive_visual_event_id = visual_event_id

    def on_proactive_request(self, now: float, proactive_type: str) -> None:
        """Start cooldown at dispatch, including when a slow cloud reply is dropped.

        Otherwise a timed check-in that exceeds its deadline is immediately
        retried every polling cycle while the user is still silent.
        """
        if proactive_type == "general_checkin":
            self.state.last_general_checkin_time = now

    def decide(
        self,
        dialogue_context: Dict,
        now: float,
        trigger_source: str = "visual_event",
    ) -> ProactiveDecision:
        gate = self._can_speak(dialogue_context, now)
        if gate:
            return ProactiveDecision(False, skip_reason=gate)

        signals = self._extract_signals(dialogue_context, now)
        if trigger_source == "visual_event" and self.state.action_lock_turns_remaining > 0:
            return ProactiveDecision(
                False,
                skip_reason=f"action_locked_until_user_turns:{self.state.action_lock_turns_remaining}",
            )

        candidates = self._generate_candidates(signals, trigger_source=trigger_source)
        candidates = [c for c in candidates if c.score > 0]
        candidates.sort(key=lambda c: c.score, reverse=True)
        candidate_dicts = [c.to_dict() for c in candidates]

        if not candidates:
            return ProactiveDecision(
                False,
                skip_reason=(
                    "no_candidate: "
                    f"visual_events={signals['visual_count']}, "
                    f"salience={signals['salience']}, "
                    f"quiet={signals['quiet_duration']:.1f}s"
                ),
                candidates=candidate_dicts,
            )

        best = candidates[0]
        decision = ProactiveDecision(
            True,
            proactive_type=best.proactive_type,
            score=best.score,
            reason=best.reason,
            utterance_hint=best.utterance_hint,
            candidates=candidate_dicts,
        )
        if decision.should_speak and decision.score < self.trigger_threshold:
            decision.should_speak = False
            decision.skip_reason = f"low_score:{decision.score}<threshold:{self.trigger_threshold}"
        return decision

    def _extract_signals(self, dialogue_context: Dict, now: float) -> Dict:
        visual_events = dialogue_context.get("recent_visual_events") or []
        latest_event = visual_events[-1] if visual_events else {}
        latest_visual_text = str(
            latest_event.get("observation_short")
            or latest_event.get("description")
            or latest_event.get("activity_summary")
            or ""
        )
        latest_event_id = str(latest_event.get("event_id") or "")
        try:
            salience = int(float(latest_event.get("salience") or 0))
        except (TypeError, ValueError):
            salience = 0
        return {
            "latest_visual_text": latest_visual_text,
            "latest_event_id": latest_event_id,
            "visual_count": len(visual_events),
            "salience": max(0, min(3, salience)),
            "quiet_duration": max(0.0, now - self.state.last_user_turn_time),
            "interaction_quiet_duration": max(
                0.0,
                now - max(self.state.last_user_turn_time, self.state.last_robot_reply_time),
            ),
            "now": now,
        }

    def _generate_candidates(self, signals: Dict, trigger_source: str = "visual_event") -> List[ProactiveCandidate]:
        candidates: List[ProactiveCandidate] = []
        user_quiet = signals["quiet_duration"]
        interaction_quiet = signals["interaction_quiet_duration"]
        visual_count = signals["visual_count"]
        salience = signals["salience"]
        latest_event_id = signals["latest_event_id"]
        latest_visual_text = signals["latest_visual_text"]
        now = signals["now"]

        if trigger_source == "timer":
            enough_since_last = now - self.state.last_general_checkin_time >= self.general_quiet_s
            if interaction_quiet >= self.general_quiet_s and visual_count > 0 and enough_since_last:
                score = 3 + min(visual_count, 3)
                reason = f"对话已经安静约 {interaction_quiet:.0f} 秒。"
                candidates.append(
                    ProactiveCandidate(
                        "general_checkin",
                        score,
                        reason,
                        "结合 primary_visual_event 轻量搭话，但不要像动作评论那样强行评价。",
                    )
                )
            return candidates

        if trigger_source != "visual_event":
            return candidates
        if (
            user_quiet >= self.visual_event_quiet_s
            and latest_event_id
            and latest_event_id != self.state.last_proactive_visual_event_id
            and salience >= self.action_salience_threshold
        ):
            score = salience * 2 + min(visual_count, 3)
            reason = f"VLM salience={salience}，最新动作适合轻量评论：{latest_visual_text}"
            candidates.append(
                ProactiveCandidate(
                    "salient_action_comment",
                    score,
                    reason,
                    "围绕 primary_visual_event 自然说一句，不要泛泛搭话。",
                )
            )
        return candidates

    def _can_speak(self, dialogue_context: Dict, now: float) -> str:
        if self.state.asr_active:
            return "asr_active"
        if self.state.tts_playing:
            return "tts_playing"
        obs = dialogue_context.get("current_visual_observation")
        if not obs:
            return "no_visual_observation"
        age = obs.get("age_s")
        if age is None:
            return ""
        if float(age) > self.max_visual_age_s:
            return f"visual_too_old:{age:.1f}s"
        return ""

def build_proactive_prompt(
    decision: ProactiveDecision,
    dialogue_context: Dict,
    dialogue_history: List[Dict],
    output_mode: str = "english",
    max_dialogue_turns: int = 1,
) -> str:
    visual_events = dialogue_context.get("recent_visual_events") or []
    current_event = dialogue_context.get("current_visual_observation") or {}
    primary_event = (
        current_event.get("observation_raw")
        or current_event.get("description")
        or current_event.get("activity_summary")
        or current_event.get("observation_short")
        or ""
    )
    comment_style = str(current_event.get("comment_style") or "neutral")
    visual_lines = []
    for event in visual_events[-5:]:
        age = event.get("age_s")
        age_text = "recently" if age is None else f"about {float(age):.0f} seconds ago"
        summary = (
            event.get("observation_raw")
            or event.get("description")
            or event.get("activity_summary")
            or event.get("observation_short")
            or ""
        )
        visual_lines.append(f"- {age_text}: {summary}")
    dialogue_lines = []
    history_count = max(0, int(max_dialogue_turns))
    recent_history = dialogue_history[-history_count:] if history_count else []
    for turn in recent_history:
        user = str(turn.get("user_text") or "")
        robot = str(turn.get("reply") or "")
        if user:
            dialogue_lines.append(f"User: {user}")
        if robot:
            dialogue_lines.append(f"Robot: {robot}")
    common = [
        f"Background: {decision.reason}",
        f"Proactive type: {decision.proactive_type}",
        f"primary_visual_event: {primary_event or 'None'}",
        f"comment_style: {comment_style}",
        "Visual context:",
        "\n".join(visual_lines) if visual_lines else "- None",
        "Recent dialogue:",
        "\n".join(dialogue_lines) if dialogue_lines else "- None",
    ]
    if decision.proactive_type == "salient_action_comment":
        english_task = [
            "Task: say one natural English sentence responding to primary_visual_event.",
            "Do not describe or restate the user's action.",
            "Say something interaction-worthy instead: a light question, caring check-in, playful comment, or useful offer.",
            "Do not say 'I see you' or sound like surveillance.",
        ]
    else:
        english_task = [
            "Task: say one low-pressure English check-in explicitly grounded in primary_visual_event.",
            "Do not describe or restate the user's action.",
            "Avoid generic greetings, advice, and surveillance-like wording.",
        ]
    return "\n".join(
        [
            "You are a social desktop robot.",
            *english_task,
            'Return JSON only: {"reply":"","response_style":"neutral"}',
            "response_style options: neutral, positive, care, sad, angry. Use neutral when unsure or when emotion is not needed.",
            "",
            *common,
            "Robot proactive JSON:",
        ]
    )
