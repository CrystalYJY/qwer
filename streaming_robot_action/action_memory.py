#动作事件记忆模块
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional


@dataclass
class ActionEvent:
    activity: str
    timestamp: float
    confidence: float
    description: str = ""
    status: str = "confirmed"
    source: str = ""
    latency_s: float = 0.0
    error: str = ""
    event_id: str = ""
    observation_short: str = ""
    observation_raw: str = ""
    observed_at: float = 0.0
    received_at: float = 0.0
    salience: int = 0
    comment_style: str = "neutral"


@dataclass
class ActionMemoryState:
    current_activity: str = "unknown"
    confidence: float = 0.0
    last_updated: float = 0.0
    recent_events: List[Dict] = field(default_factory=list)


class ActionEventMemory:
    """Stores recent VLM observations for dialogue and proactive policy."""

    def __init__(
        self,
        min_confidence: float = 0.6,
        max_events: int = 20,
    ) -> None:
        self.min_confidence = min_confidence
        self.recent_events: Deque[ActionEvent] = deque(maxlen=max_events)
        self.state = ActionMemoryState()

    def update(self, event: ActionEvent) -> ActionMemoryState:
        now = event.timestamp or time.time()
        self.recent_events.append(event)
        self.state = ActionMemoryState(
            current_activity=event.activity,
            confidence=event.confidence,
            last_updated=now,
            recent_events=[
                {
                    "event_id": e.event_id,
                    "activity": e.activity,
                    "description": e.description,
                    "observation_short": e.observation_short,
                    "observation_raw": e.observation_raw,
                    "time": e.timestamp,
                    "observed_at": e.observed_at,
                    "received_at": e.received_at,
                    "confidence": e.confidence,
                    "status": e.status,
                    "source": e.source,
                    "latency_s": e.latency_s,
                    "error": e.error,
                    "salience": e.salience,
                    "comment_style": e.comment_style,
                }
                for e in list(self.recent_events)[-5:]
            ],
        )
        return self.state

    def to_dialogue_context(self, max_events: int = 3, current_time_s: Optional[float] = None, max_age_s: float = 20.0) -> Dict:
        confirmed = [
            e for e in self.recent_events
            if e.status == "confirmed" and e.confidence >= self.min_confidence and not e.error
        ]
        if current_time_s is not None:
            confirmed = [
                e for e in confirmed
                if max(0.0, current_time_s - (e.received_at or e.timestamp)) <= max_age_s
            ]
        # Parallel VLM requests may finish out of order.  Dialogue must follow
        # when the scene was observed, not the order network responses arrived.
        confirmed.sort(key=lambda e: (e.observed_at or e.timestamp, e.received_at or e.timestamp))
        recent = confirmed[-max_events:]
        current = recent[-1] if recent else None
        def timing(e: ActionEvent) -> Dict[str, Optional[float]]:
            if current_time_s is None:
                return {"memory_age_since_vlm_return_s": None, "scene_age_at_prompt_s": None}
            return {
                "memory_age_since_vlm_return_s": max(0.0, current_time_s - (e.received_at or e.timestamp)),
                "scene_age_at_prompt_s": max(0.0, current_time_s - (e.observed_at or e.timestamp)),
            }

        def as_context_event(e: ActionEvent) -> Dict:
            age = timing(e)
            return {
                "event_id": e.event_id,
                "activity_summary": e.observation_raw or e.description or e.activity,
                "description": e.observation_raw or e.description or e.activity,
                "observation_short": e.observation_short or e.description or e.activity,
                "observation_raw": e.observation_raw or e.description,
                "confidence": e.confidence, "timestamp": e.timestamp,
                "observed_at": e.observed_at or e.timestamp, "received_at": e.received_at,
                # age_s remains for older readers; it now deliberately means scene age.
                "age_s": age["scene_age_at_prompt_s"], **age,
                "source": e.source, "latency_s": e.latency_s, "salience": e.salience,
                "comment_style": e.comment_style,
            }

        return {
            "current_visual_observation": None if current is None else as_context_event(current),
            "recent_visual_events": [as_context_event(e) for e in recent],
            "dialogue_guidance": [
                "Use visual observations only when they are relevant to the user's utterance.",
                "Do not over-mention vision or interrupt the user.",
                "Treat visual descriptions as uncertain observations, not hidden intent or emotion.",
            ],
        }
