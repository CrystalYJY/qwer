from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .action_memory import ActionEvent, ActionEventMemory
from .vlm_action_client import DoubaoFrameVLMClient
from .vlm_worker import AsyncVLMWorker, VLMJob, VLMResult


@dataclass
class SubmittedActionEvent:
    event_id: str
    timestamp: float
    status: str
    num_keyframes: int
    segment_start: Optional[float] = None
    segment_end: Optional[float] = None


class ActionEventSystem:
    """Asynchronous bridge from visual keyframes to dialogue-ready action memory."""

    def __init__(
        self,
        vlm_client: DoubaoFrameVLMClient,
        memory: Optional[ActionEventMemory] = None,
        hard_timeout_s: float = 15.0,
        max_workers: int = 1,
        event_prefix: str = "vlm",
    ) -> None:
        self.vlm_client = vlm_client
        self.memory = memory or ActionEventMemory()
        self.hard_timeout_s = float(hard_timeout_s)
        self.worker = AsyncVLMWorker(vlm_client, max_workers=max_workers)
        self.event_prefix = str(event_prefix or "vlm").strip() or "vlm"
        self.pending: Dict[int, VLMJob] = {}
        self.event_log: List[Dict] = []

    def has_pending(self) -> bool:
        return bool(self.pending)

    def pending_count(self) -> int:
        return len(self.pending)

    def submit_keyframes(
        self,
        frames: List[np.ndarray],
        timestamp: float,
        segment_start: Optional[float] = None,
        segment_end: Optional[float] = None,
        segment_duration: Optional[float] = None,
        segment_peak_motion: Optional[float] = None,
        segment_mean_motion: Optional[float] = None,
    ) -> SubmittedActionEvent:
        job = self.worker.submit(
            frames,
            timestamp=timestamp,
            segment_start=segment_start,
            segment_end=segment_end,
            segment_duration=segment_duration,
            segment_peak_motion=segment_peak_motion,
            segment_mean_motion=segment_mean_motion,
        )
        self.pending[job.job_id] = job
        event_id = self._event_id(job.job_id)
        pending_event = ActionEvent(
            activity="pending_visual_analysis",
            timestamp=timestamp,
            confidence=0.0,
            description="VLM action description is pending.",
            status="pending",
            source=self.vlm_client.model,
            event_id=event_id,
            observed_at=timestamp,
        )
        self.memory.update(pending_event)
        self.event_log.append(self._row_from_event(pending_event, job, status="pending"))
        return SubmittedActionEvent(
            event_id=event_id,
            timestamp=timestamp,
            status="pending",
            num_keyframes=len(frames),
            segment_start=segment_start,
            segment_end=segment_end,
        )

    def poll(self) -> List[Dict]:
        updates: List[Dict] = []
        for result in self.worker.collect_done():
            updates.append(self._handle_result(result))
        updates.extend(self._mark_timeouts())
        return updates

    def wait_all(self) -> List[Dict]:
        updates: List[Dict] = []
        for result in self.worker.wait_all():
            if result.job.job_id in self.pending:
                updates.append(self._handle_result(result))
        return updates

    def get_dialogue_context(
        self,
        current_time_s: Optional[float] = None,
        max_events: int = 5,
        max_age_s: float = 90.0,
    ) -> Dict:
        return self.memory.to_dialogue_context(
            current_time_s=current_time_s,
            max_events=max_events,
            max_age_s=max_age_s,
        )

    def shutdown(self) -> None:
        self.worker.shutdown()

    def _handle_result(self, result: VLMResult) -> Dict:
        job = result.job
        self.pending.pop(job.job_id, None)
        status = "vlm_error" if result.error else "confirmed"
        event = result.event
        event.event_id = self._event_id(job.job_id)
        event.status = status
        event.source = self.vlm_client.model
        event.latency_s = result.latency_s
        event.error = result.error
        event.observed_at = job.timestamp
        event.received_at = job.timestamp + result.latency_s
        event.observation_raw = event.description or event.activity
        event.observation_short = self._make_observation_short(event.observation_raw)
        self.memory.update(event)
        row = self._row_from_event(event, job, status=status, result=result)
        self.event_log.append(row)
        return row

    def _mark_timeouts(self) -> List[Dict]:
        now = time.time()
        timed_out = []
        for job_id, job in list(self.pending.items()):
            age = now - job.submitted_wall_time
            if age < self.hard_timeout_s:
                continue
            self.pending.pop(job_id, None)
            event = ActionEvent(
                activity="visual_analysis_timeout",
                timestamp=job.timestamp,
                confidence=0.0,
                description=f"VLM did not return within {self.hard_timeout_s:.1f}s.",
                status="timeout",
                source=self.vlm_client.model,
                latency_s=age,
                error="hard_timeout",
                event_id=self._event_id(job.job_id),
                observed_at=job.timestamp,
                received_at=job.timestamp + age,
            )
            self.memory.update(event)
            row = self._row_from_event(event, job, status="timeout")
            self.event_log.append(row)
            timed_out.append(row)
        return timed_out

    def _row_from_event(
        self,
        event: ActionEvent,
        job: VLMJob,
        status: str,
        result: Optional[VLMResult] = None,
    ) -> Dict:
        timing = {} if result is None else (result.timing or {})
        return {
            "event_id": event.event_id,
            "job_id": job.job_id,
            "status": status,
            "timestamp": job.timestamp,
            "segment_start": job.segment_start,
            "segment_end": job.segment_end,
            "segment_duration": job.segment_duration,
            "num_keyframes": len(job.frames),
            "activity": event.activity,
            "description": event.description,
            "observation_short": event.observation_short,
            "observation_raw": event.observation_raw,
            "confidence": event.confidence,
            "salience": event.salience,
            "comment_style": event.comment_style,
            "source": event.source,
            "observed_at": event.observed_at or job.timestamp,
            "received_at": event.received_at,
            "latency_s": event.latency_s,
            "request_s": timing.get("request_s", 0.0),
            "payload_kb": timing.get("payload_kb", 0.0),
            "error": event.error,
        }

    def _make_observation_short(self, text: str) -> str:
        text = " ".join(str(text).split())
        if not text:
            return ""
        first = text.split(". ")[0].strip()
        if len(first) > 90:
            first = first[:87].rstrip() + "..."
        return first

    def _event_id(self, job_id: int) -> str:
        return f"{self.event_prefix}-{job_id}"
