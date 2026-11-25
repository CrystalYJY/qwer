from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .action_memory import ActionEvent
from .vlm_action_client import DoubaoFrameVLMClient


@dataclass
class VLMJob:
    job_id: int
    timestamp: float
    frames: List[np.ndarray]
    submitted_wall_time: float
    segment_start: Optional[float] = None
    segment_end: Optional[float] = None
    segment_duration: Optional[float] = None
    segment_peak_motion: Optional[float] = None
    segment_mean_motion: Optional[float] = None


@dataclass
class VLMResult:
    job: VLMJob
    event: ActionEvent
    latency_s: float
    completed_wall_time: float
    error: str = ""
    timing: Dict[str, float] = None


class AsyncVLMWorker:
    """Runs slow VLM calls in the background so the video loop can continue."""

    def __init__(self, client: DoubaoFrameVLMClient, max_workers: int = 1) -> None:
        self.client = client
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._next_job_id = 1
        self._futures: List[Future] = []

    def submit(
        self,
        frames: List[np.ndarray],
        timestamp: float,
        segment_start: Optional[float] = None,
        segment_end: Optional[float] = None,
        segment_duration: Optional[float] = None,
        segment_peak_motion: Optional[float] = None,
        segment_mean_motion: Optional[float] = None,
    ) -> VLMJob:
        job = VLMJob(
            job_id=self._next_job_id,
            timestamp=timestamp,
            frames=[f.copy() for f in frames],
            submitted_wall_time=time.time(),
            segment_start=segment_start,
            segment_end=segment_end,
            segment_duration=segment_duration,
            segment_peak_motion=segment_peak_motion,
            segment_mean_motion=segment_mean_motion,
        )
        self._next_job_id += 1
        self._futures.append(self.executor.submit(self._run_job, job))
        return job

    def collect_done(self) -> List[VLMResult]:
        done: List[VLMResult] = []
        remaining: List[Future] = []
        for fut in self._futures:
            if fut.done():
                done.append(fut.result())
            else:
                remaining.append(fut)
        self._futures = remaining
        return done

    def wait_all(self) -> List[VLMResult]:
        results = []
        for fut in self._futures:
            results.append(fut.result())
        self._futures = []
        return results

    def shutdown(self) -> None:
        self.executor.shutdown(wait=True)

    def _run_job(self, job: VLMJob) -> VLMResult:
        start = time.time()
        try:
            event = self.client.describe_frames(job.frames, timestamp=job.timestamp)
            return VLMResult(
                job=job,
                event=event,
                latency_s=time.time() - start,
                completed_wall_time=time.time(),
                timing=dict(self.client.last_timing),
            )
        except Exception as exc:
            event = ActionEvent(activity="unknown", timestamp=job.timestamp, confidence=0.0, description=str(exc))
            return VLMResult(
                job=job,
                event=event,
                latency_s=time.time() - start,
                completed_wall_time=time.time(),
                error=str(exc),
                timing=dict(getattr(self.client, "last_timing", {})),
            )
