#视觉缓存模块

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

import numpy as np
from PIL import Image


@dataclass
class FrameRecord:
    timestamp: float
    image: np.ndarray
    frame_id: str = "camera"


class RealTimeVisualBuffer:
    """Keeps a low-FPS rolling buffer of recent RGB frames."""

    def __init__(self, buffer_duration: float = 8.0, save_fps: float = 5.0) -> None:
        self.buffer_duration = float(buffer_duration)
        self.save_fps = float(save_fps)
        self.min_interval = 1.0 / max(self.save_fps, 1e-6)
        self.max_frames = int(round(self.buffer_duration * self.save_fps))
        self.frames: Deque[FrameRecord] = deque(maxlen=max(self.max_frames, 1))
        self._last_saved_ts: Optional[float] = None

    def add_frame(self, image: np.ndarray, timestamp: float, frame_id: str = "camera") -> bool:
        if self._last_saved_ts is not None and timestamp - self._last_saved_ts < self.min_interval:
            return False
        self.frames.append(FrameRecord(timestamp=timestamp, image=image.copy(), frame_id=frame_id))
        self._last_saved_ts = timestamp
        self._drop_old(timestamp)
        return True

    def get_window(self, now: float, window_length: float = 4.0) -> List[FrameRecord]:
        start = now - window_length
        return [f for f in self.frames if start <= f.timestamp <= now]

    def sample_keyframes(self, now: float, window_length: float = 4.0, num_frames: int = 6) -> List[FrameRecord]:
        window = self.get_window(now=now, window_length=window_length)
        if not window:
            return []
        if len(window) <= num_frames:
            return window
        positions = np.linspace(0, len(window) - 1, num_frames)
        indices = sorted({int(round(p)) for p in positions})
        return [window[i] for i in indices]

    def _drop_old(self, now: float) -> None:
        cutoff = now - self.buffer_duration
        while self.frames and self.frames[0].timestamp < cutoff:
            self.frames.popleft()


def resize_for_vlm(image: np.ndarray, max_width: int = 768) -> np.ndarray:
    h, w = image.shape[:2]
    if w <= max_width:
        return image
    scale = max_width / max(w, 1)
    pil = Image.fromarray(image)
    resized = pil.resize((max_width, int(h * scale)), Image.Resampling.BILINEAR)
    return np.asarray(resized)
