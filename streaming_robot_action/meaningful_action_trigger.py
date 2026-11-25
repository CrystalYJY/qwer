from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List


@dataclass
class MeaningfulActionDecision:
    should_trigger: bool
    score: float
    reason: str
    features: Dict[str, float] = field(default_factory=dict)


class MeaningfulActionTrigger:
    """Lightweight semantic-ish trigger before calling the VLM.

    This is intentionally model-shaped: it extracts a short-window feature vector
    from skeleton motion. The current scorer is rule-based, but the same features
    can later feed a RandomForest/XGBoost/MLP without changing the runtime wiring.
    """

    def __init__(
        self,
        window_s: float = 3.0,
        score_threshold: float = 5.0,
        min_peak_speed: float = 2.0,
        min_active_ratio: float = 0.25,
        stand_rise_threshold_m: float = 0.20,
    ) -> None:
        self.window_s = float(window_s)
        self.score_threshold = float(score_threshold)
        self.min_peak_speed = float(min_peak_speed)
        self.min_active_ratio = float(min_active_ratio)
        self.stand_rise_threshold_m = max(0.05, float(stand_rise_threshold_m))
        self.samples: Deque[Dict] = deque(maxlen=120)

    def update(self, payload: Dict, now_t: float) -> MeaningfulActionDecision:
        self.samples.append(payload)
        self._drop_old(now_t)
        features = self.extract_features(now_t)
        score, reason = self.score_features(features)
        return MeaningfulActionDecision(
            should_trigger=score >= self.score_threshold,
            score=score,
            reason=reason,
            features=features,
        )

    def _drop_old(self, now_t: float) -> None:
        while self.samples:
            sample_t = float(self.samples[0].get("demo_time_s") or now_t)
            if now_t - sample_t <= self.window_s:
                break
            self.samples.popleft()

    def extract_features(self, now_t: float) -> Dict[str, float]:
        speeds: List[Dict[str, float]] = [s.get("part_speeds") or {} for s in self.samples]
        if not speeds:
            return {
                "sample_count": 0.0,
                "max_speed": 0.0,
                "mean_peak_speed": 0.0,
                "active_ratio": 0.0,
                "active_part_count": 0.0,
                "hand_peak_speed": 0.0,
                "head_peak_speed": 0.0,
                "speed_variation": 0.0,
                "torso_vertical_rise_m": 0.0,
            }

        peaks = [max(parts.values()) if parts else 0.0 for parts in speeds]
        active_peaks = [p for p in peaks if p >= self.min_peak_speed]
        active_names = set()
        hand_peak = 0.0
        head_peak = 0.0
        for parts in speeds:
            for name, speed in parts.items():
                if speed >= self.min_peak_speed:
                    active_names.add(name)
                lowered = name.lower()
                if any(k in lowered for k in ("hand", "wrist", "elbow")):
                    hand_peak = max(hand_peak, float(speed))
                if any(k in lowered for k in ("head", "nose")):
                    head_peak = max(head_peak, float(speed))

        max_speed = max(peaks) if peaks else 0.0
        mean_peak = sum(peaks) / max(len(peaks), 1)
        active_ratio = len(active_peaks) / max(len(peaks), 1)
        mean = mean_peak
        speed_variation = sum(abs(p - mean) for p in peaks) / max(len(peaks), 1)

        return {
            "sample_count": float(len(speeds)),
            "max_speed": float(max_speed),
            "mean_peak_speed": float(mean_peak),
            "active_ratio": float(active_ratio),
            "active_part_count": float(len(active_names)),
            "hand_peak_speed": float(hand_peak),
            "head_peak_speed": float(head_peak),
            "speed_variation": float(speed_variation),
            "torso_vertical_rise_m": self._torso_vertical_rise(),
        }

    def _torso_vertical_rise(self) -> float:
        """Estimate sit-to-stand rise from Azure Kinect's vertical z axis."""
        anchors = ("pelvis", "spine_navel", "spine_chest", "spine_shoulder", "hip_left", "hip_right", "head")
        rises = []
        for anchor in anchors:
            values = []
            for sample in self.samples:
                point = (sample.get("part_positions") or {}).get(anchor)
                if isinstance(point, dict) and point.get("z") is not None:
                    values.append(float(point["z"]))
            if len(values) >= 4:
                rises.append(values[-1] - values[0])
        if not rises:
            return 0.0
        rises.sort()
        return float(rises[len(rises) // 2])

    def score_features(self, f: Dict[str, float]) -> tuple[float, str]:
        score = 0.0
        reasons: List[str] = []

        if f["max_speed"] >= self.min_peak_speed:
            score += 1.5
            reasons.append("peak motion")
        if f["max_speed"] >= self.min_peak_speed * 1.8:
            score += 1.0
            reasons.append("large peak")
        if f["active_ratio"] >= self.min_active_ratio:
            score += 1.5
            reasons.append("sustained motion")
        if f["active_part_count"] >= 3:
            score += 1.0
            reasons.append("multiple joints")
        if f["hand_peak_speed"] >= self.min_peak_speed:
            score += 1.0
            reasons.append("hand/arm action")
        if f["head_peak_speed"] >= self.min_peak_speed * 1.5 and f["hand_peak_speed"] >= self.min_peak_speed:
            score += 1.0
            reasons.append("head-hand coordination")
        if f["speed_variation"] >= self.min_peak_speed * 0.6:
            score += 1.0
            reasons.append("dynamic change")
        if f["torso_vertical_rise_m"] >= self.stand_rise_threshold_m:
            # Standing up can be slow and involves little arm movement, so it
            # is a strong event in its own right once a stable torso rise is
            # observed across several skeleton frames.
            score += 8.0
            reasons.append("sit-to-stand torso rise")

        reason = ", ".join(reasons) if reasons else "low semantic motion"
        return score, reason
