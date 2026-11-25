#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import shlex
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from .docker_haru2_bridge import run_in_container, yaml_quote


ADVANCED_TTS_MSG = "haru2_core_msgs/msg/TTSAdvancedCommand"


@dataclass(frozen=True)
class EmotionPreset:
    text: str
    emotion: Optional[int]
    pitch_type: int
    pitch_value: float
    relative_pitch: bool
    rate_type: int
    rate_value: int
    volume_type: int
    volume_value: int
    emphasis_type: int


def make_presets() -> Dict[str, EmotionPreset]:
    return {
        "neutral": EmotionPreset(
            text="This is a neutral voice test.",
            emotion=None,
            pitch_type=0,
            pitch_value=0.0,
            relative_pitch=False,
            rate_type=0,
            rate_value=0,
            volume_type=0,
            volume_value=0,
            emphasis_type=0,
        ),
        "happy": EmotionPreset(
            text="That is great news. I am really happy for you.",
            emotion=0,
            pitch_type=0,
            pitch_value=45.0,
            relative_pitch=True,
            rate_type=1,
            rate_value=125,
            volume_type=1,
            volume_value=4,
            emphasis_type=1,
        ),
        "calm": EmotionPreset(
            text="Take your time. We can go slowly and calmly.",
            emotion=2,
            pitch_type=0,
            pitch_value=-25.0,
            relative_pitch=True,
            rate_type=-1,
            rate_value=78,
            volume_type=-1,
            volume_value=-3,
            emphasis_type=0,
        ),
        "sad": EmotionPreset(
            text="I am sorry that felt difficult. I am here with you.",
            emotion=1,
            pitch_type=0,
            pitch_value=-45.0,
            relative_pitch=True,
            rate_type=-1,
            rate_value=72,
            volume_type=-1,
            volume_value=-5,
            emphasis_type=-1,
        ),
        "angry": EmotionPreset(
            text="Hey, that was intense. Let us pause for a second.",
            emotion=3,
            pitch_type=0,
            pitch_value=20.0,
            relative_pitch=True,
            rate_type=1,
            rate_value=135,
            volume_type=2,
            volume_value=6,
            emphasis_type=2,
        ),
        "high_fast": EmotionPreset(
            text="This voice should sound higher, faster, and louder.",
            emotion=None,
            pitch_type=0,
            pitch_value=70.0,
            relative_pitch=True,
            rate_type=1,
            rate_value=145,
            volume_type=2,
            volume_value=7,
            emphasis_type=2,
        ),
        "low_slow": EmotionPreset(
            text="This voice should sound lower, slower, and softer.",
            emotion=None,
            pitch_type=0,
            pitch_value=-60.0,
            relative_pitch=True,
            rate_type=-1,
            rate_value=62,
            volume_type=-1,
            volume_value=-6,
            emphasis_type=-1,
        ),
    }


def normalize_namespace(namespace: str) -> str:
    namespace = (namespace or "/haru2").strip()
    if not namespace.startswith("/"):
        namespace = "/" + namespace
    return namespace.rstrip("/")


def parse_emotions(raw: str, presets: Dict[str, EmotionPreset]) -> List[str]:
    emotions = [x.strip().lower() for x in raw.split(",") if x.strip()]
    unknown = [x for x in emotions if x not in presets]
    if unknown:
        raise ValueError(f"Unknown emotion(s): {', '.join(unknown)}. Available: {', '.join(presets)}")
    return emotions or ["neutral", "high_fast", "low_slow", "happy", "calm", "sad", "angry"]


def estimate_wait_s(text: str, minimum: float) -> float:
    words = max(1, len(text.split()))
    return max(minimum, min(12.0, words / 2.4 + 1.0))


def build_tts_config(preset: EmotionPreset, use_emotion: bool, emotion_only: bool) -> str:
    fields = []
    if not emotion_only:
        fields.extend(
            [
                "add_pitch: true",
                f"relative_pitch: {'true' if preset.relative_pitch else 'false'}",
                f"pitch_type: {preset.pitch_type}",
                f"pitch_value: {preset.pitch_value}",
                "add_rate: true",
                f"rate_type: {preset.rate_type}",
                f"rate_value: {preset.rate_value}",
                "add_volume: true",
                f"volume_type: {preset.volume_type}",
                f"volume_value: {preset.volume_value}",
                "add_emphasis: true",
                f"emphasis_type: {preset.emphasis_type}",
            ]
        )
    if use_emotion and preset.emotion is not None:
        fields.extend(["add_emotion: true", f"emotion: {preset.emotion}"])
    return "{" + ", ".join(fields) + "}"


def build_payload(
    text: str,
    preset: EmotionPreset,
    language_code: str,
    disable_lipsync: bool,
    use_emotion: bool,
    emotion_only: bool,
) -> str:
    global_config = build_tts_config(preset, use_emotion=use_emotion, emotion_only=emotion_only)
    return (
        "{"
        f"global_config: {global_config}, "
        f"spurts: [{{config: {{}}, text: {yaml_quote(text)}}}], "
        f"disable_lipsync: {'true' if disable_lipsync else 'false'}, "
        f"language_code: {yaml_quote(language_code)}"
        "}"
    )


def publish_advanced_tts(
    container: str,
    setup: str,
    topic: str,
    text: str,
    preset: EmotionPreset,
    language_code: str,
    disable_lipsync: bool,
    use_emotion: bool,
    emotion_only: bool,
    dry_run: bool,
) -> int:
    payload = build_payload(
        text=text,
        preset=preset,
        language_code=language_code,
        disable_lipsync=disable_lipsync,
        use_emotion=use_emotion,
        emotion_only=emotion_only,
    )
    command = f"ros2 topic pub --once {shlex.quote(topic)} {ADVANCED_TTS_MSG} {shlex.quote(payload)}"
    return run_in_container(container, setup, command, dry_run=dry_run)


def run_sequence(
    container: str,
    setup: str,
    topic: str,
    emotions: Iterable[str],
    presets: Dict[str, EmotionPreset],
    text: str,
    language_code: str,
    disable_lipsync: bool,
    use_emotion: bool,
    emotion_only: bool,
    pause_s: float,
    dry_run: bool,
) -> None:
    for emotion_name in emotions:
        preset = presets[emotion_name]
        utterance = text or preset.text
        print(f"[docker_tts_emotion_test] {emotion_name}: {utterance}", flush=True)
        rc = publish_advanced_tts(
            container=container,
            setup=setup,
            topic=topic,
            text=utterance,
            preset=preset,
            language_code=language_code,
            disable_lipsync=disable_lipsync,
            use_emotion=use_emotion,
            emotion_only=emotion_only,
            dry_run=dry_run,
        )
        if rc != 0:
            print(f"[docker_tts_emotion_test] publish failed for {emotion_name}", flush=True)
            return
        time.sleep(estimate_wait_s(utterance, pause_s))


def main() -> None:
    presets = make_presets()
    parser = argparse.ArgumentParser(description="Publish English advanced TTS emotion tests from inside the Haru2 Docker container.")
    parser.add_argument("--container", default="haru-simulator-web-server-1")
    parser.add_argument("--robot-namespace", default="/haru2")
    parser.add_argument("--topic", default="", help="Override advanced TTS topic. Defaults to <namespace>/cmd_tts_advanced.")
    parser.add_argument("--setup", default="source /opt/ros/jazzy/setup.bash")
    parser.add_argument("--language-code", default="en")
    parser.add_argument("--emotion", default="", help="One emotion or comma-separated list. Default: neutral,happy,calm,sad,angry.")
    parser.add_argument("--text", default="", help="English text to speak for every selected emotion.")
    parser.add_argument("--pause-s", type=float, default=3.5)
    parser.add_argument("--disable-lipsync", action="store_true")
    parser.add_argument("--no-emotion-tag", action="store_true", help="Only test pitch/rate/volume/emphasis, without high-level emotion.")
    parser.add_argument("--emotion-only", action="store_true", help="Only test built-in high-level emotion, without manual pitch/rate/volume/emphasis.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    namespace = normalize_namespace(args.robot_namespace)
    topic = args.topic or f"{namespace}/cmd_tts_advanced"
    emotions = parse_emotions(args.emotion, presets)
    run_sequence(
        container=args.container,
        setup=args.setup,
        topic=topic,
        emotions=emotions,
        presets=presets,
        text=args.text.strip(),
        language_code=args.language_code,
        disable_lipsync=args.disable_lipsync,
        use_emotion=not args.no_emotion_tag,
        emotion_only=args.emotion_only,
        pause_s=args.pause_s,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
