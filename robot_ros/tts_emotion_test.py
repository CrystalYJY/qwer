#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List

import rclpy
from rclpy.node import Node
from std_msgs.msg import Header

try:
    from haru2_core_msgs.msg import TTSAdvancedCommand, TTSConfig, TTSSpurt, TTSStatus
except ImportError:
    try:
        from idmind_tabletop_msgs.msg import TTSAdvancedCommand, TTSConfig, TTSSpurt, TTSStatus
    except ImportError as exc:
        raise RuntimeError(
            "Robot ROS messages are not available. Source the ROS2 workspace before running this script."
        ) from exc


@dataclass(frozen=True)
class EmotionPreset:
    text: str
    emotion: int | None
    pitch_type: int
    rate_type: int
    volume_type: int
    emphasis_type: int


def normalize_namespace(namespace: str) -> str:
    namespace = (namespace or "/haru2").strip()
    if not namespace.startswith("/"):
        namespace = "/" + namespace
    return namespace.rstrip("/")


def make_presets() -> Dict[str, EmotionPreset]:
    return {
        "neutral": EmotionPreset(
            "This is a neutral voice test.",
            None,
            TTSConfig.PITCH_DEFAULT,
            TTSConfig.RATE_DEFAULT,
            TTSConfig.VOLUME_DEFAULT,
            TTSConfig.EMPHASIS_NONE,
        ),
        "happy": EmotionPreset(
            "That is great news. I am really happy for you.",
            TTSConfig.EMOTION_HAPPY,
            TTSConfig.PITCH_HIGH,
            TTSConfig.RATE_MEDIUM,
            TTSConfig.VOLUME_MEDIUM,
            TTSConfig.EMPHASIS_MODERATED,
        ),
        "calm": EmotionPreset(
            "Take your time. We can go slowly and calmly.",
            TTSConfig.EMOTION_CALM,
            TTSConfig.PITCH_SLOW,
            TTSConfig.RATE_SLOW,
            TTSConfig.VOLUME_SOFT,
            TTSConfig.EMPHASIS_NONE,
        ),
        "sad": EmotionPreset(
            "I am sorry that felt difficult. I am here with you.",
            TTSConfig.EMOTION_SAD,
            TTSConfig.PITCH_SLOW,
            TTSConfig.RATE_SLOW,
            TTSConfig.VOLUME_SOFT,
            TTSConfig.EMPHASIS_REDUCED,
        ),
        "angry": EmotionPreset(
            "Hey, that was intense. Let us pause for a second.",
            TTSConfig.EMOTION_ANGRY,
            TTSConfig.PITCH_MEDIUM,
            TTSConfig.RATE_MEDIUM,
            TTSConfig.VOLUME_LOUD,
            TTSConfig.EMPHASIS_STRONG,
        ),
    }


def build_config(preset: EmotionPreset, use_emotion: bool = True) -> TTSConfig:
    config = TTSConfig()
    config.add_pitch = True
    config.pitch_type = preset.pitch_type
    config.relative_pitch = False
    config.pitch_value = 0.0

    config.add_rate = True
    config.rate_type = preset.rate_type
    config.rate_value = 0

    config.add_volume = True
    config.volume_type = preset.volume_type
    config.volume_value = 0

    config.add_emphasis = True
    config.emphasis_type = preset.emphasis_type

    if use_emotion and preset.emotion is not None:
        config.add_emotion = True
        config.emotion = preset.emotion
    return config


def build_advanced_command(
    text: str,
    preset: EmotionPreset,
    language_code: str,
    disable_lipsync: bool,
    use_emotion: bool,
    node: Node,
) -> TTSAdvancedCommand:
    msg = TTSAdvancedCommand()
    msg.header = Header()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.global_config = build_config(preset, use_emotion=use_emotion)
    msg.disable_lipsync = disable_lipsync
    msg.language_code = language_code

    spurt = TTSSpurt()
    spurt.config = TTSConfig()
    spurt.text = text
    msg.spurts = [spurt]
    return msg


class EmotionTTSNode(Node):
    def __init__(self, namespace: str, topic: str | None) -> None:
        super().__init__("haru2_tts_emotion_test")
        self.namespace = normalize_namespace(namespace)
        self.topic_name = topic or f"{self.namespace}/cmd_tts_advanced"
        self.status_topic = f"{self.namespace}/tts_status"
        self.publisher = self.create_publisher(TTSAdvancedCommand, self.topic_name, 10)
        self.last_status = None
        self.create_subscription(TTSStatus, self.status_topic, self._on_status, 10)

    def _on_status(self, msg: TTSStatus) -> None:
        self.last_status = msg.status

    def publish_emotion(
        self,
        emotion_name: str,
        text: str,
        preset: EmotionPreset,
        language_code: str,
        disable_lipsync: bool,
        use_emotion: bool,
    ) -> None:
        msg = build_advanced_command(
            text=text,
            preset=preset,
            language_code=language_code,
            disable_lipsync=disable_lipsync,
            use_emotion=use_emotion,
            node=self,
        )
        self.publisher.publish(msg)
        self.get_logger().info(f"Published {emotion_name} advanced TTS to {self.topic_name}: {text}")


def parse_emotions(raw: str, presets: Dict[str, EmotionPreset]) -> List[str]:
    values = [item.strip().lower() for item in raw.split(",") if item.strip()]
    unknown = [item for item in values if item not in presets]
    if unknown:
        raise ValueError(f"Unknown emotion(s): {', '.join(unknown)}. Available: {', '.join(presets)}")
    return values or ["neutral", "happy", "calm", "sad", "angry"]


def estimate_wait_s(text: str, minimum: float) -> float:
    words = max(1, len(text.split()))
    return max(minimum, min(12.0, words / 2.4 + 1.0))


def run_sequence(
    node: EmotionTTSNode,
    emotions: Iterable[str],
    presets: Dict[str, EmotionPreset],
    text: str,
    language_code: str,
    disable_lipsync: bool,
    use_emotion: bool,
    pause_s: float,
) -> None:
    for emotion_name in emotions:
        preset = presets[emotion_name]
        utterance = text or preset.text
        node.publish_emotion(
            emotion_name=emotion_name,
            text=utterance,
            preset=preset,
            language_code=language_code,
            disable_lipsync=disable_lipsync,
            use_emotion=use_emotion,
        )
        wait_until = time.time() + estimate_wait_s(utterance, pause_s)
        while rclpy.ok() and time.time() < wait_until:
            rclpy.spin_once(node, timeout_sec=0.1)


def main() -> None:
    presets = make_presets()
    parser = argparse.ArgumentParser(description="Publish English advanced TTS commands with emotion/prosody.")
    parser.add_argument("--robot-namespace", default="/haru2")
    parser.add_argument("--topic", default="", help="Override advanced TTS topic. Defaults to <namespace>/cmd_tts_advanced.")
    parser.add_argument("--language-code", default="en")
    parser.add_argument("--emotion", default="", help="One emotion or comma-separated list. Default: neutral,happy,calm,sad,angry.")
    parser.add_argument("--text", default="", help="English text to speak for every selected emotion.")
    parser.add_argument("--pause-s", type=float, default=3.5)
    parser.add_argument("--disable-lipsync", action="store_true")
    parser.add_argument("--no-emotion-tag", action="store_true", help="Only test pitch/rate/volume/emphasis, without high-level emotion.")
    args = parser.parse_args()

    emotions = parse_emotions(args.emotion, presets)
    rclpy.init()
    node = EmotionTTSNode(args.robot_namespace, args.topic or None)
    try:
        time.sleep(0.5)
        run_sequence(
            node=node,
            emotions=emotions,
            presets=presets,
            text=args.text.strip(),
            language_code=args.language_code,
            disable_lipsync=args.disable_lipsync,
            use_emotion=not args.no_emotion_tag,
            pause_s=args.pause_s,
        )
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
