#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import shlex
from typing import Dict, List

from .docker_haru2_bridge import run_in_container


SHOWCASE_PROFILES: Dict[str, Dict] = {
    "neutral": {
        "text": "This is neutral. I am giving you a simple response.",
        "config": {"pitch_value": 0.0, "rate_value": 100, "volume_value": 0, "emphasis_type": 0},
        "steps": [
            {"cmd_neck_pitch_pos": 0.14, "cmd_left_eye_stroke_pos": 1.0, "cmd_right_eye_stroke_pos": 1.0, "play_time": 140, "sleep": 0.20},
            {"cmd_neck_pitch_pos": -0.04, "cmd_left_eye_stroke_pos": 0.0, "cmd_right_eye_stroke_pos": 0.0, "play_time": 140, "sleep": 0.24},
            {"cmd_neck_pitch_pos": 0.08, "play_time": 150, "sleep": 0.24},
            {"cmd_neck_pitch_pos": 0.0, "play_time": 150, "sleep": 0.55},
        ],
    },
    "care": {
        "text": "I am here with you. Take your time.",
        "config": {"emotion": 2, "pitch_value": -35.0, "rate_value": 75, "volume_value": -4, "emphasis_type": 0},
        "steps": [
            {"cmd_neck_pitch_pos": -0.22, "cmd_neck_roll_pos": 0.18, "cmd_left_eye_stroke_pos": 0.65, "cmd_right_eye_stroke_pos": 0.65, "play_time": 220, "sleep": 0.45},
            {"cmd_neck_pitch_pos": -0.10, "cmd_neck_roll_pos": 0.08, "play_time": 220, "sleep": 0.42},
            {"cmd_neck_pitch_pos": -0.18, "cmd_neck_roll_pos": -0.08, "play_time": 220, "sleep": 0.40},
            {"cmd_neck_pitch_pos": -0.04, "cmd_neck_roll_pos": 0.0, "cmd_left_eye_stroke_pos": 0.0, "cmd_right_eye_stroke_pos": 0.0, "play_time": 220, "sleep": 0.55},
        ],
    },
    "encouragement": {
        "text": "Great job. That was really nice.",
        "config": {"emotion": 0, "pitch_value": 50.0, "rate_value": 125, "volume_value": 4, "emphasis_type": 2},
        "steps": [
            {"cmd_neck_pitch_pos": 0.26, "cmd_base_pos": 0.12, "cmd_left_eye_stroke_pos": 1.0, "cmd_right_eye_stroke_pos": 1.0, "play_time": 120, "sleep": 0.14},
            {"cmd_neck_pitch_pos": -0.08, "cmd_base_pos": -0.08, "cmd_left_eye_stroke_pos": 0.0, "cmd_right_eye_stroke_pos": 0.0, "play_time": 120, "sleep": 0.14},
            {"cmd_neck_pitch_pos": 0.24, "cmd_base_pos": 0.08, "cmd_left_eye_stroke_pos": 1.0, "cmd_right_eye_stroke_pos": 1.0, "play_time": 120, "sleep": 0.14},
            {"cmd_neck_pitch_pos": 0.0, "cmd_base_pos": 0.0, "cmd_left_eye_stroke_pos": 0.0, "cmd_right_eye_stroke_pos": 0.0, "play_time": 150, "sleep": 0.65},
        ],
    },
    "playful": {
        "text": "Hey, that was fun. I like this little moment.",
        "config": {"emotion": 0, "pitch_value": 65.0, "rate_value": 132, "volume_value": 3, "emphasis_type": 1},
        "steps": [
            {"cmd_neck_roll_pos": 0.24, "cmd_base_pos": 0.18, "cmd_left_eye_yaw_pos": 0.18, "cmd_right_eye_yaw_pos": 0.18, "play_time": 160, "sleep": 0.22},
            {"cmd_neck_roll_pos": -0.22, "cmd_base_pos": -0.18, "cmd_left_eye_yaw_pos": -0.18, "cmd_right_eye_yaw_pos": -0.18, "play_time": 160, "sleep": 0.22},
            {"cmd_neck_roll_pos": 0.18, "cmd_base_pos": 0.12, "cmd_left_eye_stroke_pos": 1.0, "cmd_right_eye_stroke_pos": 1.0, "play_time": 140, "sleep": 0.18},
            {"cmd_neck_roll_pos": 0.0, "cmd_base_pos": 0.0, "cmd_left_eye_yaw_pos": 0.0, "cmd_right_eye_yaw_pos": 0.0, "cmd_left_eye_stroke_pos": 0.0, "cmd_right_eye_stroke_pos": 0.0, "play_time": 160, "sleep": 0.65},
        ],
    },
    "offer_help": {
        "text": "I can help with that if you want.",
        "config": {"pitch_value": 15.0, "rate_value": 92, "volume_value": 1, "emphasis_type": 1},
        "steps": [
            {"cmd_base_pos": 0.28, "cmd_neck_pitch_pos": -0.20, "cmd_left_eye_stroke_pos": 1.0, "cmd_right_eye_stroke_pos": 1.0, "play_time": 190, "sleep": 0.30},
            {"cmd_base_pos": 0.10, "cmd_neck_pitch_pos": 0.08, "cmd_left_eye_stroke_pos": 0.0, "cmd_right_eye_stroke_pos": 0.0, "play_time": 190, "sleep": 0.28},
            {"cmd_base_pos": 0.22, "cmd_neck_pitch_pos": -0.12, "play_time": 190, "sleep": 0.30},
            {"cmd_base_pos": 0.0, "cmd_neck_pitch_pos": 0.0, "play_time": 190, "sleep": 0.65},
        ],
    },
}

RESET_STEP = {
    "cmd_base_pos": 0.0,
    "cmd_neck_pitch_pos": 0.0,
    "cmd_neck_roll_pos": 0.0,
    "cmd_left_eye_yaw_pos": 0.0,
    "cmd_right_eye_yaw_pos": 0.0,
    "cmd_left_eye_stroke_pos": 0.0,
    "cmd_right_eye_stroke_pos": 0.0,
    "play_time": 180,
    "sleep": 0.3,
}


def normalize_namespace(namespace: str) -> str:
    namespace = (namespace or "/haru2").strip()
    if not namespace.startswith("/"):
        namespace = "/" + namespace
    return namespace.rstrip("/")


def parse_profiles(raw: str) -> List[str]:
    profiles = [x.strip().lower() for x in raw.split(",") if x.strip()]
    unknown = [x for x in profiles if x not in SHOWCASE_PROFILES]
    if unknown:
        raise ValueError(f"Unknown profile(s): {', '.join(unknown)}. Available: {', '.join(SHOWCASE_PROFILES)}")
    return profiles or ["neutral", "care", "encouragement", "playful", "offer_help"]


def build_container_runner(namespace: str, profiles: Dict[str, Dict], pause_s: float, language_code: str) -> str:
    profiles_json = json.dumps(profiles, ensure_ascii=False)
    namespace_json = json.dumps(namespace)
    language_json = json.dumps(language_code)
    pause_json = json.dumps(max(0.0, float(pause_s)))
    return f'''
import json
import time

import rclpy
from rclpy.node import Node
from haru2_core_msgs.msg import MotorCommand, TTSAdvancedCommand, TTSConfig, TTSSpurt

namespace = {namespace_json}
language_code = {language_json}
pause_s = {pause_json}
profiles = json.loads({json.dumps(profiles_json)})
reset_step = json.loads({json.dumps(json.dumps(RESET_STEP))})

topic_names = sorted({{
    key
    for profile in profiles.values()
    for step in profile["steps"]
    for key in step
    if key not in ("play_time", "sleep")
}} | {{key for key in reset_step if key not in ("play_time", "sleep")}})

rclpy.init()
node = Node("haru2_emotion_showcase")
motor_pubs = {{
    topic_name: node.create_publisher(MotorCommand, f"{{namespace}}/{{topic_name}}", 10)
    for topic_name in topic_names
}}
tts_pub = node.create_publisher(TTSAdvancedCommand, f"{{namespace}}/cmd_tts_advanced", 10)
time.sleep(0.8)

def make_config(raw):
    config = TTSConfig()
    if "emotion" in raw:
        config.add_emotion = True
        config.emotion = int(raw["emotion"])
    config.add_pitch = True
    config.relative_pitch = True
    config.pitch_type = 0
    config.pitch_value = float(raw.get("pitch_value", 0.0))
    config.add_rate = True
    config.rate_type = 0
    config.rate_value = int(raw.get("rate_value", 100))
    config.add_volume = True
    config.volume_type = 0
    config.volume_value = int(raw.get("volume_value", 0))
    config.add_emphasis = True
    config.emphasis_type = int(raw.get("emphasis_type", 0))
    return config

def publish_tts(text, raw_config):
    msg = TTSAdvancedCommand()
    msg.global_config = make_config(raw_config)
    msg.language_code = language_code
    msg.disable_lipsync = False
    spurt = TTSSpurt()
    spurt.config = TTSConfig()
    spurt.text = text
    msg.spurts = [spurt]
    tts_pub.publish(msg)
    rclpy.spin_once(node, timeout_sec=0.03)

def publish_motor(topic_name, position, play_time):
    msg = MotorCommand()
    msg.position = float(position)
    msg.play_time = max(0, min(int(play_time), 250))
    msg.relative = False
    msg.disable_eyes_roll_sync = False
    motor_pubs[topic_name].publish(msg)

def run_step(step):
    play_time = int(step.get("play_time", 180))
    for topic_name, position in step.items():
        if topic_name in ("play_time", "sleep"):
            continue
        publish_motor(topic_name, position, play_time)
    rclpy.spin_once(node, timeout_sec=0.02)
    time.sleep(float(step.get("sleep", 0.2)))

try:
    for profile_name, profile in profiles.items():
        print(f"[emotion_showcase] {{profile_name}}: {{profile['text']}}", flush=True)
        publish_tts(profile["text"], profile["config"])
        time.sleep(0.05)
        for step in profile["steps"]:
            run_step(step)
        time.sleep(pause_s)
    print("[emotion_showcase] reset", flush=True)
    run_step(reset_step)
finally:
    node.destroy_node()
    rclpy.shutdown()
'''


def main() -> None:
    parser = argparse.ArgumentParser(description="Show Haru2 emotion profiles with synchronized English TTS and motion.")
    parser.add_argument("--container", default="haru-simulator-web-server-1")
    parser.add_argument("--robot-namespace", default="/haru2")
    parser.add_argument("--setup", default="source /opt/ros/jazzy/setup.bash")
    parser.add_argument("--profiles", default="", help="Comma-separated profiles. Default: neutral,care,encouragement,playful,offer_help.")
    parser.add_argument("--pause-s", type=float, default=1.0)
    parser.add_argument("--language-code", default="en")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    selected_names = parse_profiles(args.profiles)
    selected_profiles = {name: SHOWCASE_PROFILES[name] for name in selected_names}
    runner = build_container_runner(
        namespace=normalize_namespace(args.robot_namespace),
        profiles=selected_profiles,
        pause_s=args.pause_s,
        language_code=args.language_code,
    )
    command = f"python3 -c {shlex.quote(runner)}"
    rc = run_in_container(args.container, args.setup, command, dry_run=args.dry_run)
    if rc != 0:
        raise SystemExit(rc)


if __name__ == "__main__":
    main()
