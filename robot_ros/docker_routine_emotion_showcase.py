#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import shlex
from typing import Dict, List

from .docker_haru2_bridge import run_in_container


ROUTINE_PROFILES: Dict[str, Dict] = {
    "neutral": {
        "routine_id": 2036,
        "routine_name": "nodding",
        "text": "Sure, I understand what you mean. Let me stay with you for a moment while we figure this out together.",
        "duration_s": 10.5,
        "config": {},
    },
    "happy": {
        "routine_id": 2034,
        "routine_name": "harumoji_happy",
        "text": "Great job. That was really nice to see, and I am genuinely happy for you. Keep going, you are doing well.",
        "duration_s": 10.5,
        "config": {"emotion": 0},
    },
    "care": {
        "routine_id": 2084,
        "routine_name": "sympathetic",
        "text": "I am here with you. Take your time, breathe slowly, and do not rush yourself. We can go step by step.",
        "duration_s": 12.0,
        "config": {"emotion": 2},
    },
    "sad": {
        "routine_id": 2065,
        "routine_name": "dejected",
        "text": "I am sorry. That sounds difficult, and I wish things felt a little easier right now. I will stay quietly with you.",
        "duration_s": 12.0,
        "config": {"emotion": 1},
    },
    "playful": {
        "routine_id": 2092,
        "routine_name": "harumoji_wink",
        "text": "Hey, that was fun. I saw that little moment, and honestly, it made this feel much more lively.",
        "duration_s": 9.5,
        "config": {"emotion": 0},
    },
    "offer_help": {
        "routine_id": 2094,
        "routine_name": "lean_forward",
        "text": "I can help with that if you want. Tell me what part is bothering you, and I will try to make it easier.",
        "duration_s": 10.5,
        "config": {},
    },
}


def normalize_namespace(namespace: str) -> str:
    namespace = (namespace or "/haru2").strip()
    if not namespace.startswith("/"):
        namespace = "/" + namespace
    return namespace.rstrip("/")


def parse_profiles(raw: str) -> List[str]:
    profiles = [x.strip().lower() for x in raw.split(",") if x.strip()]
    unknown = [x for x in profiles if x not in ROUTINE_PROFILES]
    if unknown:
        raise ValueError(f"Unknown profile(s): {', '.join(unknown)}. Available: {', '.join(ROUTINE_PROFILES)}")
    return profiles or ["neutral", "happy", "care", "sad", "playful", "offer_help"]


def build_container_runner(namespace: str, profiles: Dict[str, Dict], pause_s: float, language_code: str, disable_lipsync: bool) -> str:
    profiles_json = json.dumps(profiles, ensure_ascii=False)
    namespace_json = json.dumps(namespace)
    language_json = json.dumps(language_code)
    pause_json = json.dumps(max(0.0, float(pause_s)))
    disable_lipsync_py = "True" if disable_lipsync else "False"
    return f'''
import json
import time

import rclpy
from rclpy.node import Node
from haru2_core_msgs.msg import TTSAdvancedCommand, TTSConfig, TTSSpurt
from haru2_core_msgs.srv import AdvancedRoutine
from std_srvs.srv import Empty

namespace = {namespace_json}
language_code = {language_json}
pause_s = {pause_json}
disable_lipsync = {disable_lipsync_py}
profiles = json.loads({json.dumps(profiles_json)})

rclpy.init()
node = Node("haru2_routine_emotion_showcase")
tts_pub = node.create_publisher(TTSAdvancedCommand, f"{{namespace}}/cmd_tts_advanced", 10)
routine_client = node.create_client(AdvancedRoutine, f"{{namespace}}/execute_advanced_routine")
stop_client = node.create_client(Empty, f"{{namespace}}/stop_routines")

if not routine_client.wait_for_service(timeout_sec=4.0):
    raise RuntimeError(f"Routine service not available: {{namespace}}/execute_advanced_routine")

time.sleep(0.5)

def make_config(raw):
    config = TTSConfig()
    if "emotion" in raw:
        config.add_emotion = True
        config.emotion = int(raw["emotion"])
    return config

def publish_tts(text, raw_config):
    msg = TTSAdvancedCommand()
    msg.global_config = make_config(raw_config)
    msg.language_code = language_code
    msg.disable_lipsync = disable_lipsync
    spurt = TTSSpurt()
    spurt.config = TTSConfig()
    spurt.text = text
    msg.spurts = [spurt]
    tts_pub.publish(msg)
    rclpy.spin_once(node, timeout_sec=0.03)

def call_routine(routine_id):
    request = AdvancedRoutine.Request()
    request.routine = int(routine_id)
    request.disable_audio = True
    request.disable_base_yaw = False
    request.disable_neck_pitch = False
    request.disable_neck_roll = False
    request.disable_left_eye_roll = False
    request.disable_right_eye_roll = False
    request.disable_left_eye_yaw = False
    request.disable_right_eye_yaw = False
    request.disable_left_eye_stroke = False
    request.disable_right_eye_stroke = False
    request.disable_eyes_lcd = False
    request.disable_eyes_led = False
    request.disable_mouth_led = False
    request.relative = False
    future = routine_client.call_async(request)
    rclpy.spin_once(node, timeout_sec=0.05)
    return future

def stop_routines():
    if stop_client.wait_for_service(timeout_sec=0.3):
        stop_client.call_async(Empty.Request())
        rclpy.spin_once(node, timeout_sec=0.05)

try:
    for profile_name, profile in profiles.items():
        print(
            f"[routine_emotion_showcase] {{profile_name}} -> "
            f"{{profile['routine_id']}} {{profile['routine_name']}}: {{profile['text']}}",
            flush=True,
        )
        stop_routines()
        time.sleep(0.1)
        call_routine(profile["routine_id"])
        time.sleep(0.15)
        publish_tts(profile["text"], profile["config"])
        end_time = time.time() + float(profile.get("duration_s", 4.0))
        while time.time() < end_time:
            rclpy.spin_once(node, timeout_sec=0.05)
        time.sleep(pause_s)
    stop_routines()
finally:
    node.destroy_node()
    rclpy.shutdown()
'''


def main() -> None:
    parser = argparse.ArgumentParser(description="Show Haru2 routine-library expressions while speaking English TTS.")
    parser.add_argument("--container", default="haru-simulator-web-server-1")
    parser.add_argument("--robot-namespace", default="/haru2")
    parser.add_argument("--setup", default="source /opt/ros/jazzy/setup.bash")
    parser.add_argument("--profiles", default="", help="Comma-separated profiles. Default: neutral,happy,care,sad,playful,offer_help.")
    parser.add_argument("--pause-s", type=float, default=0.8)
    parser.add_argument("--language-code", default="en")
    parser.add_argument("--enable-lipsync", action="store_true", help="Keep TTS lip sync enabled. Default disables it so routine mouth expression is visible.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    names = parse_profiles(args.profiles)
    profiles = {name: ROUTINE_PROFILES[name] for name in names}
    runner = build_container_runner(
        namespace=normalize_namespace(args.robot_namespace),
        profiles=profiles,
        pause_s=args.pause_s,
        language_code=args.language_code,
        disable_lipsync=not args.enable_lipsync,
    )
    command = f"python3 -c {shlex.quote(runner)}"
    rc = run_in_container(args.container, args.setup, command, dry_run=args.dry_run)
    if rc != 0:
        raise SystemExit(rc)


if __name__ == "__main__":
    main()
