#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import shlex
import time
from typing import Dict, Iterable, List

from .docker_haru2_bridge import run_in_container


MOTOR_MSG = "haru2_core_msgs/msg/MotorCommand"


PROFILES: Dict[str, List[Dict]] = {
    "neutral": [
        {"cmd_neck_pitch_pos": 0.08, "cmd_left_eye_stroke_pos": 1.0, "cmd_right_eye_stroke_pos": 1.0, "play_time": 120, "sleep": 0.12},
        {"cmd_neck_pitch_pos": 0.0, "cmd_left_eye_stroke_pos": 0.0, "cmd_right_eye_stroke_pos": 0.0, "play_time": 130, "sleep": 0.45},
    ],
    "care": [
        {"cmd_neck_pitch_pos": -0.10, "cmd_neck_roll_pos": 0.10, "cmd_left_eye_stroke_pos": 0.6, "cmd_right_eye_stroke_pos": 0.6, "play_time": 180, "sleep": 0.35},
        {"cmd_neck_pitch_pos": -0.04, "cmd_neck_roll_pos": 0.04, "play_time": 180, "sleep": 0.25},
        {"cmd_neck_pitch_pos": 0.0, "cmd_neck_roll_pos": 0.0, "cmd_left_eye_stroke_pos": 0.0, "cmd_right_eye_stroke_pos": 0.0, "play_time": 180, "sleep": 0.55},
    ],
    "encouragement": [
        {"cmd_neck_pitch_pos": 0.18, "cmd_left_eye_stroke_pos": 1.0, "cmd_right_eye_stroke_pos": 1.0, "play_time": 110, "sleep": 0.10},
        {"cmd_neck_pitch_pos": -0.04, "cmd_left_eye_stroke_pos": 0.0, "cmd_right_eye_stroke_pos": 0.0, "play_time": 110, "sleep": 0.12},
        {"cmd_neck_pitch_pos": 0.14, "cmd_left_eye_stroke_pos": 1.0, "cmd_right_eye_stroke_pos": 1.0, "play_time": 110, "sleep": 0.10},
        {"cmd_neck_pitch_pos": 0.0, "cmd_left_eye_stroke_pos": 0.0, "cmd_right_eye_stroke_pos": 0.0, "play_time": 130, "sleep": 0.55},
    ],
    "playful": [
        {"cmd_neck_roll_pos": 0.16, "cmd_left_eye_yaw_pos": 0.12, "cmd_right_eye_yaw_pos": 0.12, "play_time": 160, "sleep": 0.18},
        {"cmd_neck_roll_pos": -0.12, "cmd_left_eye_yaw_pos": -0.10, "cmd_right_eye_yaw_pos": -0.10, "play_time": 160, "sleep": 0.18},
        {"cmd_neck_roll_pos": 0.0, "cmd_left_eye_yaw_pos": 0.0, "cmd_right_eye_yaw_pos": 0.0, "play_time": 160, "sleep": 0.55},
    ],
    "offer_help": [
        {"cmd_base_pos": 0.18, "cmd_neck_pitch_pos": -0.10, "cmd_left_eye_stroke_pos": 1.0, "cmd_right_eye_stroke_pos": 1.0, "play_time": 180, "sleep": 0.20},
        {"cmd_base_pos": 0.05, "cmd_neck_pitch_pos": 0.04, "cmd_left_eye_stroke_pos": 0.0, "cmd_right_eye_stroke_pos": 0.0, "play_time": 170, "sleep": 0.18},
        {"cmd_base_pos": 0.0, "cmd_neck_pitch_pos": 0.0, "play_time": 180, "sleep": 0.55},
    ],
    "reset": [
        {"cmd_base_pos": 0.0, "cmd_neck_pitch_pos": 0.0, "cmd_neck_roll_pos": 0.0, "cmd_left_eye_yaw_pos": 0.0, "cmd_right_eye_yaw_pos": 0.0, "cmd_left_eye_stroke_pos": 0.0, "cmd_right_eye_stroke_pos": 0.0, "play_time": 180, "sleep": 0.35},
    ],
}


def normalize_namespace(namespace: str) -> str:
    namespace = (namespace or "/haru2").strip()
    if not namespace.startswith("/"):
        namespace = "/" + namespace
    return namespace.rstrip("/")


def parse_profiles(raw: str) -> List[str]:
    profiles = [x.strip().lower() for x in raw.split(",") if x.strip()]
    unknown = [x for x in profiles if x not in PROFILES]
    if unknown:
        raise ValueError(f"Unknown profile(s): {', '.join(unknown)}. Available: {', '.join(PROFILES)}")
    return profiles or ["neutral", "care", "encouragement", "playful", "offer_help"]


def motor_payload(position: float, play_time: int) -> str:
    return (
        "{"
        f"position: {float(position)}, "
        f"play_time: {max(0, min(int(play_time), 250))}, "
        "relative: false, "
        "disable_eyes_roll_sync: false"
        "}"
    )


def publish_motor(container: str, setup: str, namespace: str, topic_name: str, position: float, play_time: int, dry_run: bool) -> int:
    topic = f"{namespace}/{topic_name}"
    command = f"ros2 topic pub --once {shlex.quote(topic)} {MOTOR_MSG} {shlex.quote(motor_payload(position, play_time))}"
    return run_in_container(container, setup, command, dry_run=dry_run)


def run_step(container: str, setup: str, namespace: str, step: Dict, dry_run: bool) -> bool:
    play_time = int(step.get("play_time", 180))
    for topic_name, value in step.items():
        if topic_name in {"play_time", "sleep"}:
            continue
        rc = publish_motor(container, setup, namespace, topic_name, float(value), play_time, dry_run=dry_run)
        if rc != 0:
            return False
    time.sleep(float(step.get("sleep", 0.2)))
    return True


def run_profiles(container: str, setup: str, namespace: str, profiles: Iterable[str], pause_s: float, dry_run: bool) -> None:
    profile_names = list(profiles)
    run_names = profile_names if "reset" in profile_names else [*profile_names, "reset"]
    selected_profiles = {name: PROFILES[name] for name in run_names}
    container_code = build_container_runner(namespace, selected_profiles, pause_s)
    command = f"python3 -c {shlex.quote(container_code)}"
    rc = run_in_container(container, setup, command, dry_run=dry_run)
    if rc != 0:
        print("[docker_emotion_motion_test] container motion runner failed", flush=True)


def build_container_runner(namespace: str, selected_profiles: Dict[str, List[Dict]], pause_s: float) -> str:
    profiles_json = json.dumps(selected_profiles, ensure_ascii=False)
    namespace_json = json.dumps(namespace)
    pause_json = json.dumps(max(0.0, float(pause_s)))
    return f'''
import json
import time

import rclpy
from rclpy.node import Node
from haru2_core_msgs.msg import MotorCommand

namespace = {namespace_json}
profiles = json.loads({json.dumps(profiles_json)})
pause_s = {pause_json}

topic_names = sorted({{
    key
    for steps in profiles.values()
    for step in steps
    for key in step
    if key not in ("play_time", "sleep")
}})

rclpy.init()
node = Node("haru2_emotion_motion_profile_test")
publishers = {{
    topic_name: node.create_publisher(MotorCommand, f"{{namespace}}/{{topic_name}}", 10)
    for topic_name in topic_names
}}
time.sleep(0.5)

def publish_motor(topic_name, position, play_time):
    msg = MotorCommand()
    msg.position = float(position)
    msg.play_time = max(0, min(int(play_time), 250))
    msg.relative = False
    msg.disable_eyes_roll_sync = False
    publishers[topic_name].publish(msg)

try:
    for profile_name, steps in profiles.items():
        print(f"[container_motion] profile={{profile_name}}", flush=True)
        for step in steps:
            play_time = int(step.get("play_time", 180))
            for topic_name, position in step.items():
                if topic_name in ("play_time", "sleep"):
                    continue
                publish_motor(topic_name, position, play_time)
            rclpy.spin_once(node, timeout_sec=0.02)
            time.sleep(float(step.get("sleep", 0.2)))
        time.sleep(pause_s)
finally:
    node.destroy_node()
    rclpy.shutdown()
'''


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Haru2 emotion motion profiles inside the Docker simulator.")
    parser.add_argument("--container", default="haru-simulator-web-server-1")
    parser.add_argument("--robot-namespace", default="/haru2")
    parser.add_argument("--setup", default="source /opt/ros/jazzy/setup.bash")
    parser.add_argument("--profiles", default="", help="Comma-separated profiles. Default: neutral,care,encouragement,playful,offer_help.")
    parser.add_argument("--pause-s", type=float, default=0.7)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_profiles(
        container=args.container,
        setup=args.setup,
        namespace=normalize_namespace(args.robot_namespace),
        profiles=parse_profiles(args.profiles),
        pause_s=args.pause_s,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
