#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


MOTOR_MSG = "haru2_core_msgs/msg/MotorCommand"
TTS_MSG = "haru2_core_msgs/msg/TTSCommand"


def yaml_quote(text: str) -> str:
    return '"' + str(text).replace("\\", "\\\\").replace('"', '\\"') + '"'


def run_in_container(container: str, setup: str, command: str, dry_run: bool = False) -> int:
    script = f"{setup}\n{command}"
    docker_cmd = ["docker", "exec", container, "bash", "-lc", script]
    if dry_run:
        print("[dry-run]", " ".join(shlex.quote(part) for part in docker_cmd), flush=True)
        return 0
    result = subprocess.run(docker_cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.stdout.strip():
        print(result.stdout.strip(), flush=True)
    if result.stderr.strip():
        print(result.stderr.strip(), flush=True)
    return result.returncode


class Haru2DockerBridge:
    def __init__(
        self,
        container: str,
        namespace: str = "/haru2",
        setup: str = "source /opt/ros/jazzy/setup.bash",
        dry_run: bool = False,
    ) -> None:
        self.container = container
        self.namespace = namespace.rstrip("/")
        self.setup = setup
        self.dry_run = dry_run

    def topic(self, name: str) -> str:
        return f"{self.namespace}/{name}"

    def publish_motor(self, motor_topic: str, position: float, play_time: int = 200) -> None:
        payload = (
            "{"
            f"position: {float(position)}, "
            f"play_time: {max(0, min(int(play_time), 250))}, "
            "relative: false, "
            "disable_eyes_roll_sync: false"
            "}"
        )
        command = f"ros2 topic pub --once {shlex.quote(self.topic(motor_topic))} {MOTOR_MSG} {shlex.quote(payload)}"
        rc = run_in_container(self.container, self.setup, command, dry_run=self.dry_run)
        if rc != 0:
            print(f"[robot_docker_bridge] motor publish failed: {motor_topic}", flush=True)

    def speak(self, text: str, language_code: str = "", disable_lipsync: bool = False) -> None:
        payload = (
            "{"
            f"message: {yaml_quote(text)}, "
            f"disable_lipsync: {'true' if disable_lipsync else 'false'}, "
            f"language_code: {yaml_quote(language_code)}"
            "}"
        )
        command = f"ros2 topic pub --once {shlex.quote(self.topic('cmd_tts'))} {TTS_MSG} {shlex.quote(payload)}"
        rc = run_in_container(self.container, self.setup, command, dry_run=self.dry_run)
        if rc != 0:
            print("[robot_docker_bridge] tts publish failed", flush=True)

    def test_motion(self) -> None:
        self.publish_motor("cmd_base_pos", 0.5)
        time.sleep(1.1)
        self.publish_motor("cmd_base_pos", -0.5)
        time.sleep(1.1)
        self.publish_motor("cmd_base_pos", 0.0)
        time.sleep(0.5)
        self.publish_motor("cmd_neck_pitch_pos", 0.35)
        time.sleep(0.8)
        self.publish_motor("cmd_neck_pitch_pos", -0.25)
        time.sleep(0.8)
        self.publish_motor("cmd_neck_pitch_pos", 0.0)

    def look_forward(self) -> None:
        self.publish_motor("cmd_base_pos", 0.0, play_time=180)
        self.publish_motor("cmd_neck_pitch_pos", 0.0, play_time=180)
        self.publish_motor("cmd_neck_roll_pos", 0.0, play_time=180)
        self.publish_motor("cmd_left_eye_yaw_pos", 0.0, play_time=160)
        self.publish_motor("cmd_right_eye_yaw_pos", 0.0, play_time=160)

    def listen_motion(self) -> None:
        self.publish_motor("cmd_neck_roll_pos", 0.12, play_time=180)
        self.publish_motor("cmd_left_eye_yaw_pos", 0.12, play_time=160)
        self.publish_motor("cmd_right_eye_yaw_pos", 0.12, play_time=160)
        time.sleep(0.25)
        self.publish_motor("cmd_neck_roll_pos", 0.0, play_time=180)
        self.publish_motor("cmd_left_eye_yaw_pos", 0.0, play_time=160)
        self.publish_motor("cmd_right_eye_yaw_pos", 0.0, play_time=160)

    def reply_motion(self) -> None:
        self.publish_motor("cmd_neck_pitch_pos", 0.22, play_time=160)
        time.sleep(0.25)
        self.publish_motor("cmd_neck_pitch_pos", 0.0, play_time=160)
        self.blink()

    def proactive_motion(self) -> None:
        self.publish_motor("cmd_base_pos", 0.18, play_time=200)
        self.publish_motor("cmd_neck_pitch_pos", -0.10, play_time=180)
        time.sleep(0.35)
        self.publish_motor("cmd_base_pos", 0.0, play_time=200)
        self.publish_motor("cmd_neck_pitch_pos", 0.0, play_time=180)
        self.blink()

    def blink(self) -> None:
        self.publish_motor("cmd_left_eye_stroke_pos", 1.0, play_time=100)
        self.publish_motor("cmd_right_eye_stroke_pos", 1.0, play_time=100)
        time.sleep(0.10)
        self.publish_motor("cmd_left_eye_stroke_pos", 0.0, play_time=100)
        self.publish_motor("cmd_right_eye_stroke_pos", 0.0, play_time=100)

    def look_at_target(
        self,
        point_3d: Tuple[float, float, float],
        yaw_scale: float = 4.0,
        pitch_scale: float = 2.5,
    ) -> None:
        base_yaw, neck_pitch, eye_yaw = compute_track_commands(
            point_3d,
            yaw_scale=yaw_scale,
            pitch_scale=pitch_scale,
        )
        self.publish_motor("cmd_base_pos", base_yaw, play_time=180)
        self.publish_motor("cmd_neck_pitch_pos", neck_pitch, play_time=180)
        self.publish_motor("cmd_left_eye_yaw_pos", eye_yaw, play_time=160)
        self.publish_motor("cmd_right_eye_yaw_pos", eye_yaw, play_time=160)


def load_turns(path: Path) -> List[Dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except json.JSONDecodeError:
        return []


def load_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


class SkeletonTargetWatcher:
    """Subscribe to skeletons locally; motor commands are still sent through Docker."""

    def __init__(self, skeleton_topic: str, prefer_part: str = "nose") -> None:
        import rclpy
        from rclpy.node import Node
        from strawberry_ros_msgs.msg import Skeletons

        self.rclpy = rclpy
        self.Node = Node
        self.Skeletons = Skeletons
        self.skeleton_topic = skeleton_topic
        self.prefer_part = prefer_part.lower()
        self.latest_target: Optional[Tuple[float, float, float]] = None
        self.latest_motion: Dict[str, object] = {}
        self.latest_wall = 0.0
        self._prev_parts: Dict[str, Tuple[float, float, float]] = {}
        self._prev_wall = 0.0
        self.lock = threading.Lock()
        self.node = None
        self.thread = None

    def start(self) -> None:
        if not self.rclpy.ok():
            self.rclpy.init()
        self.node = self.Node("docker_haru2_skeleton_watcher")
        self.node.create_subscription(self.Skeletons, self.skeleton_topic, self._on_skeletons, 10)
        self.node.get_logger().info(f"Watching skeleton topic for Docker bridge: {self.skeleton_topic}")
        self.thread = threading.Thread(target=self.rclpy.spin, args=(self.node,), daemon=True)
        self.thread.start()

    def stop(self, shutdown_context: bool = True) -> None:
        if self.node is not None:
            self.node.destroy_node()
        if shutdown_context and self.rclpy.ok():
            self.rclpy.shutdown()

    def get_latest(self, max_age_s: float) -> Optional[Tuple[float, float, float]]:
        with self.lock:
            if self.latest_target is None:
                return None
            if time.time() - self.latest_wall > max_age_s:
                return None
            return self.latest_target

    def get_motion(self, max_age_s: float) -> Optional[Dict[str, object]]:
        with self.lock:
            if not self.latest_motion:
                return None
            if time.time() - self.latest_wall > max_age_s:
                return None
            return dict(self.latest_motion)

    def _on_skeletons(self, msg) -> None:
        if not msg.skeletons:
            return
        skeleton = msg.skeletons[0]
        parts = {str(part.name or "").lower(): part for part in skeleton.body_parts}
        target = parts.get(self.prefer_part) or parts.get("head")
        if target is None:
            return
        pos = target.pose.position
        now = time.time()
        current_parts: Dict[str, Tuple[float, float, float]] = {}
        for name, part in parts.items():
            part_pos = part.pose.position
            current_parts[name] = (float(part_pos.x), float(part_pos.y), float(part_pos.z))
        part_speeds = self._compute_part_speeds(current_parts, now)
        with self.lock:
            self.latest_target = (float(pos.x), float(pos.y), float(pos.z))
            self.latest_motion = {
                "wall_time": now,
                "motion_score": max(part_speeds.values()) if part_speeds else 0.0,
                "part_speeds": part_speeds,
                "part_positions": {
                    name: {"x": point[0], "y": point[1], "z": point[2]}
                    for name, point in current_parts.items()
                    if self._is_motion_part(name)
                },
            }
            self.latest_wall = now
            self._prev_parts = current_parts
            self._prev_wall = now

    def _compute_part_speeds(self, current_parts: Dict[str, Tuple[float, float, float]], now: float) -> Dict[str, float]:
        if not self._prev_parts or self._prev_wall <= 0:
            return {}
        dt = max(now - self._prev_wall, 1e-3)
        speeds: Dict[str, float] = {}
        for name, point in current_parts.items():
            if not self._is_motion_part(name):
                continue
            prev = self._prev_parts.get(name)
            if prev is None:
                continue
            dx = point[0] - prev[0]
            dy = point[1] - prev[1]
            dz = point[2] - prev[2]
            speeds[name] = round(math.sqrt(dx * dx + dy * dy + dz * dz) / dt, 4)
        return speeds

    @staticmethod
    def _is_motion_part(name: str) -> bool:
        lowered = name.lower()
        keywords = ("head", "nose", "hand", "wrist", "elbow")
        return any(keyword in lowered for keyword in keywords)


def compute_track_commands(
    point_3d: Tuple[float, float, float],
    yaw_scale: float = 4.0,
    pitch_scale: float = 2.5,
) -> Tuple[float, float, float]:
    x, y, z = point_3d
    yaw = math.atan2(-y, x) * yaw_scale
    pitch = -math.atan2(z, math.sqrt(y * y + x * x)) * pitch_scale
    base_yaw = clamp(yaw, -0.55, 0.55)
    eye_yaw = clamp(yaw * 0.35, -0.35, 0.35)
    neck_pitch = clamp(pitch, -0.45, 0.45)
    return base_yaw, neck_pitch, eye_yaw


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def extract_tts_text(reply: str) -> str:
    """Use the English line for TTS when the dialogue model returns bilingual text."""
    reply = str(reply or "").strip()
    if not reply:
        return ""

    lines = [line.strip() for line in reply.splitlines() if line.strip()]
    for line in lines:
        match = re.match(r"^[\s*`_#>-]*(?:English|EN|英文)\s*[:：\-]\s*(.+)$", line, flags=re.IGNORECASE)
        if match:
            return clean_tts_label(match.group(1))

    # Handle one-line outputs like "中文: ... English: ..."
    match = re.search(r"(?:English|EN|英文)\s*[:：\-]\s*(.+)$", reply, flags=re.IGNORECASE | re.DOTALL)
    if match:
        text = match.group(1).strip()
        text = re.split(r"(?:中文|Chinese)\s*[:：\-]", text, maxsplit=1, flags=re.IGNORECASE)[0]
        return clean_tts_label(text)

    return clean_tts_label(reply)


def clean_tts_label(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"^[\s*`_#>\"'“”‘’]+", "", text)
    text = re.sub(r"[\s*`_#>\"'“”‘’]+$", "", text)
    # Defensive cleanup in case the label was partially left in the generated text.
    text = re.sub(r"^(?:English|EN|英文)\s*[:：\-]\s*", "", text, flags=re.IGNORECASE).strip()
    return text


def main() -> None:
    p = argparse.ArgumentParser(
        description="Control Haru2 simulator by publishing ROS2 topics inside the Docker container."
    )
    p.add_argument("--container", default="haru-simulator-web-server-1")
    p.add_argument("--robot-namespace", default="/haru2")
    p.add_argument("--setup", default="source /opt/ros/jazzy/setup.bash")
    p.add_argument("--dialogue-json", default="streaming_camera_outputs/camera_asr_dialogue/dialogue_turns.json")
    p.add_argument("--dialogue-log", default="streaming_camera_outputs/camera_asr_dialogue/dialogue.log")
    p.add_argument("--poll-s", type=float, default=0.5)
    p.add_argument("--language-code", default="en")
    p.add_argument("--disable-lipsync", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--replay-existing", action="store_true")
    p.add_argument("--test-motion", action="store_true")
    p.add_argument("--test-tts", default="")
    p.add_argument("--no-motion", action="store_true")
    p.add_argument("--no-listen-motion", action="store_true")
    p.add_argument("--no-tts", action="store_true")
    p.add_argument("--track-skeleton", action="store_true")
    p.add_argument("--skeleton-topic", default="/strawberry/azure/skeletons")
    p.add_argument("--track-interval-s", type=float, default=1.0)
    p.add_argument("--track-max-age-s", type=float, default=2.0)
    p.add_argument("--track-yaw-scale", type=float, default=4.0)
    p.add_argument("--track-pitch-scale", type=float, default=2.5)
    args = p.parse_args()

    bridge = Haru2DockerBridge(
        container=args.container,
        namespace=args.robot_namespace,
        setup=args.setup,
        dry_run=args.dry_run,
    )

    if args.test_motion:
        bridge.test_motion()
        return
    if args.test_tts:
        bridge.speak(args.test_tts, language_code=args.language_code, disable_lipsync=args.disable_lipsync)
        return

    skeleton_watcher = None
    next_track_wall = time.time() + 0.2
    if args.track_skeleton:
        skeleton_watcher = SkeletonTargetWatcher(args.skeleton_topic)
        skeleton_watcher.start()
        print(f"[robot_docker_bridge] skeleton tracking enabled: {args.skeleton_topic}", flush=True)

    dialogue_path = Path(args.dialogue_json)
    dialogue_log_path = Path(args.dialogue_log)
    seen_count = 0
    initialized = bool(args.replay_existing)
    seen_log_size = len(load_text(dialogue_log_path))
    print(f"[robot_docker_bridge] watching {dialogue_path}", flush=True)
    if not args.no_listen_motion:
        print(f"[robot_docker_bridge] also watching {dialogue_log_path} for USER_ASR listen motions", flush=True)
    print(f"[robot_docker_bridge] container={args.container} namespace={args.robot_namespace}", flush=True)
    if not args.replay_existing:
        print("[robot_docker_bridge] will skip the first snapshot and wait for new replies", flush=True)

    try:
        while True:
            if skeleton_watcher is not None and not args.no_motion and time.time() >= next_track_wall:
                next_track_wall = time.time() + max(0.3, args.track_interval_s)
                target = skeleton_watcher.get_latest(max_age_s=args.track_max_age_s)
                if target is not None:
                    print(f"[robot_docker_bridge] track target={target}", flush=True)
                    bridge.look_at_target(
                        target,
                        yaw_scale=args.track_yaw_scale,
                        pitch_scale=args.track_pitch_scale,
                    )

            if not args.no_motion and not args.no_listen_motion:
                log_text = load_text(dialogue_log_path)
                if len(log_text) < seen_log_size:
                    seen_log_size = 0
                new_log = log_text[seen_log_size:]
                seen_log_size = len(log_text)
                if "USER_ASR" in new_log:
                    print("[robot_docker_bridge] USER_ASR detected -> listen motion", flush=True)
                    bridge.listen_motion()

            turns = load_turns(dialogue_path)
            if not initialized:
                seen_count = len(turns)
                initialized = True
                print(f"[robot_docker_bridge] skipped {seen_count} existing replies; waiting for new replies", flush=True)
                time.sleep(args.poll_s)
                continue
            if len(turns) < seen_count:
                seen_count = 0
            for idx, turn in enumerate(turns[seen_count:], start=seen_count):
                reply = str(turn.get("reply") or "").strip()
                if not reply:
                    continue
                tts_text = extract_tts_text(reply)
                tag = "proactive" if turn.get("proactive") else "dialogue"
                print(f"[robot_docker_bridge] new {tag} reply #{idx + 1}: {reply}", flush=True)
                if tts_text != reply:
                    print(f"[robot_docker_bridge] TTS English: {tts_text}", flush=True)
                if not args.no_motion:
                    if tag == "proactive":
                        bridge.proactive_motion()
                    else:
                        bridge.reply_motion()
                if not args.no_tts:
                    bridge.speak(tts_text, language_code=args.language_code, disable_lipsync=args.disable_lipsync)
            seen_count = len(turns)
            time.sleep(args.poll_s)
    except KeyboardInterrupt:
        print("[robot_docker_bridge] stopped", flush=True)
    finally:
        if skeleton_watcher is not None:
            skeleton_watcher.stop()


if __name__ == "__main__":
    main()
