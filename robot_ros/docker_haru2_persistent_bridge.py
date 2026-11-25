#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import importlib
import json
import math
import re
import shlex
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .docker_haru2_bridge import (
    SkeletonTargetWatcher,
    clean_tts_label,
    compute_track_commands,
    extract_tts_text,
    load_text,
    load_turns,
)

EMOTION_STYLE_PROFILES = {
    "positive": {"routine_id": 2034, "routine_name": "harumoji_happy", "emotion": 0},
    "care": {"routine_id": 2084, "routine_name": "sympathetic", "emotion": 2},
    "sad": {"routine_id": 2065, "routine_name": "dejected", "emotion": 1},
    "angry": {"routine_id": 2021, "routine_name": "harumoji_angry", "emotion": 3},
}


CONTAINER_NODE_CODE = r'''
import json
import select
import sys
import time

import rclpy
from rclpy.node import Node


def load_haru_interfaces(package_name: str) -> None:
    """Load the ROS interface package available on the selected robot setup."""
    msg_module = importlib.import_module(f"{package_name}.msg")
    srv_module = importlib.import_module(f"{package_name}.srv")
    globals()["MotorCommand"] = msg_module.MotorCommand
    globals()["TTSAdvancedCommand"] = msg_module.TTSAdvancedCommand
    globals()["TTSCommand"] = msg_module.TTSCommand
    globals()["TTSConfig"] = msg_module.TTSConfig
    globals()["TTSSpurt"] = msg_module.TTSSpurt
    globals()["TTSStatus"] = msg_module.TTSStatus
    globals()["AdvancedRoutine"] = srv_module.AdvancedRoutine


class Haru2CommandNode(Node):
    def __init__(self, namespace="/haru2"):
        super().__init__("haru2_persistent_command_bridge")
        self.namespace = namespace.rstrip("/")
        self.motor_pubs = {
            "cmd_base_pos": self.create_publisher(MotorCommand, self.topic("cmd_base_pos"), 10),
            "cmd_neck_pitch_pos": self.create_publisher(MotorCommand, self.topic("cmd_neck_pitch_pos"), 10),
            "cmd_neck_roll_pos": self.create_publisher(MotorCommand, self.topic("cmd_neck_roll_pos"), 10),
            "cmd_left_eye_yaw_pos": self.create_publisher(MotorCommand, self.topic("cmd_left_eye_yaw_pos"), 10),
            "cmd_right_eye_yaw_pos": self.create_publisher(MotorCommand, self.topic("cmd_right_eye_yaw_pos"), 10),
            "cmd_left_eye_stroke_pos": self.create_publisher(MotorCommand, self.topic("cmd_left_eye_stroke_pos"), 10),
            "cmd_right_eye_stroke_pos": self.create_publisher(MotorCommand, self.topic("cmd_right_eye_stroke_pos"), 10),
        }
        self.tts_pub = self.create_publisher(TTSCommand, self.topic("cmd_tts"), 10)
        self.advanced_tts_pub = self.create_publisher(TTSAdvancedCommand, self.topic("cmd_tts_advanced"), 10)
        self.routine_client = self.create_client(AdvancedRoutine, self.topic("execute_advanced_routine"))
        self.tts_status_sub = self.create_subscription(
            TTSStatus,
            self.topic("tts_status"),
            self._on_tts_status,
            10,
        )
        self._last_tts_status = None
        self._tts_playing = False
        self._last_tts_status_emit = 0.0
        self.get_logger().info(f"Persistent Haru2 command bridge ready under {self.namespace}")

    def topic(self, name):
        return f"{self.namespace}/{name}"

    def publish_motor(self, topic, position, play_time=180):
        pub = self.motor_pubs.get(topic)
        if pub is None:
            self.get_logger().warn(f"Unknown motor topic: {topic}")
            return
        msg = MotorCommand()
        msg.position = float(position)
        msg.play_time = max(0, min(int(play_time), 250))
        msg.relative = False
        msg.disable_eyes_roll_sync = False
        pub.publish(msg)

    def _on_tts_status(self, msg):
        self._emit_tts_status(msg.status)

    def _emit_tts_status(self, status, force=False):
        now = time.time()
        if not force and status == self._last_tts_status:
            return
        self._last_tts_status = status
        self._last_tts_status_emit = now
        self._tts_playing = status == getattr(TTSStatus, "PLAYING", 2)
        status_names = {
            getattr(TTSStatus, "IDLE", -1): "IDLE",
            getattr(TTSStatus, "PLAYING", -1): "PLAYING",
            getattr(TTSStatus, "STOPPED", -1): "STOPPED",
            getattr(TTSStatus, "ERROR", -1): "ERROR",
        }
        payload = {
            "status": int(status),
            "status_name": status_names.get(status, str(status)),
            "playing": self._tts_playing,
            "wall_time": now,
        }
        print("__TTS_STATUS__ " + json.dumps(payload, ensure_ascii=False), flush=True)

    def emit_tts_heartbeat(self):
        if self._tts_playing and time.time() - self._last_tts_status_emit >= 0.3:
            self._emit_tts_status(self._last_tts_status, force=True)

    def speak(self, text, language_code="en", disable_lipsync=False):
        msg = TTSCommand()
        msg.message = str(text)
        msg.language_code = str(language_code or "")
        msg.disable_lipsync = bool(disable_lipsync)
        self.tts_pub.publish(msg)

    def speak_advanced(self, text, language_code="en", disable_lipsync=False, emotion=None):
        msg = TTSAdvancedCommand()
        config = TTSConfig()
        if emotion is not None:
            config.add_emotion = True
            config.emotion = int(emotion)
        msg.global_config = config
        msg.language_code = str(language_code or "")
        msg.disable_lipsync = bool(disable_lipsync)
        spurt = TTSSpurt()
        spurt.config = TTSConfig()
        spurt.text = str(text)
        msg.spurts = [spurt]
        self.advanced_tts_pub.publish(msg)

    def execute_routine(self, routine_id, disable_audio=True):
        if routine_id is None:
            return
        if not self.routine_client.wait_for_service(timeout_sec=0.2):
            self.get_logger().warn(f"Routine service not available: {self.topic('execute_advanced_routine')}")
            return
        request = AdvancedRoutine.Request()
        request.routine = int(routine_id)
        request.disable_audio = bool(disable_audio)
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
        self.routine_client.call_async(request)

    def speak_with_expression(self, text, language_code="en", disable_lipsync=True, emotion=None, routine_id=None):
        self.execute_routine(routine_id, disable_audio=True)
        time.sleep(0.15)
        self.speak_advanced(text, language_code=language_code, disable_lipsync=disable_lipsync, emotion=emotion)


def main():
    namespace = sys.argv[1] if len(sys.argv) > 1 else "/haru2"
    message_package = sys.argv[2] if len(sys.argv) > 2 else "haru2_core_msgs"
    load_haru_interfaces(message_package)
    rclpy.init()
    node = Haru2CommandNode(namespace=namespace)
    # Give DDS discovery a tiny moment once, not for every command.
    time.sleep(0.4)
    print("__HARU2_BRIDGE_READY__", flush=True)
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            node.emit_tts_heartbeat()
            ready, _, _ = select.select([sys.stdin], [], [], 0.001)
            if not ready:
                continue
            line = sys.stdin.readline()
            if line == "":
                break
            line = line.strip()
            if not line:
                continue
            try:
                cmd = json.loads(line)
                kind = cmd.get("kind")
                if kind == "motor":
                    node.publish_motor(cmd["topic"], cmd.get("position", 0.0), cmd.get("play_time", 180))
                elif kind == "tts":
                    node.speak(cmd.get("text", ""), cmd.get("language_code", "en"), cmd.get("disable_lipsync", False))
                elif kind == "expression_tts":
                    node.speak_with_expression(
                        cmd.get("text", ""),
                        cmd.get("language_code", "en"),
                        cmd.get("disable_lipsync", True),
                        cmd.get("emotion"),
                        cmd.get("routine_id"),
                    )
                elif kind == "batch":
                    for item in cmd.get("items", []):
                        if item.get("kind") == "motor":
                            node.publish_motor(item["topic"], item.get("position", 0.0), item.get("play_time", 180))
                        elif item.get("kind") == "tts":
                            node.speak(item.get("text", ""), item.get("language_code", "en"), item.get("disable_lipsync", False))
                elif kind == "shutdown":
                    break
            except Exception as exc:
                print(f"__ERR__ {exc}", flush=True)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
'''


class PersistentHaru2Bridge:
    def __init__(
        self,
        container: str,
        namespace: str = "/haru2",
        setup: str = "source /opt/ros/jazzy/setup.bash",
        ssh_target: str = "",
        message_package: str = "haru2_core_msgs",
        local: bool = False,
        dry_run: bool = False,
        tts_status_path: Optional[Path] = None,
    ) -> None:
        self.container = container
        self.namespace = namespace
        self.setup = setup
        self.ssh_target = str(ssh_target or "").strip()
        self.message_package = str(message_package or "haru2_core_msgs").strip() or "haru2_core_msgs"
        self.local = bool(local)
        self.dry_run = dry_run
        self.proc: Optional[subprocess.Popen] = None
        self.lock = threading.Lock()
        self.motion_until = 0.0
        self.tts_status_path = tts_status_path
        self._stdout_thread: Optional[threading.Thread] = None
        self._stdout_stop = threading.Event()

    def start(self) -> None:
        if self.dry_run:
            print("[dry-run] persistent bridge would start", flush=True)
            return
        command = (
            f"{self.setup}\n"
            f"python3 -u -c {shlex.quote(CONTAINER_NODE_CODE)} {shlex.quote(self.namespace)} {shlex.quote(self.message_package)}"
        )
        if self.ssh_target:
            cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", self.ssh_target, "bash", "-lc", command]
        elif self.local:
            cmd = ["bash", "-lc", command]
        else:
            cmd = ["docker", "exec", "-i", self.container, "bash", "-lc", command]
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert self.proc.stdout is not None
        deadline = time.time() + 8.0
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if line:
                print(f"[haru2_container] {line.rstrip()}", flush=True)
                if "__HARU2_BRIDGE_READY__" in line:
                    self._stdout_thread = threading.Thread(target=self._stdout_reader_loop, daemon=True)
                    self._stdout_thread.start()
                    return
            if self.proc.poll() is not None:
                raise RuntimeError("Persistent Haru2 bridge exited during startup.")
        raise TimeoutError("Persistent Haru2 bridge did not become ready.")

    def stop(self) -> None:
        if self.proc is None:
            return
        try:
            self._stdout_stop.set()
            self.send({"kind": "shutdown"})
        except Exception:
            pass
        try:
            self.proc.terminate()
        except Exception:
            pass

    def send(self, cmd: Dict) -> None:
        if self.dry_run:
            print("[dry-run]", json.dumps(cmd, ensure_ascii=False), flush=True)
            return
        if self.proc is None or self.proc.stdin is None:
            raise RuntimeError("Persistent bridge is not started.")
        with self.lock:
            self.proc.stdin.write(json.dumps(cmd, ensure_ascii=False) + "\n")
            self.proc.stdin.flush()

    def _stdout_reader_loop(self) -> None:
        if self.proc is None or self.proc.stdout is None:
            return
        for line in self.proc.stdout:
            if self._stdout_stop.is_set():
                break
            line = line.rstrip()
            if not line:
                continue
            if line.startswith("__TTS_STATUS__ "):
                self._handle_tts_status_line(line[len("__TTS_STATUS__ "):])
            else:
                print(f"[haru2_container] {line}", flush=True)

    def _handle_tts_status_line(self, payload_text: str) -> None:
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            print(f"[persistent_bridge] bad tts status: {payload_text}", flush=True)
            return
        if self.tts_status_path is None:
            return
        self.tts_status_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.tts_status_path.with_suffix(self.tts_status_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(self.tts_status_path)

    def motor(self, topic: str, position: float, play_time: int = 180) -> None:
        self.send({"kind": "motor", "topic": topic, "position": position, "play_time": play_time})

    def batch(self, items: List[Dict]) -> None:
        self.send({"kind": "batch", "items": items})

    def speak(self, text: str, language_code: str = "en", disable_lipsync: bool = False) -> None:
        self.send({
            "kind": "tts",
            "text": text,
            "language_code": language_code,
            "disable_lipsync": disable_lipsync,
        })

    def speak_with_expression(
        self,
        text: str,
        response_style: str,
        language_code: str = "en",
        disable_lipsync: bool = True,
        enable_routine: bool = True,
    ) -> bool:
        profile = EMOTION_STYLE_PROFILES.get(normalize_response_style(response_style))
        if not profile:
            return False
        self.send({
            "kind": "expression_tts",
            "text": text,
            "language_code": language_code,
            "disable_lipsync": disable_lipsync,
            "emotion": profile.get("emotion"),
            "routine_id": profile.get("routine_id") if enable_routine else None,
        })
        return True

    def look_at_target(self, target: Tuple[float, float, float], yaw_scale: float, pitch_scale: float) -> None:
        if time.time() < self.motion_until:
            return
        base_yaw, neck_pitch, eye_yaw = compute_track_commands(
            target,
            yaw_scale=yaw_scale,
            pitch_scale=pitch_scale,
        )
        self.batch([
            {"kind": "motor", "topic": "cmd_base_pos", "position": base_yaw, "play_time": 160},
            {"kind": "motor", "topic": "cmd_neck_pitch_pos", "position": neck_pitch, "play_time": 160},
            {"kind": "motor", "topic": "cmd_left_eye_yaw_pos", "position": eye_yaw, "play_time": 140},
            {"kind": "motor", "topic": "cmd_right_eye_yaw_pos", "position": eye_yaw, "play_time": 140},
        ])

    def _reserve_motion(self, duration_s: float) -> None:
        self.motion_until = max(self.motion_until, time.time() + max(0.0, duration_s))

    def listen_motion(self) -> None:
        self._reserve_motion(0.9)
        self.batch([
            {"kind": "motor", "topic": "cmd_neck_pitch_pos", "position": -0.12, "play_time": 130},
            {"kind": "motor", "topic": "cmd_neck_roll_pos", "position": 0.12, "play_time": 130},
            {"kind": "motor", "topic": "cmd_left_eye_yaw_pos", "position": 0.10, "play_time": 110},
            {"kind": "motor", "topic": "cmd_right_eye_yaw_pos", "position": 0.10, "play_time": 110},
        ])
        time.sleep(0.22)
        self.batch([
            {"kind": "motor", "topic": "cmd_neck_pitch_pos", "position": 0.07, "play_time": 120},
            {"kind": "motor", "topic": "cmd_neck_roll_pos", "position": 0.0, "play_time": 120},
            {"kind": "motor", "topic": "cmd_left_eye_stroke_pos", "position": 1.0, "play_time": 80},
            {"kind": "motor", "topic": "cmd_right_eye_stroke_pos", "position": 1.0, "play_time": 80},
        ])
        time.sleep(0.12)
        self.batch([
            {"kind": "motor", "topic": "cmd_neck_pitch_pos", "position": 0.0, "play_time": 120},
            {"kind": "motor", "topic": "cmd_left_eye_stroke_pos", "position": 0.0, "play_time": 80},
            {"kind": "motor", "topic": "cmd_right_eye_stroke_pos", "position": 0.0, "play_time": 80},
        ])

    def thinking_motion(self) -> None:
        self._reserve_motion(1.0)
        self.batch([
            {"kind": "motor", "topic": "cmd_neck_roll_pos", "position": -0.18, "play_time": 160},
            {"kind": "motor", "topic": "cmd_left_eye_yaw_pos", "position": -0.16, "play_time": 140},
            {"kind": "motor", "topic": "cmd_right_eye_yaw_pos", "position": -0.16, "play_time": 140},
        ])
        time.sleep(0.32)
        self.batch([
            {"kind": "motor", "topic": "cmd_neck_roll_pos", "position": 0.0, "play_time": 150},
            {"kind": "motor", "topic": "cmd_left_eye_yaw_pos", "position": 0.0, "play_time": 130},
            {"kind": "motor", "topic": "cmd_right_eye_yaw_pos", "position": 0.0, "play_time": 130},
        ])

    def reply_motion(self) -> None:
        self._reserve_motion(0.9)
        self.batch([
            {"kind": "motor", "topic": "cmd_neck_pitch_pos", "position": 0.24, "play_time": 130},
            {"kind": "motor", "topic": "cmd_left_eye_stroke_pos", "position": 1.0, "play_time": 90},
            {"kind": "motor", "topic": "cmd_right_eye_stroke_pos", "position": 1.0, "play_time": 90},
        ])
        time.sleep(0.12)
        self.batch([
            {"kind": "motor", "topic": "cmd_neck_pitch_pos", "position": 0.0, "play_time": 130},
            {"kind": "motor", "topic": "cmd_left_eye_stroke_pos", "position": 0.0, "play_time": 90},
            {"kind": "motor", "topic": "cmd_right_eye_stroke_pos", "position": 0.0, "play_time": 90},
        ])

    def proactive_motion(self) -> None:
        self._reserve_motion(1.15)
        self.batch([
            {"kind": "motor", "topic": "cmd_base_pos", "position": 0.28, "play_time": 180},
            {"kind": "motor", "topic": "cmd_neck_pitch_pos", "position": -0.14, "play_time": 160},
            {"kind": "motor", "topic": "cmd_left_eye_stroke_pos", "position": 1.0, "play_time": 90},
            {"kind": "motor", "topic": "cmd_right_eye_stroke_pos", "position": 1.0, "play_time": 90},
        ])
        time.sleep(0.30)
        self.batch([
            {"kind": "motor", "topic": "cmd_base_pos", "position": -0.10, "play_time": 160},
            {"kind": "motor", "topic": "cmd_neck_pitch_pos", "position": 0.0, "play_time": 150},
            {"kind": "motor", "topic": "cmd_left_eye_stroke_pos", "position": 0.0, "play_time": 90},
            {"kind": "motor", "topic": "cmd_right_eye_stroke_pos", "position": 0.0, "play_time": 90},
        ])
        time.sleep(0.18)
        self.batch([
            {"kind": "motor", "topic": "cmd_base_pos", "position": 0.0, "play_time": 160},
        ])

    def speaking_motion(self, text: str) -> None:
        words = max(3, len(str(text).split()))
        cycles = min(4, max(2, words // 7))
        self._reserve_motion(0.42 * cycles + 0.2)
        for idx in range(cycles):
            pitch = 0.08 if idx % 2 == 0 else -0.03
            eye = 0.06 if idx % 2 == 0 else -0.06
            self.batch([
                {"kind": "motor", "topic": "cmd_neck_pitch_pos", "position": pitch, "play_time": 150},
                {"kind": "motor", "topic": "cmd_left_eye_yaw_pos", "position": eye, "play_time": 140},
                {"kind": "motor", "topic": "cmd_right_eye_yaw_pos", "position": eye, "play_time": 140},
            ])
            time.sleep(0.32)
        self.batch([
            {"kind": "motor", "topic": "cmd_neck_pitch_pos", "position": 0.0, "play_time": 150},
            {"kind": "motor", "topic": "cmd_left_eye_yaw_pos", "position": 0.0, "play_time": 140},
            {"kind": "motor", "topic": "cmd_right_eye_yaw_pos", "position": 0.0, "play_time": 140},
        ])


def normalize_response_style(style: object) -> str:
    value = str(style or "neutral").strip().lower()
    if value not in {"neutral", "positive", "care", "sad", "angry"}:
        return "neutral"
    return value


def main() -> None:
    p = argparse.ArgumentParser(description="Low-latency persistent Docker bridge for Haru2 simulator.")
    p.add_argument("--container", default="haru-simulator-web-server-1")
    p.add_argument("--local", action="store_true", help="Run the ROS command node on this host instead of in Docker.")
    p.add_argument("--ssh-target", default="", help="Run the ROS command node on the robot through SSH.")
    p.add_argument("--message-package", default="haru2_core_msgs", choices=["haru2_core_msgs", "idmind_tabletop_msgs"])
    p.add_argument("--robot-namespace", default="/haru2")
    p.add_argument("--setup", default="source /opt/ros/jazzy/setup.bash")
    p.add_argument("--dialogue-json", default="streaming_camera_outputs/camera_asr_dialogue/dialogue_turns.json")
    p.add_argument("--dialogue-log", default="streaming_camera_outputs/camera_asr_dialogue/dialogue.log")
    p.add_argument("--tts-status-json", default="")
    p.add_argument("--poll-s", type=float, default=0.2)
    p.add_argument("--language-code", default="en")
    p.add_argument("--disable-lipsync", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--replay-existing", action="store_true")
    p.add_argument("--no-motion", action="store_true")
    p.add_argument("--no-listen-motion", action="store_true")
    p.add_argument("--no-speaking-motion", action="store_true")
    p.add_argument("--no-tts", action="store_true")
    p.add_argument("--no-emotion-expression", action="store_true")
    p.add_argument("--track-skeleton", action="store_true")
    p.add_argument("--skeleton-topic", default="/strawberry/azure/skeletons")
    p.add_argument("--track-interval-s", type=float, default=0.35)
    p.add_argument("--track-max-age-s", type=float, default=2.0)
    p.add_argument("--track-yaw-scale", type=float, default=4.0)
    p.add_argument("--track-pitch-scale", type=float, default=2.5)
    p.add_argument("--skeleton-motion-json", default="")
    args = p.parse_args()

    dialogue_path = Path(args.dialogue_json)
    tts_status_path = Path(args.tts_status_json) if args.tts_status_json else dialogue_path.parent / "tts_status.json"
    bridge = PersistentHaru2Bridge(
        args.container,
        args.robot_namespace,
        args.setup,
        ssh_target=args.ssh_target,
        message_package=args.message_package,
        local=args.local,
        dry_run=args.dry_run,
        tts_status_path=tts_status_path,
    )
    bridge.start()

    def run_dialogue_motion(tag: str, tts_text: str) -> None:
        if tag == "proactive":
            bridge.proactive_motion()
        else:
            bridge.thinking_motion()
            bridge.reply_motion()
        if not args.no_speaking_motion:
            time.sleep(0.15)
            bridge.speaking_motion(tts_text)

    skeleton_watcher = None
    if args.track_skeleton:
        skeleton_watcher = SkeletonTargetWatcher(args.skeleton_topic)
        skeleton_watcher.start()
        print(f"[persistent_bridge] skeleton tracking enabled: {args.skeleton_topic}", flush=True)

    dialogue_log_path = Path(args.dialogue_log)
    skeleton_motion_path = Path(args.skeleton_motion_json) if args.skeleton_motion_json else None
    seen_count = 0
    initialized = bool(args.replay_existing)
    seen_log_size = len(load_text(dialogue_log_path))
    next_track_wall = time.time() + 0.2
    print(f"[persistent_bridge] watching {dialogue_path}", flush=True)

    try:
        while True:
            now = time.time()
            if skeleton_watcher is not None and not args.no_motion and now >= next_track_wall:
                next_track_wall = now + max(0.1, args.track_interval_s)
                target = skeleton_watcher.get_latest(max_age_s=args.track_max_age_s)
                if target is not None:
                    bridge.look_at_target(target, args.track_yaw_scale, args.track_pitch_scale)
                if skeleton_motion_path is not None:
                    motion = skeleton_watcher.get_motion(max_age_s=args.track_max_age_s)
                    if motion is not None:
                        skeleton_motion_path.parent.mkdir(parents=True, exist_ok=True)
                        skeleton_motion_path.write_text(json.dumps(motion, ensure_ascii=False), encoding="utf-8")

            if not args.no_motion and not args.no_listen_motion:
                log_text = load_text(dialogue_log_path)
                if len(log_text) < seen_log_size:
                    seen_log_size = 0
                new_log = log_text[seen_log_size:]
                seen_log_size = len(log_text)
                if "ASR_SPEECH" in new_log or "USER_ASR" in new_log:
                    print("[persistent_bridge] user speech detected -> listen motion", flush=True)
                    bridge.listen_motion()

            turns = load_turns(dialogue_path)
            if not initialized:
                seen_count = len(turns)
                initialized = True
                print(f"[persistent_bridge] skipped {seen_count} existing replies; waiting for new replies", flush=True)
                time.sleep(args.poll_s)
                continue
            if len(turns) < seen_count:
                seen_count = 0
            for idx, turn in enumerate(turns[seen_count:], start=seen_count):
                reply = str(turn.get("reply") or "").strip()
                if not reply:
                    continue
                tag = "proactive" if turn.get("proactive") else "dialogue"
                response_style = normalize_response_style(turn.get("response_style") or "neutral")
                tts_text = extract_tts_text(reply)
                print(f"[persistent_bridge] new {tag} reply #{idx + 1} style={response_style}: {reply}", flush=True)
                use_expression = response_style != "neutral" and not args.no_emotion_expression
                if not args.no_motion and not use_expression:
                    threading.Thread(target=run_dialogue_motion, args=(tag, tts_text), daemon=True).start()
                if not args.no_tts:
                    if use_expression:
                        handled = bridge.speak_with_expression(
                            tts_text,
                            response_style,
                            args.language_code,
                            disable_lipsync=True,
                            enable_routine=not args.no_motion,
                        )
                        if not handled:
                            bridge.speak(tts_text, args.language_code, args.disable_lipsync)
                    else:
                        bridge.speak(tts_text, args.language_code, args.disable_lipsync)
            seen_count = len(turns)
            time.sleep(args.poll_s)
    except KeyboardInterrupt:
        print("[persistent_bridge] stopped", flush=True)
    finally:
        if skeleton_watcher is not None:
            skeleton_watcher.stop()
        bridge.stop()


if __name__ == "__main__":
    main()
