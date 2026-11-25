#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import math
import time
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node

try:
    from haru2_core_msgs.msg import MotorCommand
except ImportError:
    from idmind_tabletop_msgs.msg import MotorCommand

from strawberry_ros_msgs.msg import Skeletons


class Haru2HeadTracker(Node):
    """Continuously look at the detected user's head.

    This node must run in a ROS2 environment that can see both:
    - skeleton topic, e.g. /strawberry/azure/skeletons
    - Haru2 motor topics, e.g. /haru2/cmd_base_pos
    """

    def __init__(
        self,
        skeleton_topic: str,
        robot_namespace: str,
        publish_hz: float,
        lost_timeout_s: float,
        smoothing: float,
        deadzone: float,
        yaw_scale: float,
        pitch_scale: float,
        base_gain: float,
        eye_gain: float,
        verbose: bool,
    ) -> None:
        super().__init__("haru2_head_tracker")
        self.skeleton_topic = skeleton_topic
        self.robot_namespace = normalize_namespace(robot_namespace)
        self.publish_period = 1.0 / max(0.5, float(publish_hz))
        self.lost_timeout_s = float(lost_timeout_s)
        self.smoothing = max(0.0, min(float(smoothing), 0.95))
        self.deadzone = max(0.0, float(deadzone))
        self.yaw_scale = float(yaw_scale)
        self.pitch_scale = float(pitch_scale)
        self.base_gain = float(base_gain)
        self.eye_gain = float(eye_gain)
        self.verbose = verbose

        self.base_pub = self.create_publisher(MotorCommand, self.topic("cmd_base_pos"), 10)
        self.neck_pitch_pub = self.create_publisher(MotorCommand, self.topic("cmd_neck_pitch_pos"), 10)
        self.left_eye_yaw_pub = self.create_publisher(MotorCommand, self.topic("cmd_left_eye_yaw_pos"), 10)
        self.right_eye_yaw_pub = self.create_publisher(MotorCommand, self.topic("cmd_right_eye_yaw_pos"), 10)
        self.neck_roll_pub = self.create_publisher(MotorCommand, self.topic("cmd_neck_roll_pos"), 10)

        self.sub = self.create_subscription(Skeletons, self.skeleton_topic, self.on_skeletons, 10)
        self.timer = self.create_timer(self.publish_period, self.on_timer)

        self.last_target: Optional[Tuple[float, float, float]] = None
        self.last_seen_wall = 0.0
        self.last_cmd = {
            "base": 0.0,
            "neck_pitch": 0.0,
            "eye_yaw": 0.0,
            "neck_roll": 0.0,
        }
        self.last_publish_wall = 0.0

        self.get_logger().info(f"Tracking skeleton topic: {self.skeleton_topic}")
        self.get_logger().info(f"Publishing Haru2 motor topics under: {self.robot_namespace}")

    def topic(self, name: str) -> str:
        return f"{self.robot_namespace}/{name}"

    def on_skeletons(self, msg: Skeletons) -> None:
        if not msg.skeletons:
            return
        skeleton = msg.skeletons[0]
        head = None
        nose = None
        for part in skeleton.body_parts:
            name = str(part.name or "").lower()
            if name == "head":
                head = part
            elif name == "nose":
                nose = part
        target = nose or head
        if target is None:
            return
        pos = target.pose.position
        self.last_target = (float(pos.x), float(pos.y), float(pos.z))
        self.last_seen_wall = time.time()

    def on_timer(self) -> None:
        now = time.time()
        if self.last_target is None or now - self.last_seen_wall > self.lost_timeout_s:
            self.publish_smoothed(0.0, 0.0, 0.0, reason="lost")
            return

        yaw, pitch = compute_human_gaze_commands(self.last_target)
        yaw *= self.yaw_scale
        pitch *= self.pitch_scale

        # Haru2 simulator movements are subtle for small eye-only commands, so let
        # the base participate even for moderate horizontal offsets.
        base_yaw = clamp(yaw * self.base_gain, -0.55, 0.55)
        eye_yaw = clamp(yaw * self.eye_gain, -0.35, 0.35)
        neck_pitch = clamp(pitch, -0.45, 0.45)
        self.publish_smoothed(base_yaw, neck_pitch, eye_yaw, reason="track")

    def publish_smoothed(self, base_yaw: float, neck_pitch: float, eye_yaw: float, reason: str) -> None:
        alpha = 1.0 - self.smoothing
        cmd = {
            "base": smooth(self.last_cmd["base"], base_yaw, alpha),
            "neck_pitch": smooth(self.last_cmd["neck_pitch"], neck_pitch, alpha),
            "eye_yaw": smooth(self.last_cmd["eye_yaw"], eye_yaw, alpha),
            "neck_roll": 0.0,
        }

        if all(abs(cmd[k] - self.last_cmd[k]) < self.deadzone for k in cmd):
            return

        self.publish_motor(self.base_pub, cmd["base"], play_time=180)
        self.publish_motor(self.neck_pitch_pub, cmd["neck_pitch"], play_time=180)
        self.publish_motor(self.left_eye_yaw_pub, cmd["eye_yaw"], play_time=160)
        self.publish_motor(self.right_eye_yaw_pub, cmd["eye_yaw"], play_time=160)
        self.publish_motor(self.neck_roll_pub, cmd["neck_roll"], play_time=180)
        self.last_cmd = cmd

        if self.verbose:
            self.get_logger().info(
                f"{reason}: base={cmd['base']:+.3f} pitch={cmd['neck_pitch']:+.3f} eye={cmd['eye_yaw']:+.3f}"
            )

    @staticmethod
    def publish_motor(pub, position: float, play_time: int = 180) -> None:
        msg = MotorCommand()
        msg.position = float(position)
        msg.play_time = max(0, min(int(play_time), 250))
        msg.relative = False
        msg.disable_eyes_roll_sync = False
        pub.publish(msg)


def compute_human_gaze_commands(point_3d: Tuple[float, float, float]) -> Tuple[float, float]:
    x, y, z = point_3d
    # Same convention as the user's LookController human mode.
    yaw = math.atan2(-y, x)
    pitch = -math.atan2(z, math.sqrt(y * y + x * x))
    return yaw, pitch


def smooth(prev: float, target: float, alpha: float) -> float:
    return prev * (1.0 - alpha) + target * alpha


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalize_namespace(namespace: str) -> str:
    namespace = (namespace or "/haru2").strip()
    if not namespace.startswith("/"):
        namespace = "/" + namespace
    return namespace.rstrip("/")


def main() -> None:
    p = argparse.ArgumentParser(description="Continuously make Haru2 look at the detected user's head.")
    p.add_argument("--skeleton-topic", default="/strawberry/azure/skeletons")
    p.add_argument("--robot-namespace", default="/haru2")
    p.add_argument("--publish-hz", type=float, default=4.0)
    p.add_argument("--lost-timeout-s", type=float, default=1.5)
    p.add_argument("--smoothing", type=float, default=0.65)
    p.add_argument("--deadzone", type=float, default=0.015)
    p.add_argument("--yaw-scale", type=float, default=3.0)
    p.add_argument("--pitch-scale", type=float, default=2.0)
    p.add_argument("--base-gain", type=float, default=0.8)
    p.add_argument("--eye-gain", type=float, default=0.35)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    rclpy.init()
    node = Haru2HeadTracker(
        skeleton_topic=args.skeleton_topic,
        robot_namespace=args.robot_namespace,
        publish_hz=args.publish_hz,
        lost_timeout_s=args.lost_timeout_s,
        smoothing=args.smoothing,
        deadzone=args.deadzone,
        yaw_scale=args.yaw_scale,
        pitch_scale=args.pitch_scale,
        base_gain=args.base_gain,
        eye_gain=args.eye_gain,
        verbose=args.verbose,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
