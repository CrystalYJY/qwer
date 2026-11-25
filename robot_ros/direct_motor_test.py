#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Safe, direct ROS motor smoke test for the physical Haru2 robot.

This publishes MotorCommand messages from the host ROS environment, using the
same /haru2/cmd_* topics as the working reference program.  It intentionally
does not send TTS or inspect dialogue files.
"""

from __future__ import annotations

import argparse
import time

import rclpy

from .robot_adapter import init_robot_adapter


def pause_with_callbacks(node, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=min(0.1, max(0.0, deadline - time.monotonic())))


def test_motor(node, name: str, amplitude: float, hold_s: float = 2.2) -> None:
    """Make one motor's positive, negative, and neutral positions obvious."""
    print(f"[motor_test] {name}: +{amplitude:.2f}", flush=True)
    node.send_motor(name, amplitude, play_time=250)
    pause_with_callbacks(node, hold_s)
    print(f"[motor_test] {name}: -{amplitude:.2f}", flush=True)
    node.send_motor(name, -amplitude, play_time=250)
    pause_with_callbacks(node, hold_s)
    print(f"[motor_test] {name}: center", flush=True)
    node.send_motor(name, 0.0, play_time=250)
    pause_with_callbacks(node, hold_s)


def main() -> None:
    parser = argparse.ArgumentParser(description="Send a short, safe direct-ROS Haru2 motor test (no TTS).")
    parser.add_argument("--robot-namespace", default="/haru2")
    args = parser.parse_args()

    node = init_robot_adapter(namespace=args.robot_namespace)
    try:
        print(f"[motor_test] direct ROS publisher ready: {args.robot_namespace}", flush=True)
        print("[motor_test] centering all tested motors...", flush=True)
        for motor_name in (
            "base",
            "neck_roll",
            "neck_pitch",
            "left_eye_yaw",
            "left_eye_roll",
            "right_eye_yaw",
            "right_eye_roll",
        ):
            node.send_motor(motor_name, 0.0, play_time=250)
        pause_with_callbacks(node, 1.5)

        test_motor(node, "base", 0.50, hold_s=2.5)
        test_motor(node, "neck_roll", 0.35)
        test_motor(node, "neck_pitch", 0.35)
        test_motor(node, "left_eye_yaw", 0.35)
        test_motor(node, "left_eye_roll", 0.35)
        test_motor(node, "right_eye_yaw", 0.35)
        test_motor(node, "right_eye_roll", 0.35)

        pause_with_callbacks(node, 1.5)
        print("[motor_test] completed", flush=True)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
