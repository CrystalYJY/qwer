#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Direct physical Haru2 test: eye-roll focus pulse plus matching warm-amber LEDs.

No camera, VLM, ASR, dialogue file, or skeleton tracking is used.  This makes
it safe to verify whether the two eye-roll motors and base LED topic are wired
correctly on the physical robot.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import rclpy

try:
    from .robot_adapter import init_robot_adapter
except ImportError:
    # Support direct execution: python3 robot_ros/eye_roll_led_test.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from robot_ros.robot_adapter import init_robot_adapter


def spin_for(node, seconds: float) -> None:
    deadline = time.monotonic() + max(0.0, seconds)
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=min(0.1, deadline - time.monotonic()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Haru2 eye roll + base/eye LEDs without tracking.")
    parser.add_argument("--robot-namespace", default="/haru2")
    parser.add_argument("--left-roll", type=float, default=-1.0, help="Physical left eye: 0 -> -1 -> 0.")
    parser.add_argument("--right-roll", type=float, default=1.0, help="Physical right eye: 0 -> +1 -> 0.")
    parser.add_argument("--play-time", type=int, default=25, help="Haru driver units; 25 is about 0.225 seconds.")
    parser.add_argument("--hold-s", type=float, default=0.80)
    parser.add_argument("--gap-s", type=float, default=1.20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--base-rgb", default="150,90,15", help="Visible base RGB.")
    parser.add_argument("--eye-rgb", default="250,150,25",
                        help="Eye-strip input RGB, compensated for its 0.6 driver attenuation.")
    parser.add_argument("--calibrate-eye", action="store_true",
                        help="Cycle four compensated base/eye colour pairs for physical calibration.")
    parser.add_argument("--calibration-hold-s", type=float, default=3.0)
    args = parser.parse_args()

    try:
        base_rgb = tuple(int(value.strip()) for value in args.base_rgb.split(","))
        eye_rgb = tuple(int(value.strip()) for value in args.eye_rgb.split(","))
        if (len(base_rgb) != 3 or len(eye_rgb) != 3
                or any(value < 0 or value > 255 for value in (*base_rgb, *eye_rgb))):
            raise ValueError
    except ValueError as exc:
        raise SystemExit("--base-rgb and --eye-rgb must be RGB, e.g. 150,90,15") from exc

    node = init_robot_adapter(namespace=args.robot_namespace)
    try:
        print(
            f"[eye_roll_led_test] ready: namespace={args.robot_namespace}, "
            f"base_input={base_rgb}, eye_input={eye_rgb}",
            flush=True,
        )
        node.send_motor("left_eye_roll", 0.0, play_time=args.play_time, disable_eyes_roll_sync=True)
        node.send_motor("right_eye_roll", 0.0, play_time=args.play_time, disable_eyes_roll_sync=True)
        node.set_base_led_color(0, 0, 0)
        node.set_eye_led_color(0, 0, 0)
        spin_for(node, 0.8)

        if args.calibrate_eye:
            # Each pair describes the intended visible base colour first and
            # the eye input second. Eye input is approximately visible / 0.6.
            # These colours deliberately avoid blue, which the eye strip
            # renders much more strongly than the base.
            candidates = (
                ("1: warm amber", (150, 90, 15), (250, 150, 25)),
                ("2: yellow-green", (120, 145, 0), (200, 242, 0)),
                ("3: coral orange", (150, 50, 25), (250, 83, 42)),
                ("4: soft purple", (100, 45, 105), (167, 75, 175)),
            )
            for name, candidate_base, candidate_eye in candidates:
                print(
                    f"[eye_roll_led_test] calibration {name}: "
                    f"base={candidate_base}, eye_input={candidate_eye}",
                    flush=True,
                )
                node.set_base_led_color(*candidate_base)
                node.set_eye_led_color(*candidate_eye)
                spin_for(node, args.calibration_hold_s)
            node.set_base_led_color(0, 0, 0)
            node.set_eye_led_color(0, 0, 0)
            print("[eye_roll_led_test] calibration completed", flush=True)
            return

        for index in range(max(1, args.repeats)):
            print(f"[eye_roll_led_test] pulse {index + 1}/{max(1, args.repeats)}: base/eye LEDs + eye roll", flush=True)
            node.set_base_led_color(*base_rgb)
            node.set_eye_led_color(*eye_rgb)
            node.send_motor("left_eye_roll", args.left_roll, play_time=args.play_time, disable_eyes_roll_sync=True)
            node.send_motor("right_eye_roll", args.right_roll, play_time=args.play_time, disable_eyes_roll_sync=True)
            spin_for(node, args.hold_s)
            node.send_motor("left_eye_roll", 0.0, play_time=args.play_time, disable_eyes_roll_sync=True)
            node.send_motor("right_eye_roll", 0.0, play_time=args.play_time, disable_eyes_roll_sync=True)
            node.set_base_led_color(0, 0, 0)
            node.set_eye_led_color(0, 0, 0)
            spin_for(node, args.gap_s)

        print("[eye_roll_led_test] completed", flush=True)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
