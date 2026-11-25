#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import time

import rclpy

from .robot_adapter import init_robot_adapter


def main() -> None:
    p = argparse.ArgumentParser(description="Send a small robot motion/TTS test.")
    p.add_argument("--text", default="你好，我准备好了。")
    p.add_argument("--language-code", default="")
    p.add_argument("--robot-namespace", default="/haru2")
    p.add_argument("--no-tts", action="store_true")
    p.add_argument("--no-motion", action="store_true")
    args = p.parse_args()

    node = init_robot_adapter(namespace=args.robot_namespace)
    try:
        if not args.no_motion:
            node.obvious_test_motion()
        if not args.no_tts:
            node.speak(args.text, language_code=args.language_code)
        end_t = time.time() + 3.0
        while time.time() < end_t:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
