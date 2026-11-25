#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import time
from typing import Dict

import rclpy
from rclpy.node import Node
from std_msgs.msg import Header

try:
    from haru2_core_msgs.msg import LedsArray, MotorCommand, TTSCommand, TTSAdvancedCommand, TTSConfig, TTSSpurt, TTSStatus
except ImportError:
    try:
        from idmind_tabletop_msgs.msg import LedsArray, MotorCommand, TTSCommand, TTSAdvancedCommand, TTSConfig, TTSSpurt, TTSStatus
    except ImportError as exc:  # pragma: no cover - only happens outside the robot ROS env.
        raise RuntimeError(
            "Robot ROS messages are not available. Source the ROS2 workspace before running this script."
        ) from exc


class RobotAdapter(Node):
    """Small ROS2 adapter for speech and simple social motions."""

    def __init__(self, node_name: str = "streaming_dialogue_robot_adapter", namespace: str = "/haru2") -> None:
        super().__init__(node_name)
        self.namespace = normalize_namespace(namespace)
        self.tts_pub = self.create_publisher(TTSCommand, self.topic("cmd_tts"), 10)
        self.tts_advanced_pub = self.create_publisher(TTSAdvancedCommand, self.topic("cmd_tts_advanced"), 10)
        self.base_led_pub = self.create_publisher(LedsArray, self.topic("cmd_base_leds"), 10)
        self.left_eye_led_pub = self.create_publisher(LedsArray, self.topic("cmd_left_eye_leds"), 10)
        self.right_eye_led_pub = self.create_publisher(LedsArray, self.topic("cmd_right_eye_leds"), 10)
        self.tts_status_sub = self.create_subscription(
            TTSStatus,
            self.topic("tts_status"),
            self._on_tts_status,
            10,
        )
        self.motor_pubs: Dict[str, object] = {
            "base": self.create_publisher(MotorCommand, self.topic("cmd_base_pos"), 10),
            "neck_pitch": self.create_publisher(MotorCommand, self.topic("cmd_neck_pitch_pos"), 10),
            "neck_roll": self.create_publisher(MotorCommand, self.topic("cmd_neck_roll_pos"), 10),
            "left_eye_yaw": self.create_publisher(MotorCommand, self.topic("cmd_left_eye_yaw_pos"), 10),
            "right_eye_yaw": self.create_publisher(MotorCommand, self.topic("cmd_right_eye_yaw_pos"), 10),
            "left_eye_roll": self.create_publisher(MotorCommand, self.topic("cmd_left_eye_roll_pos"), 10),
            "right_eye_roll": self.create_publisher(MotorCommand, self.topic("cmd_right_eye_roll_pos"), 10),
            "left_eye_stroke": self.create_publisher(MotorCommand, self.topic("cmd_left_eye_stroke_pos"), 10),
            "right_eye_stroke": self.create_publisher(MotorCommand, self.topic("cmd_right_eye_stroke_pos"), 10),
        }
        self.tts_playing = False
        self._last_tts_status = None
        self.tts_status = None
        self.get_logger().info(f"Robot namespace: {self.namespace}")

    def topic(self, name: str) -> str:
        return f"{self.namespace}/{name}"

    def _on_tts_status(self, msg: TTSStatus) -> None:
        if msg.status == self._last_tts_status:
            return
        self._last_tts_status = msg.status
        self.tts_status = int(msg.status)
        self.tts_playing = msg.status == getattr(TTSStatus, "PLAYING", 2)
        self.get_logger().info(f"TTS status: {msg.status}")

    def speak(self, text: str, language_code: str = "", disable_lipsync: bool = False, response_style: str = "neutral") -> None:
        """Use the robot's advanced TTS emotion tags when a dialogue style warrants it."""
        style = str(response_style or "neutral").strip().lower()
        emotion_map = {
            "positive": ("EMOTION_HAPPY", "PITCH_HIGH", "RATE_MEDIUM", "VOLUME_MEDIUM", "EMPHASIS_MODERATED"),
            "care": ("EMOTION_CALM", "PITCH_SLOW", "RATE_SLOW", "VOLUME_SOFT", "EMPHASIS_NONE"),
            "sad": ("EMOTION_SAD", "PITCH_SLOW", "RATE_SLOW", "VOLUME_SOFT", "EMPHASIS_REDUCED"),
            "angry": ("EMOTION_ANGRY", "PITCH_MEDIUM", "RATE_MEDIUM", "VOLUME_LOUD", "EMPHASIS_STRONG"),
        }
        if style in emotion_map:
            emotion, pitch, rate, volume, emphasis = emotion_map[style]
            msg = TTSAdvancedCommand()
            msg.header = Header()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.language_code = language_code
            msg.disable_lipsync = disable_lipsync
            config = TTSConfig()
            config.add_emotion = True
            config.emotion = getattr(TTSConfig, emotion)
            config.add_pitch = True; config.pitch_type = getattr(TTSConfig, pitch)
            config.add_rate = True; config.rate_type = getattr(TTSConfig, rate)
            config.add_volume = True; config.volume_type = getattr(TTSConfig, volume)
            config.add_emphasis = True; config.emphasis_type = getattr(TTSConfig, emphasis)
            msg.global_config = config
            spurt = TTSSpurt(); spurt.config = TTSConfig(); spurt.text = text
            msg.spurts = [spurt]
            self.tts_advanced_pub.publish(msg)
            self.get_logger().info(f"Advanced TTS sent ({style}): {text}")
            return
        msg = TTSCommand()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.message = text
        msg.language_code = language_code
        msg.disable_lipsync = disable_lipsync
        self.tts_pub.publish(msg)
        self.get_logger().info(f"TTS sent: {text}")

    def send_motor(
        self,
        motor_name: str,
        position: float,
        play_time: int = 300,
        relative: bool = False,
        disable_eyes_roll_sync: bool = False,
    ) -> None:
        pub = self.motor_pubs.get(motor_name)
        if pub is None:
            self.get_logger().error(f"Unknown motor: {motor_name}")
            return
        msg = MotorCommand()
        msg.position = float(position)
        msg.play_time = max(0, min(int(play_time), 250))
        msg.relative = bool(relative)
        msg.disable_eyes_roll_sync = bool(disable_eyes_roll_sync)
        pub.publish(msg)

    def set_base_led_color(self, red: int, green: int, blue: int) -> None:
        """Set the physical base LED ring to one solid RGB color."""
        msg = LedsArray()
        msg.force = True
        msg.r = [max(0, min(255, int(red)))]
        msg.g = [max(0, min(255, int(green)))]
        msg.b = [max(0, min(255, int(blue)))]
        self.base_led_pub.publish(msg)

    @staticmethod
    def _solid_led_message(red: int, green: int, blue: int, count: int) -> LedsArray:
        msg = LedsArray()
        msg.force = True
        color = (max(0, min(255, int(red))), max(0, min(255, int(green))), max(0, min(255, int(blue))))
        msg.r = [color[0]] * count
        msg.g = [color[1]] * count
        msg.b = [color[2]] * count
        return msg

    def set_eye_led_color(self, red: int, green: int, blue: int) -> None:
        """Set both 64-pixel eye-frame LED arrays to one solid RGB color."""
        self.left_eye_led_pub.publish(self._solid_led_message(red, green, blue, count=64))
        self.right_eye_led_pub.publish(self._solid_led_message(red, green, blue, count=64))

    def look_forward(self) -> None:
        self.send_motor("base", 0.0, play_time=200)
        self.send_motor("neck_pitch", 0.0, play_time=200)
        self.send_motor("neck_roll", 0.0, play_time=200)
        self.send_motor("left_eye_yaw", 0.0, play_time=250)
        self.send_motor("right_eye_yaw", 0.0, play_time=250)

    def blink(self) -> None:
        self.send_motor("left_eye_stroke", 100.0, play_time=120)
        self.send_motor("right_eye_stroke", 100.0, play_time=120)
        time.sleep(0.12)
        self.send_motor("left_eye_stroke", 0.0, play_time=120)
        self.send_motor("right_eye_stroke", 0.0, play_time=120)

    def small_nod(self) -> None:
        self.send_motor("neck_pitch", 0.35, play_time=200)
        time.sleep(0.4)
        self.send_motor("neck_pitch", -0.25, play_time=200)
        time.sleep(0.35)
        self.send_motor("neck_pitch", 0.0, play_time=200)

    def thinking_motion(self) -> None:
        self.send_motor("neck_roll", 0.25, play_time=200)
        self.send_motor("left_eye_yaw", 0.25, play_time=200)
        self.send_motor("right_eye_yaw", 0.25, play_time=200)
        time.sleep(0.35)
        self.look_forward()

    def acknowledge(self) -> None:
        self.look_forward()
        time.sleep(0.1)
        self.blink()
        self.small_nod()

    def obvious_test_motion(self) -> None:
        self.look_forward()
        time.sleep(0.6)

        self.send_motor("base", 0.5, play_time=200)
        self.send_motor("left_eye_yaw", 0.35, play_time=200)
        self.send_motor("right_eye_yaw", 0.35, play_time=200)
        time.sleep(1.2)

        self.send_motor("base", -0.5, play_time=200)
        self.send_motor("left_eye_yaw", -0.35, play_time=200)
        self.send_motor("right_eye_yaw", -0.35, play_time=200)
        time.sleep(1.2)

        self.send_motor("base", 0.0, play_time=200)
        self.send_motor("left_eye_yaw", 0.0, play_time=200)
        self.send_motor("right_eye_yaw", 0.0, play_time=200)
        time.sleep(0.5)

        self.send_motor("neck_pitch", 0.35, play_time=200)
        time.sleep(0.8)
        self.send_motor("neck_pitch", -0.25, play_time=200)
        time.sleep(0.8)
        self.send_motor("neck_pitch", 0.0, play_time=200)
        time.sleep(0.4)

        self.send_motor("neck_roll", 0.30, play_time=200)
        time.sleep(0.8)
        self.send_motor("neck_roll", -0.30, play_time=200)
        time.sleep(0.8)
        self.send_motor("neck_roll", 0.0, play_time=200)
        time.sleep(0.4)

        for _ in range(2):
            self.blink()
            time.sleep(0.25)


def normalize_namespace(namespace: str) -> str:
    namespace = (namespace or "/haru2").strip()
    if not namespace.startswith("/"):
        namespace = "/" + namespace
    return namespace.rstrip("/")


def init_robot_adapter(namespace: str = "/haru2") -> RobotAdapter:
    if not rclpy.ok():
        rclpy.init()
    node = RobotAdapter(namespace=namespace)
    # Give ROS discovery a short moment before the first publish.
    time.sleep(0.5)
    return node
