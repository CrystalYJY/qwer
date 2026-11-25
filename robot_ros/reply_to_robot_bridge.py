#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from pathlib import Path
from typing import Dict, List

import rclpy

from .robot_adapter import init_robot_adapter


class DirectSkeletonTracker:
    """Skeleton subscription attached to the same node/executor as robot output."""

    def __init__(self, node, topic: str, smoothing_alpha: float, deadband_m: float, max_step_m: float) -> None:
        from strawberry_ros_msgs.msg import Skeletons

        self.latest_target = None
        self.latest_wall = 0.0
        self.latest_motion = {}
        self._previous_parts = {}
        self._previous_wall = 0.0
        self.callback_count = 0
        self.valid_target_count = 0
        self.last_missing_reason = "waiting_for_first_skeleton"
        self.last_part_names = []
        self.filtered_target = None
        self.smoothing_alpha = max(0.01, min(float(smoothing_alpha), 1.0))
        self.deadband_m = max(0.0, float(deadband_m))
        self.max_step_m = max(0.001, float(max_step_m))
        self.subscription = node.create_subscription(Skeletons, topic, self._on_skeletons, 10)
        node.get_logger().info(f"Direct skeleton tracking subscription: {topic}")

    def _on_skeletons(self, msg) -> None:
        self.callback_count += 1
        if not msg.skeletons:
            self.last_missing_reason = "empty_skeletons"
            return
        parts = {str(part.name or "").lower(): part for part in msg.skeletons[0].body_parts}
        self.last_part_names = sorted(parts)
        # Match the calibrated legacy head-tracking controller: use the Azure
        # `head` joint first.  Nose jitters more when the face rotates; the
        # remaining upper-body landmarks are only fallbacks during short gaps.
        target = parts.get("head") or parts.get("nose") or parts.get("neck") or parts.get("spine_shoulder")
        if target is None:
            self.last_missing_reason = "no_head_landmark"
            return
        now = time.time()
        positions = {
            name: (float(part.pose.position.x), float(part.pose.position.y), float(part.pose.position.z))
            for name, part in parts.items()
        }
        dt = max(now - self._previous_wall, 1e-3)
        speeds = {}
        motion_joint_keys = ("head", "nose", "hand", "wrist", "elbow", "spine", "pelvis", "hip", "knee", "ankle", "foot")
        for name, point in positions.items():
            previous = self._previous_parts.get(name)
            if previous is None or not any(key in name for key in motion_joint_keys):
                continue
            speeds[name] = round(math.dist(point, previous) / dt, 4)
        point = target.pose.position
        raw_target = (float(point.x), float(point.y), float(point.z))
        if self.filtered_target is None:
            self.filtered_target = raw_target
        else:
            delta = tuple(raw - old for raw, old in zip(raw_target, self.filtered_target))
            distance = math.sqrt(sum(value * value for value in delta))
            if distance > self.deadband_m:
                # Reject one-frame jumps, then apply an exponential low-pass filter.
                scale = min(1.0, self.max_step_m / distance)
                bounded = tuple(old + value * scale for old, value in zip(self.filtered_target, delta))
                self.filtered_target = tuple(
                    old + self.smoothing_alpha * (new - old)
                    for old, new in zip(self.filtered_target, bounded)
                )
        self.latest_target = self.filtered_target
        self.latest_wall = now
        self.valid_target_count += 1
        self.last_missing_reason = ""
        self.latest_motion = {
            "wall_time": now,
            "motion_score": max(speeds.values()) if speeds else 0.0,
            "part_speeds": speeds,
            "part_positions": {
                name: {"x": point[0], "y": point[1], "z": point[2]}
                for name, point in positions.items()
                if any(key in name for key in motion_joint_keys)
            },
        }
        self._previous_parts = positions
        self._previous_wall = now

    def get_latest(self, max_age_s: float):
        if self.latest_target is None or time.time() - self.latest_wall > max_age_s:
            return None
        return self.latest_target

    def get_motion(self, max_age_s: float):
        if not self.latest_motion or time.time() - self.latest_wall > max_age_s:
            return None
        return dict(self.latest_motion)


def compute_track_commands(
    point_3d,
    yaw_scale: float,
    pitch_scale: float,
    eye_only_yaw_threshold: float = 0.5,
    base_yaw_limit: float = 0.55,
    eye_yaw_limit: float = 0.5,
    neck_pitch_limit: float = 0.5,
):
    """Legacy-compatible human tracking: eyes for small yaw, base for large yaw."""
    x, y, z = point_3d
    yaw = math.atan2(-y, x) * yaw_scale
    pitch = -math.atan2(z, math.sqrt(y * y + x * x)) * pitch_scale
    # This is the allocation used by qwe9128's calibrated LookController.
    # Never compound a small eye correction with a base turn.
    if abs(yaw) <= eye_only_yaw_threshold:
        base_yaw = 0.0
        eye_yaw = max(-eye_yaw_limit, min(eye_yaw_limit, yaw))
    else:
        base_yaw = max(-base_yaw_limit, min(base_yaw_limit, yaw))
        eye_yaw = 0.0
    neck_pitch = max(-neck_pitch_limit, min(neck_pitch_limit, pitch))
    return base_yaw, neck_pitch, eye_yaw


def load_turns(path: Path) -> List[Dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except json.JSONDecodeError:
        # The dialogue process may be rewriting the JSON file.
        return []


def load_recent_focus_trigger(path: Path, max_age_s: float) -> Dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if time.time() - float(payload.get("wall_time") or 0.0) <= max_age_s:
            return payload
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
        pass
    return {}


def parse_rgb(value: str) -> tuple[int, int, int]:
    try:
        channels = [max(0, min(255, int(part.strip()))) for part in str(value).split(",")]
        if len(channels) == 3:
            return tuple(channels)
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("RGB color must be three comma-separated values, e.g. 0,120,255")


def write_tts_status(path: Path, node) -> None:
    """Heartbeat for the Conda dialogue process' ASR suppression logic."""
    status = getattr(node, "tts_status", None)
    playing = bool(getattr(node, "tts_playing", False))
    payload = {
        "wall_time": time.time(),
        "status": status,
        "status_name": "PLAYING" if playing else ("STOPPED" if status is not None else "UNKNOWN"),
        "playing": playing,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Watch dialogue_turns.json and send robot TTS/motion for each new robot reply."
    )
    p.add_argument("--dialogue-json", default="streaming_camera_outputs/camera_asr_dialogue/dialogue_turns.json")
    p.add_argument("--poll-s", type=float, default=0.3)
    p.add_argument("--spin-period-s", type=float, default=0.02)
    p.add_argument("--language-code", default="")
    p.add_argument("--robot-namespace", default="/haru2")
    p.add_argument("--tts-status-json", default="")
    p.add_argument("--track-skeleton", action="store_true")
    p.add_argument("--skeleton-topic", default="/strawberry/azure/skeletons")
    p.add_argument("--track-interval-s", type=float, default=0.10)
    p.add_argument("--track-max-age-s", type=float, default=5.0,
                   help="Keep the last valid gaze target briefly through detector dropouts; never send a reset command.")
    p.add_argument("--track-yaw-scale", type=float, default=1.0)
    p.add_argument("--track-pitch-scale", type=float, default=1.0)
    p.add_argument("--track-eye-only-yaw-threshold", type=float, default=0.5)
    p.add_argument("--track-base-yaw-limit", type=float, default=0.55)
    p.add_argument("--track-eye-yaw-limit", type=float, default=0.5)
    p.add_argument("--track-neck-pitch-limit", type=float, default=0.5)
    p.add_argument("--track-smoothing-alpha", type=float, default=0.32)
    p.add_argument("--track-deadband-m", type=float, default=0.025)
    p.add_argument("--track-max-step-m", type=float, default=0.30)
    p.add_argument("--focus-motion-threshold", type=float, default=5.0,
                   help="Skeleton speed (m/s) that triggers one small eye-roll focus gesture.")
    p.add_argument("--focus-cooldown-s", type=float, default=15.0)
    p.add_argument("--focus-left-eye-roll", type=float, default=-1.0,
                   help="Focus target for the left eye roll (physical Haru2 direction is negative).")
    p.add_argument("--focus-right-eye-roll", type=float, default=1.0,
                   help="Focus target for the right eye roll (physical Haru2 direction is positive).")
    # Eye LED channels are attenuated by 0.6 in the installed Haru2 driver.
    # These values therefore have the same visible output: warm amber
    # base (150, 90, 15), eye input (250, 150, 25).
    p.add_argument("--focus-base-led-color", type=parse_rgb, default=(150, 90, 15),
                   help="Visible base RGB for a visually detected significant action.")
    p.add_argument("--focus-eye-led-color", type=parse_rgb, default=(250, 150, 25),
                   help="Eye-strip input RGB, compensated for the driver attenuation.")
    p.add_argument("--focus-base-led-reset-color", type=parse_rgb, default=(0, 0, 0))
    p.add_argument("--visual-focus-json", default="")
    p.add_argument("--skeleton-motion-json", default="")
    p.add_argument("--timeline-jsonl", default="")
    p.add_argument("--reply-motion", action="store_true",
                   help="Opt in to the acknowledgement gesture. Disabled by default because it conflicts with continuous tracking.")
    p.add_argument("--disable-lipsync", action="store_true")
    p.add_argument("--no-tts", action="store_true")
    p.add_argument("--no-motion", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    path = Path(args.dialogue_json)
    tts_status_path = Path(args.tts_status_json) if args.tts_status_json else path.parent / "tts_status.json"
    node = None if args.dry_run else init_robot_adapter(namespace=args.robot_namespace)
    seen_count = 0
    skeleton_tracker = None
    next_track_wall = time.monotonic()
    last_track_report = 0.0
    skeleton_motion_path = Path(args.skeleton_motion_json) if args.skeleton_motion_json else None
    timeline_path = Path(args.timeline_jsonl) if args.timeline_jsonl else path.parent / "robot_timeline.jsonl"
    next_turn_poll = time.monotonic()
    next_status_write = time.monotonic()
    last_motion_write = 0.0
    tracking_was_stale = False
    last_missing_reason = None
    last_focus_wall = -1e9
    focus_reset_wall = None
    visual_focus_path = Path(args.visual_focus_json) if args.visual_focus_json else path.parent / "visual_focus_trigger.json"
    last_visual_focus_wall = 0.0

    def timeline(event: str, **data) -> None:
        timeline_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"wall_time": time.time(), "monotonic": time.monotonic(), "source": "robot_bridge", "event": event, **data}
        with timeline_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    if node is not None and args.track_skeleton:
        skeleton_tracker = DirectSkeletonTracker(node, args.skeleton_topic, args.track_smoothing_alpha, args.track_deadband_m, args.track_max_step_m)
        print(f"[robot_bridge] direct skeleton tracking: {args.skeleton_topic}", flush=True)

    print(f"[robot_bridge] watching {path}", flush=True)
    if args.dry_run:
        print("[robot_bridge] dry-run enabled; no ROS commands will be published", flush=True)

    try:
        while True:
            now = time.monotonic()
            if skeleton_tracker is not None and now >= next_track_wall:
                next_track_wall = now + max(0.05, args.track_interval_s)
                target = skeleton_tracker.get_latest(max_age_s=args.track_max_age_s)
                if target is not None:
                    tracking_was_stale = False
                    base_yaw, neck_pitch, eye_yaw = compute_track_commands(
                        target,
                        yaw_scale=args.track_yaw_scale,
                        pitch_scale=args.track_pitch_scale,
                        eye_only_yaw_threshold=args.track_eye_only_yaw_threshold,
                        base_yaw_limit=args.track_base_yaw_limit,
                        eye_yaw_limit=args.track_eye_yaw_limit,
                        neck_pitch_limit=args.track_neck_pitch_limit,
                    )
                    node.send_motor("base", base_yaw, play_time=160)
                    node.send_motor("neck_pitch", neck_pitch, play_time=160)
                    node.send_motor("left_eye_yaw", eye_yaw, play_time=140)
                    node.send_motor("right_eye_yaw", eye_yaw, play_time=140)
                    timeline("track_command", target={"x": target[0], "y": target[1], "z": target[2]},
                             commands={"base": base_yaw, "neck_pitch": neck_pitch, "eye_yaw": eye_yaw},
                             skeleton_age_s=max(0.0, time.time() - skeleton_tracker.latest_wall), callback_count=skeleton_tracker.callback_count)
                    if now - last_track_report >= 2.0:
                        last_track_report = now
                        print(
                            "[robot_bridge] TRACK target=({:.2f}, {:.2f}, {:.2f}) cmd=(base={:.2f}, pitch={:.2f}, eye={:.2f})".format(
                                *target, base_yaw, neck_pitch, eye_yaw
                            ),
                            flush=True,
                        )
                elif not tracking_was_stale:
                    tracking_was_stale = True
                    timeline("tracking_stale", max_age_s=args.track_max_age_s, callback_count=skeleton_tracker.callback_count,
                             valid_target_count=skeleton_tracker.valid_target_count,
                             reason=skeleton_tracker.last_missing_reason, part_names=skeleton_tracker.last_part_names)
                if skeleton_tracker.last_missing_reason != last_missing_reason:
                    last_missing_reason = skeleton_tracker.last_missing_reason
                    if last_missing_reason:
                        timeline("skeleton_target_missing", reason=last_missing_reason,
                                 callback_count=skeleton_tracker.callback_count,
                                 valid_target_count=skeleton_tracker.valid_target_count,
                                 part_names=skeleton_tracker.last_part_names)
                if skeleton_motion_path is not None and now - last_motion_write >= 0.2:
                    last_motion_write = now
                    motion = skeleton_tracker.get_motion(max_age_s=args.track_max_age_s)
                    if motion is not None:
                        skeleton_motion_path.parent.mkdir(parents=True, exist_ok=True)
                        temporary = skeleton_motion_path.with_suffix(skeleton_motion_path.suffix + ".tmp")
                        temporary.write_text(json.dumps(motion, ensure_ascii=False), encoding="utf-8")
                        temporary.replace(skeleton_motion_path)

                motion = skeleton_tracker.get_motion(max_age_s=args.track_max_age_s)
                visual_focus = load_recent_focus_trigger(visual_focus_path, max_age_s=2.0)
                visual_new = visual_focus is not None and float(visual_focus.get("wall_time") or 0.0) > last_visual_focus_wall
                skeleton_large = motion is not None and float(motion.get("motion_score") or 0.0) >= args.focus_motion_threshold
                if (skeleton_large or visual_new) and now - last_focus_wall >= args.focus_cooldown_s:
                    # This only uses eye roll: base, neck, and eye yaw remain under continuous tracking.
                    # Haru's driver interprets play_time=25 as about 225 ms.
                    # A previous value of 180 took 1.62 s, then was reset after
                    # 300 ms, so the eye never reached its ±1 target.
                    node.send_motor("left_eye_roll", args.focus_left_eye_roll, play_time=25, disable_eyes_roll_sync=True)
                    node.send_motor("right_eye_roll", args.focus_right_eye_roll, play_time=25, disable_eyes_roll_sync=True)
                    node.set_base_led_color(*args.focus_base_led_color)
                    node.set_eye_led_color(*args.focus_eye_led_color)
                    last_focus_wall = now
                    focus_reset_wall = now + 0.50
                    if visual_new:
                        last_visual_focus_wall = float(visual_focus["wall_time"])
                    timeline("eye_focus", trigger_source=(visual_focus or {}).get("source", "skeleton"),
                             motion_score=(visual_focus or motion or {}).get("motion_score", (visual_focus or {}).get("motion")),
                             cooldown_s=args.focus_cooldown_s, base_led_rgb=args.focus_base_led_color,
                             eye_led_input_rgb=args.focus_eye_led_color,
                             visible_led_rgb=args.focus_base_led_color,
                             left_eye_roll=args.focus_left_eye_roll, right_eye_roll=args.focus_right_eye_roll)

            if focus_reset_wall is not None and now >= focus_reset_wall:
                node.send_motor("left_eye_roll", 0.0, play_time=25, disable_eyes_roll_sync=True)
                node.send_motor("right_eye_roll", 0.0, play_time=25, disable_eyes_roll_sync=True)
                node.set_base_led_color(*args.focus_base_led_reset_color)
                node.set_eye_led_color(*args.focus_base_led_reset_color)
                focus_reset_wall = None
                timeline("eye_focus_reset", base_led_rgb=args.focus_base_led_reset_color, eye_led_rgb=args.focus_base_led_reset_color)

            if now >= next_turn_poll:
                next_turn_poll = now + max(0.05, args.poll_s)
                turns = load_turns(path)
                if len(turns) < seen_count:
                    seen_count = 0

                for idx, turn in enumerate(turns[seen_count:], start=seen_count):
                    reply = str(turn.get("reply") or "").strip()
                    if not reply:
                        continue
                    proactive = bool(turn.get("proactive"))
                    response_style = str(turn.get("response_style") or "neutral")
                    tag = "proactive" if proactive else "dialogue"
                    print(f"[robot_bridge] new {tag} reply #{idx + 1}: {reply}", flush=True)

                    if node is not None and args.reply_motion and not args.no_motion:
                        # Acknowledge contains sleeps; never let it stop skeleton callbacks.
                        threading.Thread(target=node.acknowledge, daemon=True).start()
                        timeline("reply_motion_start", turn_index=idx + 1, proactive=proactive)

                    if node is not None and not args.no_tts:
                        node.speak(reply, language_code=args.language_code, disable_lipsync=args.disable_lipsync, response_style=response_style)
                        timeline("tts_request", turn_index=idx + 1, proactive=proactive, response_style=response_style, reply=reply)

                seen_count = len(turns)
            if node is not None:
                rclpy.spin_once(node, timeout_sec=max(0.001, args.spin_period_s))
                if now >= next_status_write:
                    next_status_write = now + 0.1
                    write_tts_status(tts_status_path, node)
            else:
                time.sleep(max(0.01, args.spin_period_s))
    except KeyboardInterrupt:
        print("[robot_bridge] stopped", flush=True)
    except Exception:
        # The launcher terminates all child processes together.  ROS can have
        # already invalidated the context before spin_once wakes up.
        if rclpy.ok():
            raise
    finally:
        if node is not None:
            try:
                node.destroy_node()
            finally:
                if rclpy.ok():
                    try:
                        rclpy.shutdown()
                    except Exception:
                        pass


if __name__ == "__main__":
    main()
