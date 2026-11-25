#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
import time
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

def compressed_image_to_rgb(msg) -> np.ndarray:
    image = PILImage.open(BytesIO(bytes(msg.data))).convert("RGB")
    return np.asarray(image)


def image_msg_to_rgb(msg) -> np.ndarray:
    height = int(msg.height)
    width = int(msg.width)
    encoding = str(msg.encoding).lower()
    data = np.frombuffer(msg.data, dtype=np.uint8)

    if encoding in {"rgb8", "bgr8"}:
        image = data.reshape((height, width, 3))
        if encoding == "bgr8":
            image = image[..., ::-1]
        return image.copy()
    if encoding in {"rgba8", "bgra8"}:
        image = data.reshape((height, width, 4))[..., :3]
        if encoding == "bgra8":
            image = image[..., ::-1]
        return image.copy()
    if encoding in {"mono8", "8uc1"}:
        gray = data.reshape((height, width))
        return np.repeat(gray[..., None], 3, axis=2)
    raise ValueError(f"Unsupported image encoding: {msg.encoding}")

def main() -> None:
    parser = argparse.ArgumentParser(description="ROS2 camera frame bridge: camera topic -> JPEG frame directory.")
    parser.add_argument("--image-topic", default="/azure_kinect/rgb/image_raw/compressed")
    parser.add_argument("--compressed", action="store_true", default=True)
    parser.add_argument("--raw-image", dest="compressed", action="store_false")
    parser.add_argument("--output-dir", default="streaming_camera_outputs/frame_bridge")
    parser.add_argument("--save-fps", type=float, default=5.0)
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--qos", choices=["reliable", "best_effort", "sensor_data"], default="sensor_data")
    parser.add_argument("--latest-only", action="store_true", help="Keep only latest.jpg; avoids frame-history I/O lag on removable disks.")
    parser.add_argument("--timeline-jsonl", default="", help="Append saved-frame timing records here.")
    args = parser.parse_args()

    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
        from sensor_msgs.msg import CompressedImage, Image
    except Exception as exc:
        raise RuntimeError("ROS2 Python dependencies are not available. Run with system ROS2 Python.") from exc

    class CameraFrameBridge(Node):
        def __init__(self) -> None:
            super().__init__("ros_camera_frame_bridge")
            self.output_dir = Path(args.output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.frame_idx = self._initial_frame_idx()
            self.last_save_wall = 0.0
            self.start_wall = time.time()
            self.timeline_path = Path(args.timeline_jsonl) if args.timeline_jsonl else None
            msg_type = CompressedImage if args.compressed else Image
            self.subscription = self.create_subscription(msg_type, args.image_topic, self.on_image, self._qos_profile())
            self.get_logger().info(f"Saving camera frames from {args.image_topic} to {self.output_dir} (qos={args.qos})")

        @staticmethod
        def _qos_profile():
            if args.qos in {"sensor_data", "best_effort"}:
                return QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
            return QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        def on_image(self, msg) -> None:
            now = time.time()
            if now - self.last_save_wall < 1.0 / max(args.save_fps, 1e-6):
                return
            try:
                frame = compressed_image_to_rgb(msg) if args.compressed else image_msg_to_rgb(msg)
            except Exception as exc:
                self.get_logger().warn(f"frame conversion failed: {exc!r}")
                return
            self.last_save_wall = now
            sim_t = now - self.start_wall
            self.frame_idx += 1
            filename = "latest.jpg" if args.latest_only else f"frame_{self.frame_idx:06d}.jpg"
            tmp_path = self.output_dir / f".{filename}.tmp"
            final_path = self.output_dir / filename
            PILImage.fromarray(frame).save(tmp_path, format="JPEG", quality=90)
            tmp_path.rename(final_path)
            latest = {
                "frame_idx": self.frame_idx,
                "timestamp": sim_t,
                "wall_time": now,
                "source_stamp_s": float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9,
                "path": str(final_path.resolve()),
                "topic": args.image_topic,
            }
            (self.output_dir / "latest.json").write_text(json.dumps(latest, ensure_ascii=False), encoding="utf-8")
            if self.timeline_path is not None:
                self.timeline_path.parent.mkdir(parents=True, exist_ok=True)
                with self.timeline_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"source": "camera_bridge", "event": "camera_frame_saved", **latest}, ensure_ascii=False) + "\n")
            if not args.latest_only:
                self._prune_old()
            if self.frame_idx == 1:
                print(f"[t={sim_t:06.1f}] CAMERA_READY saved {final_path}", flush=True)

        def _prune_old(self) -> None:
            frames = sorted(self.output_dir.glob("frame_*.jpg"))
            extra = len(frames) - args.max_frames
            for path in frames[:max(0, extra)]:
                path.unlink(missing_ok=True)

        def _initial_frame_idx(self) -> int:
            max_idx = 0
            pattern = re.compile(r"frame_(\d+)\.jpg$")
            for path in self.output_dir.glob("frame_*.jpg"):
                match = pattern.match(path.name)
                if match:
                    max_idx = max(max_idx, int(match.group(1)))
            return max_idx

    rclpy.init()
    node = CameraFrameBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        # SIGTERM during launcher cleanup can make spin raise this after ROS
        # has already shut down; it is not a camera failure.
        if exc.__class__.__name__ != "ExternalShutdownException":
            raise
    finally:
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
