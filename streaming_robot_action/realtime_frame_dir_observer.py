#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image as PILImage

from .action_event_system import ActionEventSystem
from .visual_buffer import RealTimeVisualBuffer, resize_for_vlm
from .vlm_action_client import DoubaoFrameVLMClient


def log(t: float, kind: str, message: str) -> None:
    print(f"[t={t:06.1f}] {kind:<13} {message}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Observe a live frame directory and run VLM action memory.")
    p.add_argument("--frame-dir", default="streaming_camera_outputs/frame_bridge")
    p.add_argument("--output-dir", default="streaming_camera_outputs/frame_dir_observer")
    p.add_argument("--buffer-duration-s", type=float, default=8.0)
    p.add_argument("--save-fps", type=float, default=5.0)
    p.add_argument("--first-submit-delay-s", type=float, default=1.0)
    p.add_argument("--vlm-interval-s", type=float, default=10.0)
    p.add_argument("--window-s", type=float, default=6.0)
    p.add_argument("--keyframes-first", type=int, default=1)
    p.add_argument("--keyframes-normal", type=int, default=4)
    p.add_argument("--max-image-width", type=int, default=512)
    p.add_argument("--hard-timeout-s", type=float, default=15.0)
    p.add_argument("--vlm-timeout-s", type=float, default=20.0)
    p.add_argument("--model", default="qwen3.6-flash")
    p.add_argument("--api-provider", choices=["", "ark", "dashscope"], default="dashscope")
    p.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    p.add_argument("--ignore-env-proxy", action="store_true", help="Ignore HTTP_PROXY/HTTPS_PROXY/ALL_PROXY from the shell.")
    p.add_argument("--poll-s", type=float, default=0.1)
    args = p.parse_args()

    frame_dir = Path(args.frame_dir)
    output_dir = Path(args.output_dir)
    keyframes_dir = output_dir / "keyframes"
    output_dir.mkdir(parents=True, exist_ok=True)
    keyframes_dir.mkdir(parents=True, exist_ok=True)

    buffer = RealTimeVisualBuffer(buffer_duration=args.buffer_duration_s, save_fps=args.save_fps)
    vlm = DoubaoFrameVLMClient(
        model=args.model,
        timeout_s=args.vlm_timeout_s,
        api_provider=args.api_provider or None,
        base_url=args.base_url or None,
        ignore_env_proxy=args.ignore_env_proxy,
    )
    action_system = ActionEventSystem(vlm, hard_timeout_s=args.hard_timeout_s)
    event_log: List[dict] = []
    start_wall = time.time()
    next_submit_t = args.first_submit_delay_s
    submissions = 0
    last_frame_idx = -1
    camera_ready = False

    try:
        while True:
            now_t = time.time() - start_wall
            latest_path = frame_dir / "latest.json"
            if latest_path.exists():
                try:
                    latest = json.loads(latest_path.read_text(encoding="utf-8"))
                    frame_idx = int(latest["frame_idx"])
                    if frame_idx != last_frame_idx:
                        image_path = Path(latest["path"])
                        if not image_path.exists():
                            time.sleep(args.poll_s)
                            continue
                        image = np.asarray(PILImage.open(image_path).convert("RGB"))
                        buffer.add_frame(image, timestamp=now_t, frame_id="frame_dir")
                        last_frame_idx = frame_idx
                        if not camera_ready:
                            camera_ready = True
                            log(now_t, "CAMERA_READY", f"frame_idx={frame_idx}")
                except Exception as exc:
                    log(now_t, "CAMERA_ERR", repr(exc))

            for update in action_system.poll():
                event_log.append(update)
                if update["status"] == "confirmed":
                    log(now_t, "VLM_RETURN", f"{update['observation_short'] or update['activity']} | conf={update['confidence']:.2f} | latency={update['latency_s']:.2f}s")
                    write_state(output_dir, action_system, event_log, now_t)
                else:
                    log(now_t, "VLM_UPDATE", f"{update['status']} | {update.get('error', '')}")

            if camera_ready and now_t + 1e-6 >= next_submit_t:
                if action_system.has_pending():
                    log(now_t, "VLM_SKIP", "previous VLM request is still pending")
                else:
                    num_frames = args.keyframes_first if submissions == 0 else args.keyframes_normal
                    samples = buffer.sample_keyframes(now=now_t, window_length=args.window_s, num_frames=num_frames)
                    frames = [resize_for_vlm(f.image, max_width=args.max_image_width) for f in samples]
                    for idx, image in enumerate(frames):
                        PILImage.fromarray(image).save(keyframes_dir / f"t{now_t:.1f}_k{idx:02d}.jpg", quality=90)
                    submitted = action_system.submit_keyframes(frames, timestamp=now_t)
                    submissions += 1
                    log(now_t, "VLM_SUBMIT", f"{submitted.event_id} | keyframes={submitted.num_keyframes}")
                    write_state(output_dir, action_system, event_log, now_t)
                next_submit_t += args.vlm_interval_s

            time.sleep(args.poll_s)
    except KeyboardInterrupt:
        log(time.time() - start_wall, "STOP", "keyboard interrupt")
    finally:
        write_state(output_dir, action_system, event_log, time.time() - start_wall)
        action_system.shutdown()


def write_state(output_dir: Path, action_system: ActionEventSystem, event_log: List[dict], now_t: float) -> None:
    (output_dir / "camera_action_updates.json").write_text(json.dumps(event_log, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "camera_dialogue_context.json").write_text(
        json.dumps(action_system.get_dialogue_context(current_time_s=now_t), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
