#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image as PILImage

from .action_event_system import ActionEventSystem
from .baidu_translate_client import BaiduTranslateClient
from .dialogue_llm_client import DialogueLLMClient
from .dialogue_prompt_builder import build_dialogue_prompt
from .meaningful_action_trigger import MeaningfulActionTrigger
from .proactive_policy import ProactivePolicy, build_proactive_prompt
from .visual_buffer import RealTimeVisualBuffer, resize_for_vlm
from .vlm_action_client import DoubaoFrameVLMClient
from .xunfei_asr_client import ASRTextQueue, XunfeiStreamingASR, load_xunfei_credentials
from .xunfei_rtasr_llm_client import XunfeiRTASRLLMStreaming


RUN_OUTPUT_FILES = (
    "vision.log",
    "dialogue.log",
    "system.log",
    "dialogue_turns.json",
    "vlm_frame_evidence.json",
)

RUN_OUTPUT_DIRS = (
    "keyframes",
    "trigger_frames",
)


def archive_previous_run(output_dir: Path) -> Path | None:
    existing_files = []
    for name in RUN_OUTPUT_FILES:
        path = output_dir / name
        if path.is_file() and path.stat().st_size > 0:
            existing_files.append(path)
    existing_dirs = []
    for name in RUN_OUTPUT_DIRS:
        path = output_dir / name
        if path.is_dir() and any(path.iterdir()):
            existing_dirs.append(path)
    if not existing_files and not existing_dirs:
        return None

    archive_dir = output_dir / "archive" / datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir.mkdir(parents=True, exist_ok=True)
    for path in existing_files:
        shutil.copy2(path, archive_dir / path.name)
    for path in existing_dirs:
        shutil.copytree(path, archive_dir / path.name, dirs_exist_ok=True)
    return archive_dir


class DemoLogger:
    def __init__(self, output_dir: Path, console_mode: str = "all") -> None:
        self.output_dir = output_dir
        self.console_mode = console_mode
        self.vision_log = output_dir / "vision.log"
        self.dialogue_log = output_dir / "dialogue.log"
        self.system_log = output_dir / "system.log"
        self.timeline_log = output_dir / "main_timeline.jsonl"
        for path in (self.vision_log, self.dialogue_log, self.system_log, self.timeline_log):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")

    def log(self, t: float, kind: str, message: str, channel: str = "system") -> None:
        line = f"[t={t:06.1f}] {kind:<13} {message}"
        if channel == "vision":
            path = self.vision_log
        elif channel == "dialogue":
            path = self.dialogue_log
        else:
            path = self.system_log
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        with self.timeline_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"wall_time": time.time(), "demo_time_s": t, "source": "main", "event": kind, "channel": channel, "message": message}, ensure_ascii=False) + "\n")
        if self._show_on_console(channel):
            print(line, flush=True)

    def _show_on_console(self, channel: str) -> bool:
        if self.console_mode == "all":
            return True
        if self.console_mode == "vision":
            return channel in {"vision", "system"}
        if self.console_mode == "dialogue":
            return channel in {"dialogue", "system"}
        if self.console_mode == "quiet":
            return False
        return True


class AdaptiveVisualScheduler:
    """Triggers extra VLM submissions when motion becomes strong."""

    def __init__(
        self,
        enabled: bool,
        base_interval_s: float,
        min_event_interval_s: float = 2.5,
        motion_threshold: float = 8.0,
        ema_alpha: float = 0.35,
    ) -> None:
        self.enabled = bool(enabled)
        self.base_interval_s = float(base_interval_s)
        self.min_event_interval_s = float(min_event_interval_s)
        self.motion_threshold = float(motion_threshold)
        self.ema_alpha = float(ema_alpha)
        self.motion_ema = 0.0
        self.last_event_submit_s = -1e9
        self._prev_gray: Optional[np.ndarray] = None

    def update(self, image: np.ndarray, now_t: float) -> Optional[dict]:
        if not self.enabled:
            return None

        gray = self._small_gray(image)
        if self._prev_gray is None:
            self._prev_gray = gray
            return None

        diff = float(np.mean(np.abs(gray - self._prev_gray)))
        self._prev_gray = gray
        self.motion_ema = self.ema_alpha * diff + (1.0 - self.ema_alpha) * self.motion_ema

        if self.should_trigger(self.motion_ema, self.motion_threshold, now_t):
            return {
                "state": "event",
                "motion": self.motion_ema,
                "source": "frame_diff",
                "interval_s": self.min_event_interval_s,
            }
        return None

    def update_external_motion(self, score: float, now_t: float, threshold: float, source: str) -> Optional[dict]:
        if not self.enabled:
            return None
        if self.should_trigger(float(score), float(threshold), now_t):
            return {
                "state": "event",
                "motion": float(score),
                "source": source,
                "interval_s": self.min_event_interval_s,
            }
        return None

    def should_trigger(self, score: float, threshold: float, now_t: float) -> bool:
        if score < threshold:
            return False
        if now_t - self.last_event_submit_s < self.min_event_interval_s:
            return False
        self.last_event_submit_s = now_t
        return True

    def mark_event_submitted(self, now_t: float) -> None:
        self.last_event_submit_s = now_t

    def current_interval(self, now_t: float) -> float:
        return self.base_interval_s

    @staticmethod
    def _small_gray(image: np.ndarray) -> np.ndarray:
        pil = PILImage.fromarray(image).convert("L").resize((64, 48), PILImage.Resampling.BILINEAR)
        return np.asarray(pil, dtype=np.float32)


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config-json", default="")
    pre_args, _ = pre_parser.parse_known_args()

    p = argparse.ArgumentParser(
        description="Live camera + Xunfei ASR + action-aware dialogue terminal demo.",
        parents=[pre_parser],
    )
    p.add_argument("--frame-dir", default="streaming_camera_outputs/frame_bridge")
    p.add_argument("--output-dir", default="streaming_camera_outputs/camera_asr_dialogue")
    p.add_argument("--buffer-duration-s", type=float, default=8.0)
    p.add_argument("--save-fps", type=float, default=5.0)
    p.add_argument("--first-submit-delay-s", type=float, default=1.0)
    p.add_argument("--vlm-interval-s", type=float, default=10.0)
    p.add_argument("--adaptive-visual-schedule", action="store_true")
    p.add_argument("--frame-diff-motion-trigger", action="store_true",
                   help="Also trigger motion events from raw camera frame differences. Disabled by default for entity runs.")
    p.add_argument("--adaptive-min-event-interval-s", type=float, default=2.5)
    p.add_argument("--adaptive-motion-threshold", type=float, default=8.0)
    p.add_argument("--skeleton-motion-json", default="")
    p.add_argument("--visual-focus-json", default="",
                   help="Small JSON heartbeat sent to the robot bridge for visually detected large actions.")
    p.add_argument("--skeleton-motion-threshold", type=float, default=0.12)
    p.add_argument("--skeleton-motion-max-age-s", type=float, default=2.0)
    p.add_argument("--meaningful-action-trigger", action="store_true")
    p.add_argument("--meaningful-action-window-s", type=float, default=3.0)
    p.add_argument("--meaningful-action-score-threshold", type=float, default=5.0)
    p.add_argument("--meaningful-action-min-peak-speed", type=float, default=2.0)
    p.add_argument("--meaningful-action-min-active-ratio", type=float, default=0.25,
                   help="Fraction of the recent skeleton window that must contain high-speed motion.")
    p.add_argument("--meaningful-action-settle-s", type=float, default=1.2,
                   help="Wait this long after the last strong skeleton sample before analysing one complete action episode.")
    p.add_argument("--meaningful-action-stand-rise-m", type=float, default=0.20,
                   help="Vertical torso rise (metres) that identifies a sit-to-stand transition.")
    p.add_argument("--window-s", type=float, default=6.0)
    p.add_argument("--keyframes-first", type=int, default=1)
    p.add_argument("--keyframes-normal", type=int, default=4)
    p.add_argument("--max-image-width", type=int, default=512)
    p.add_argument("--hard-timeout-s", type=float, default=12.0)
    p.add_argument("--vlm-timeout-s", type=float, default=8.0)
    p.add_argument("--vlm-model", default="qwen3.6-flash")
    p.add_argument("--vlm-max-inflight", type=int, default=2)
    p.add_argument("--vlm-priority-max-inflight", type=int, default=1)
    p.add_argument("--vlm-priority-motion-threshold", type=float, default=6.0)
    p.add_argument("--vlm-priority-pending-max-age-s", type=float, default=3.0)
    p.add_argument("--vlm-api-provider", choices=["", "ark", "dashscope"], default="dashscope")
    p.add_argument("--vlm-base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    p.add_argument("--dialogue-model", default="doubao-seed-1-6-flash-250828")
    p.add_argument("--dialogue-base-url", default="https://ark.cn-beijing.volces.com/api/v3")
    p.add_argument("--dialogue-timeout-s", type=float, default=20.0)
    p.add_argument("--dialogue-user-deadline-s", type=float, default=12.0,
                   help="Emit a local fallback if a user-facing LLM request is still pending after this time.")
    p.add_argument("--dialogue-proactive-deadline-s", type=float, default=8.0,
                   help="Discard a proactive LLM response which is no longer timely after this time.")
    p.add_argument("--dialogue-output-mode", choices=["english"], default="english")
    p.add_argument("--dialogue-turn-delay-s", type=float, default=1.4)
    p.add_argument("--baidu-translate-replies", action="store_true",
                   help="Translate English robot replies to Chinese in a background worker; robot TTS remains English.")
    p.add_argument("--baidu-translate-timeout-s", type=float, default=3.0)
    p.add_argument("--ignore-env-proxy", action="store_true")
    p.add_argument("--mic-device", default="plughw:2,0")
    p.add_argument("--xunfei-config", default="")
    p.add_argument("--asr-engine", choices=["iat", "rtasr_llm"], default="iat")
    p.add_argument("--asr-verbose", action="store_true")
    p.add_argument("--asr-energy-threshold", type=float, default=300.0)
    p.add_argument("--asr-min-speech-frames", type=int, default=3)
    p.add_argument("--asr-max-silence-frames", type=int, default=8)
    p.add_argument("--asr-frame-ms", type=int, default=40)
    p.add_argument("--asr-connect-timeout-s", type=float, default=8.0)
    p.add_argument("--asr-final-wait-s", type=float, default=1.2)
    p.add_argument("--asr-min-recognition-s", type=float, default=1.2)
    p.add_argument("--asr-suppress-after-robot-s", type=float, default=3.0)
    p.add_argument("--asr-tts-status-hold-s", type=float, default=1.5)
    p.add_argument("--tts-status-json", default="")
    p.add_argument("--tts-status-max-age-s", type=float, default=2.0)
    p.add_argument("--no-asr-partial-on-silence", action="store_true")
    p.add_argument("--asr-level-meter", action="store_true")
    p.add_argument("--console-mode", choices=["all", "vision", "dialogue", "quiet"], default="all")
    p.add_argument("--max-dialogue-turns", type=int, default=3)
    p.add_argument("--max-visual-events", type=int, default=3)
    p.add_argument("--visual-memory-age-s", type=float, default=90.0)
    p.add_argument("--enable-proactive", action="store_true")
    p.add_argument("--proactive-check-interval-s", type=float, default=2.0)
    p.add_argument("--proactive-quiet-s", type=float, default=20.0)
    p.add_argument("--proactive-visual-event-quiet-s", type=float, default=2.0)
    p.add_argument("--proactive-action-lock-user-turns", type=int, default=2)
    p.add_argument("--proactive-action-salience-threshold", type=int, default=2)
    p.add_argument("--proactive-trigger-threshold", type=int, default=5,
                   help="Minimum proactive-policy score; 4 permits a timer check-in with one usable VLM event.")
    p.add_argument("--deferred-visual-proactive-max-age-s", type=float, default=15.0,
                   help="Keep one salient visual event briefly when it arrives during a user turn or robot speech.")
    p.add_argument("--proactive-debug", action="store_true")
    p.add_argument("--poll-s", type=float, default=0.1)

    if pre_args.config_json:
        config_path = Path(pre_args.config_json)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        valid_keys = {action.dest for action in p._actions}
        defaults = {key: value for key, value in config.items() if key in valid_keys}
        unknown_keys = sorted(set(config) - valid_keys)
        if unknown_keys:
            print(f"[WARN] Ignoring unknown config keys: {', '.join(unknown_keys)}", flush=True)
        p.set_defaults(**defaults)

    args = p.parse_args()

    frame_dir = Path(args.frame_dir)
    output_dir = Path(args.output_dir)
    archive_dir = archive_previous_run(output_dir)
    keyframes_dir = output_dir / "keyframes"
    trigger_frames_dir = output_dir / "trigger_frames"
    skeleton_motion_path = Path(args.skeleton_motion_json) if args.skeleton_motion_json else output_dir / "skeleton_motion.json"
    visual_focus_path = Path(args.visual_focus_json) if args.visual_focus_json else output_dir / "visual_focus_trigger.json"
    tts_status_path = Path(args.tts_status_json) if args.tts_status_json else output_dir / "tts_status.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    for run_dir in (keyframes_dir, trigger_frames_dir):
        if run_dir.exists():
            shutil.rmtree(run_dir)
    keyframes_dir.mkdir(parents=True, exist_ok=True)
    trigger_frames_dir.mkdir(parents=True, exist_ok=True)
    logger = DemoLogger(output_dir, console_mode=args.console_mode)
    if archive_dir is not None:
        logger.log(0.0, "ARCHIVE", str(archive_dir), channel="system")

    def write_visual_focus_trigger(schedule_event: dict, now_t: float) -> None:
        """Tell the ROS bridge about a system-level significant movement."""
        payload = {"wall_time": time.time(), "demo_time_s": now_t, "source": schedule_event.get("source"),
                   "motion": schedule_event.get("motion"), "reason": schedule_event.get("reason", "")}
        temporary = visual_focus_path.with_suffix(visual_focus_path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temporary.replace(visual_focus_path)

    buffer = RealTimeVisualBuffer(buffer_duration=args.buffer_duration_s, save_fps=args.save_fps)
    scheduler = AdaptiveVisualScheduler(
        enabled=args.adaptive_visual_schedule,
        base_interval_s=args.vlm_interval_s,
        min_event_interval_s=args.adaptive_min_event_interval_s,
        motion_threshold=args.adaptive_motion_threshold,
    )
    meaningful_trigger = MeaningfulActionTrigger(
        window_s=args.meaningful_action_window_s,
        score_threshold=args.meaningful_action_score_threshold,
        min_peak_speed=args.meaningful_action_min_peak_speed,
        min_active_ratio=args.meaningful_action_min_active_ratio,
        stand_rise_threshold_m=args.meaningful_action_stand_rise_m,
    )
    vlm = DoubaoFrameVLMClient(
        model=args.vlm_model,
        timeout_s=args.vlm_timeout_s,
        api_provider=args.vlm_api_provider or None,
        base_url=args.vlm_base_url or None,
        ignore_env_proxy=args.ignore_env_proxy,
    )
    priority_vlm = DoubaoFrameVLMClient(
        model=args.vlm_model,
        timeout_s=args.vlm_timeout_s,
        api_provider=args.vlm_api_provider or None,
        base_url=args.vlm_base_url or None,
        ignore_env_proxy=args.ignore_env_proxy,
    )
    action_system = ActionEventSystem(
        vlm,
        hard_timeout_s=args.hard_timeout_s,
        max_workers=max(1, args.vlm_max_inflight),
        event_prefix="vlm",
    )
    priority_action_system = ActionEventSystem(
        priority_vlm,
        memory=action_system.memory,
        hard_timeout_s=args.hard_timeout_s,
        max_workers=max(1, args.vlm_priority_max_inflight),
        event_prefix="pvlm",
    )
    dialogue_client = DialogueLLMClient(
        model=args.dialogue_model,
        base_url=args.dialogue_base_url,
        timeout_s=args.dialogue_timeout_s,
        ignore_env_proxy=args.ignore_env_proxy,
    )
    translation_client = None
    if args.baidu_translate_replies:
        try:
            translation_client = BaiduTranslateClient(timeout_s=args.baidu_translate_timeout_s)
            logger.log(0.0, "TRANSLATE_READY", "Baidu English-to-Chinese background translation enabled", channel="system")
        except RuntimeError as exc:
            logger.log(0.0, "TRANSLATE_OFF", str(exc), channel="system")
    proactive_policy = ProactivePolicy(
        max_visual_age_s=30.0,
        visual_event_quiet_s=args.proactive_visual_event_quiet_s,
        general_quiet_s=args.proactive_quiet_s,
        action_lock_user_turns=args.proactive_action_lock_user_turns,
        action_salience_threshold=args.proactive_action_salience_threshold,
        trigger_threshold=args.proactive_trigger_threshold,
    )

    asr_queue = ASRTextQueue()
    credentials = load_xunfei_credentials(legacy_config=args.xunfei_config)
    asr_status_start = time.time()

    def on_asr_status(kind: str, message: str) -> None:
        proactive_policy.on_asr_status(kind, time.time() - asr_status_start)
        logger.log(time.time() - asr_status_start, kind, message, channel="dialogue")

    asr_cls = XunfeiRTASRLLMStreaming if args.asr_engine == "rtasr_llm" else XunfeiStreamingASR
    asr = asr_cls(
        credentials,
        mic_device=args.mic_device,
        on_final_text=asr_queue.callback,
        on_status=on_asr_status,
        verbose=args.asr_verbose,
        energy_threshold=args.asr_energy_threshold,
        min_speech_frames=args.asr_min_speech_frames,
        max_silence_frames=args.asr_max_silence_frames,
        frame_ms=args.asr_frame_ms,
        connect_timeout_s=args.asr_connect_timeout_s,
        final_wait_s=args.asr_final_wait_s,
        min_recognition_s=args.asr_min_recognition_s,
        emit_partial_on_silence=not args.no_asr_partial_on_silence,
        ignore_env_proxy=args.ignore_env_proxy,
        level_meter=args.asr_level_meter,
    )

    event_log: List[dict] = []
    dialogue_log: List[dict] = []
    vlm_evidence_log: List[dict] = []
    start_wall = time.time()
    next_submit_t = args.first_submit_delay_s
    submissions = 0
    last_frame_idx = -1
    camera_ready = False
    next_proactive_check_t = args.proactive_check_interval_s
    pending_user_turn = None
    queued_user_turn = None
    # User answers and proactive check-ins call the same LLM, but a stalled
    # check-in must never make a real user wait in the executor queue.
    user_dialogue_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="user_dialogue_llm")
    proactive_dialogue_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="proactive_dialogue_llm")
    translation_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="baidu_translate")
    pending_translations = []
    pending_user_job = None
    pending_proactive_job = None
    stale_proactive_jobs = []
    # A salient action is not allowed to interrupt ASR/TTS.  Previously such
    # events were simply lost; retain only the newest one and replay it as
    # soon as the interaction is actually free.
    deferred_visual_proactive = None
    pending_motion_trigger = False
    pending_motion_evidence = None
    pending_priority_evidence = None
    motion_episode = None
    last_skeleton_motion_wall = 0.0
    last_tts_status_key = None
    robot_output_busy_until = -1e9
    last_camera_image = None
    last_camera_frame_info = None

    asr.start()
    logger.log(0.0, "ASR_READY", f"engine={args.asr_engine} mic={args.mic_device}", channel="dialogue")
    logger.log(0.0, "LOG_FILES", f"vision={logger.vision_log} dialogue={logger.dialogue_log}", channel="system")

    def request_llm(prompt: str) -> str:
        # One bounded network attempt.  The main real-time loop never waits on it.
        return dialogue_client.reply(prompt, retries=0).strip()

    def queue_translation(turn: dict, now_t: float) -> None:
        if translation_client is None:
            return
        turn["translation_status"] = "pending"
        pending_translations.append({
            "turn": turn,
            "future": translation_executor.submit(translation_client.translate_en_to_zh, turn["reply"]),
            "started_t": now_t,
        })

    def finish_translations(now_t: float) -> None:
        for job in list(pending_translations):
            if not job["future"].done():
                continue
            pending_translations.remove(job)
            turn = job["turn"]
            try:
                translated, request_latency_ms = job["future"].result()
                turn["reply_zh"] = translated
                turn["translation_status"] = "ok"
                turn["translation_latency_ms"] = round(float(request_latency_ms), 1)
                turn["translation_total_latency_s"] = round(now_t - job["started_t"], 3)
                logger.log(now_t, "TRANSLATE_ZH", f"{translated} | latency={request_latency_ms:.0f}ms", channel="dialogue")
            except Exception as exc:
                turn["translation_status"] = "error"
                turn["translation_error"] = str(exc)
                logger.log(now_t, "TRANSLATE_ERR", str(exc), channel="system")
            write_state(output_dir, dialogue_log, vlm_evidence_log)

    def try_proactive_reply(now_t: float, context: dict, trigger_source: str) -> bool:
        nonlocal pending_proactive_job
        if pending_proactive_job is not None or pending_user_job is not None:
            return False
        if any(not job["future"].done() for job in stale_proactive_jobs):
            if args.proactive_debug:
                logger.log(now_t, "PROACTIVE_SKIP", "previous_proactive_request_still_running", channel="dialogue")
            return False
        if now_t < robot_output_busy_until:
            if args.proactive_debug:
                logger.log(now_t, "PROACTIVE_SKIP", f"robot_output_busy:{robot_output_busy_until - now_t:.1f}s", channel="dialogue")
            return False
        if trigger_source == "timer" and (pending_motion_trigger or pending_priority_evidence is not None or has_pending_motion_vlm(vlm_evidence_log)):
            if args.proactive_debug:
                logger.log(now_t, "PROACTIVE_SKIP", "timer | pending_motion_vlm", channel="dialogue")
            return False

        decision = proactive_policy.decide(context, now_t, trigger_source=trigger_source)
        if args.proactive_debug:
            for candidate in decision.candidates:
                logger.log(
                    now_t,
                    "PROACTIVE_CANDIDATE",
                    (
                        f"{trigger_source} | {candidate.get('proactive_type')} "
                        f"score={candidate.get('score')} | {candidate.get('reason')}"
                    ),
                    channel="dialogue",
                )
        if not decision.should_speak:
            if args.proactive_debug:
                logger.log(
                    now_t,
                    "PROACTIVE_SKIP",
                    f"{trigger_source} | {decision.skip_reason or f'score={decision.score}'}",
                    channel="dialogue",
                )
            return False

        logger.log(now_t, "PROACTIVE_SELECTED", f"{trigger_source} | {decision.proactive_type} | score={decision.score}", channel="dialogue")
        logger.log(now_t, "PROACTIVE_TRIGGER", f"{decision.proactive_type} | score={decision.score} | {decision.reason}", channel="dialogue")
        proactive_policy.on_proactive_request(now_t, decision.proactive_type)
        prompt = build_proactive_prompt(
            decision,
            context,
            dialogue_log,
            output_mode=args.dialogue_output_mode,
            max_dialogue_turns=args.max_dialogue_turns,
        )
        visual_observation = context.get("current_visual_observation") or {}
        pending_proactive_job = {
            "future": proactive_dialogue_executor.submit(request_llm, prompt), "started_t": now_t,
            "context": context, "decision": decision, "trigger_source": trigger_source,
            "visual_event_id": str(visual_observation.get("event_id") or ""),
        }
        logger.log(now_t, "DIALOGUE_REQUEST", f"proactive | deadline={args.dialogue_proactive_deadline_s:.1f}s", channel="dialogue")
        return True

    def dispatch_user_dialogue(now_t: float, user_text: str, user_speech_duration: float, context: dict) -> dict:
        prompt = build_dialogue_prompt(context, user_text, current_time_s=now_t, dialogue_history=dialogue_log,
                                       max_visual_events=args.max_visual_events, max_dialogue_turns=args.max_dialogue_turns,
                                       output_mode=args.dialogue_output_mode)
        logger.log(now_t, "DIALOGUE_REQUEST", f"user | deadline={args.dialogue_user_deadline_s:.1f}s", channel="dialogue")
        return {"future": user_dialogue_executor.submit(request_llm, prompt), "started_t": now_t,
                "user_text": user_text, "speech_duration_s": user_speech_duration, "context": context}

    def finish_dialogue_jobs(now_t: float) -> None:
        """Publish ready responses; stale proactive jobs are intentionally dropped."""
        nonlocal pending_user_job, pending_proactive_job, robot_output_busy_until
        # A future cannot safely be force-cancelled after its HTTP request has
        # started.  Reap it explicitly and do not queue further proactive
        # requests behind it; user replies use their separate worker.
        for stale_job in list(stale_proactive_jobs):
            if not stale_job["future"].done():
                continue
            stale_proactive_jobs.remove(stale_job)
            elapsed = now_t - stale_job["started_t"]
            try:
                stale_job["future"].result()
                outcome = "late_result_discarded"
            except Exception as exc:
                outcome = f"late_error:{exc!r}"
            logger.log(now_t, "PROACTIVE_REAP", f"{outcome} | elapsed={elapsed:.2f}s", channel="dialogue")
        if pending_user_job is not None:
            job = pending_user_job
            if job["future"].done() or now_t - job["started_t"] >= args.dialogue_user_deadline_s:
                if job["future"].done():
                    try: raw = job["future"].result()
                    except Exception as exc:
                        logger.log(now_t, "DIALOGUE_ERR", repr(exc), channel="dialogue"); raw = fallback_dialogue_reply(args.dialogue_output_mode)
                else:
                    raw = fallback_dialogue_reply(args.dialogue_output_mode)
                    logger.log(now_t, "DIALOGUE_TIMEOUT", "user fallback emitted; late cloud result discarded", channel="dialogue")
                reply, style = parse_robot_reply_payload(raw, default_style="neutral")
                latency = now_t - job["started_t"]
                obs = job["context"].get("current_visual_observation") or {}
                logger.log(now_t, "ROBOT", f"{reply} | dialogue_latency={latency:.2f}s", channel="dialogue")
                proactive_policy.on_robot_reply(now_t)
                speech = estimate_robot_speech_s(reply, args.asr_suppress_after_robot_s)
                logger.log(now_t, "ASR_SUPPRESS", f"robot_reply | {speech:.1f}s adaptive hold", channel="dialogue")
                robot_output_busy_until = max(robot_output_busy_until, now_t + speech); asr.suppress_for(speech)
                turn = {"time_s": job["started_t"], "turn_type": "dialogue", "user_text": job["user_text"],
                    "speech_duration_s": job["speech_duration_s"], "reply": reply, "response_style": style, "raw_llm_reply": raw,
                    "dialogue_latency_s": latency, "primary_visual_event_id": str(obs.get("event_id") or ""),
                    "llm_visual_context": build_visual_memory_used(job["context"].get("recent_visual_events", []), vlm_evidence_log)}
                dialogue_log.append(turn); queue_translation(turn, now_t)
                write_state(output_dir, dialogue_log, vlm_evidence_log); pending_user_job = None
        if pending_proactive_job is not None:
            job = pending_proactive_job
            if not job["future"].done() and now_t - job["started_t"] >= args.dialogue_proactive_deadline_s:
                logger.log(now_t, "PROACTIVE_DROP", "deadline exceeded; wait for in-flight request before another proactive attempt", channel="dialogue")
                stale_proactive_jobs.append(job)
                pending_proactive_job = None
            elif job["future"].done():
                try: raw = job["future"].result()
                except Exception as exc:
                    logger.log(now_t, "DIALOGUE_ERR", repr(exc), channel="dialogue"); raw = ""
                invalid = proactive_policy.state.asr_active or pending_user_turn or pending_user_job or not asr_queue.queue.empty() or now_t < robot_output_busy_until
                if not raw or invalid:
                    logger.log(now_t, "PROACTIVE_DROP", "user/asr/tts activity or request failure", channel="dialogue")
                else:
                    reply, style = parse_robot_reply_payload(raw, default_style=style_from_comment_style(comment_style_from_context(job["context"])))
                    decision = job["decision"]; latency = now_t - job["started_t"]
                    logger.log(now_t, "PROACTIVE_ROBOT", f"{reply} | dialogue_latency={latency:.2f}s", channel="dialogue")
                    proactive_policy.on_robot_reply(now_t, proactive_type=decision.proactive_type, visual_event_id=job["visual_event_id"])
                    speech = estimate_robot_speech_s(reply, args.asr_suppress_after_robot_s)
                    logger.log(now_t, "ASR_SUPPRESS", f"robot_reply | {speech:.1f}s adaptive hold", channel="dialogue")
                    robot_output_busy_until = max(robot_output_busy_until, now_t + speech); asr.suppress_for(speech)
                    turn = {"time_s": job["started_t"], "turn_type":"proactive", "user_text":"", "speech_duration_s":0.0,
                        "reply":reply, "response_style":style, "raw_llm_reply":raw, "dialogue_latency_s":latency, "proactive":True,
                        "proactive_type":decision.proactive_type, "proactive_reason":decision.reason, "proactive_trigger_source":job["trigger_source"],
                        "primary_visual_event_id":job["visual_event_id"], "llm_visual_context":build_visual_memory_used(job["context"].get("recent_visual_events", []), vlm_evidence_log)}
                    dialogue_log.append(turn); queue_translation(turn, now_t)
                    write_state(output_dir, dialogue_log, vlm_evidence_log)
                pending_proactive_job = None

    def update_tts_status(now_t: float) -> bool:
        nonlocal last_tts_status_key, robot_output_busy_until
        status = read_recent_tts_status(tts_status_path, args.tts_status_max_age_s)
        playing = bool(status and status.get("playing"))
        proactive_policy.state.tts_playing = playing
        if playing:
            asr.suppress_for(args.asr_tts_status_hold_s)
            robot_output_busy_until = max(robot_output_busy_until, now_t + args.asr_tts_status_hold_s)

        status_key = None if status is None else (status.get("status"), playing)
        if status_key != last_tts_status_key:
            last_tts_status_key = status_key
            if status is None:
                logger.log(now_t, "TTS_STATUS", "missing_or_stale", channel="dialogue")
            else:
                logger.log(
                    now_t,
                    "TTS_STATUS",
                    f"{status.get('status_name', status.get('status'))} playing={playing}",
                    channel="dialogue",
                )
        return playing

    try:
        while True:
            now_t = time.time() - start_wall
            tts_playing = update_tts_status(now_t)
            skeleton_motion = read_recent_skeleton_motion(skeleton_motion_path, args.skeleton_motion_max_age_s)
            skeleton_wall = float((skeleton_motion or {}).get("wall_time") or 0.0)
            schedule_event = None
            if skeleton_motion is not None and skeleton_wall > last_skeleton_motion_wall:
                last_skeleton_motion_wall = skeleton_wall
                if args.meaningful_action_trigger:
                    skeleton_motion["demo_time_s"] = now_t
                    decision = meaningful_trigger.update(skeleton_motion, now_t)
                    instantaneous_peak = float(skeleton_motion.get("motion_score") or 0.0)
                    # A semantic window may remain positive after movement
                    # stops.  A new/continuing episode therefore requires a
                    # current high-speed skeleton sample as well.
                    stand_transition = float(decision.features.get("torso_vertical_rise_m") or 0.0) >= args.meaningful_action_stand_rise_m
                    if decision.should_trigger and (instantaneous_peak >= args.meaningful_action_min_peak_speed or stand_transition):
                        if motion_episode is None:
                            motion_episode = {
                                "started_t": now_t, "last_strong_t": now_t, "score": decision.score,
                                "reason": decision.reason, "features": decision.features,
                                "uploads_started": 0, "first_returned": False, "second_ready": False,
                            }
                            logger.log(now_t, "MOTION_EPISODE_START", f"score={decision.score:.1f} | {decision.reason}", channel="vision")
                        else:
                            motion_episode.update({"last_strong_t": now_t, "score": max(float(motion_episode["score"]), decision.score),
                                                   "reason": decision.reason, "features": decision.features})

            # Fast path: upload immediately when the first strong movement is
            # seen.  Do not wait for the action burst to finish.
            if motion_episode is not None and motion_episode["uploads_started"] == 0:
                motion_episode["uploads_started"] = 1
                schedule_event = {
                    "state": "event", "motion": float(motion_episode["score"]), "source": "meaningful_action",
                    "interval_s": args.adaptive_min_event_interval_s, "episode_upload_index": 1,
                    "reason": f"episode first upload; {motion_episode['reason']}",
                    "features": motion_episode.get("features") or {},
                }

            # Second upload is allowed only after the first VLM response and
            # only while the same burst is still actively moving.  This caps a
            # whole continuous action episode at two VLM uploads.
            if motion_episode is not None and motion_episode.get("second_ready"):
                motion_episode["second_ready"] = False
                motion_episode["uploads_started"] = 2
                schedule_event = {
                    "state": "event", "motion": float(motion_episode["score"]), "source": "meaningful_action",
                    "interval_s": args.adaptive_min_event_interval_s, "episode_upload_index": 2,
                    "reason": f"episode follow-up upload; {motion_episode['reason']}",
                    "features": motion_episode.get("features") or {},
                }

            if motion_episode is not None and now_t - float(motion_episode["last_strong_t"]) >= args.meaningful_action_settle_s:
                duration_s = now_t - float(motion_episode["started_t"])
                logger.log(now_t, "MOTION_EPISODE_END", f"duration={duration_s:.1f}s | uploads={motion_episode['uploads_started']}", channel="vision")
                motion_episode = None

            if schedule_event is not None:
                    trigger_frame = save_trigger_frame(
                        trigger_frames_dir,
                        last_camera_image,
                        now_t,
                        schedule_event["source"],
                    )
                    pending_motion_evidence = build_schedule_evidence(
                        schedule_event,
                        now_t,
                        trigger_frame=trigger_frame,
                        frame_info=last_camera_frame_info,
                        skeleton_motion=skeleton_motion,
                        meaningful_features=schedule_event.get("features"),
                    )
                    logger.log(
                        now_t,
                        "VIS_SCHEDULE",
                        (
                            f"{schedule_event['state']} | source={schedule_event['source']} "
                            f"| motion={schedule_event['motion']:.3f} | interval={schedule_event['interval_s']:.1f}s"
                            f" | {schedule_event.get('reason', '')}"
                        ),
                        channel="vision",
                    )
                    if schedule_event["state"] == "event":
                        write_visual_focus_trigger(schedule_event, now_t)
                        if is_priority_motion_trigger(pending_motion_evidence, args.vlm_priority_motion_threshold):
                            pending_priority_evidence = pending_motion_evidence
                            logger.log(
                                now_t,
                                "VLM_PRIORITY_PENDING",
                                f"source={schedule_event['source']} | motion={schedule_event['motion']:.3f}",
                                channel="vision",
                            )
                        else:
                            pending_motion_trigger = True
                            next_submit_t = min(next_submit_t, now_t + 0.5)

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
                        last_camera_image = image
                        last_camera_frame_info = {
                            "frame_idx": frame_idx,
                            "source_path": str(image_path),
                            "timestamp_s": now_t,
                        }
                        saved = buffer.add_frame(image, timestamp=now_t, frame_id="frame_dir")
                        # Raw frame difference is easily caused by Kinect auto
                        # exposure, tracking noise, or camera vibration. In the
                        # entity experiment it is opt-in; skeleton semantics
                        # above remains the only motion-event source by default.
                        if saved and args.frame_diff_motion_trigger:
                            schedule_event = scheduler.update(image, now_t)
                            if schedule_event is not None:
                                trigger_frame = save_trigger_frame(
                                    trigger_frames_dir,
                                    image,
                                    now_t,
                                    schedule_event["source"],
                                )
                                pending_motion_evidence = build_schedule_evidence(
                                    schedule_event,
                                    now_t,
                                    trigger_frame=trigger_frame,
                                    frame_info=last_camera_frame_info,
                                )
                                logger.log(
                                    now_t,
                                    "VIS_SCHEDULE",
                                    (
                                        f"{schedule_event['state']} | source={schedule_event['source']} | motion={schedule_event['motion']:.2f} "
                                        f"| interval={schedule_event['interval_s']:.1f}s"
                                    ),
                                    channel="vision",
                                )
                                if schedule_event["state"] == "event":
                                    write_visual_focus_trigger(schedule_event, now_t)
                                    if is_priority_motion_trigger(pending_motion_evidence, args.vlm_priority_motion_threshold):
                                        pending_priority_evidence = pending_motion_evidence
                                        logger.log(
                                            now_t,
                                            "VLM_PRIORITY_PENDING",
                                            f"source={schedule_event['source']} | motion={schedule_event['motion']:.3f}",
                                            channel="vision",
                                        )
                                    else:
                                        pending_motion_trigger = True
                                        next_submit_t = min(next_submit_t, now_t + 0.5)
                        last_frame_idx = frame_idx
                        if not camera_ready:
                            camera_ready = True
                            logger.log(now_t, "CAMERA_READY", f"frame_idx={frame_idx}", channel="vision")
                except json.JSONDecodeError:
                    # The bridge may be rewriting latest.json at this exact moment.
                    pass
                except Exception as exc:
                    logger.log(now_t, "CAMERA_ERR", repr(exc), channel="vision")

            for update in priority_action_system.poll() + action_system.poll():
                event_log.append(update)
                if update["status"] == "confirmed":
                    logger.log(
                        now_t,
                        "VLM_RETURN",
                        (
                            f"{update['observation_short'] or update['activity']} | conf={update['confidence']:.2f} "
                            f"| salience={update.get('salience', 0)} | style={update.get('comment_style', '')} "
                            f"| latency={update['latency_s']:.2f}s"
                        ),
                        channel="vision",
                    )
                else:
                    logger.log(now_t, "VLM_UPDATE", f"{update['status']} | {update.get('error', '')}", channel="vision")
                update_vlm_evidence_from_event(vlm_evidence_log, update)
                write_state(output_dir, dialogue_log, vlm_evidence_log)
                # The first action upload has returned.  If the same action
                # burst is still genuinely active, allow exactly one fresh
                # follow-up upload.  No return => no repeated uploads.
                evidence = find_vlm_evidence(vlm_evidence_log, str(update.get("event_id") or "")) or {}
                trigger = evidence.get("trigger") or {}
                if (
                    update.get("status") == "confirmed"
                    and str(trigger.get("source") or "") == "meaningful_action"
                    and motion_episode is not None
                    and motion_episode.get("uploads_started") == 1
                    and now_t - float(motion_episode["last_strong_t"]) < args.meaningful_action_settle_s
                ):
                    motion_episode["first_returned"] = True
                    motion_episode["second_ready"] = True
                    logger.log(now_t, "MOTION_EPISODE_FOLLOWUP", "first VLM returned while action is continuing", channel="vision")
                if args.enable_proactive and update["status"] == "confirmed":
                    context = action_system.get_dialogue_context(
                        current_time_s=now_t,
                        max_events=args.max_visual_events,
                        max_age_s=args.visual_memory_age_s,
                    )
                    context = enrich_visual_context_timing(context, now_t, vlm_evidence_log)
                    visual_busy = (
                        pending_user_turn is not None
                        or pending_user_job is not None
                        or pending_proactive_job is not None
                        or now_t < robot_output_busy_until
                        or proactive_policy.state.asr_active
                        or proactive_policy.state.tts_playing
                    )
                    try:
                        salience = int(float(update.get("salience") or 0))
                    except (TypeError, ValueError):
                        salience = 0
                    if visual_busy and salience >= args.proactive_action_salience_threshold:
                        deferred_visual_proactive = {
                            "event_id": str(update.get("event_id") or ""),
                            "queued_t": now_t,
                            "observation": str(update.get("observation_short") or update.get("activity") or ""),
                        }
                        logger.log(
                            now_t,
                            "PROACTIVE_DEFER",
                            "visual_event | interaction busy; retain newest salient action",
                            channel="dialogue",
                        )
                    elif not visual_busy:
                        try_proactive_reply(now_t, context, trigger_source="visual_event")

            if camera_ready and pending_priority_evidence is not None:
                if priority_action_system.pending_count() < args.vlm_priority_max_inflight:
                    priority_age = now_t - float(pending_priority_evidence.get("trigger_time_s") or now_t)
                    if priority_age > args.vlm_priority_pending_max_age_s:
                        logger.log(
                            now_t,
                            "VLM_PRIORITY_DROP",
                            f"stale_pending_priority:{priority_age:.1f}s",
                            channel="vision",
                        )
                        pending_priority_evidence = None
                        time.sleep(args.poll_s)
                        continue
                    trigger_evidence = dict(pending_priority_evidence)
                    trigger_evidence["priority"] = {
                        "enabled": True,
                        "reason": "strong motion used dedicated priority VLM worker",
                        "priority_pending_before_submit": priority_action_system.pending_count(),
                        "priority_limit": args.vlm_priority_max_inflight,
                    }
                    submitted = submit_vlm_observation(
                        priority_action_system,
                        buffer,
                        keyframes_dir,
                        now_t,
                        args.window_s,
                        args.keyframes_normal,
                        args.max_image_width,
                        vlm_evidence_log,
                        schedule_state="motion_event_priority",
                        trigger_evidence=trigger_evidence,
                    )
                    if submitted is not None:
                        logger.log(
                            now_t,
                            "VLM_SUBMIT",
                            f"{submitted.event_id} | keyframes={submitted.num_keyframes} | schedule=motion_event_priority",
                            channel="vision",
                        )
                        pending_priority_evidence = None
                    else:
                        logger.log(now_t, "VLM_SKIP", "priority no keyframes available yet", channel="vision")

            if camera_ready and now_t + 1e-6 >= next_submit_t:
                pending_count = action_system.pending_count()
                if pending_count >= args.vlm_max_inflight:
                    logger.log(
                        now_t,
                        "VLM_SKIP",
                        f"max inflight VLM requests reached ({action_system.pending_count()}/{args.vlm_max_inflight})",
                        channel="vision",
                    )
                    if pending_motion_trigger:
                        scheduler.mark_event_submitted(now_t)
                        pending_motion_trigger = False
                        pending_motion_evidence = None
                    next_submit_t = now_t + min(2.0, scheduler.current_interval(now_t))
                else:
                    schedule_state = "motion_event" if pending_motion_trigger else "base"
                    trigger_evidence = pending_motion_evidence if pending_motion_trigger else None
                    submitted = submit_vlm_observation(
                        action_system,
                        buffer,
                        keyframes_dir,
                        now_t,
                        args.window_s,
                        args.keyframes_first if submissions == 0 else args.keyframes_normal,
                        args.max_image_width,
                        vlm_evidence_log,
                        schedule_state=schedule_state,
                        trigger_evidence=trigger_evidence,
                    )
                    if submitted is not None:
                        submissions += 1
                        interval_s = scheduler.current_interval(now_t)
                        logger.log(
                            now_t,
                            "VLM_SUBMIT",
                            f"{submitted.event_id} | keyframes={submitted.num_keyframes} | next_base_interval={interval_s:.1f}s | schedule={schedule_state}",
                            channel="vision",
                        )
                        if pending_motion_trigger:
                            scheduler.mark_event_submitted(now_t)
                            pending_motion_trigger = False
                            pending_motion_evidence = None
                        next_submit_t = now_t + interval_s
                    else:
                        logger.log(now_t, "VLM_SKIP", "no keyframes available yet", channel="vision")
                        next_submit_t = now_t + min(2.0, scheduler.current_interval(now_t))

            item = asr_queue.get_nowait()
            if item:
                if tts_playing:
                    logger.log(now_t, "ASR_SUPPRESS", f"tts_playing | {item['text']}", channel="dialogue")
                    time.sleep(args.poll_s)
                    continue
                if pending_user_turn:
                    pending_user_turn["text"] = merge_user_utterances(pending_user_turn["text"], item["text"])
                    pending_user_turn["duration_s"] += float(item.get("duration_s") or 0.0)
                    pending_user_turn["last_time_s"] = now_t
                    logger.log(now_t, "USER_ASR_BUFFER", f"{pending_user_turn['text']}", channel="dialogue")
                    proactive_policy.on_user_turn(pending_user_turn["text"], now_t)
                else:
                    pending_user_turn = {
                        "text": item["text"],
                        "duration_s": float(item.get("duration_s") or 0.0),
                        "first_time_s": now_t,
                        "last_time_s": now_t,
                    }
                    logger.log(now_t, "USER_ASR_BUFFER", f"{item['text']} | speech={item['duration_s']:.2f}s", channel="dialogue")
                    proactive_policy.on_user_turn(str(item["text"]), now_t)

            # Process new ASR state first, so a ready proactive reply cannot
            # interrupt an utterance that arrived in this polling cycle.
            finish_dialogue_jobs(now_t)
            finish_translations(now_t)

            if pending_user_turn and now_t - pending_user_turn["last_time_s"] >= args.dialogue_turn_delay_s:
                user_text = pending_user_turn["text"]
                user_speech_duration = pending_user_turn["duration_s"]
                pending_user_turn = None
                logger.log(now_t, "USER_ASR", f"{user_text} | speech={user_speech_duration:.2f}s", channel="dialogue")
                proactive_policy.on_user_dialogue_turn(user_text, now_t)
                context = action_system.get_dialogue_context(
                    current_time_s=now_t,
                    max_events=args.max_visual_events,
                    max_age_s=args.visual_memory_age_s,
                )
                context = enrich_visual_context_timing(context, now_t, vlm_evidence_log)
                obs = context.get("current_visual_observation")
                visual_events = context.get("recent_visual_events", [])
                if obs:
                    logger.log(now_t, "MEMORY_READ", f"{obs.get('observation_short')} | scene_age={obs.get('scene_age_at_prompt_s', 0):.1f}s | memory_age={obs.get('memory_age_since_vlm_return_s', 0):.1f}s", channel="dialogue")
                else:
                    logger.log(now_t, "MEMORY_READ", "none", channel="dialogue")
                if visual_events:
                    logger.log(now_t, "VISUAL_MEMORY", f"{len(visual_events)} events", channel="dialogue")
                    for idx, event in enumerate(visual_events, 1):
                        age = event.get("scene_age_at_prompt_s", event.get("age_s"))
                        age_text = "unknown" if age is None else f"{age:.1f}s"
                        summary = event.get("observation_short") or event.get("description") or event.get("activity_summary") or ""
                        logger.log(now_t, "VISUAL_MEM", f"{idx}. age={age_text} | {summary}", channel="dialogue")
                else:
                    logger.log(now_t, "VISUAL_MEMORY", "0 events", channel="dialogue")

                if pending_user_job is None:
                    pending_user_job = dispatch_user_dialogue(now_t, user_text, user_speech_duration, context)
                else:
                    # Keep a second utterance for the next worker slot instead of
                    # freezing or silently losing it while cloud dialogue is slow.
                    queued_user_turn = {"text": user_text, "duration_s": user_speech_duration}
                    logger.log(now_t, "USER_QUEUE", "dialogue request still running", channel="dialogue")

                if camera_ready and action_system.pending_count() < args.vlm_max_inflight:
                    submitted = submit_vlm_observation(
                        action_system,
                        buffer,
                        keyframes_dir,
                        time.time() - start_wall,
                        args.window_s,
                        args.keyframes_normal,
                        args.max_image_width,
                        vlm_evidence_log,
                        schedule_state="after_user",
                        trigger_evidence={
                            "source": "after_user",
                            "reason": "A VLM observation was requested after the user's dialogue turn.",
                            "user_text": user_text,
                        },
                    )
                    if submitted is not None:
                        logger.log(time.time() - start_wall, "VLM_SUBMIT", f"{submitted.event_id} | after_user | keyframes={submitted.num_keyframes}", channel="vision")

            if pending_user_job is None and queued_user_turn is not None:
                queued = queued_user_turn
                queued_user_turn = None
                context = enrich_visual_context_timing(
                    action_system.get_dialogue_context(current_time_s=now_t, max_events=args.max_visual_events, max_age_s=args.visual_memory_age_s),
                    now_t, vlm_evidence_log,
                )
                pending_user_job = dispatch_user_dialogue(now_t, queued["text"], queued["duration_s"], context)

            # Give a deferred salient action priority over a silence timer.
            # The exact VLM event is restored as the primary observation so a
            # later neutral base frame cannot overwrite what the robot says.
            if args.enable_proactive and deferred_visual_proactive is not None:
                deferred_age = now_t - float(deferred_visual_proactive["queued_t"])
                if deferred_age > args.deferred_visual_proactive_max_age_s:
                    logger.log(now_t, "PROACTIVE_DEFER_DROP", f"visual_event stale:{deferred_age:.1f}s", channel="dialogue")
                    deferred_visual_proactive = None
                elif (
                    pending_user_turn is None
                    and pending_user_job is None
                    and pending_proactive_job is None
                    and now_t >= robot_output_busy_until
                    and not proactive_policy.state.asr_active
                    and not proactive_policy.state.tts_playing
                ):
                    context = action_system.get_dialogue_context(
                        current_time_s=now_t,
                        max_events=args.max_visual_events,
                        max_age_s=args.visual_memory_age_s,
                    )
                    context = enrich_visual_context_timing(context, now_t, vlm_evidence_log)
                    event_id = deferred_visual_proactive["event_id"]
                    preserved = next(
                        (event for event in context.get("recent_visual_events", []) if str(event.get("event_id") or "") == event_id),
                        None,
                    )
                    if preserved is None:
                        logger.log(now_t, "PROACTIVE_DEFER_DROP", "visual_event no longer in visual memory", channel="dialogue")
                        deferred_visual_proactive = None
                    else:
                        context["current_visual_observation"] = preserved
                        if try_proactive_reply(now_t, context, trigger_source="visual_event"):
                            logger.log(now_t, "PROACTIVE_REPLAY", f"visual_event | {deferred_visual_proactive['observation']}", channel="dialogue")
                            deferred_visual_proactive = None

            now_t = time.time() - start_wall
            if args.enable_proactive and now_t >= next_proactive_check_t:
                next_proactive_check_t += args.proactive_check_interval_s
                motion_preempts_timer = motion_episode is not None or pending_priority_evidence is not None or has_pending_motion_vlm(vlm_evidence_log) or (
                    pending_proactive_job is not None and pending_proactive_job.get("trigger_source") == "visual_event"
                ) or deferred_visual_proactive is not None
                if pending_user_turn or pending_user_job is not None or motion_preempts_timer:
                    if args.proactive_debug:
                        reason = "motion_episode_or_priority" if motion_preempts_timer else "pending_user_turn_or_job"
                        logger.log(now_t, "PROACTIVE_SKIP", reason, channel="dialogue")
                    time.sleep(args.poll_s)
                    continue
                context = action_system.get_dialogue_context(
                    current_time_s=now_t,
                    max_events=args.max_visual_events,
                    max_age_s=args.visual_memory_age_s,
                )
                context = enrich_visual_context_timing(context, now_t, vlm_evidence_log)
                try_proactive_reply(now_t, context, trigger_source="timer")

            time.sleep(args.poll_s)
    except KeyboardInterrupt:
        logger.log(time.time() - start_wall, "STOP", "keyboard interrupt", channel="system")
    finally:
        asr.stop()
        write_state(output_dir, dialogue_log, vlm_evidence_log)
        user_dialogue_executor.shutdown(wait=False, cancel_futures=True)
        proactive_dialogue_executor.shutdown(wait=False, cancel_futures=True)
        translation_executor.shutdown(wait=False, cancel_futures=True)
        priority_action_system.shutdown()
        action_system.shutdown()


def submit_vlm_observation(
    action_system: ActionEventSystem,
    buffer: RealTimeVisualBuffer,
    keyframes_dir: Path,
    now_t: float,
    window_s: float,
    num_frames: int,
    max_image_width: int,
    vlm_evidence_log: List[dict],
    schedule_state: str = "base",
    trigger_evidence: Optional[dict] = None,
):
    samples = buffer.sample_keyframes(now=now_t, window_length=window_s, num_frames=num_frames)
    frames = [resize_for_vlm(f.image, max_width=max_image_width) for f in samples]
    if not frames:
        return None
    submitted = action_system.submit_keyframes(frames, timestamp=now_t)
    keyframe_records = []
    for idx, (sample, image) in enumerate(zip(samples, frames)):
        filename = f"{submitted.event_id}_k{idx:02d}_submit{now_t:.1f}_src{sample.timestamp:.1f}.jpg"
        path = keyframes_dir / filename
        PILImage.fromarray(image).save(path, quality=90)
        keyframe_records.append(
            {
                "index": idx,
                "path": str(path.relative_to(keyframes_dir.parent)),
                "absolute_path": str(path),
                "sample_timestamp_s": sample.timestamp,
                "submit_time_s": now_t,
                "frame_id": sample.frame_id,
                "resized_shape": list(image.shape),
            }
        )
    vlm_evidence_log.append(
        {
            "event_id": submitted.event_id,
            "job_id": int(submitted.event_id.split("-")[-1]) if "-" in submitted.event_id else None,
            "status": "pending",
            "submitted_at_s": now_t,
            "schedule": schedule_state,
            "window_s": window_s,
            "requested_keyframes": num_frames,
            "num_keyframes": submitted.num_keyframes,
            "keyframes": keyframe_records,
            "trigger": trigger_evidence or {
                "source": schedule_state,
                "reason": "Periodic base VLM observation." if schedule_state == "base" else "",
            },
            "analysis_trace": {
                "why_uploaded": make_upload_reason(schedule_state, trigger_evidence),
                "trigger_frame": (trigger_evidence or {}).get("trigger_frame"),
                "vlm_input_keyframes": keyframe_records,
            },
            "vlm_result": {},
        }
    )
    return submitted


def make_upload_reason(schedule_state: str, trigger_evidence: Optional[dict]) -> str:
    if schedule_state.startswith("motion_event"):
        source = (trigger_evidence or {}).get("source", "motion_event")
        reason = (trigger_evidence or {}).get("reason", "")
        return f"Motion-triggered VLM upload from {source}: {reason}".strip()
    if schedule_state == "after_user":
        return "VLM upload after user dialogue turn."
    return "Periodic base VLM upload."


def save_trigger_frame(
    trigger_frames_dir: Path,
    image: Optional[np.ndarray],
    now_t: float,
    source: str,
) -> Optional[dict]:
    if image is None:
        return None
    safe_source = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(source)) or "trigger"
    path = trigger_frames_dir / f"{safe_source}_t{now_t:.1f}.jpg"
    PILImage.fromarray(image).save(path, quality=90)
    return {
        "path": str(path.relative_to(trigger_frames_dir.parent)),
        "absolute_path": str(path),
        "timestamp_s": now_t,
        "shape": list(image.shape),
    }


def build_schedule_evidence(
    schedule_event: dict,
    now_t: float,
    trigger_frame: Optional[dict] = None,
    frame_info: Optional[dict] = None,
    skeleton_motion: Optional[dict] = None,
    meaningful_features: Optional[dict] = None,
) -> dict:
    evidence = {
        "source": schedule_event.get("source", ""),
        "state": schedule_event.get("state", ""),
        "motion": schedule_event.get("motion"),
        "interval_s": schedule_event.get("interval_s"),
        "reason": schedule_event.get("reason", ""),
        "trigger_time_s": now_t,
        "trigger_frame": trigger_frame,
        "camera_frame": frame_info,
    }
    if skeleton_motion:
        evidence["skeleton_motion"] = {
            "motion_score": skeleton_motion.get("motion_score"),
            "part_speeds": skeleton_motion.get("part_speeds"),
            "demo_time_s": skeleton_motion.get("demo_time_s"),
            "wall_time": skeleton_motion.get("wall_time"),
        }
    if meaningful_features:
        evidence["meaningful_action_features"] = meaningful_features
    return evidence


def update_vlm_evidence_from_event(vlm_evidence_log: List[dict], event_update: dict) -> None:
    event_id = str(event_update.get("event_id") or "")
    if not event_id:
        return
    for record in reversed(vlm_evidence_log):
        if record.get("event_id") != event_id:
            continue
        record["status"] = event_update.get("status", record.get("status"))
        record["vlm_result"] = {
            "activity": event_update.get("activity"),
            "description": event_update.get("description"),
            "observation_short": event_update.get("observation_short"),
            "observation_raw": event_update.get("observation_raw"),
            "confidence": event_update.get("confidence"),
            "salience": event_update.get("salience"),
            "comment_style": event_update.get("comment_style"),
            "source": event_update.get("source"),
            "observed_at": event_update.get("observed_at"),
            "received_at": event_update.get("received_at"),
            "latency_s": event_update.get("latency_s"),
            "request_s": event_update.get("request_s"),
            "payload_kb": event_update.get("payload_kb"),
            "error": event_update.get("error"),
        }
        return


def build_visual_memory_used(visual_events: List[dict], vlm_evidence_log: List[dict]) -> List[dict]:
    memory = []
    for idx, event in enumerate(visual_events, 1):
        if not isinstance(event, dict):
            continue
        event_id = str(event.get("event_id") or "")
        vlm_record = find_vlm_evidence(vlm_evidence_log, event_id)
        result = (vlm_record or {}).get("vlm_result") or {}
        description = (
            event.get("observation_raw")
            or event.get("description")
            or result.get("description")
            or ""
        )
        memory.append(
            {
                "index": idx,
                "event_id": event_id,
                "description": description,
                "age_s": event.get("age_s"),
                "scene_age_at_prompt_s": event.get("scene_age_at_prompt_s"),
                "memory_age_since_vlm_return_s": event.get("memory_age_since_vlm_return_s"),
                "scene_observed_at_s": event.get("scene_observed_at_s"),
                "observed_at_s": event.get("observed_at"),
                "received_at_s": event.get("received_at"),
                "confidence": event.get("confidence"),
                "salience": event.get("salience"),
                "comment_style": event.get("comment_style"),
                "vlm_schedule": (vlm_record or {}).get("schedule"),
                "vlm_upload_trigger": ((vlm_record or {}).get("trigger") or {}),
                "trigger_frame": ((vlm_record or {}).get("analysis_trace") or {}).get("trigger_frame"),
                "vlm_input_keyframes": (vlm_record or {}).get("keyframes", []),
                "vlm_latency_s": result.get("latency_s"),
            }
        )
    return memory


def enrich_visual_context_timing(context: dict, prompt_time_s: float, vlm_evidence_log: List[dict]) -> dict:
    """Attach capture-based ages using the actual saved keyframe timestamps.

    `observed_at` in the action system is the VLM submission time.  The newest
    keyframe is the closest available proxy for when the physical scene occurred.
    """
    for event in list(context.get("recent_visual_events") or []) + ([context["current_visual_observation"]] if context.get("current_visual_observation") else []):
        if not isinstance(event, dict):
            continue
        record = find_vlm_evidence(vlm_evidence_log, str(event.get("event_id") or ""))
        frames = (record or {}).get("keyframes") or []
        samples = [float(frame.get("sample_timestamp_s")) for frame in frames if frame.get("sample_timestamp_s") is not None]
        scene_time = max(samples) if samples else float(event.get("observed_at") or event.get("timestamp") or prompt_time_s)
        event["scene_observed_at_s"] = scene_time
        event["scene_age_at_prompt_s"] = max(0.0, prompt_time_s - scene_time)
        event["memory_age_since_vlm_return_s"] = max(0.0, prompt_time_s - float(event.get("received_at") or event.get("timestamp") or prompt_time_s))
        event["age_s"] = event["scene_age_at_prompt_s"]
    return context


def find_vlm_evidence(vlm_evidence_log: List[dict], event_id: str) -> Optional[dict]:
    if not event_id:
        return None
    for record in reversed(vlm_evidence_log):
        if record.get("event_id") == event_id:
            return record
    return None


def has_pending_motion_vlm(vlm_evidence_log: List[dict]) -> bool:
    for record in reversed(vlm_evidence_log):
        status = str(record.get("status") or "")
        if status != "pending":
            continue
        if str(record.get("schedule") or "").startswith("motion_event"):
            return True
        trigger = record.get("trigger") or {}
        if str(trigger.get("source") or "") in {"meaningful_action", "frame_diff", "skeleton"}:
            return True
    return False


def is_priority_motion_trigger(trigger_evidence: Optional[dict], motion_threshold: float) -> bool:
    if not trigger_evidence:
        return False
    try:
        motion = float(trigger_evidence.get("motion") or 0.0)
    except (TypeError, ValueError):
        motion = 0.0
    source = str(trigger_evidence.get("source") or "")
    return source in {"meaningful_action", "frame_diff", "skeleton"} and motion >= float(motion_threshold)


def parse_robot_reply_payload(raw_text: str, default_style: str = "neutral") -> tuple[str, str]:
    text = str(raw_text or "").strip()
    style = normalize_response_style(default_style)
    if not text:
        return "", style
    payload = None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if 0 <= start < end:
            try:
                payload = json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                payload = None
    if isinstance(payload, dict):
        reply = str(payload.get("reply") or payload.get("text") or "").strip()
        parsed_style = normalize_response_style(payload.get("response_style") or payload.get("style") or style)
        return (reply or text, parsed_style)
    return text, style


def normalize_response_style(style: object) -> str:
    value = str(style or "neutral").strip().lower()
    if value not in {"neutral", "positive", "care", "sad", "angry"}:
        return "neutral"
    return value


def comment_style_from_context(context: dict) -> str:
    obs = context.get("current_visual_observation") or {}
    return str(obs.get("comment_style") or "neutral")


def style_from_comment_style(comment_style: str) -> str:
    return normalize_response_style(comment_style)


def read_recent_skeleton_motion(path: Path, max_age_s: float) -> Optional[dict]:
    try:
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        wall_time = float(payload.get("wall_time") or 0.0)
        if wall_time <= 0 or time.time() - wall_time > max_age_s:
            return None
        return payload
    except Exception:
        return None


def read_recent_tts_status(path: Path, max_age_s: float) -> Optional[dict]:
    try:
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        wall_time = float(payload.get("wall_time") or 0.0)
        if wall_time <= 0 or time.time() - wall_time > max_age_s:
            return None
        return payload
    except Exception:
        return None


def estimate_robot_speech_s(reply: str, minimum_s: float) -> float:
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+", str(reply or ""))
    # Haru's actual English TTS rate is about 2 words/s in the entity setup,
    # then its speaker/microphone path continues to leak tail audio after the
    # status topic reports STOPPED.  The previous 2.5 words/s + 0.8 s estimate
    # ended around 2.5 s too early for a 25-word reply (as the echo log showed).
    # Keep this adaptive by length, with a 2.5 s acoustic tail margin.
    estimated = len(words) / 2.0 + 2.5
    return max(float(minimum_s), min(24.0, estimated))


def fallback_dialogue_reply(output_mode: str) -> str:
    return "Sorry, my connection hiccuped. Could you say that again?"


def merge_user_utterances(previous: str, current: str) -> str:
    previous = str(previous or "").strip()
    current = str(current or "").strip()
    if not previous:
        return current
    if not current:
        return previous
    if current in previous:
        return previous
    if previous.endswith(("，", "。", "？", "！", ",", ".", "?", "!")):
        return previous + " " + current
    return previous + "，" + current


def write_state(
    output_dir: Path,
    dialogue_log: List[dict],
    vlm_evidence_log: Optional[List[dict]] = None,
) -> None:
    (output_dir / "dialogue_turns.json").write_text(json.dumps(dialogue_log, ensure_ascii=False, indent=2), encoding="utf-8")
    if vlm_evidence_log is not None:
        (output_dir / "vlm_frame_evidence.json").write_text(json.dumps(vlm_evidence_log, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
