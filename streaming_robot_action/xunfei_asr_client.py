#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import base64
import hashlib
import hmac
import importlib.util
import json
import os
import queue
import ssl
import struct
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import mktime
from typing import Callable, Optional
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time

import websocket


STATUS_FIRST_FRAME = 0
STATUS_CONTINUE_FRAME = 1


@dataclass
class XunfeiCredentials:
    appid: str
    api_key: str
    api_secret: str


def load_xunfei_credentials(config_path: str = "abc_eval_config.json", legacy_config: str = "") -> XunfeiCredentials:
    """Load credentials without hard-coding secrets into this project."""
    appid = os.getenv("XUNFEI_APPID", "").strip()
    api_key = os.getenv("XUNFEI_API_KEY", "").strip()
    api_secret = os.getenv("XUNFEI_API_SECRET", "").strip()

    path = Path(config_path)
    if path.exists():
        cfg = json.loads(path.read_text(encoding="utf-8"))
        appid = appid or str(cfg.get("xunfei_appid", "")).strip()
        api_key = api_key or str(cfg.get("xunfei_api_key", "")).strip()
        api_secret = api_secret or str(cfg.get("xunfei_api_secret", "")).strip()

    if legacy_config:
        legacy_path = Path(legacy_config)
        if legacy_path.exists():
            spec = importlib.util.spec_from_file_location("legacy_xunfei_config", legacy_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                appid = appid or str(getattr(module, "XUNFEI_APPID", "")).strip()
                api_key = api_key or str(getattr(module, "XUNFEI_API_KEY", "")).strip()
                api_secret = api_secret or str(getattr(module, "XUNFEI_API_SECRET", "")).strip()

    if not appid or not api_key or not api_secret:
        raise RuntimeError(
            "Xunfei credentials missing. Set xunfei_appid/xunfei_api_key/xunfei_api_secret "
            "in abc_eval_config.json, export XUNFEI_* env vars, or pass --xunfei-config."
        )
    return XunfeiCredentials(appid=appid, api_key=api_key, api_secret=api_secret)


class XunfeiStreamingASR:
    """Small streaming ASR wrapper based on the user's previous Xunfei implementation."""

    def __init__(
        self,
        credentials: XunfeiCredentials,
        mic_device: str = "plughw:2,0",
        on_final_text: Optional[Callable[[str, float], None]] = None,
        on_status: Optional[Callable[[str, str], None]] = None,
        verbose: bool = False,
        energy_threshold: float = 300.0,
        min_speech_frames: int = 3,
        max_silence_frames: int = 8,
        frame_ms: int = 40,
        connect_timeout_s: float = 8.0,
        final_wait_s: float = 1.2,
        min_recognition_s: float = 1.2,
        emit_partial_on_silence: bool = True,
        ignore_env_proxy: bool = False,
        level_meter: bool = False,
    ) -> None:
        self.credentials = credentials
        self.mic_device = mic_device
        self.on_final_text = on_final_text
        self.on_status = on_status
        self.verbose = verbose
        self.energy_threshold = float(energy_threshold)
        self.min_speech_frames = int(min_speech_frames)
        self.max_silence_frames = int(max_silence_frames)
        self.frame_ms = int(frame_ms)
        self.connect_timeout_s = float(connect_timeout_s)
        self.final_wait_s = float(final_wait_s)
        self.min_recognition_s = float(min_recognition_s)
        self.emit_partial_on_silence = bool(emit_partial_on_silence)
        self.ignore_env_proxy = bool(ignore_env_proxy)
        self.level_meter = level_meter

        self.current_text = ""
        self.last_final_text = ""
        self.last_speech_duration = 0.0
        self.speech_start_time: Optional[float] = None
        self.speech_end_time: Optional[float] = None
        self.recognition_start_wall: Optional[float] = None
        self.recognition_connected_wall: Optional[float] = None
        self.last_emitted_text = ""
        self.last_session_end_wall = 0.0
        self.suppress_until_wall = 0.0

        self.is_running = False
        self.is_recognizing = False
        self.frame_status = STATUS_FIRST_FRAME
        self.ws = None
        self.ws_connected = False
        self.ws_ready_event = threading.Event()
        self.final_result_event = threading.Event()
        self.session_id = 0
        self.arecord_process: Optional[subprocess.Popen] = None
        self.audio_buffer = []
        self.max_buffer_frames = max(15, int(5000 / max(10, self.frame_ms)))
        self.audio_lock = threading.Lock()
        self.restart_cooldown_s = 0.35

        self.common_args = {"app_id": credentials.appid}
        self.business_args = {
            "domain": "iat",
            "language": "zh_cn",
            "accent": "mandarin",
            "vinfo": 1,
            "vad_eos": 60000,
        }

    def start(self) -> None:
        if self.is_running:
            return
        self.is_running = True
        threading.Thread(target=self._audio_monitor_loop, daemon=True).start()

    def stop(self) -> None:
        self.is_running = False
        if self.arecord_process:
            self.arecord_process.terminate()
        if self.ws:
            self.ws.close()

    def suppress_for(self, seconds: float) -> None:
        """Temporarily ignore new speech triggers, e.g. while the robot is speaking."""
        self.suppress_until_wall = max(self.suppress_until_wall, time.time() + max(0.0, float(seconds)))
        self._reset_detection_state()

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)

    def _warn(self, message: str) -> None:
        print(message, flush=True)

    def _status(self, kind: str, message: str) -> None:
        if self.on_status:
            self.on_status(kind, message)

    def _create_url(self) -> str:
        host = "iat-api.xfyun.cn"
        path = "/v2/iat"
        date = format_date_time(mktime(datetime.now().timetuple()))
        signature_origin = f"host: {host}\ndate: {date}\nGET {path} HTTP/1.1"
        signature_sha = hmac.new(
            self.credentials.api_secret.encode("utf-8"),
            signature_origin.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).digest()
        signature = base64.b64encode(signature_sha).decode("utf-8")
        authorization_origin = (
            f'api_key="{self.credentials.api_key}", algorithm="hmac-sha256", '
            f'headers="host date request-line", signature="{signature}"'
        )
        authorization = base64.b64encode(authorization_origin.encode("utf-8")).decode("utf-8")
        return "wss://iat-api.xfyun.cn/v2/iat?" + urlencode(
            {"authorization": authorization, "date": date, "host": host}
        )

    def _audio_monitor_loop(self) -> None:
        frame_size = int(16000 * 2 * max(10, self.frame_ms) / 1000)
        interval = max(0.01, self.frame_ms / 1000.0)
        min_speech_frames = self.min_speech_frames
        max_silence_frames = self.max_silence_frames
        energy_threshold = self.energy_threshold
        silence_frames = 0
        speech_frames = 0
        non_speech_count = 0
        non_speech_tolerance = 5
        last_level_print = 0.0

        cmd = ["arecord", "-D", self.mic_device, "-f", "S16_LE", "-r", "16000", "-c", "1", "-t", "raw"]
        self.arecord_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        time.sleep(0.5)
        if self.arecord_process.poll() is not None:
            stderr = b""
            if self.arecord_process.stderr:
                stderr = self.arecord_process.stderr.read()
            self._warn(f"ASR arecord failed for {self.mic_device}: {stderr.decode('utf-8', errors='ignore').strip()}")
            return

        while self.is_running:
            try:
                buf = b""
                while len(buf) < frame_size:
                    chunk = self.arecord_process.stdout.read(frame_size - len(buf))
                    if not chunk:
                        break
                    buf += chunk
                if len(buf) < frame_size:
                    break

                samples = struct.unpack(f"<{len(buf) // 2}h", buf)
                energy = sum(abs(s) for s in samples) / len(samples)
                has_speech = energy > energy_threshold
                is_suppressed = time.time() < self.suppress_until_wall
                if (
                    self.is_recognizing
                    and not self.ws_connected
                    and self.recognition_start_wall is not None
                    and time.time() - self.recognition_start_wall > self.connect_timeout_s
                ):
                    self._warn("ASR websocket connection timeout")
                    self._status("ASR_TIMEOUT", "websocket connection timeout")
                    self._reset_detection_state()
                in_restart_cooldown = (
                    not self.is_recognizing
                    and self.last_session_end_wall > 0
                    and time.time() - self.last_session_end_wall < self.restart_cooldown_s
                )
                if self.level_meter and time.time() - last_level_print >= 0.5:
                    last_level_print = time.time()
                    level_message = f"energy={energy:.1f} threshold={energy_threshold:.1f} speech={has_speech}"
                    print(f"ASR_LEVEL {level_message}", flush=True)
                    self._status("ASR_LEVEL", level_message)

                if has_speech and not in_restart_cooldown and not is_suppressed:
                    speech_frames += 1
                    silence_frames = 0
                    non_speech_count = 0
                    if self.speech_start_time is None:
                        self.speech_start_time = time.time()
                    if not self.is_recognizing and speech_frames >= min_speech_frames:
                        self._log("ASR speech detected")
                        self._status("ASR_SPEECH", "speech detected")
                        if self._start_recognition():
                            self._buffer_audio_frame(buf)
                        else:
                            self._reset_detection_state()
                            speech_frames = 0
                            silence_frames = 0
                            non_speech_count = 0
                    else:
                        if not self.is_recognizing:
                            self._buffer_audio_frame(buf)
                        else:
                            self._send_audio_frame(buf)
                else:
                    silence_frames += 1
                    if not self.is_recognizing:
                        self._buffer_audio_frame(buf)
                        non_speech_count += 1
                        if non_speech_count > non_speech_tolerance:
                            speech_frames = 0
                            non_speech_count = 0
                    else:
                        self._send_audio_frame(buf)
                        if silence_frames == 1:
                            self.speech_end_time = time.time()
                        required_silence = max_silence_frames
                        if len(self.current_text) > 5:
                            required_silence = max(7, max_silence_frames // 2)
                        connected_age = (
                            time.time() - self.recognition_connected_wall
                            if self.recognition_connected_wall is not None
                            else 0.0
                        )
                        if silence_frames >= required_silence and connected_age >= self.min_recognition_s:
                            self._stop_recognition()
                            speech_frames = 0
                            silence_frames = 0
                            non_speech_count = 0

                time.sleep(interval)
            except Exception as exc:
                self._log(f"ASR monitor error: {exc!r}")
                break

    def _finalize_current_text(self) -> None:
        text = self.current_text.strip()
        if self.speech_start_time and self.speech_end_time:
            self.last_speech_duration = max(0.0, self.speech_end_time - self.speech_start_time)
        self.speech_start_time = None
        self.speech_end_time = None
        if text and text != self.last_emitted_text:
            self.last_final_text = text
            self.last_emitted_text = text
            self._log(f"ASR final text: {text}")
            self._status("ASR_TEXT", text)
            if self.on_final_text:
                self.on_final_text(text, self.last_speech_duration)
        else:
            self._log("ASR finalized with empty text")
            self._status("ASR_EMPTY", "finalized with empty text")
        self.current_text = ""
        self.last_session_end_wall = time.time()
        with self.audio_lock:
            self.audio_buffer.clear()

    def _reset_detection_state(self) -> None:
        self.speech_start_time = None
        self.speech_end_time = None
        self.recognition_start_wall = None
        self.recognition_connected_wall = None
        self.current_text = ""
        self.is_recognizing = False
        self.ws_connected = False
        with self.audio_lock:
            self.audio_buffer.clear()
        self.last_session_end_wall = time.time()

    def _buffer_audio_frame(self, audio_data: bytes) -> None:
        if not audio_data:
            return
        with self.audio_lock:
            self.audio_buffer.append(audio_data)
            self.audio_buffer = self.audio_buffer[-self.max_buffer_frames:]

    def _pop_audio_buffer(self) -> list:
        with self.audio_lock:
            frames = list(self.audio_buffer)
            self.audio_buffer.clear()
        return frames

    def _start_recognition(self) -> bool:
        self.current_text = ""
        self.is_recognizing = True
        self.frame_status = STATUS_FIRST_FRAME
        self.ws_connected = False
        self.ws_ready_event.clear()
        self.final_result_event.clear()
        self.recognition_start_wall = time.time()
        self.recognition_connected_wall = None
        self.session_id += 1
        session_id = self.session_id
        old_ws = self.ws
        if old_ws:
            try:
                old_ws.close()
            except Exception:
                pass

        def on_message(ws, message):
            if session_id != self.session_id:
                return
            try:
                data = json.loads(message)
                if data.get("code") != 0:
                    self._warn(f"ASR server error: {data}")
                    self._status("ASR_ERROR", str(data))
                    return
                result_data = data["data"]["result"]["ws"]
                text = ""
                for item in result_data:
                    for word in item["cw"]:
                        text += word["w"]
                if text:
                    self.current_text += text
                    if data["data"].get("status") == 2:
                        self.last_final_text = self.current_text
                        self.final_result_event.set()
            except Exception as exc:
                self._log(f"ASR parse error: {exc!r}")

        def on_error(ws, error):
            if session_id != self.session_id:
                return
            self._warn(f"ASR websocket error: {error!r}")
            self._status("ASR_ERROR", repr(error))

        def on_close(ws, *args):
            if session_id != self.session_id:
                return
            self.ws_connected = False
            self.final_result_event.set()
            self._log("ASR websocket closed")
            self._status("ASR_CLOSED", "websocket closed")

        def on_open(ws):
            if session_id != self.session_id or not self.is_recognizing:
                try:
                    ws.close()
                except Exception:
                    pass
                return
            self.ws_connected = True
            self.recognition_connected_wall = time.time()
            self._log("ASR websocket connected")
            self._status("ASR_CONNECTED", "websocket connected")
            self._send_first_frame()
            self.ws_ready_event.set()
            buffered_frames = self._pop_audio_buffer()
            self._status("ASR_BUFFER", f"flush_frames={len(buffered_frames)}")
            for buffered in buffered_frames:
                self._send_audio_frame(buffered)

        websocket.enableTrace(False)
        self.ws = websocket.WebSocketApp(
            self._create_url(),
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )
        self.ws.on_open = on_open
        threading.Thread(target=self._run_websocket_forever, args=(self.ws,), daemon=True).start()
        return True

    def _run_websocket_forever(self, ws: websocket.WebSocketApp) -> None:
        if not self.ignore_env_proxy:
            ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
            return

        proxy_keys = ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]
        old_values = {key: os.environ.get(key) for key in proxy_keys}
        try:
            for key in proxy_keys:
                os.environ.pop(key, None)
            ws.run_forever(
                sslopt={"cert_reqs": ssl.CERT_NONE},
                http_proxy_host=None,
                http_proxy_port=None,
                http_no_proxy=["*"],
            )
        finally:
            for key, value in old_values.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    def _stop_recognition(self) -> None:
        if not self.is_recognizing:
            return
        self.is_recognizing = False
        if self.emit_partial_on_silence and self.current_text.strip():
            self._finalize_current_text()
            self._send_last_frame()
        else:
            self._send_last_frame()
            self.final_result_event.wait(timeout=self.final_wait_s)
            self._finalize_current_text()

        def close_later() -> None:
            time.sleep(0.5)
            if self.ws:
                self.ws.close()

        threading.Thread(target=close_later, daemon=True).start()

    def _send_first_frame(self) -> None:
        if not self.ws_connected:
            return
        try:
            self.ws.send(json.dumps({
                "common": self.common_args,
                "business": self.business_args,
                "data": {"status": 0, "format": "audio/L16;rate=16000", "audio": "", "encoding": "raw"},
            }))
            self.frame_status = STATUS_CONTINUE_FRAME
        except Exception as exc:
            self.ws_connected = False
            self._warn(f"ASR send first frame failed: {exc!r}")

    def _send_audio_frame(self, audio_data: bytes) -> None:
        if not self.is_recognizing:
            return
        if not self.ws_connected:
            self._buffer_audio_frame(audio_data)
            return
        if self.frame_status == STATUS_FIRST_FRAME:
            self._send_first_frame()
        try:
            self.ws.send(json.dumps({
                "data": {
                    "status": 1,
                    "format": "audio/L16;rate=16000",
                    "audio": base64.b64encode(audio_data).decode("utf-8"),
                    "encoding": "raw",
                }
            }))
        except Exception as exc:
            self.ws_connected = False
            self._warn(f"ASR send audio frame failed: {exc!r}")

    def _send_last_frame(self) -> None:
        if not self.ws_connected:
            return
        try:
            self.ws.send(json.dumps({
                "data": {"status": 2, "format": "audio/L16;rate=16000", "audio": "", "encoding": "raw"}
            }))
        except Exception as exc:
            self.ws_connected = False
            self._warn(f"ASR send last frame failed: {exc!r}")


class ASRTextQueue:
    def __init__(self) -> None:
        self.queue: "queue.Queue[dict]" = queue.Queue()

    def callback(self, text: str, duration_s: float) -> None:
        self.queue.put({"text": text, "duration_s": duration_s, "received_wall": time.time()})

    def get_nowait(self) -> Optional[dict]:
        try:
            return self.queue.get_nowait()
        except queue.Empty:
            return None
