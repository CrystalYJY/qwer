#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import queue
import ssl
import struct
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Callable, Dict, Optional
from urllib.parse import urlencode, quote

import websocket

from .xunfei_asr_client import XunfeiCredentials


class XunfeiRTASRLLMStreaming:
    """Xunfei realtime transcription large-model ASR client.

    Official protocol summary:
    - WebSocket: wss://office-api-ast-dx.iflyaisol.com/ast/communicate/v1
    - Audio: 16 kHz, 16-bit, mono PCM, binary frames, recommended 40 ms / 1280 bytes.
    - Auth: sorted query params + HMAC-SHA1 with accessKeySecret.
    """

    def __init__(
        self,
        credentials: XunfeiCredentials,
        mic_device: str = "plughw:2,0",
        on_final_text: Optional[Callable[[str, float], None]] = None,
        on_status: Optional[Callable[[str, str], None]] = None,
        verbose: bool = False,
        energy_threshold: float = 300.0,
        min_speech_frames: int = 3,
        max_silence_frames: int = 9,
        frame_ms: int = 40,
        connect_timeout_s: float = 8.0,
        final_wait_s: float = 2.5,
        min_recognition_s: float = 1.2,
        emit_partial_on_silence: bool = True,
        ignore_env_proxy: bool = False,
        level_meter: bool = False,
        lang: str = "autodialect",
        vad_mode: int = 2,
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
        self.level_meter = bool(level_meter)
        self.lang = lang
        self.vad_mode = int(vad_mode)

        self.current_text = ""
        self.last_final_text = ""
        self.last_speech_duration = 0.0
        self.last_emitted_text = ""
        self.speech_start_time: Optional[float] = None
        self.speech_end_time: Optional[float] = None
        self.recognition_start_wall: Optional[float] = None
        self.recognition_connected_wall: Optional[float] = None
        self.last_session_end_wall = 0.0
        self.suppress_until_wall = 0.0

        self.is_running = False
        self.is_recognizing = False
        self.ws = None
        self.ws_connected = False
        self.ws_ready_event = threading.Event()
        self.final_result_event = threading.Event()
        self.session_id = ""
        self.local_session_counter = 0
        self.arecord_process: Optional[subprocess.Popen] = None
        self.audio_buffer = []
        self.max_buffer_frames = max(15, int(5000 / max(10, self.frame_ms)))
        self.audio_lock = threading.Lock()
        self.restart_cooldown_s = 0.35
        self.segments: Dict[int, str] = {}

    def start(self) -> None:
        if self.is_running:
            return
        self.is_running = True
        threading.Thread(target=self._audio_monitor_loop, daemon=True).start()

    def stop(self) -> None:
        self.is_running = False
        if self.arecord_process:
            self.arecord_process.terminate()
        ws = self.ws
        if ws:
            try:
                ws.close()
            except Exception:
                pass

    def suppress_for(self, seconds: float) -> None:
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
        # The new RTASR LLM API uses accessKeyId/accessKeySecret terminology.
        # Existing XUNFEI_API_KEY / XUNFEI_API_SECRET map to those fields.
        utc = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%dT%H:%M:%S%z")
        params = {
            "accessKeyId": self.credentials.api_key,
            "appId": self.credentials.appid,
            "uuid": self.session_id,
            "utc": utc,
            "audio_encode": "pcm_s16le",
            "lang": self.lang,
            "samplerate": "16000",
            "eng_vad_mdn": str(self.vad_mode),
        }
        base_parts = []
        for key in sorted(params):
            base_parts.append(f"{quote(str(key), safe='')}={quote(str(params[key]), safe='')}")
        base_string = "&".join(base_parts)
        digest = hmac.new(
            self.credentials.api_secret.encode("utf-8"),
            base_string.encode("utf-8"),
            hashlib.sha1,
        ).digest()
        params["signature"] = base64.b64encode(digest).decode("utf-8")
        return "wss://office-api-ast-dx.iflyaisol.com/ast/communicate/v1?" + urlencode(params)

    def _audio_monitor_loop(self) -> None:
        frame_size = int(16000 * 2 * max(10, self.frame_ms) / 1000)
        interval = max(0.01, self.frame_ms / 1000.0)
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
            self._warn(f"RTASR arecord failed for {self.mic_device}: {stderr.decode('utf-8', errors='ignore').strip()}")
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
                has_speech = energy > self.energy_threshold
                now = time.time()
                is_suppressed = now < self.suppress_until_wall
                if (
                    self.is_recognizing
                    and not self.ws_connected
                    and self.recognition_start_wall is not None
                    and now - self.recognition_start_wall > self.connect_timeout_s
                ):
                    self._warn("RTASR websocket connection timeout")
                    self._status("ASR_TIMEOUT", "rtasr websocket connection timeout")
                    self._reset_detection_state()
                in_restart_cooldown = (
                    not self.is_recognizing
                    and self.last_session_end_wall > 0
                    and now - self.last_session_end_wall < self.restart_cooldown_s
                )
                if self.level_meter and now - last_level_print >= 0.5:
                    last_level_print = now
                    msg = f"energy={energy:.1f} threshold={self.energy_threshold:.1f} speech={has_speech} engine=rtasr_llm"
                    print(f"ASR_LEVEL {msg}", flush=True)
                    self._status("ASR_LEVEL", msg)

                if has_speech and not in_restart_cooldown and not is_suppressed:
                    speech_frames += 1
                    silence_frames = 0
                    non_speech_count = 0
                    if self.speech_start_time is None:
                        self.speech_start_time = now
                    if not self.is_recognizing and speech_frames >= self.min_speech_frames:
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
                            self.speech_end_time = now
                        # Do not shorten the end-of-turn silence merely because
                        # text has already arrived.  The old behaviour reduced a
                        # 9-frame setting to 7 frames (280 ms), which cuts a
                        # natural mid-sentence pause into a second ASR session.
                        required_silence = self.max_silence_frames
                        connected_age = now - self.recognition_connected_wall if self.recognition_connected_wall else 0.0
                        if silence_frames >= required_silence and connected_age >= self.min_recognition_s:
                            self._stop_recognition()
                            speech_frames = 0
                            silence_frames = 0
                            non_speech_count = 0

                time.sleep(interval)
            except Exception as exc:
                self._log(f"RTASR monitor error: {exc!r}")
                break

    def _start_recognition(self) -> bool:
        self.current_text = ""
        self.segments.clear()
        self.is_recognizing = True
        self.ws_connected = False
        self.ws_ready_event.clear()
        self.final_result_event.clear()
        self.recognition_start_wall = time.time()
        self.recognition_connected_wall = None
        self.session_id = str(uuid.uuid4())
        self.local_session_counter += 1
        local_counter = self.local_session_counter
        old_ws = self.ws
        if old_ws:
            try:
                old_ws.close()
            except Exception:
                pass

        def on_message(ws, message):
            if local_counter != self.local_session_counter:
                return
            try:
                data = json.loads(message)
                msg_type = data.get("msg_type") or data.get("action")
                if msg_type in {"started", "handshake"}:
                    self._status("ASR_STARTED", data.get("sid") or self.session_id)
                    return
                if msg_type == "error" or data.get("code") not in {None, "0", 0}:
                    self._warn(f"RTASR server error: {data}")
                    self._status("ASR_ERROR", str(data))
                    return
                if data.get("msg_type") == "result" and data.get("res_type") not in {None, "asr"}:
                    return
                text = self._extract_text(data)
                if text:
                    seg_key = self._extract_segment_key(data)
                    self.segments[int(seg_key)] = text
                    self.current_text = self._clean_incremental_repeats(
                        "".join(self.segments[k] for k in sorted(self.segments))
                    )
                    self._status("ASR_PARTIAL", self.current_text)
                    if self._is_last_result(data):
                        self.final_result_event.set()
            except Exception as exc:
                self._log(f"RTASR parse error: {exc!r}")

        def on_error(ws, error):
            if local_counter != self.local_session_counter:
                return
            self.ws_connected = False
            self._warn(f"RTASR websocket error: {error!r}")
            self._status("ASR_ERROR", repr(error))

        def on_close(ws, *args):
            if local_counter != self.local_session_counter:
                return
            self.ws_connected = False
            self.final_result_event.set()
            close_code = args[0] if len(args) > 0 else None
            close_reason = args[1] if len(args) > 1 else ""
            self._status("ASR_CLOSED", f"rtasr websocket closed | code={close_code} reason={close_reason}")

        def on_open(ws):
            if local_counter != self.local_session_counter or not self.is_recognizing:
                try:
                    ws.close()
                except Exception:
                    pass
                return
            self.ws_connected = True
            self.recognition_connected_wall = time.time()
            self._status("ASR_CONNECTED", "rtasr websocket connected")
            buffered_frames = self._pop_audio_buffer()
            self._status("ASR_BUFFER", f"flush_frames={len(buffered_frames)} engine=rtasr_llm")
            for buffered in buffered_frames:
                self._send_audio_frame(buffered)
                if not self.ws_connected:
                    break

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
        self._send_end_frame()
        self.final_result_event.wait(timeout=self.final_wait_s)
        self._finalize_current_text()

        # Bind this delayed close to the exact websocket/session that just
        # ended. Previously it used self.ws after the delay; a new utterance
        # could replace self.ws in 350 ms and the old timer then closed the new
        # connection, producing websocket-client's NoneType.sock error.
        closing_ws = self.ws
        closing_counter = self.local_session_counter

        def close_later() -> None:
            time.sleep(0.5)
            if closing_counter != self.local_session_counter or closing_ws is None:
                return
            try:
                closing_ws.close()
            except Exception:
                pass

        threading.Thread(target=close_later, daemon=True).start()

    def _finalize_current_text(self) -> None:
        text = self.current_text.strip()
        if self.speech_start_time and self.speech_end_time:
            self.last_speech_duration = max(0.0, self.speech_end_time - self.speech_start_time)
        self.speech_start_time = None
        self.speech_end_time = None
        if text and text != self.last_emitted_text:
            self.last_final_text = text
            self.last_emitted_text = text
            self._status("ASR_TEXT", text)
            if self.on_final_text:
                self.on_final_text(text, self.last_speech_duration)
        else:
            self._status("ASR_EMPTY", "rtasr finalized with empty text")
        self.current_text = ""
        self.segments.clear()
        self.last_session_end_wall = time.time()
        with self.audio_lock:
            self.audio_buffer.clear()

    def _reset_detection_state(self) -> None:
        self.speech_start_time = None
        self.speech_end_time = None
        self.recognition_start_wall = None
        self.recognition_connected_wall = None
        self.current_text = ""
        self.segments.clear()
        self.is_recognizing = False
        self.ws_connected = False
        with self.audio_lock:
            self.audio_buffer.clear()
        self.last_session_end_wall = time.time()

    def _is_ws_open(self) -> bool:
        ws = self.ws
        sock = getattr(ws, "sock", None) if ws is not None else None
        return bool(self.ws_connected and ws is not None and sock is not None and getattr(sock, "connected", False))

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

    def _send_audio_frame(self, audio_data: bytes) -> None:
        if not self.is_recognizing:
            return
        if not self._is_ws_open():
            self.ws_connected = False
            self._buffer_audio_frame(audio_data)
            return
        try:
            self.ws.send(audio_data, opcode=websocket.ABNF.OPCODE_BINARY)
        except Exception as exc:
            self.ws_connected = False
            self._warn(f"RTASR send audio frame failed: {exc!r}")

    def _send_end_frame(self) -> None:
        if not self._is_ws_open():
            self.ws_connected = False
            return
        try:
            self.ws.send(json.dumps({"end": True, "sessionId": self.session_id}))
        except Exception as exc:
            self.ws_connected = False
            self._warn(f"RTASR send end frame failed: {exc!r}")

    def _extract_text(self, data: Dict) -> str:
        pieces = []
        st = (((data.get("data") or {}).get("cn") or {}).get("st") or {})
        for rt in st.get("rt") or []:
            for ws_item in rt.get("ws") or []:
                for cw in ws_item.get("cw") or []:
                    word = cw.get("w")
                    if word:
                        pieces.append(str(word))
        return "".join(pieces)

    def _clean_incremental_repeats(self, text: str) -> str:
        text = str(text or "").strip()
        if len(text) < 8:
            return text
        # RTASR partial results are often cumulative. If several cumulative
        # partials were accidentally concatenated, the prefix can be segmented
        # into repeated prefixes of the final suffix:
        #   感觉现在 + 感觉现在这个 + 感觉现在这个反应还挺快的
        # -> 感觉现在这个反应还挺快的
        candidates = []
        for i in range(1, len(text)):
            suffix = text[i:].strip()
            prefix = text[:i].strip()
            if len(suffix) < 4:
                continue
            if self._is_incremental_prefix_chain(prefix, suffix):
                candidates.append(suffix)
            elif self._is_incremental_prefix_chain_with_leading_noise(prefix, suffix):
                candidates.append(suffix)
        if candidates:
            return min(candidates, key=len)
        return text

    def _is_incremental_prefix_chain_with_leading_noise(self, prefix: str, suffix: str) -> bool:
        prefix_norm = self._normalize_for_repeat_check(prefix)
        suffix_norm = self._normalize_for_repeat_check(suffix)
        if len(prefix_norm) < 8 or len(suffix_norm) < 6:
            return False
        max_noise_chars = min(8, max(0, len(prefix_norm) - 4))
        for noise_chars in range(1, max_noise_chars + 1):
            if self._is_incremental_prefix_chain(prefix_norm[noise_chars:], suffix_norm):
                return True
        return False

    def _is_incremental_prefix_chain(self, prefix: str, suffix: str) -> bool:
        prefix_norm = self._normalize_for_repeat_check(prefix)
        suffix_norm = self._normalize_for_repeat_check(suffix)
        if len(prefix_norm) < 2 or len(suffix_norm) < 4:
            return False
        chunks = 0
        rest = prefix_norm
        while rest:
            matched = ""
            for n in range(min(len(suffix_norm), len(rest)), 1, -1):
                candidate = suffix_norm[:n]
                if rest.startswith(candidate):
                    matched = candidate
                    break
            if not matched:
                return False
            chunks += 1
            rest = rest[len(matched):]
        return chunks >= 1

    def _normalize_for_repeat_check(self, text: str) -> str:
        drop = set("，。？！?,.!;；:：、 \t\n\r")
        return "".join(ch for ch in str(text or "") if ch not in drop)

    def _extract_segment_key(self, data: Dict) -> int:
        seg_id = self._extract_seg_id(data)
        if seg_id is not None:
            return seg_id
        st = (((data.get("data") or {}).get("cn") or {}).get("st") or {})
        try:
            return int(st.get("bg"))
        except Exception:
            return 0

    def _extract_seg_id(self, data: Dict) -> Optional[int]:
        try:
            seg = (data.get("data") or {}).get("seg_id")
            return None if seg is None else int(seg)
        except Exception:
            return None

    def _is_last_result(self, data: Dict) -> bool:
        return bool((data.get("data") or {}).get("ls"))


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
