#豆包 VLM 多帧动作理解客户端
from __future__ import annotations

import base64
from io import BytesIO
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from .action_memory import ActionEvent


def build_observer_prompt() -> str:
    return (
        "You are the visual observer for a social desktop robot.\n"
        "Describe only visible user action across these time-ordered frames: posture, body/hand motion, and object interaction.\n"
        "Do not infer hidden intent or emotion.\n"
        "Rate salience for whether the robot may naturally comment: 0=static/routine, 1=minor motion, 2=noticeable action, 3=strong or socially directed action.\n"
        "Choose comment_style from: neutral, positive, care, sad, angry. Use neutral when unsure.\n"
        'Return JSON only: {"observation":"","confidence":0.0,"salience":0,"comment_style":"neutral"}'
    )


def image_to_data_url(image: np.ndarray) -> str:
    buf = BytesIO()
    Image.fromarray(image).save(buf, format="JPEG", quality=88)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def parse_action_event(text: str, timestamp: float) -> ActionEvent:
    raw = text.strip()
    match = re.search(r"\{.*\}", raw, flags=re.S)
    if match:
        raw = match.group(0)
    try:
        obj = json.loads(raw)
    except Exception:
        return ActionEvent(activity="unknown", timestamp=timestamp, confidence=0.0, description=text)

    observation = str(obj.get("observation") or obj.get("description") or "").strip()
    activity = observation or "unknown"
    try:
        confidence = float(obj.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    desc = observation
    try:
        salience = int(float(obj.get("salience", 0)))
    except (TypeError, ValueError):
        salience = 0
    salience = max(0, min(3, salience))
    comment_style = normalize_comment_style(obj.get("comment_style") or obj.get("response_style") or "neutral")
    return ActionEvent(
        activity=activity,
        timestamp=timestamp,
        confidence=confidence,
        description=desc,
        salience=salience,
        comment_style=comment_style,
    )


def normalize_comment_style(style: object) -> str:
    value = str(style or "neutral").strip().lower()
    if value not in {"neutral", "positive", "care", "sad", "angry"}:
        return "neutral"
    return value


class DoubaoFrameVLMClient:
    def __init__(
        self,
        config_path: str = "abc_eval_config.json",
        model: Optional[str] = None,
        timeout_s: Optional[float] = None,
        base_url: Optional[str] = None,
        api_key_env: Optional[str] = None,
        api_provider: Optional[str] = None,
        ignore_env_proxy: bool = False,
    ) -> None:
        cfg = {}
        path = Path(config_path)
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                cfg = json.load(f)
        self.model = str(model or cfg.get("model", "doubao-seed-2-0-pro-260215"))
        default_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1" if self.model.startswith("qwen") else "https://ark.cn-beijing.volces.com/api/v3"
        self.base_url = str(base_url or cfg.get("base_url", default_base_url))
        if api_provider:
            self.api_provider = api_provider
        elif "dashscope" in self.base_url or self.model.startswith("qwen"):
            self.api_provider = "dashscope"
        else:
            self.api_provider = "ark"
        self.timeout_s = timeout_s
        key_env = api_key_env or ("DASHSCOPE_API_KEY" if self.api_provider == "dashscope" else "ARK_API_KEY")
        api_key = str(cfg.get("api_key", "")).strip()
        if self.api_provider == "dashscope":
            api_key = str(cfg.get("dashscope_api_key", "")).strip()
        if not api_key or "请替换" in api_key:
            api_key = (
                str(cfg.get("dashscope_api_key_env", "")).strip()
                if self.api_provider == "dashscope"
                else str(cfg.get("api_key_env", "")).strip()
            ) or os.getenv(key_env, "")
        if not api_key:
            raise RuntimeError(f"API key missing. Set it in config or export {key_env}.")
        from openai import OpenAI

        kwargs = {"base_url": self.base_url, "api_key": api_key, "max_retries": 0}
        if timeout_s is not None and timeout_s > 0:
            kwargs["timeout"] = timeout_s
        if ignore_env_proxy:
            import httpx

            kwargs["http_client"] = httpx.Client(trust_env=False, timeout=timeout_s if timeout_s else None)
        self.client = OpenAI(**kwargs)
        self.last_timing: Dict[str, float] = {}

    def describe_frames(self, frames: List[np.ndarray], timestamp: float) -> ActionEvent:
        total_start = time.time()
        encode_start = time.time()
        prompt = build_observer_prompt()
        text_type = "text" if self.api_provider == "dashscope" else "input_text"
        image_type = "image_url" if self.api_provider == "dashscope" else "input_image"
        content = [{"type": text_type, "text": prompt}]
        payload_chars = len(content[0]["text"])
        for image in frames:
            data_url = image_to_data_url(image)
            payload_chars += len(data_url)
            if self.api_provider == "dashscope":
                content.insert(-1, {"type": image_type, "image_url": {"url": data_url}})
            else:
                content.append({"type": image_type, "image_url": data_url})
        encode_s = time.time() - encode_start

        request_start = time.time()
        if self.api_provider == "dashscope":
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                temperature=0,
            )
        else:
            resp = self.client.responses.create(
                model=self.model,
                input=[{"role": "user", "content": content}],
                temperature=0,
            )
        request_s = time.time() - request_start

        parse_start = time.time()
        if self.api_provider == "dashscope":
            text = resp.choices[0].message.content or ""
        else:
            text = getattr(resp, "output_text", "") or ""
        event = parse_action_event(text, timestamp=timestamp)
        parse_s = time.time() - parse_start
        self.last_timing = {
            "encode_s": encode_s,
            "request_s": request_s,
            "parse_s": parse_s,
            "total_s": time.time() - total_start,
            "payload_kb": payload_chars / 1024.0,
            "num_frames": float(len(frames)),
            "model": self.model,
            "api_provider": self.api_provider,
        }
        return event
