#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

from .dialogue_prompt_builder import build_dialogue_prompt


class DialogueLLMClient:
    def __init__(
        self,
        model: str = "doubao-seed-1-6-flash-250828",
        base_url: str = "https://ark.cn-beijing.volces.com/api/v3",
        config_path: str = "abc_eval_config.json",
        timeout_s: Optional[float] = 20.0,
        ignore_env_proxy: bool = False,
    ) -> None:
        cfg = {}
        path = Path(config_path)
        if path.exists():
            cfg = json.loads(path.read_text(encoding="utf-8"))
        api_key = str(cfg.get("api_key", "")).strip()
        if not api_key or "请替换" in api_key:
            api_key = str(cfg.get("api_key_env", "")).strip() or os.getenv("ARK_API_KEY", "")
        if not api_key:
            raise RuntimeError("API key missing. Set api_key in config or export ARK_API_KEY.")

        from openai import OpenAI

        kwargs = {"base_url": base_url, "api_key": api_key, "max_retries": 0}
        if timeout_s is not None and timeout_s > 0:
            kwargs["timeout"] = timeout_s
        if ignore_env_proxy:
            import httpx

            kwargs["http_client"] = httpx.Client(trust_env=False, timeout=timeout_s if timeout_s else None)
        self.client = OpenAI(**kwargs)
        self.model = model

    def reply(self, prompt: str, retries: int = 2) -> str:
        last_error = None
        for attempt in range(max(1, retries + 1)):
            try:
                resp = self.client.responses.create(
                    model=self.model,
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": prompt,
                                }
                            ],
                        }
                    ],
                    temperature=0.4,
                )
                return getattr(resp, "output_text", "") or ""
            except Exception as exc:
                last_error = exc
                if attempt >= retries:
                    break
                time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"Dialogue LLM request failed after retries: {last_error!r}") from last_error


def main() -> None:
    p = argparse.ArgumentParser(description="Generate a robot dialogue reply from action memory context.")
    p.add_argument("--context-json", required=True)
    p.add_argument("--user-utterance", required=True)
    p.add_argument("--current-time-s", type=float, default=None)
    p.add_argument("--model", default="doubao-seed-1-6-flash-250828")
    p.add_argument("--base-url", default="https://ark.cn-beijing.volces.com/api/v3")
    p.add_argument("--timeout-s", type=float, default=20.0)
    p.add_argument("--output", default="")
    args = p.parse_args()

    context = json.loads(Path(args.context_json).read_text(encoding="utf-8"))
    prompt = build_dialogue_prompt(context, args.user_utterance, args.current_time_s)
    client = DialogueLLMClient(model=args.model, base_url=args.base_url, timeout_s=args.timeout_s)
    reply = client.reply(prompt)
    if args.output:
        Path(args.output).write_text(reply, encoding="utf-8")
        print(f"Dialogue reply: {args.output}")
    else:
        print(reply)


if __name__ == "__main__":
    main()
