#!/usr/bin/env python3
"""Small, dependency-free Baidu Text Translation client (English -> Chinese)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import time
import urllib.error
import urllib.parse
import urllib.request


class BaiduTranslateClient:
    endpoint = "https://fanyi-api.baidu.com/api/trans/vip/translate"

    def __init__(self, appid: str = "", appkey: str = "", timeout_s: float = 3.0) -> None:
        self.appid = str(appid or os.getenv("BAIDU_TRANSLATE_APPID", "")).strip()
        self.appkey = str(appkey or os.getenv("BAIDU_TRANSLATE_APPKEY", "")).strip()
        self.timeout_s = max(0.1, float(timeout_s))
        if not self.appid or not self.appkey:
            raise RuntimeError("Baidu translation credentials missing: set BAIDU_TRANSLATE_APPID and BAIDU_TRANSLATE_APPKEY.")

    def translate_en_to_zh(self, text: str) -> tuple[str, float]:
        text = str(text or "").strip()
        if not text:
            return "", 0.0
        salt = secrets.token_hex(8)
        sign = hashlib.md5(f"{self.appid}{text}{salt}{self.appkey}".encode("utf-8")).hexdigest()
        payload = urllib.parse.urlencode({
            "appid": self.appid, "q": text, "from": "en", "to": "zh", "salt": salt, "sign": sign,
        }).encode("utf-8")
        request = urllib.request.Request(self.endpoint, data=payload, headers={"Content-Type": "application/x-www-form-urlencoded"})
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Baidu translation network error: {exc.reason}") from exc
        latency_ms = (time.perf_counter() - started) * 1000.0
        try:
            result = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Baidu translation returned invalid JSON.") from exc
        if result.get("error_code"):
            raise RuntimeError(f"Baidu translation error {result['error_code']}: {result.get('error_msg', 'unknown error')}")
        translated = "\n".join(str(item.get("dst") or "") for item in result.get("trans_result") or []).strip()
        if not translated:
            raise RuntimeError("Baidu translation returned no translated text.")
        return translated, latency_ms


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure Baidu English-to-Chinese translation latency.")
    parser.add_argument("text", nargs="?", default="That sounds great. Let's try a gentle stretch together.")
    parser.add_argument("--timeout-s", type=float, default=3.0)
    args = parser.parse_args()
    translated, latency_ms = BaiduTranslateClient(timeout_s=args.timeout_s).translate_en_to_zh(args.text)
    print(json.dumps({"source_en": args.text, "reply_zh": translated, "translation_latency_ms": round(latency_ms, 1)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
