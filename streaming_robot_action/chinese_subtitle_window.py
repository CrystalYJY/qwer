#!/usr/bin/env python3
"""Large local window showing the newest Chinese robot-reply translation."""

from __future__ import annotations

import argparse
import json
import tkinter as tk
from pathlib import Path


class ChineseSubtitleWindow:
    def __init__(self, dialogue_json: Path, poll_ms: int, font_size: int, topmost: bool) -> None:
        self.dialogue_json = dialogue_json
        self.poll_ms = max(100, poll_ms)
        self.last_key = None

        self.root = tk.Tk()
        self.root.title("机器人中文翻译")
        self.root.configure(bg="#161616")
        self.root.geometry("940x260+80+80")
        self.root.minsize(600, 180)
        self.root.attributes("-topmost", bool(topmost))
        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)

        # This font family is installed system-wide and contains the complete
        # Simplified-Chinese glyph set, including full-width punctuation.
        self.cjk_font = ("Droid Sans Fallback", font_size)
        self.header_font = ("Droid Sans Fallback", 15, "bold")
        header = tk.Label(
            self.root, text="机器人回复 · 中文翻译", anchor="w",
            fg="#FFC15A", bg="#161616", font=self.header_font, padx=22, pady=13,
        )
        header.pack(fill="x")
        self.text = tk.Label(
            self.root, text="等待机器人回复…", justify="left", anchor="nw", wraplength=880,
            fg="#F5F5F5", bg="#161616", font=self.cjk_font, padx=25, pady=15,
        )
        self.text.pack(fill="both", expand=True)
        self.root.bind("<Escape>", lambda _event: self.root.destroy())

    def start(self) -> None:
        self._poll()
        self.root.mainloop()

    def _poll(self) -> None:
        try:
            turns = json.loads(self.dialogue_json.read_text(encoding="utf-8"))
            if isinstance(turns, list) and turns:
                latest = turns[-1] if isinstance(turns[-1], dict) else {}
                key = (latest.get("time_s"), latest.get("reply"), latest.get("translation_status"), latest.get("reply_zh"))
                if key != self.last_key:
                    self.last_key = key
                    status = str(latest.get("translation_status") or "")
                    translated = str(latest.get("reply_zh") or "").strip()
                    if translated:
                        self.text.configure(text=translated, fg="#F5F5F5")
                    elif status == "pending":
                        self.text.configure(text="正在翻译这句英文回复…", fg="#B8B8B8")
                    elif status == "error":
                        self.text.configure(text="翻译暂时失败（机器人英文回复未受影响）。", fg="#FF9D9D")
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass
        if self.root.winfo_exists():
            self.root.after(self.poll_ms, self._poll)


def main() -> None:
    parser = argparse.ArgumentParser(description="Display latest robot Chinese translation in a large local window.")
    parser.add_argument("--dialogue-json", required=True)
    parser.add_argument("--poll-ms", type=int, default=200)
    parser.add_argument("--font-size", type=int, default=28)
    parser.add_argument("--not-topmost", action="store_true")
    args = parser.parse_args()
    ChineseSubtitleWindow(Path(args.dialogue_json), args.poll_ms, args.font_size, not args.not_topmost).start()


if __name__ == "__main__":
    main()
