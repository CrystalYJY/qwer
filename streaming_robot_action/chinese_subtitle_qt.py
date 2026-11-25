#!/usr/bin/env python3
"""Qt large-font Chinese subtitle window for robot reply translations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QFontDatabase
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget


class SubtitleWindow(QWidget):
    def __init__(self, dialogue_json: Path, poll_ms: int, font_size: int, topmost: bool) -> None:
        flags = Qt.Window | (Qt.WindowStaysOnTopHint if topmost else Qt.Widget)
        super().__init__(None, flags)
        self.dialogue_json = dialogue_json
        self.last_key = None
        self.setWindowTitle("机器人中文翻译")
        self.resize(940, 260)
        self.setMinimumSize(600, 180)
        self.setStyleSheet("background:#161616;")

        # QFontDatabase confirms whether Qt can actually resolve this family;
        # it avoids Tk's bitmap fallback that produced square glyphs.
        family = "Noto Sans CJK SC"
        if family not in QFontDatabase().families():
            family = "Droid Sans Fallback"
        title_font = QFont(family, 15, QFont.Bold)
        body_font = QFont(family, font_size)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(25, 13, 25, 15)
        self.header = QLabel("机器人回复 · 中文翻译")
        self.header.setFont(title_font)
        self.header.setStyleSheet("color:#FFC15A;")
        self.text = QLabel("等待机器人回复…")
        self.text.setFont(body_font)
        self.text.setWordWrap(True)
        self.text.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.text.setStyleSheet("color:#F5F5F5;")
        layout.addWidget(self.header)
        layout.addWidget(self.text, 1)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(max(100, poll_ms))
        self.poll()

    def poll(self) -> None:
        try:
            turns = json.loads(self.dialogue_json.read_text(encoding="utf-8"))
            if not isinstance(turns, list) or not turns:
                return
            latest = turns[-1] if isinstance(turns[-1], dict) else {}
            key = (latest.get("time_s"), latest.get("reply"), latest.get("translation_status"), latest.get("reply_zh"))
            if key == self.last_key:
                return
            self.last_key = key
            translated = str(latest.get("reply_zh") or "").strip()
            status = str(latest.get("translation_status") or "")
            if translated:
                self.text.setText(translated)
                self.text.setStyleSheet("color:#F5F5F5;")
            elif status == "pending":
                self.text.setText("正在翻译这句英文回复…")
                self.text.setStyleSheet("color:#B8B8B8;")
            elif status == "error":
                self.text.setText("翻译暂时失败（机器人英文回复未受影响）。")
                self.text.setStyleSheet("color:#FF9D9D;")
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Show Chinese robot translations in a Qt window.")
    parser.add_argument("--dialogue-json", required=True)
    parser.add_argument("--poll-ms", type=int, default=200)
    parser.add_argument("--font-size", type=int, default=28)
    parser.add_argument("--not-topmost", action="store_true")
    args = parser.parse_args()
    app = QApplication(sys.argv)
    window = SubtitleWindow(Path(args.dialogue_json), args.poll_ms, args.font_size, not args.not_topmost)
    window.show()
    raise SystemExit(app.exec_())


if __name__ == "__main__":
    main()
