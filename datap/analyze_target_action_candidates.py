#!/usr/bin/env python3
"""Extract possible A/B/C/D target actions from the five experiment logs."""

from __future__ import annotations

import json
import re
from pathlib import Path


DATA_ROOT = Path(r"E:\qwer\new")
PATTERN = re.compile(
    r"screen|computer|laptop|paper|document|write|writing|keyboard|mouse|"
    r"desk|table|cup|bottle|drink|phone|pen|object|stretch|arm|shoulder|"
    r"neck|lean|chair|stand|turn|look|search|find|organ|arrang|file|read|book",
    re.IGNORECASE,
)


def main() -> None:
    for run_dir in sorted(path for path in DATA_ROOT.iterdir() if path.is_dir()):
        evidence_path = run_dir / "camera_asr_dialogue" / "vlm_frame_evidence.json"
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        print(f"\n### {run_dir.name}")
        for event in evidence:
            description = (event.get("vlm_result") or {}).get("description", "")
            if PATTERN.search(description):
                print(
                    event["event_id"],
                    f'{event.get("submitted_at_s", 0):.1f}s',
                    event.get("schedule", ""),
                    ":",
                    description,
                )


if __name__ == "__main__":
    main()
