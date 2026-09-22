"""Optional run traces. Nothing is written unless TRACE_DIR is set."""

import json
import os
import secrets
import time
from pathlib import Path


class Trace:
    def __init__(self, url, goal):
        directory = os.environ.get("TRACE_DIR")
        self.directory = Path(directory) if directory else None
        self.run_id = f"{time.strftime('%Y%m%dT%H%M%S')}-{secrets.token_hex(2)}"
        self.url, self.goal = url, goal
        self.steps = 0
        self.pending = None

    def step(self, goal, request, result, latency_ms, *, awaiting_text=False):
        self.flush()
        self.steps += 1
        self.pending = {
            "step": self.steps,
            "goal": goal,
            "request": request,
            "answers": result["answers"],
            "latency_ms": latency_ms,
            "usage": result.get("usage"),
        }
        if not awaiting_text:
            self.flush()

    def type_text(self, context, text):
        if self.pending:
            self.pending["type_text"] = {"context": context, "text": text}
        self.flush()

    def flush(self):
        record, self.pending = self.pending, None
        if record and self.directory:
            self.directory.mkdir(parents=True, exist_ok=True)
            with open(self.directory / f"{self.run_id}.jsonl", "a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def finish(self, status, elapsed_ms):
        self.flush()
        if self.directory:
            self.directory.mkdir(parents=True, exist_ok=True)
            meta = {"goal": self.goal, "url": self.url, "status": status, "steps": self.steps, "elapsed_ms": elapsed_ms}
            (self.directory / f"{self.run_id}.meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )
