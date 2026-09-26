"""Optional run traces. Nothing is written unless TRACE_DIR is set."""

import json
import os
import secrets
import statistics
import time
from contextlib import contextmanager
from pathlib import Path

PHASES = ("snapshot", "model", "execute", "wait")


class Trace:
    def __init__(self, url, goal):
        directory = os.environ.get("TRACE_DIR")
        self.directory = Path(directory) if directory else None
        self.run_id = f"{time.strftime('%Y%m%dT%H%M%S')}-{secrets.token_hex(2)}"
        self.url, self.goal = url, goal
        self.steps = 0
        self.pending = None
        self.timing = None
        self.phase = None
        self.verified = None
        self.meta = None
        self.latencies = []
        self.input_tokens = None

    def start_step(self, t_ms, url):
        self.end_step()
        self.timing, self.phase = {}, None
        self.write({"event": "step_start", "step": self.steps + 1, "t_ms": t_ms, "url": url})

    @contextmanager
    def timed(self, phase):
        started = time.perf_counter()
        self.phase = phase
        try:
            yield
            self.phase = None
        finally:
            if self.timing is not None:
                self.timing[phase] = self.timing.get(phase, 0) + time.perf_counter() - started

    def end_step(self, error=None):
        if self.timing is None:
            return
        timing, self.timing = self.timing, None
        if error is None:
            timing = {**dict.fromkeys(PHASES, 0), **timing}
        fields = {f"{phase}_ms": round(seconds * 1000) for phase, seconds in timing.items()}
        if error is not None:
            fields.update(failed_phase=self.phase, error=str(error))
        if self.pending:
            self.pending.update(event="step", **fields)
            self.flush()
        elif error is not None:
            self.write({"event": "step_failed", "step": self.steps + 1, **fields})

    def step(self, goal, request, result, latency_ms, *, awaiting_text=False, retries=None):
        self.flush()
        self.steps += 1
        self.latencies.append(latency_ms)
        tokens = (result.get("usage") or {}).get("input_tokens")
        if isinstance(tokens, int):
            self.input_tokens = (self.input_tokens or 0) + tokens
        self.pending = {
            "step": self.steps,
            "goal": goal,
            "request": request,
            "answers": result["answers"],
            "latency_ms": latency_ms,
            "usage": result.get("usage"),
        }
        if retries:
            self.pending["retries"] = retries
        if not awaiting_text and self.timing is None:
            self.flush()

    def type_text(self, context, text):
        if self.pending:
            self.pending["type_text"] = {"context": context, "text": text}
        if self.timing is None:
            self.flush()

    def flush(self):
        record, self.pending = self.pending, None
        if record:
            self.write(record)

    def write(self, record):
        if self.directory:
            self.directory.mkdir(parents=True, exist_ok=True)
            with open(self.directory / f"{self.run_id}.jsonl", "a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def stats(self):
        return {
            "jev_latency_p50_ms": round(statistics.median(self.latencies)) if self.latencies else None,
            "jev_latency_max_ms": max(self.latencies, default=None),
            "input_tokens_total": self.input_tokens,
        }

    @property
    def finished(self):
        return self.meta is not None

    def finish(self, status, elapsed_ms, error=None, raw_response=None, reason=None):
        self.end_step()
        self.flush()
        if self.finished:
            return
        self.meta = {
            "goal": self.goal,
            "url": self.url,
            "status": status,
            "steps": self.steps,
            "elapsed_ms": elapsed_ms,
            "verified": self.verified,
        }
        if error is not None:
            self.meta["error"] = error
        if raw_response is not None:
            self.meta["raw_response"] = raw_response
        if reason is not None:
            self.meta["reason"] = reason
        self.write_meta()

    def set_verified(self, value):
        self.verified = None if value is None else bool(value)
        if self.finished:
            self.meta["verified"] = self.verified
            self.write_meta()

    def write_meta(self):
        if self.directory:
            self.directory.mkdir(parents=True, exist_ok=True)
            (self.directory / f"{self.run_id}.meta.json").write_text(
                json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )
