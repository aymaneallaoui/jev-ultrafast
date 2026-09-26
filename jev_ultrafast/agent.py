"""The complete agent loop. Typed choices, observable state, bounded execution."""

import base64
import time
from pathlib import Path

from .browser import Browser, StalePage, TargetRefused
from .model import TransientModelError, action_space, choose, field_context, field_text, text_model
from .questions import MAX_STEPS
from .tracing import Trace

EXPAND_POLL_MS = 50
EXPAND_LIMIT_MS = 400
EMPTY_RETRIES = 5
EMPTY_RETRY_MS = 100
DECISION_RETRY_MS = 1000
REFUSAL_LIMIT = 3
clock = time.monotonic
sleep = time.sleep


def elements(page):
    return {a["node"] for a in page["actions"] if "node" in a}


def expanded(action, page):
    same = [a for a in page["actions"] if a.get("node") == action["node"]] or [
        a for a in page["actions"] if a.get("role") == action["role"] and a["label"] == action["label"]
    ]
    return any(a.get("expanded") == "true" for a in same)


class BudgetExhausted(ValueError):
    pass


class Agent:
    def __init__(self, url, goals, *, record_dir=None, screenshots=False):
        task = goals.strip() if isinstance(goals, str) else "\n".join(goals).strip()
        if not task:
            raise ValueError("Supply a task")
        plan = [task]
        self.pending_text = None
        self.trace = Trace(url, task)
        self.browser = Browser(url)
        self.record_dir = Path(record_dir) if record_dir else None
        self.screenshots = screenshots or bool(record_dir)
        try:
            page = self.browser.observe(screenshot=self.screenshots)
        except Exception:
            self.browser.close()
            raise
        self.state = dict(
            browser=self.browser,
            goal="\n".join(plan),
            page=page,
            decision=None,
            history=[],
            refusals=[],
            blocked_reason=None,
            status="ready",
            plan=plan,
            plan_index=0,
            decisions=[],
            text_calls=[],
            elapsed_ms=0,
            started_at=None,
            record=bool(self.record_dir),
        )
        if self.record_dir:
            self.record_dir.mkdir(parents=True, exist_ok=True)
            (self.record_dir / "000000.jpg").write_bytes(base64.b64decode(page["screenshot"]))

    def snapshot(self):
        return {
            **{k: v for k, v in self.state.items() if k != "browser"},
            "elements": action_space(self.state["page"]["actions"])[0],
        }

    def command(self, name, body=None):
        body = body or {}
        state = self.state
        if name == "tick":
            try:
                self.command("predict", {})
                return self.command("act", {"fingerprint": state["page"]["fingerprint"]})
            except StalePage:
                state["decision"] = None
                if state["status"] != "blocked":
                    state["status"] = "ready"
                state["page"] = self.observe(state["page"])
                state["elapsed_ms"] = round((time.perf_counter() - state["started_at"]) * 1000)
                return self.snapshot()
        elif name == "predict":
            if not state["browser"]:
                raise ValueError("Start a demo first")
            if state["started_at"] is None:
                state["started_at"] = time.perf_counter()
            self.trace.start_step(round((time.perf_counter() - state["started_at"]) * 1000), state["page"]["url"])
            try:
                self.predict()
            except Exception as error:
                self.trace.end_step(error)
                raise
        elif name == "act":
            try:
                self.act(body)
            except Exception as error:
                self.trace.end_step(error)
                raise
            self.trace.end_step()
        else:
            raise ValueError("Unknown command")
        return self.snapshot()

    def predict(self):
        state, timed = self.state, self.trace.timed
        with timed("snapshot"):
            if not state["browser"].fresh(state["page"]):
                state["page"] = self.observe(state["page"])
        state["decision"] = None
        if state["status"] in {"done", "blocked"}:
            raise ValueError("This run has stopped. Start a fresh demo.")
        if len(state["decisions"]) >= MAX_STEPS * 2:
            raise BudgetExhausted("Reached the demo's model-call budget")
        retries = []
        while True:
            attempt_started = time.perf_counter()
            try:
                with timed("model"):
                    state["decision"] = choose(state["page"], state["goal"], state["history"], self.trace, retries)
                break
            except TransientModelError as error:
                if retries:
                    raise
                retries.append(
                    {
                        "error": str(error),
                        "after_ms": round((time.perf_counter() - attempt_started) * 1000),
                        **({"raw": error.raw} if getattr(error, "raw", None) is not None else {}),
                    }
                )
            with timed("model"):
                sleep(DECISION_RETRY_MS / 1000)
            with timed("snapshot"):
                if not state["browser"].fresh(state["page"]):
                    state["page"] = self.observe(state["page"])
        state["decisions"].append(
            {
                **state["decision"],
                **({"retries": retries} if retries else {}),
                "fingerprint": state["page"]["fingerprint"],
                "elapsed_ms": round((time.perf_counter() - state["started_at"]) * 1000),
            }
        )
        state["status"] = "predicted"

    def act(self, body):
        state, timed = self.state, self.trace.timed
        decision, page = state["decision"], state["page"]
        if not decision or body.get("fingerprint") != page["fingerprint"]:
            raise ValueError("Observe and choose before acting")
        # Consume once, before any mutation or model call. A retry cannot double-click.
        state["decision"] = None
        selected = decision["choice"]
        if selected in {"DONE", "BLOCKED"}:
            with timed("execute"):
                fresh = state["browser"].fresh(page)
            if not fresh:
                state["status"] = "ready"
                raise StalePage("Page changed since the decision. Choose again.")
            state["status"] = "done" if selected == "DONE" else "blocked"
            state["plan_index"] = int(selected == "DONE")
            state["elapsed_ms"] = round((time.perf_counter() - state["started_at"]) * 1000)
            return
        action = next(a for a in page["actions"] if a["id"] == selected)
        if len(state["history"]) >= MAX_STEPS:
            state["status"] = "blocked"
            raise BudgetExhausted(f"Stopped at the {MAX_STEPS}-action demo budget")
        text, helper = None, None
        if action["kind"] == "fill":
            with timed("text"):
                if not state["browser"].fresh(page):
                    raise StalePage("Page changed before text generation. Choose again.")
                context = field_context(state["goal"], action, page, state["history"])
                if self.pending_text and self.pending_text[0] == context:
                    _, text, helper = self.pending_text
                else:
                    text, helper = self.generate_text(context, action["label"])
                    self.pending_text = (context, text, helper)
                self.trace.type_text(context, text)
        # Browser.act checks freshness immediately before input, including after text generation.
        with timed("execute"):
            try:
                state["browser"].act(action, page, text=text)
            except TargetRefused as error:
                self.refused(action, error)
                raise
        self.pending_text = None
        state["elapsed_ms"] = round((time.perf_counter() - state["started_at"]) * 1000)
        # Record execution before observing. A stale post-action observation must not erase the action.
        state["history"].append(
            {
                "step": len(state["history"]) + 1,
                "action": action["label"],
                "kind": action["kind"],
                "choice": selected,
                "probability": decision["probabilities"][selected],
                "confidence": decision["confidence"],
                "latency_ms": decision["latency_ms"],
                "text": text,
                "text_helper": helper["model"] if helper else None,
                "text_latency_ms": helper["latency_ms"] if helper else 0,
                "operation": decision["operation"],
                "target": decision["target"],
                "page_changed": None,
                "wait_ms": 0,
                "url": page["url"],
                "usage": decision["usage"],
                "executed_ms": round((time.perf_counter() - state["started_at"]) * 1000),
                "elapsed_ms": state["elapsed_ms"],
            }
        )
        with timed("wait"):
            state["page"] = self.observe(page)
            wait_ms = 0
            if action["kind"] == "click" and action.get("expanded") == "false":
                wait_ms = self.await_expansion(action, page)
        state["elapsed_ms"] = round((time.perf_counter() - state["started_at"]) * 1000)
        state["history"][-1].update(
            page_changed=state["page"]["fingerprint"] != page["fingerprint"],
            url=state["page"]["url"],
            elapsed_ms=state["elapsed_ms"],
            wait_ms=wait_ms,
        )
        if state["record"]:
            (self.record_dir / f"{state['elapsed_ms']:06d}.jpg").write_bytes(
                base64.b64decode(state["page"]["screenshot"])
            )
        repeated = state["history"][-3:]
        state["status"] = (
            "blocked"
            if len(repeated) == 3 and all(h["page_changed"] is False and h["kind"] != "wait" for h in repeated)
            else "ready"
        )
        if state["status"] == "blocked":
            state["blocked_reason"] = "No page change after 3 actions"

    def refused(self, action, error):
        state = self.state
        target = action["node"] if type(action.get("node")) is int else action["label"]
        refusals = state.setdefault("refusals", [])
        refusals.append(
            {"target": target, "action": action["label"], "kind": action["kind"], "error": str(error),
             "after_step": len(state["history"])}
        )
        streak = refusals[-REFUSAL_LIMIT:]
        if len(streak) == REFUSAL_LIMIT and all(
            r["target"] == target and r["after_step"] == len(state["history"]) for r in streak
        ):
            state["status"] = "blocked"
            state["blocked_reason"] = f"Target refused {REFUSAL_LIMIT} times: {action['label']}: {error}"

    def observe(self, previous):
        page = self.state["browser"].observe(screenshot=self.screenshots)
        for _ in range(EMPTY_RETRIES):
            if elements(page) or page["text"].strip() or (page["url"] == previous["url"] and not elements(previous)):
                break
            sleep(EMPTY_RETRY_MS / 1000)
            page = self.state["browser"].observe(screenshot=self.screenshots)
        return page

    def await_expansion(self, action, before):
        """Observe until the clicked control reports expanded and new elements appear. Nothing is executed."""
        started, count = clock(), len(elements(before))
        while True:
            waited = round((clock() - started) * 1000)
            page = self.state["page"]
            if (expanded(action, page) and len(elements(page)) > count) or waited >= EXPAND_LIMIT_MS:
                return waited
            sleep(min(EXPAND_POLL_MS, EXPAND_LIMIT_MS - waited) / 1000)
            self.state["page"] = self.state["browser"].observe(screenshot=self.screenshots)

    def generate_text(self, context, field):
        for attempt in range(2):
            started = time.perf_counter()
            try:
                text, helper = field_text(context)
            except TransientModelError as error:
                self.state["text_calls"].append(
                    {
                        "model": text_model(),
                        "latency_ms": round((time.perf_counter() - started) * 1000),
                        "field": field,
                        "failed": True,
                        "error": str(error),
                    }
                )
                if attempt:
                    raise
                continue
            self.state["text_calls"].append({**helper, "field": field, "value": text})
            return text, helper

    def run(self):
        while self.state["status"] not in {"done", "blocked"}:
            try:
                snapshot = self.command("tick")
            except BudgetExhausted as error:
                self.finish_trace("max_steps", error)
                raise
            except Exception as error:
                self.finish_trace("error", error)
                raise
            if snapshot["status"] in {"done", "blocked"}:
                self.finish_trace(snapshot["status"].upper(), reason=self.state.get("blocked_reason"))
            yield snapshot

    def finish_trace(self, status, error=None, reason=None):
        started = self.state["started_at"]
        elapsed_ms = round((time.perf_counter() - started) * 1000) if started else 0
        self.trace.finish(
            status, elapsed_ms, None if error is None else str(error), getattr(error, "raw", None), reason
        )

    def close(self):
        self.finish_trace("closed")
        self.browser.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
