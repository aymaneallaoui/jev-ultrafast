"""Offline contracts for a dynamic operation/target policy. No paid APIs."""

import base64
import json
import threading
import time
from copy import deepcopy
from datetime import date
from unittest.mock import Mock

import pytest

from jev_ultrafast import agent as loop
from jev_ultrafast import model
from jev_ultrafast.browser import StalePage, TargetRefused, browser_operation, fingerprint
from jev_ultrafast.tracing import Trace


def page():
    state = {
        "url": "https://example.test/",
        "title": "Search",
        "text": "Search",
        "scroll": {"y": 0},
        "actions": [
            {"id": "e1", "kind": "fill", "label": "Search", "role": "textbox", "value": "", "node": 10},
            {"id": "e2", "kind": "click", "label": "Open Search", "role": "textbox", "value": "", "node": 10},
            {"id": "e3", "kind": "click", "label": "Go", "role": "button", "value": "", "node": 20},
            {"id": "wait", "kind": "wait", "label": "Wait"},
        ],
    }
    state["fingerprint"] = fingerprint(state)
    return state


def choice(ids, selected):
    return {"choice": selected, "confidence": 1.0, "probabilities": {i: float(i == selected) for i in ids}}


def decision(action="e1"):
    return {
        "choice": action,
        "operation": "TYPE_TEXT",
        "target": "1",
        "confidence": 1.0,
        "probabilities": {action: 1.0},
        "latency_ms": 10,
        "usage": {},
    }


@pytest.mark.parametrize("mutation", ["unknown", "nan", "missing", "negative", "non_max", "confidence"])
def test_invalid_choice_is_rejected(mutation):
    a = choice(["a", "b"], "a")
    if mutation == "unknown":
        a["choice"] = "invented"
    elif mutation == "nan":
        a["probabilities"]["a"] = float("nan")
    elif mutation == "missing":
        del a["probabilities"]["b"]
    elif mutation == "negative":
        a["probabilities"]["b"] = -1
    elif mutation == "non_max":
        a["choice"] = "b"
    else:
        a["confidence"] = 5
    with pytest.raises(ValueError, match="Invalid TypeSafe"):
        model.validate_choice(a, {"a", "b"})


def test_one_index_per_node_with_operation_specific_targets():
    elements, targets, controls = model.action_space(page()["actions"])
    assert len(elements) == 2
    assert elements[0]["operations"] == ["TYPE_TEXT", "CLICK"]
    assert targets["TYPE_TEXT"]["1"]["id"] == "e1"
    assert targets["CLICK"]["1"]["id"] == "e2"
    assert targets["CLICK"]["2"]["id"] == "e3"
    assert "WAIT" in controls


def test_all_heads_are_one_request_and_only_matching_head_executes(monkeypatch):
    calls = []

    def post(_url, _key, body):
        calls.append(body)
        return {
            "model": "test",
            "answers": {
                "operation": choice(body["questions"]["operation"]["criteria"], "TYPE_TEXT"),
                "type_text_target": choice(["1"], "1"),
                "click_target": {"choice": "invented"},
            },
        }

    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", post)
    d = model.choose(page(), "Find a book", [])
    assert len(calls) == 1
    assert d["operation"] == "TYPE_TEXT" and d["target"] == "1" and d["choice"] == "e1"
    assert set(calls[0]["questions"]) == {"operation", "click_target", "type_text_target"}


def test_click_cannot_consume_a_text_target(monkeypatch):
    def post(_url, _key, body):
        return {
            "model": "test",
            "answers": {
                "operation": choice(body["questions"]["operation"]["criteria"], "CLICK"),
                "type_text_target": choice(["1"], "1"),
                "click_target": choice(["1", "2", "999"], "999"),
            },
        }

    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", post)
    with pytest.raises(ValueError, match="Invalid TypeSafe"):
        model.choose(page(), "Find a book", [])


def test_target_head_receives_control_state_and_full_next_step_rules(monkeypatch):
    p = page()
    p["actions"].insert(0, {
        "id": "toggle", "kind": "click", "label": "Free cancellation", "node": 30,
        "role": "checkbox", "checked": "true", "selected": False,
    })

    def post(_url, _key, body):
        questions = body["questions"]
        target = questions["click_target"]
        assert target["criteria"]["1"]["checked"] == "true"
        assert target["criteria"]["1"]["selected"] is False
        assert questions["operation"]["instructions"]["rules"] in target["instructions"]["rules"]
        return {
            "model": "test",
            "answers": {
                "operation": choice(questions["operation"]["criteria"], "CLICK"),
                "click_target": choice(target["criteria"], "3"),
            },
        }

    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", post)
    d = model.choose(p, "Search with free cancellation", [])
    assert d["choice"] == "e3"


def click_response(_url, _key, body):
    return {
        "model": "test",
        "usage": {"input_tokens": 7},
        "answers": {
            "operation": choice(body["questions"]["operation"]["criteria"], "CLICK"),
            "click_target": choice(["1", "2"], "2"),
        },
    }


def test_trace_writes_one_line_per_choice_without_credentials(monkeypatch, tmp_path):
    monkeypatch.setenv("TYPESAFE_API_KEY", "secret-key")
    monkeypatch.setenv("TRACE_DIR", str(tmp_path))
    monkeypatch.setattr(model, "post_json", click_response)
    trace = Trace("https://example.test/", "Find a book")
    model.choose(page(), "Find a book", [], trace)
    lines = (tmp_path / f"{trace.run_id}.jsonl").read_text().splitlines()
    assert len(lines) == 1
    line = json.loads(lines[0])
    assert set(line) == {"step", "goal", "request", "answers", "latency_ms", "usage"}
    assert line["step"] == 1 and line["goal"] == "Find a book" and line["usage"] == {"input_tokens": 7}
    assert "secret-key" not in lines[0]


def test_trace_writes_nothing_without_trace_dir(monkeypatch, tmp_path):
    monkeypatch.delenv("TRACE_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", click_response)
    model.choose(page(), "Find a book", [], Trace("https://example.test/", "Find a book"))
    assert not any(tmp_path.iterdir())


def test_quoted_task_text_still_uses_the_llm(monkeypatch):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "test")
    post = Mock(return_value={"choices": [{"message": {"content": '{"text":"Zurich"}'}}]})
    monkeypatch.setattr(model, "post_json", post)
    context = model.field_context('Fly from "Zurich" to London', page()["actions"][0], page(), [])
    assert model.field_text(context)[0] == "Zurich"
    assert post.call_count == 1
    sent = json.loads(post.call_args.args[2]["messages"][1]["content"])
    assert sent["goal"] == 'Fly from "Zurich" to London'


def test_missing_text_credential_stops_before_guessing(monkeypatch):
    monkeypatch.delenv("TEXT_MODEL_API_KEY", raising=False)
    with pytest.raises(ValueError, match="TEXT_MODEL_API_KEY"):
        model.field_text({"goal": 'Enter "Zurich"'})


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.delenv("TRACE_DIR", raising=False)
    a = loop.Agent.__new__(loop.Agent)
    a.screenshots = False
    a.pending_text = None
    a.trace = Trace("https://example.test/", "Find a book")
    p = page()
    a.state = {
        "browser": Mock(fresh=Mock(return_value=True), observe=Mock(return_value=p)),
        "page": p,
        "decision": decision(),
        "goal": "Find a book",
        "history": [],
        "decisions": [],
        "status": "predicted",
        "started_at": time.perf_counter(),
        "record": False,
        "text_calls": [],
    }
    return a


def test_stale_decision_is_consumed_before_any_mutation(runner):
    runner.state["browser"].fresh.return_value = False
    with pytest.raises(StalePage):
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    runner.state["browser"].act.assert_not_called()
    assert runner.state["decision"] is None


def test_generated_text_reused_only_for_identical_retry_context(runner, monkeypatch):
    helper = Mock(return_value=("book", {"model": "test", "latency_ms": 10}))
    monkeypatch.setattr(loop, "field_text", helper)
    runner.state["browser"].act.side_effect = [StalePage("Changed before input"), None]
    with pytest.raises(StalePage):
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    runner.state["decision"] = decision()
    runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    assert helper.call_count == 1
    assert runner.state["browser"].act.call_count == 2  # The first call rejects before any browser input.
    assert runner.pending_text is None


def test_changed_field_context_does_not_reuse_generated_text(runner, monkeypatch):
    helper = Mock(return_value=("book", {"model": "test", "latency_ms": 10}))
    monkeypatch.setattr(loop, "field_text", helper)
    runner.state["browser"].act.side_effect = [StalePage("Changed before input"), None]
    with pytest.raises(StalePage):
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    runner.state["page"]["text"] = "Different page context"
    runner.state["decision"] = decision()
    runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    assert helper.call_count == 2


def test_loading_waits_do_not_trigger_no_progress_stop(runner):
    for _ in range(5):
        runner.state["decision"] = decision("wait")
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    assert len(runner.state["history"]) == 5 and runner.state["status"] == "ready"


def test_stale_observation_preserves_executed_action(runner):
    runner.state["decision"] = decision("e3")
    runner.state["browser"].observe.side_effect = StalePage("changed")
    with pytest.raises(StalePage):
        runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    assert runner.state["history"][-1]["action"] == "Go"
    runner.state["browser"].act.assert_called_once()


def test_observation_is_one_atomic_browser_read(monkeypatch):
    import jev_ultrafast.browser as browser

    p = page()
    cdp = Mock(return_value={"result": {"value": p}})
    monkeypatch.setattr(browser, "cdp", cdp)
    actual = browser_operation({"operation": "observe", "session": "test", "screenshot": False})
    assert actual["actions"] == p["actions"]
    assert cdp.call_count == 1
    assert cdp.call_args.args[0] == "Runtime.evaluate"


def test_executor_rejects_a_stale_page_before_browser_input(monkeypatch):
    import jev_ultrafast.browser as browser

    b = browser.Browser.__new__(browser.Browser)
    b.fresh = Mock(return_value=False)
    operation = Mock()
    monkeypatch.setattr(browser, "browser_operation", operation)
    with pytest.raises(StalePage):
        b.act(page()["actions"][0], page(), "book")
    operation.assert_not_called()


@pytest.mark.parametrize("response", [{"exceptionDetails": {}}, {"result": {}}])
def test_interrupted_dropdown_mutation_cannot_be_retried_as_stale(monkeypatch, response):
    import jev_ultrafast.browser as browser

    # A navigation can destroy the evaluation result after the change event already fired.
    if "exceptionDetails" in response:
        response["exceptionDetails"] = {"text": "Execution context destroyed"}
    cdp = Mock(return_value=response)
    monkeypatch.setattr(browser, "cdp", cdp)
    with pytest.raises(RuntimeError, match="Dropdown execution"):
        browser_operation({"operation": "act", "session": "test", "action": {
            "id": "e1", "kind": "select", "node": 1, "value": "Design",
        }})
    assert cdp.call_count == 1


def test_fingerprint_tracks_values_and_identity_not_screenshots():
    p = page()
    other = deepcopy(p)
    other["screenshot"] = "changed"
    assert fingerprint(p) == fingerprint(other)
    other["actions"][0]["node"] = 99
    assert fingerprint(p) != fingerprint(other)


@pytest.mark.parametrize("changed", ["Departure", "Where from?", "Where to?", "year"])
def test_flight_verification_rejects_wrong_trip(changed):
    from examples.flights import verify

    actual = {
        "url": "https://www.google.com/travel/flights/search?tfs=example",
        "text": "Track prices from Zürich to London departing 2026-09-20",
        "actions": [
            {"label": k, "value": v}
            for k, v in [
                ("Change ticket type. One way", "One way"),
                ("Where from?", "Zürich"),
                ("Where to?", "London"),
                ("Departure", "Sun, Sep 20"),
                ("Nonstop flight on Sunday, September 20. Select flight", ""),
            ]
        ],
    }
    assert verify(actual, date(2026, 9, 20))["passed"]
    if changed == "year":
        actual["text"] = actual["text"].replace("2026", "2027")
    else:
        next(a for a in actual["actions"] if a["label"] == changed)["value"] = "wrong"
    assert not verify(actual, date(2026, 9, 20))["passed"]


def test_flight_date_forms_reproduce_the_recorded_run():
    from examples.flights import date_forms, goal

    forms = date_forms(date(2026, 9, 20))
    assert forms == {
        "goal": "September 20, 2026",
        "iso": "2026-09-20",
        "departure": "Sun, Sep 20",
        "flight": "Sunday, September 20",
    }
    assert goal(date(2026, 9, 20)) == (
        "Find one-way flights from Zurich to London on September 20, 2026, for one adult in economy. "
        "Stop when matching flight options are visible. Do not select or book a flight."
    )
    recorded_tfs = "CBwQAhooEgoyMDI2LTA5LTIwagwIAxIIL20vMDg5NjZyDAgDEggvbS8wNGpwbEABSAFwAYIBCwj___________8BmAEC"
    assert forms["iso"].encode() in base64.urlsafe_b64decode(recorded_tfs + "=" * (-len(recorded_tfs) % 4))
    assert date_forms(date(2026, 3, 1))["departure"] == "Sun, Mar 1"


@pytest.mark.parametrize(
    "content", ["Thinking: Zurich", '{"text":null}', '{"text":"Zurich","extra":true}', '{"text":123}']
)
def test_text_helper_rejects_invalid_values(monkeypatch, content):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", Mock(return_value={"choices": [{"message": {"content": content}}]}))
    with pytest.raises(ValueError, match="nothing typed"):
        model.field_text({"goal": "Find a flight"})


def traced(runner, monkeypatch, tmp_path):
    monkeypatch.setenv("TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    runner.trace = Trace("https://example.test/", "Find a book")
    runner.browser = runner.state["browser"]
    runner.state["status"] = "ready"
    return lambda: json.loads((tmp_path / f"{runner.trace.run_id}.meta.json").read_text())


def test_close_without_run_writes_closed_meta(runner, monkeypatch, tmp_path):
    meta = traced(runner, monkeypatch, tmp_path)
    runner.close()
    assert meta()["status"] == "closed" and meta()["verified"] is None and "error" not in meta()
    runner.trace.set_verified(False)
    assert meta()["verified"] is False
    runner.trace.set_verified(None)
    assert meta()["verified"] is None


def test_close_after_run_keeps_run_status_and_verification(runner, monkeypatch, tmp_path):
    meta = traced(runner, monkeypatch, tmp_path)
    monkeypatch.setattr(model, "post_json", lambda _url, _key, body: {
        "model": "test", "answers": {"operation": choice(body["questions"]["operation"]["criteria"], "DONE")},
    })
    list(runner.run())
    runner.trace.set_verified(True)
    runner.close()
    assert meta()["status"] == "DONE" and meta()["verified"] is True and meta()["steps"] == 1


def test_budget_exhaustion_is_traced_as_max_steps(runner, monkeypatch, tmp_path):
    meta = traced(runner, monkeypatch, tmp_path)
    runner.state["decisions"] = [{}] * (loop.MAX_STEPS * 2)
    with pytest.raises(ValueError, match="model-call budget"):
        list(runner.run())
    assert meta()["status"] == "max_steps" and meta()["error"] == "Reached the demo's model-call budget"


def test_other_exceptions_are_traced_as_errors(runner, monkeypatch, tmp_path):
    meta = traced(runner, monkeypatch, tmp_path)
    runner.state["browser"].fresh.side_effect = RuntimeError("boom")
    with pytest.raises(RuntimeError, match="boom"):
        list(runner.run())
    assert meta()["status"] == "error" and meta()["error"] == "boom"
    start, failed = trace_lines(runner, tmp_path)
    assert start["event"] == "step_start" and failed["event"] == "step_failed" and failed["step"] == 1
    assert failed["failed_phase"] == "snapshot" and failed["error"] == "boom" and "request" not in failed


def trace_lines(runner, tmp_path):
    return [json.loads(line) for line in (tmp_path / f"{runner.trace.run_id}.jsonl").read_text().splitlines()]


def type_response(_url, _key, body):
    return {
        "model": "test",
        "usage": {"input_tokens": 5},
        "answers": {
            "operation": choice(body["questions"]["operation"]["criteria"], "TYPE_TEXT"),
            "type_text_target": choice(["1"], "1"),
        },
    }


def test_each_step_traces_a_start_line_then_its_timed_decision(runner, monkeypatch, tmp_path):
    traced(runner, monkeypatch, tmp_path)
    replies = iter([click_response, type_response])
    monkeypatch.setattr(model, "post_json", lambda *args: next(replies)(*args))
    monkeypatch.setattr(loop, "field_text", Mock(return_value=("book", {"model": "test", "latency_ms": 10})))
    runner.command("tick")
    runner.command("tick")
    lines = trace_lines(runner, tmp_path)
    assert [(line["event"], line["step"]) for line in lines] == [
        ("step_start", 1), ("step", 1), ("step_start", 2), ("step", 2),
    ]
    assert lines[0]["url"] == "https://example.test/" and 0 <= lines[0]["t_ms"] <= lines[2]["t_ms"]
    for line in lines[1::2]:
        assert "request" in line and "failed_phase" not in line
        for phase in ("snapshot", "model", "execute", "wait"):
            assert isinstance(line[f"{phase}_ms"], int) and line[f"{phase}_ms"] >= 0
    assert "text_ms" not in lines[1] and "type_text" not in lines[1]
    assert isinstance(lines[3]["text_ms"], int) and lines[3]["type_text"]["text"] == "book"
    assert runner.trace.steps == 2 and runner.trace.stats()["input_tokens_total"] == 12


def test_failed_execution_still_writes_its_decision_line(runner, monkeypatch, tmp_path):
    meta = traced(runner, monkeypatch, tmp_path)
    monkeypatch.setattr(model, "post_json", click_response)
    runner.state["browser"].act.side_effect = RuntimeError("CDP call timed out")
    with pytest.raises(RuntimeError, match="CDP call timed out"):
        list(runner.run())
    start, step = trace_lines(runner, tmp_path)
    assert start["event"] == "step_start" and step["event"] == "step" and step["step"] == 1
    assert step["failed_phase"] == "execute" and step["error"] == "CDP call timed out"
    assert step["answers"]["operation"]["choice"] == "CLICK" and step["execute_ms"] >= 0 and "wait_ms" not in step
    assert meta()["status"] == "error" and meta()["steps"] == 1


def test_navigation_during_prediction_reobserves_without_action(runner):
    runner.state["browser"].fresh.side_effect = StalePage("Document navigating")
    runner.command("tick")
    assert runner.state["status"] == "ready"
    assert runner.state["decision"] is None
    runner.state["browser"].act.assert_not_called()


@pytest.fixture
def fake_time(monkeypatch):
    now, sleeps = [0.0], []

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    monkeypatch.setattr(loop, "clock", lambda: now[0])
    monkeypatch.setattr(loop, "sleep", sleep)
    return sleeps


def combobox_page(expanded="false", options=0):
    p = page()
    p["actions"][3:3] = [
        {"id": "e4", "kind": "click", "label": "Where to?", "role": "combobox", "value": "", "node": 30,
         "expanded": expanded},
        *({"id": f"e{5 + i}", "kind": "click", "label": f"City {i}", "role": "option", "value": "", "node": 40 + i}
          for i in range(options)),
    ]
    p["fingerprint"] = fingerprint(p)
    return p


def empty_page():
    p = page()
    p["text"], p["actions"] = "", p["actions"][-1:]
    p["fingerprint"] = fingerprint(p)
    return p


def act(runner):
    return runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})


def test_click_on_collapsed_control_observes_until_it_expands(runner, fake_time):
    runner.state["page"], runner.state["decision"] = combobox_page(), decision("e4")
    expanded = combobox_page("true", options=2)
    # Expanded without new elements is not enough; the third poll has both.
    runner.state["browser"].observe.side_effect = [
        combobox_page(), combobox_page("true"), combobox_page(), expanded,
    ]
    act(runner)
    runner.state["browser"].act.assert_called_once()
    assert runner.state["browser"].observe.call_count == 4
    assert fake_time == [0.05] * 3
    assert runner.state["page"] is expanded
    assert runner.state["history"][-1]["wait_ms"] == 150


def test_expansion_wait_stops_at_the_limit(runner, fake_time):
    runner.state["page"], runner.state["decision"] = combobox_page(), decision("e4")
    runner.state["browser"].observe.return_value = combobox_page()
    act(runner)
    runner.state["browser"].act.assert_called_once()
    assert runner.state["browser"].observe.call_count == 1 + loop.EXPAND_LIMIT_MS // loop.EXPAND_POLL_MS
    assert runner.state["history"][-1]["wait_ms"] == loop.EXPAND_LIMIT_MS


def test_clicks_without_collapsed_state_do_not_wait(runner, fake_time):
    runner.state["decision"] = decision("e3")
    act(runner)
    assert runner.state["browser"].observe.call_count == 1 and fake_time == []
    assert runner.state["history"][-1]["wait_ms"] == 0


def test_empty_snapshot_is_reobserved_before_the_model_sees_it(runner, fake_time, monkeypatch):
    runner.state["decision"] = decision("e3")
    runner.state["browser"].observe.side_effect = [empty_page(), empty_page(), page()]
    act(runner)
    assert fake_time == [0.1, 0.1]
    sent = []
    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", lambda url, key, body: sent.append(body) or click_response(url, key, body))
    runner.command("predict")
    assert len(sent) == 1 and len(sent[0]["state"]["elements"]) == 2


def test_persistently_empty_snapshot_proceeds_after_retries(runner, fake_time):
    runner.state["decision"] = decision("e3")
    runner.state["browser"].observe.return_value = empty_page()
    act(runner)
    assert runner.state["browser"].observe.call_count == 1 + loop.EMPTY_RETRIES
    assert fake_time == [0.1] * loop.EMPTY_RETRIES
    assert runner.state["page"]["actions"] == empty_page()["actions"]


def test_transient_decision_failure_is_retried_once(runner, fake_time, monkeypatch, tmp_path):
    monkeypatch.setenv("TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    runner.trace = Trace("https://example.test/", "Find a book")
    replies = iter([lambda *_: {"model": "test", "answers": {"operation": {"choice": "invented"}}}, click_response])
    monkeypatch.setattr(model, "post_json", lambda *args: next(replies)(*args))
    runner.command("predict")
    assert fake_time == [1.0]
    assert len(runner.state["decisions"]) == 1 and runner.trace.steps == 1
    runner.state["browser"].act.assert_not_called()
    act(runner)
    lines = (tmp_path / f"{runner.trace.run_id}.jsonl").read_text().splitlines()
    assert [json.loads(line)["event"] for line in lines] == ["step_start", "step"]
    line = json.loads(lines[1])
    assert line["step"] == 1
    assert [r["error"] for r in line["retries"]] == ["Invalid TypeSafe response; no action executed."]
    assert runner.state["decisions"][0]["retries"] == line["retries"]


def test_second_transient_decision_failure_raises(runner, fake_time, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    post = Mock(side_effect=model.ModelConnectionError("Model connection failed; no action executed."))
    monkeypatch.setattr(model, "post_json", post)
    with pytest.raises(RuntimeError, match="Model connection failed"):
        runner.command("predict")
    assert post.call_count == 2 and fake_time == [1.0]
    assert runner.state["decisions"] == [] and runner.trace.steps == 0


def test_non_transient_decision_failure_is_not_retried(runner, fake_time, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    post = Mock(side_effect=RuntimeError("Model provider returned HTTP 500; no action executed."))
    monkeypatch.setattr(model, "post_json", post)
    with pytest.raises(RuntimeError, match="HTTP 500"):
        runner.command("predict")
    assert post.call_count == 1 and fake_time == []


def text_reply(content):
    return {"choices": [{"message": {"content": content}}]}


@pytest.mark.parametrize(
    "failure",
    [text_reply("Thinking: book"), model.ModelConnectionError("Model connection failed; no action executed.")],
)
def test_text_helper_retries_once_and_records_both_attempts(runner, monkeypatch, failure):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "test")
    post = Mock(side_effect=[failure, text_reply('{"text":"book"}')])
    monkeypatch.setattr(model, "post_json", post)
    act(runner)
    assert post.call_count == 2 and all(c.kwargs["deadline"] > time.monotonic() - 1 for c in post.call_args_list)
    calls = runner.state["text_calls"]
    assert [c.get("failed", False) for c in calls] == [True, False] and calls[1]["value"] == "book"
    runner.state["browser"].act.assert_called_once()
    assert runner.state["browser"].act.call_args.kwargs["text"] == "book"


def test_text_helper_failing_twice_reports_raw_content(runner, monkeypatch):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "test")
    post = Mock(return_value=text_reply("x" * 500))
    monkeypatch.setattr(model, "post_json", post)
    with pytest.raises(ValueError, match="nothing typed") as error:
        act(runner)
    assert f"Model returned: '{'x' * 300}'" in str(error.value) and "x" * 301 not in str(error.value)
    assert post.call_count == 2 and [c["failed"] for c in runner.state["text_calls"]] == [True, True]
    runner.state["browser"].act.assert_not_called()
    assert runner.pending_text is None


def test_post_json_timeout_is_per_call(monkeypatch):
    import httpx

    response = Mock(status_code=200, is_error=False, json=Mock(return_value={}))
    client = Mock(post=Mock(return_value=response))
    monkeypatch.setattr(model, "CLIENT", client)
    model.post_json("https://example.test", "key", {})
    model.post_json("https://example.test", "key", {}, timeout=20)
    assert [c.kwargs["timeout"] for c in client.post.call_args_list] == [httpx.USE_CLIENT_DEFAULT, 20]
    client.post.side_effect = httpx.ReadTimeout("slow")
    with pytest.raises(model.TransientModelError, match="Model connection failed"):
        model.post_json("https://example.test", "key", {}, timeout=20)


def invalid_response(*_):
    return {"model": "test", "answers": {"operation": {"choice": "invented", "note": "x" * 3000}}}


def test_decision_retry_keeps_the_raw_invalid_response(runner, fake_time, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "secret-key")
    replies = iter([invalid_response, click_response])
    monkeypatch.setattr(model, "post_json", lambda *args: next(replies)(*args))
    runner.command("predict")
    raw = runner.state["decisions"][0]["retries"][0]["raw"]
    assert raw == json.dumps(invalid_response(), ensure_ascii=False)[: model.RAW_LIMIT]
    assert len(raw) == 2000 and "secret-key" not in raw


def test_connection_retry_has_no_raw_response(runner, fake_time, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    replies = iter(
        [Mock(side_effect=model.ModelConnectionError("Model connection failed; no action executed.")), click_response]
    )
    monkeypatch.setattr(model, "post_json", lambda *args: next(replies)(*args))
    runner.command("predict")
    assert "raw" not in runner.state["decisions"][0]["retries"][0]


def test_final_invalid_response_lands_in_trace_meta(runner, fake_time, monkeypatch, tmp_path):
    meta = traced(runner, monkeypatch, tmp_path)
    monkeypatch.setattr(model, "post_json", invalid_response)
    with pytest.raises(ValueError, match="Invalid TypeSafe"):
        list(runner.run())
    assert meta()["status"] == "error"
    assert meta()["raw_response"] == json.dumps(invalid_response(), ensure_ascii=False)[: model.RAW_LIMIT]


def refused():
    return TargetRefused("Target changed or is covered. Observe again.")


def test_same_target_refused_three_times_blocks_with_reason(runner, monkeypatch, tmp_path):
    meta = traced(runner, monkeypatch, tmp_path)
    monkeypatch.setattr(model, "post_json", click_response)
    runner.state["browser"].act.side_effect = refused()
    states = list(runner.run())
    reason = "Target refused 3 times: Go: Target changed or is covered. Observe again."
    assert [s["status"] for s in states] == ["ready", "ready", "blocked"]
    assert runner.state["blocked_reason"] == states[-1]["blocked_reason"] == reason
    assert len(runner.state["decisions"]) == 3 and runner.state["history"] == []
    assert [r["target"] for r in runner.state["refusals"]] == [20] * 3
    assert meta()["status"] == "BLOCKED" and meta()["reason"] == reason and meta()["steps"] == 3
    steps = [line for line in trace_lines(runner, tmp_path) if line["event"] == "step"]
    assert [(line["failed_phase"], line["error"]) for line in steps] == [
        ("execute", "Target changed or is covered. Observe again.")
    ] * 3


def test_successful_action_resets_the_refusal_streak(runner, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    runner.state["status"] = "ready"
    monkeypatch.setattr(model, "post_json", click_response)
    runner.state["browser"].act.side_effect = [refused(), refused(), None, refused(), refused(), refused()]
    for _ in range(5):
        runner.command("tick")
    assert runner.state["status"] == "ready" and len(runner.state["history"]) == 1
    runner.command("tick")
    assert runner.state["status"] == "blocked" and runner.state["blocked_reason"].startswith("Target refused 3")


def test_alternating_refused_targets_stay_bounded_by_the_budget(runner, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    runner.state["status"] = "ready"
    targets = iter(["1", "2"] * loop.MAX_STEPS)

    def post(_url, _key, body):
        return {
            "model": "test",
            "answers": {
                "operation": choice(body["questions"]["operation"]["criteria"], "CLICK"),
                "click_target": choice(["1", "2"], next(targets)),
            },
        }

    monkeypatch.setattr(model, "post_json", post)
    runner.state["browser"].act.side_effect = refused()
    with pytest.raises(loop.BudgetExhausted):
        list(runner.run())
    assert len(runner.state["decisions"]) == loop.MAX_STEPS * 2
    assert {r["target"] for r in runner.state["refusals"]} == {10, 20}
    assert runner.state["status"] == "ready" and "blocked_reason" not in runner.state


def test_page_change_refusals_do_not_count_toward_the_target_streak(runner, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    runner.state["status"] = "ready"
    monkeypatch.setattr(model, "post_json", click_response)
    runner.state["browser"].act.side_effect = StalePage("Page changed since this decision. Observe again.")
    for _ in range(4):
        runner.command("tick")
    assert runner.state["status"] == "ready" and "refusals" not in runner.state


def test_blank_first_snapshot_of_a_new_page_is_reobserved(runner, fake_time):
    runner.state["decision"] = decision("e3")
    elsewhere = empty_page()
    elsewhere["url"] = "https://other.test/article"
    loaded = page()
    loaded["url"] = elsewhere["url"]
    runner.state["browser"].observe.side_effect = [elsewhere, elsewhere, loaded]
    act(runner)
    assert fake_time == [0.1, 0.1]
    assert runner.state["page"] is loaded


def test_filled_field_offers_press_enter_as_its_own_operation():
    actions = page()["actions"] + [
        {"id": "e4", "kind": "enter", "label": "Press Enter in Search", "role": "textbox", "value": "q", "node": 10},
    ]
    elements, targets, _ = model.action_space(actions)
    assert targets["PRESS_ENTER"] == {"1": actions[-1]}
    assert elements[0]["label"] == "Search" and "PRESS_ENTER" in elements[0]["operations"]


@pytest.mark.parametrize(
    ("content", "text"),
    [
        ('{"text": "Rust"}\n```', "Rust"),
        ('```json\n{"text": "Rust"}\n```', "Rust"),
        ('Sure: {"text": "a {b} c"} done', "a {b} c"),
    ],
)
def test_text_helper_accepts_the_first_json_object_around_fences(monkeypatch, content, text):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", lambda *a, **k: {"choices": [{"message": {"content": content}}]})
    assert model.field_text({"goal": "g"})[0] == text


@pytest.mark.parametrize("content", ["no json here", '{"text": "x", "extra": 1}', '["text"]', None])
def test_text_helper_still_rejects_invalid_values(monkeypatch, content):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", lambda *a, **k: {"choices": [{"message": {"content": content}}]})
    with pytest.raises(model.InvalidModelResponse):
        model.field_text({"goal": "g"})


def guard_page():
    p = page()
    p["actions"] = [{"id": f"e{n}", "kind": "click", "label": label, "role": "button", "value": "", "node": n}
                    for n, label in enumerate(["Advanced", "Buy It Now", "Price", "Search"], start=1)]
    return p


def guard_decision(choice, probabilities):
    return {"choice": choice, "operation": "CLICK", "target": str(int(choice[1:])), "confidence": 0.9,
            "probabilities": probabilities, "target_ids": {f"e{n}": str(n) for n in range(1, 5)}}


def test_loop_guard_breaks_an_a_b_cycle_with_a_third_target(runner):
    runner.state["page"] = guard_page()
    advanced, buy = {"key": ["CLICK", "Advanced"]}, {"key": ["CLICK", "Buy It Now"]}
    runner.state["decisions"] = [advanced, buy, advanced]
    decision = guard_decision("e2", {"e1": 0.3, "e2": 0.4, "e3": 0.2, "e4": 0.1})
    key = runner.loop_guard(decision, ["CLICK", "Buy It Now"])
    assert decision["choice"] == "e3" and decision["target"] == "3" and key == ["CLICK", "Price"]
    assert decision["loop_guard"] == {"pattern": "cycle", "replaced": "Buy It Now", "with": "Price", "probability": 0.2}


def test_loop_guard_repeat_without_change_and_repeated_refusals(runner):
    runner.state["page"] = guard_page()
    runner.state["decisions"] = []
    runner.state["history"] = [{"operation": "CLICK", "action": "Search", "page_changed": False}] * 2
    decision = guard_decision("e4", {"e1": 0.1, "e2": 0.2, "e3": 0.05, "e4": 0.65})
    assert runner.loop_guard(decision, ["CLICK", "Search"]) == ["CLICK", "Buy It Now"]
    assert decision["loop_guard"]["pattern"] == "repeat_no_change"
    runner.state["history"] = []
    runner.state["refusals"] = [{"action": "Advanced", "after_step": 0}] * 2
    decision = guard_decision("e1", {"e1": 0.7, "e2": 0.1, "e3": 0.15, "e4": 0.05})
    assert runner.loop_guard(decision, ["CLICK", "Advanced"]) == ["CLICK", "Price"]
    assert decision["loop_guard"]["pattern"] == "refused_twice"


def test_loop_guard_leaves_progress_alone(runner):
    runner.state["page"] = guard_page()
    advanced, price = {"key": ["CLICK", "Advanced"]}, {"key": ["CLICK", "Price"]}
    runner.state["decisions"] = [advanced, price, advanced]
    runner.state["history"] = [{"operation": "CLICK", "action": "Search", "page_changed": True}] * 2
    decision = guard_decision("e2", {"e1": 0.3, "e2": 0.4, "e3": 0.2, "e4": 0.1})
    assert runner.loop_guard(decision, ["CLICK", "Buy It Now"]) == ["CLICK", "Buy It Now"]
    assert decision["choice"] == "e2" and "loop_guard" not in decision


def test_loop_guard_is_off_without_the_flag_and_never_touches_done(runner, monkeypatch):
    monkeypatch.delenv("JEV_LOOP_GUARD", raising=False)
    runner.state["page"] = guard_page()
    assert runner.decision_key({"choice": "DONE", "operation": "DONE", "target_ids": {}}) is None
    assert runner.decision_key(guard_decision("e1", {"e1": 1.0})) == ["CLICK", "Advanced"]


def gate_decision(operation, probabilities):
    return {"choice": operation, "operation": operation, "target": None, "confidence": 0.5, "probabilities": {},
            "target_ids": {}, "operation_probabilities": probabilities,
            "raw_answers": {"click_target": {"choice": "2", "confidence": 0.8, "probabilities": {"1": 0.1, "2": 0.9}}}}


def test_low_confidence_done_becomes_the_runner_up_operation_with_its_validated_target(runner, monkeypatch):
    monkeypatch.setenv("JEV_DONE_MIN_CONF", "0.72")
    decision = gate_decision("DONE", {"DONE": 0.6, "BLOCKED": 0.05, "CLICK": 0.3, "WAIT": 0.05})
    runner.confidence_gate(decision)
    assert (decision["operation"], decision["choice"], decision["target"]) == ("CLICK", "e3", "2")
    assert decision["confidence_gate"] == {
        "from": "DONE", "probability": 0.6, "threshold": 0.72, "to": "CLICK", "to_probability": 0.3}
    assert decision["probabilities"] == {"e2": 0.1, "e3": 0.9}


def test_confident_done_and_unset_thresholds_pass_through(runner, monkeypatch):
    monkeypatch.setenv("JEV_DONE_MIN_CONF", "0.72")
    confident = gate_decision("DONE", {"DONE": 0.9, "CLICK": 0.1})
    runner.confidence_gate(confident)
    assert confident["operation"] == "DONE" and "confidence_gate" not in confident
    monkeypatch.delenv("JEV_BLOCKED_MIN_CONF", raising=False)
    blocked = gate_decision("BLOCKED", {"BLOCKED": 0.3, "CLICK": 0.7})
    runner.confidence_gate(blocked)
    assert blocked["operation"] == "BLOCKED"


def test_low_confidence_blocked_can_fall_back_to_a_control(runner, monkeypatch):
    monkeypatch.setenv("JEV_BLOCKED_MIN_CONF", "0.72")
    decision = gate_decision("BLOCKED", {"BLOCKED": 0.5, "WAIT": 0.4, "DONE": 0.1})
    runner.confidence_gate(decision)
    assert (decision["operation"], decision["choice"], decision["target"]) == ("WAIT", "wait", None)


def test_gate_skips_an_invalid_fallback_head(runner, monkeypatch):
    monkeypatch.setenv("JEV_DONE_MIN_CONF", "0.72")
    decision = gate_decision("DONE", {"DONE": 0.6, "CLICK": 0.24, "WAIT": 0.16})
    decision["raw_answers"]["click_target"] = {"choice": "9", "confidence": 1.0, "probabilities": {"9": 1.0}}
    runner.confidence_gate(decision)
    assert decision["operation"] == "WAIT"
    only_click = gate_decision("DONE", {"DONE": 0.6, "CLICK": 0.4})
    only_click["raw_answers"]["click_target"] = {"choice": "9", "confidence": 1.0, "probabilities": {"9": 1.0}}
    runner.confidence_gate(only_click)
    assert only_click["operation"] == "DONE"


def test_gate_marks_a_gated_done_and_writes_the_mark_to_the_pending_trace_step(runner, monkeypatch):
    monkeypatch.setenv("JEV_DONE_MIN_CONF", "0.9")
    runner.trace.pending = {}
    decision = gate_decision("DONE", {"DONE": 0.6, "CLICK": 0.4})
    runner.confidence_gate(decision)
    assert decision["operation"] == "CLICK" and decision["done_gated"] is True
    assert "blocked_gated" not in decision
    assert runner.trace.pending["done_gated"] is True
    assert runner.trace.pending["confidence_gate"]["to_probability"] == 0.4


def test_gate_does_not_fire_when_no_alternative_clears_the_floor(runner, monkeypatch):
    monkeypatch.setenv("JEV_DONE_MIN_CONF", "0.9")
    decision = gate_decision("DONE", {"DONE": 0.88, "CLICK": 0.12})
    runner.confidence_gate(decision)
    assert decision["operation"] == "DONE"
    assert "confidence_gate" not in decision and "done_gated" not in decision


def test_gate_marks_a_gated_blocked(runner, monkeypatch):
    monkeypatch.setenv("JEV_BLOCKED_MIN_CONF", "0.9")
    decision = gate_decision("BLOCKED", {"BLOCKED": 0.6, "CLICK": 0.4})
    runner.confidence_gate(decision)
    assert decision["operation"] == "CLICK" and decision["blocked_gated"] is True and "done_gated" not in decision


def cascade_response(body, operation, target_probability=1.0, model_name="test"):
    answers = {"operation": choice(body["questions"]["operation"]["criteria"], operation)}
    if operation == "CLICK":
        rest = (1 - target_probability) / 2
        answers["click_target"] = {
            "choice": "1",
            "confidence": target_probability,
            "probabilities": {"1": target_probability, "2": rest, "3": rest},
        }
    return {"model": model_name, "answers": answers, "usage": {"input_tokens": 5}}


def cascade_page():
    state = page()
    state["actions"].append({"id": "e4", "kind": "click", "label": "More", "role": "button", "value": "", "node": 30})
    return state


def cascade_setup(monkeypatch, primary, verifier):
    calls = []

    def post(url, key, body):
        calls.append((url, key, body))
        reply = primary if url.startswith("http://primary") else verifier
        if isinstance(reply, Exception):
            raise reply
        return reply(body) if callable(reply) else reply

    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setenv("TYPESAFE_BASE_URL", "http://primary")
    monkeypatch.setattr(model, "post_json", post)
    return calls


def test_cascade_is_off_without_verifier_url(monkeypatch, tmp_path):
    monkeypatch.setenv("TRACE_DIR", str(tmp_path))
    calls = cascade_setup(monkeypatch, lambda body: cascade_response(body, "DONE"), None)
    trace = Trace("https://example.test/", "Find a book")
    d = model.choose(cascade_page(), "Find a book", [], trace)
    trace.flush()
    assert len(calls) == 1 and "cascade" not in d
    assert "cascade" not in json.loads((tmp_path / f"{trace.run_id}.jsonl").read_text().splitlines()[0])


def test_cascade_done_escalates_to_verifier(monkeypatch, tmp_path):
    monkeypatch.setenv("TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("JEV_VERIFIER_BASE_URL", "http://verifier/")
    monkeypatch.setenv("JEV_VERIFIER_API_KEY", "vkey")
    calls = cascade_setup(
        monkeypatch,
        lambda body: cascade_response(body, "DONE", model_name="small"),
        lambda body: cascade_response(body, "CLICK", model_name="large"),
    )
    trace = Trace("https://example.test/", "Find a book")
    d = model.choose(cascade_page(), "Find a book", [], trace)
    trace.flush()
    assert len(calls) == 2
    assert calls[1][0] == "http://verifier/v1/systemone" and calls[1][1] == "vkey" and calls[1][2] == calls[0][2]
    assert d["operation"] == "CLICK" and d["model"] == "large"
    assert d["cascade"]["reason"] == "done" and d["cascade"]["used"] == "verifier"
    assert d["cascade"]["primary"]["answers"]["operation"]["choice"] == "DONE"
    assert d["cascade"]["verifier"]["answers"] == d["raw_answers"]
    step = json.loads((tmp_path / f"{trace.run_id}.jsonl").read_text().splitlines()[0])
    assert step["answers"] == d["raw_answers"] and step["cascade"] == d["cascade"]


def test_cascade_blocked_escalates(monkeypatch):
    monkeypatch.setenv("JEV_VERIFIER_BASE_URL", "http://verifier")
    calls = cascade_setup(
        monkeypatch, lambda body: cascade_response(body, "BLOCKED"), lambda body: cascade_response(body, "CLICK")
    )
    d = model.choose(cascade_page(), "Find a book", [])
    assert len(calls) == 2 and d["cascade"]["reason"] == "blocked"


@pytest.mark.parametrize(
    ("probability", "threshold", "escalates"), [(0.4, None, True), (0.6, None, False), (0.6, "0.7", True)]
)
def test_cascade_target_confidence(monkeypatch, probability, threshold, escalates):
    monkeypatch.setenv("JEV_VERIFIER_BASE_URL", "http://verifier")
    if threshold:
        monkeypatch.setenv("JEV_CASCADE_TARGET_CONF", threshold)
    calls = cascade_setup(
        monkeypatch,
        lambda body: cascade_response(body, "CLICK", probability),
        lambda body: cascade_response(body, "CLICK", 0.9),
    )
    d = model.choose(cascade_page(), "Find a book", [])
    assert (len(calls) == 2) is escalates
    assert ("cascade" in d) is escalates
    if escalates:
        assert d["cascade"]["reason"] == "target_conf"


@pytest.mark.parametrize("failure", ["connection", "invalid"])
def test_cascade_falls_back_to_primary_when_verifier_fails(monkeypatch, tmp_path, failure):
    monkeypatch.setenv("TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("JEV_VERIFIER_BASE_URL", "http://verifier")

    def bad(body):
        reply = cascade_response(body, "CLICK", 0.9)
        reply["answers"]["operation"]["probabilities"]["CLICK"] = 0.2
        return reply

    verifier = model.ModelConnectionError("Model connection failed; no action executed.")
    calls = cascade_setup(
        monkeypatch, lambda body: cascade_response(body, "DONE"), verifier if failure == "connection" else bad
    )
    trace = Trace("https://example.test/", "Find a book")
    d = model.choose(cascade_page(), "Find a book", [], trace)
    trace.flush()
    assert len(calls) == 2 and d["operation"] == "DONE"
    assert d["cascade"]["used"] == "primary" and d["cascade"]["verifier"]["error"]
    step = json.loads((tmp_path / f"{trace.run_id}.jsonl").read_text().splitlines()[0])
    assert step["answers"]["operation"]["choice"] == "DONE"


def veto_run(monkeypatch, primary="DONE", verifier="CLICK", cache=None, tmp_path=None):
    monkeypatch.setenv("JEV_VERIFIER_BASE_URL", "http://verifier")
    if tmp_path:
        monkeypatch.setenv("TRACE_DIR", str(tmp_path))
    calls = cascade_setup(
        monkeypatch, lambda body: cascade_response(body, primary), lambda body: cascade_response(body, verifier)
    )
    cache = {} if cache is None else cache
    return calls, cache


def test_veto_cache_skips_the_verifier_on_a_repeated_done(monkeypatch, tmp_path):
    calls, cache = veto_run(monkeypatch, tmp_path=tmp_path)
    trace = Trace("https://example.test/", "Find a book")
    first = model.choose(cascade_page(), "Find a book", [], trace, veto_cache=cache)
    assert len(calls) == 2 and len(cache) == 1
    stored = next(iter(cache.values()))
    assert stored["result"]["answers"] == first["raw_answers"]
    second = model.choose(cascade_page(), "Find a book", [], trace, veto_cache=cache)
    trace.flush()
    assert len(calls) == 3 and calls[2][0].startswith("http://primary")
    assert second["operation"] == "CLICK" and second["cascade"]["used"] == "cache"
    assert second["cascade"]["cache"]["from_step"] == stored["step"] == 1
    assert second["cascade"]["cache"]["answers"] == stored["result"]["answers"] == second["raw_answers"]
    assert "verifier" not in second["cascade"]
    lines = [json.loads(line) for line in (tmp_path / f"{trace.run_id}.jsonl").read_text().splitlines()]
    step = next(line for line in lines if line.get("step") == 2 and "answers" in line)
    assert step["answers"] == stored["result"]["answers"]
    assert step["latency_ms"] == second["cascade"]["primary"]["latency_ms"] == second["latency_ms"]


def test_veto_cache_ignores_a_verifier_that_agrees(monkeypatch):
    calls, cache = veto_run(monkeypatch, verifier="DONE")
    model.choose(cascade_page(), "Find a book", [], veto_cache=cache)
    model.choose(cascade_page(), "Find a book", [], veto_cache=cache)
    assert len(calls) == 4 and not cache


def test_veto_cache_does_not_store_target_confidence_overrides(monkeypatch):
    monkeypatch.setenv("JEV_VERIFIER_BASE_URL", "http://verifier")
    calls = cascade_setup(
        monkeypatch, lambda body: cascade_response(body, "CLICK", 0.4), lambda body: cascade_response(body, "DONE")
    )
    cache = {}
    d = model.choose(cascade_page(), "Find a book", [], veto_cache=cache)
    assert d["cascade"]["reason"] == "target_conf" and d["operation"] == "DONE"
    assert len(calls) == 2 and not cache


def test_veto_cache_key_depends_on_url_and_labels(monkeypatch):
    calls, cache = veto_run(monkeypatch)
    model.choose(cascade_page(), "Find a book", [], veto_cache=cache)
    other_url = cascade_page()
    other_url["url"] = "https://example.test/next"
    other_label = cascade_page()
    other_label["actions"][2]["label"] = "Proceed"
    for changed in (other_url, other_label):
        d = model.choose(changed, "Find a book", [], veto_cache=cache)
        assert d["cascade"]["used"] == "verifier"
    assert len(calls) == 6 and len(cache) == 3


def test_veto_key_is_stable_and_label_sensitive():
    def body(labels, url="https://example.test/"):
        return {"state": {"page": {"url": url}, "elements": [{"label": label} for label in labels]}}

    assert model.veto_key(body(["a", "b"])) == model.veto_key(body(["a", "b"]))
    assert model.veto_key(body(["a", "b"])) != model.veto_key(body(["a", "c"]))
    assert model.veto_key(body(["a", "b"])) != model.veto_key(body(["a", "b"], "https://example.test/x"))


def test_veto_cache_drops_an_entry_that_no_longer_validates(monkeypatch):
    calls, cache = veto_run(monkeypatch)
    model.choose(cascade_page(), "Find a book", [], veto_cache=cache)
    key = next(iter(cache))
    cache[key]["result"]["answers"]["click_target"]["choice"] = "99"
    d = model.choose(cascade_page(), "Find a book", [], veto_cache=cache)
    assert len(calls) == 4 and d["cascade"]["used"] == "verifier"
    assert cache[key]["result"]["answers"]["click_target"]["choice"] == "1"


def test_veto_cache_can_be_disabled(monkeypatch):
    monkeypatch.setenv("JEV_VETO_CACHE", "0")
    calls, cache = veto_run(monkeypatch)
    model.choose(cascade_page(), "Find a book", [], veto_cache=cache)
    model.choose(cascade_page(), "Find a book", [], veto_cache=cache)
    assert len(calls) == 4 and not cache


def test_veto_cache_is_inert_without_a_cache_or_verifier(monkeypatch):
    calls, cache = veto_run(monkeypatch)
    d = model.choose(cascade_page(), "Find a book", [])
    assert len(calls) == 2 and d["cascade"]["used"] == "verifier" and not cache
    monkeypatch.delenv("JEV_VERIFIER_BASE_URL")
    d = model.choose(cascade_page(), "Find a book", [], veto_cache=cache)
    assert "cascade" not in d and not cache


@pytest.mark.parametrize("page_changed", [True, False])
def test_veto_cache_entry_is_cleared_only_when_the_page_changed(runner, page_changed):
    p = runner.state["page"]
    request = {"state": {"page": {"url": p["url"]}, "elements": [{"label": "Go"}]}}
    key = model.veto_key(request)
    runner.state["veto_cache"] = {key: {"result": {}, "step": 1}, ("other", "key"): {}}
    runner.state["decision"] = {**decision("e3"), "operation": "CLICK", "request": request}
    after = deepcopy(p)
    if page_changed:
        after["fingerprint"] = "changed"
    runner.state["browser"].observe.return_value = after
    runner.command("act", {"fingerprint": p["fingerprint"]})
    assert (key in runner.state["veto_cache"]) is not page_changed
    assert ("other", "key") in runner.state["veto_cache"]


TEXT_REPLY = {"choices": [{"message": {"content": '{"text":"Zurich"}'}}], "usage": {"input_tokens": 3}}


def test_text_helper_returns_value_and_metadata_within_budget(monkeypatch):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "test")
    monkeypatch.setattr(model, "post_json", lambda *a, **k: TEXT_REPLY)
    value, helper = model.field_text({"goal": "g"})
    assert value == "Zurich"
    assert helper["model"] == model.text_model() and helper["usage"] == {"input_tokens": 3}


def test_text_helper_budget_bounds_a_blocked_request(monkeypatch):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "test")
    monkeypatch.setenv("TEXT_TIMEOUT_S", "0.2")
    release = threading.Event()
    monkeypatch.setattr(model, "post_json", lambda *a, **k: release.wait(5) and TEXT_REPLY)
    started = time.perf_counter()
    with pytest.raises(model.ModelConnectionError, match="time budget; nothing typed"):
        model.field_text({"goal": "g"})
    assert time.perf_counter() - started < 1
    release.set()


def test_text_helper_budget_bounds_overload_retries(monkeypatch):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "test")
    monkeypatch.setenv("TEXT_TIMEOUT_S", "0.2")
    overloaded = Mock(return_value=Mock(status_code=503, is_error=True))
    monkeypatch.setattr(model.CLIENT, "post", overloaded)
    started = time.perf_counter()
    with pytest.raises(model.ModelConnectionError, match="time budget; nothing typed"):
        model.field_text({"goal": "g"})
    assert time.perf_counter() - started < 0.6
    assert overloaded.call_args.kwargs["timeout"].read <= 0.2


def test_text_budget_env_override(monkeypatch):
    monkeypatch.delenv("TEXT_TIMEOUT_S", raising=False)
    assert model.text_budget_s() == model.TEXT_TIMEOUT_S == 20
    monkeypatch.setenv("TEXT_TIMEOUT_S", "3.5")
    assert model.text_budget_s() == 3.5


def test_late_text_reply_is_never_used(monkeypatch):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "test")
    monkeypatch.setenv("TEXT_TIMEOUT_S", "0.1")
    finished = threading.Event()

    def slow(*a, **k):
        time.sleep(0.3)
        finished.set()
        return TEXT_REPLY

    monkeypatch.setattr(model, "post_json", slow)
    with pytest.raises(model.ModelConnectionError):
        model.field_text({"goal": "g"})
    assert finished.wait(2)
