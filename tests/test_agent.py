"""Offline contracts for a dynamic operation/target policy. No paid APIs."""

import base64
import json
import time
from copy import deepcopy
from datetime import date
from unittest.mock import Mock

import pytest

from jev_ultrafast import agent as loop
from jev_ultrafast import model
from jev_ultrafast.browser import StalePage, browser_operation, fingerprint
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
    lines = (tmp_path / f"{runner.trace.run_id}.jsonl").read_text().splitlines()
    assert len(lines) == 1
    line = json.loads(lines[0])
    assert line["step"] == 1
    assert [r["error"] for r in line["retries"]] == ["Invalid TypeSafe response; no action executed."]
    assert runner.state["decisions"][0]["retries"] == line["retries"]
    runner.state["browser"].act.assert_not_called()


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
    assert [c.kwargs["timeout"] for c in post.call_args_list] == [model.TEXT_TIMEOUT_S] * 2 == [20, 20]
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
