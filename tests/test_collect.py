"""Offline contracts for trace collection and conversion. No browser, network, or paid APIs."""

import csv
import hashlib
import json
from collections import Counter
from datetime import date
from pathlib import Path

import pytest

from jev_ultrafast.tracing import Trace
from jev_ultrafast.verifiers import date_forms, flights
from scripts import collect, traces_to_kev

ROOT = Path(__file__).resolve().parents[1]
TODAY = date(2026, 9, 28)


def test_placeholders_resolve_from_a_fixed_today():
    assert collect.resolve("on {date+10}", TODAY) == "on October 8, 2026"
    assert collect.resolve("{date+10:%Y-%m-%d}", TODAY) == "2026-10-08"
    assert collect.resolve("{weekday+3}", TODAY) == "Thursday"
    assert collect.resolve("{date+4}", TODAY) == "October 2, 2026"


@pytest.mark.parametrize("text", ["{month+1}", "{weekday+1:%A}", "{date-1}", "{date+x}"])
def test_unknown_placeholders_fail_at_load(text):
    with pytest.raises(ValueError, match="Unknown placeholder"):
        collect.resolve(text, TODAY)


def test_tasks_file_has_the_planned_distribution():
    tasks = collect.load_tasks(ROOT / "tasks.yaml", TODAY)
    assert len(tasks) == 60 and len({t["id"] for t in tasks}) == 60
    assert Counter(t["tags"][0] for t in tasks) == {
        "google_flights": 12, "wikipedia": 10, "hackernews": 6, "github": 8, "mdn": 5,
        "openstreetmap": 4, "forms": 6, "youtube": 4, "ebay": 5,
    }
    for task in tasks:
        assert "{" not in task["goal"] and "Stop when" in task["goal"]
        assert ("verify" in task) == (task["tags"][0] == "google_flights")
        if "verify" in task:
            assert set(task["verify_args"]) == {"origin", "destination", "day", "one_way", "adults"}
            assert date_forms(task["verify_args"]["day"])["goal"] in task["goal"]


def test_flight_verifier_checks_trip_type_and_passengers_when_asked():
    page = {
        "url": "https://www.google.com/travel/flights/search?tfs=example",
        "text": "departing 2026-10-12",
        "actions": [
            {"label": "Change ticket type. Round trip", "value": "Round trip"},
            {"label": "2 passengers, change number of passengers.", "role": "button"},
            {"label": "Where from?", "value": "Paris"},
            {"label": "Where to?", "value": "Rome"},
            {"label": "Departure", "value": "Mon, Oct 12"},
            {"label": "Nonstop flight on Monday, October 12. Select flight", "value": ""},
        ],
    }
    route = {"origin": "Paris", "destination": "Rome", "day": date(2026, 10, 12)}
    assert flights(page, one_way=False, adults=2, **route)["passed"]
    assert not flights(page, one_way=False, adults=1, **route)["passed"]
    assert not flights(page, one_way=True, **route)["passed"]
    assert "passengers" not in flights(page, one_way=False, **route)["checks"]


def test_trace_stats_and_summary_rows(tmp_path, monkeypatch):
    monkeypatch.delenv("TRACE_DIR", raising=False)
    trace = Trace("https://example.test/", "Find a book")
    for latency, usage in [(30, {"input_tokens": 10}), (10, {"input_tokens": 20}), (20, None)]:
        trace.step("Find a book", {}, {"answers": {}, "usage": usage}, latency)
    assert trace.stats() == {"jev_latency_p50_ms": 20, "jev_latency_max_ms": 30, "input_tokens_total": 30}
    trace.finish("DONE", 5)
    task = {"id": "t1", "tags": ["wikipedia", "section"]}
    path = tmp_path / "summary.csv"
    collect.append_summary(path, collect.summary_row(task, trace))
    collect.append_summary(path, collect.summary_row(task, None))
    rows = list(csv.DictReader(path.open()))
    assert path.read_text().count("run_id") == 1
    assert rows[0] == {
        "run_id": trace.run_id, "task_id": "t1", "tags": "wikipedia;section", "status": "DONE", "steps": "3",
        "elapsed_ms": "5", "verified": "", "jev_latency_p50_ms": "20", "jev_latency_max_ms": "30",
        "input_tokens_total": "30",
    }
    assert rows[1]["status"] == "error" and rows[1]["run_id"] == ""


def write_run(folder, run_id, status, verified, operations):
    meta = {"goal": "g", "url": "https://en.wikipedia.org/wiki/Main_Page", "status": status, "verified": verified}
    (folder / f"{run_id}.meta.json").write_text(json.dumps(meta))
    lines = []
    for n, operation in enumerate(operations, 1):
        lines.append(json.dumps({
            "step": n,
            "request": {
                "state": {"page": {"url": "u", "title": "t", "text": "x" * 2000}, "elements": [], "recent_actions": []},
                "questions": {"operation": {"type": "choice"}, "click_target": {"type": "choice"}},
            },
            "answers": {"operation": {"choice": operation}, "click_target": {"choice": "2"}},
        }))
    (folder / f"{run_id}.jsonl").write_text("\n".join(lines) + "\n")


def test_converter_keeps_verified_or_unverified_done_runs(tmp_path):
    write_run(tmp_path, "run-pass", "DONE", True, ["CLICK", "WAIT", "DONE"])
    write_run(tmp_path, "run-unchecked", "DONE", None, ["TYPE_TEXT", "BLOCKED"])
    write_run(tmp_path, "run-failed", "DONE", False, ["CLICK"])
    write_run(tmp_path, "run-blocked", "BLOCKED", None, ["CLICK"])
    (tmp_path / "summary.csv").write_text("run_id,tags\nrun-pass,wikipedia;section\n")

    counts = traces_to_kev.convert(tmp_path, tmp_path / "kev")
    assert counts["train"] + counts["heldout"] == 5
    assert counts["operation"] == {"CLICK": 1, "WAIT": 1, "DONE": 1, "TYPE_TEXT": 1, "BLOCKED": 1}
    assert counts["tag"] == {"wikipedia": 3, "section": 3, "unknown": 2}
    assert counts["host"] == {"en.wikipedia.org": 5}
    kev = tmp_path / "kev"
    records = [json.loads(line) for name in ("train", "heldout") for line in (kev / f"{name}.jsonl").open()]
    assert all(len(r["state"]["page"]["text"]) == 1500 for r in records)
    assert all(r["questions"]["click_target"]["label"] == "2" for r in records)
    assert all("label" in r["questions"]["operation"] for r in records)
    assert "label" not in json.loads((tmp_path / "run-pass.jsonl").read_text().splitlines()[0])["request"]["questions"]

    first = {name: (tmp_path / "kev" / f"{name}.jsonl").read_text() for name in ("train", "heldout")}
    trimmed = traces_to_kev.convert(tmp_path, tmp_path / "kev", drop_nonprogress=True)
    assert trimmed["operation"] == {"CLICK": 1, "DONE": 1, "TYPE_TEXT": 1}
    traces_to_kev.convert(tmp_path, tmp_path / "kev")
    assert first == {name: (tmp_path / "kev" / f"{name}.jsonl").read_text() for name in ("train", "heldout")}
    expected = int(hashlib.sha256(b"run-pass").hexdigest(), 16) % 100 < 85
    assert (traces_to_kev.split("run-pass") == "train") == expected
