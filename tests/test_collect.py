"""Offline contracts for trace collection and conversion. No browser, network, or paid APIs."""

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import date
from pathlib import Path
from types import SimpleNamespace

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
            arguments = task["verify_args"]
            assert {"origin", "destination", "day", "one_way", "adults"} <= set(arguments)
            assert date_forms(arguments["day"])["goal"] in task["goal"]


def test_round_trip_tasks_verify_the_goal_return_date():
    raw = {t["id"]: t for t in collect.yaml.safe_load((ROOT / "tasks.yaml").read_text())}
    round_trips = [t for t in raw.values() if "round_trip" in t["tags"]]
    assert round_trips
    for task in round_trips:
        returning = re.search(r"returning \{date\+(\d+)\}", task["goal"]).group(1)
        assert task["verify_args"]["return_date"] == f"{{date+{returning}:%Y-%m-%d}}"


def test_round_trip_task_without_return_date_fails_to_load(tmp_path):
    path = tmp_path / "tasks.yaml"
    path.write_text(
        "- {id: t, url: u, goal: g, tags: [google_flights], verify: flights,\n"
        "   verify_args: {origin: A, destination: B, date: '{date+10:%Y-%m-%d}', one_way: false}}\n"
    )
    with pytest.raises(ValueError, match="return_date"):
        collect.load_tasks(path, TODAY)


def flight_page(**values):
    fields = {
        "Change ticket type. Round trip": "Round trip",
        "Where from?": "Zürich",
        "Where to?": "New York, NY",
        "Departure": "Mon, Oct 12",
        "Return": "Mon, Oct 19",
        **values,
    }
    return {
        "url": "https://www.google.com/travel/flights/search?tfs=example",
        "text": "departing 2026-10-12",
        "actions": [
            *({"label": label, "value": value} for label, value in fields.items()),
            {"label": "2 passengers, change number of passengers.", "role": "button"},
            {"label": "Passenger assistance", "role": "button"},
            {"label": "Nonstop flight on Monday, October 12. Select flight", "value": ""},
        ],
    }


ROUND_TRIP = {"day": date(2026, 10, 12), "return_day": date(2026, 10, 19), "one_way": False}


@pytest.mark.parametrize("origin, destination", [("Zurich", "new york"), ("ZÜRICH", "New  York")])
def test_flight_cities_match_across_accents_case_and_suffixes(origin, destination):
    assert flights(flight_page(), origin=origin, destination=destination, **ROUND_TRIP)["passed"]
    page = flight_page(**{"Where from?": "Zurich"})
    assert flights(page, origin="Zürich", destination="New York", **ROUND_TRIP)["checks"]["origin"]


@pytest.mark.parametrize("value", ["", None, "Geneva"])
def test_flight_city_mismatch_or_empty_value_fails(value):
    checks = flights(flight_page(**{"Where from?": value}), origin="Zurich", destination="New York", **ROUND_TRIP)
    assert not checks["checks"]["origin"] and not checks["passed"]


def test_flight_passenger_count_is_parsed_from_buttons():
    route = {"origin": "Zurich", "destination": "New York", **ROUND_TRIP}
    assert flights(flight_page(), adults=2, **route)["checks"]["passengers"]
    assert not flights(flight_page(), adults=1, **route)["checks"]["passengers"]
    single = flight_page()
    single["actions"][5]["label"] = "1 passenger, change number of passengers."
    assert flights(single, adults=1, **route)["checks"]["passengers"]
    assert "passengers" not in flights(flight_page(), **route)["checks"]


def test_round_trip_checks_the_return_date():
    route = {"origin": "Zurich", "destination": "New York", "day": date(2026, 10, 12), "one_way": False}
    assert flights(flight_page(), return_day=date(2026, 10, 19), **route)["checks"]["return_date"]
    assert not flights(flight_page(), return_day=date(2026, 10, 20), **route)["passed"]
    assert not flights(flight_page(Return=""), return_day=date(2026, 10, 19), **route)["passed"]
    with pytest.raises(ValueError, match="return_day"):
        flights(flight_page(), **route)
    assert not flights(flight_page(), origin="Zurich", destination="New York", day=date(2026, 10, 12))["passed"]


@pytest.mark.parametrize("text, seconds", [("90", 90), ("45s", 45), ("90m", 5400), ("2h", 7200)])
def test_max_runtime_accepts_seconds_or_suffixes(text, seconds):
    assert collect.duration(text) == seconds


@pytest.mark.parametrize("text", ["", "0", "1.5h", "2d", "-5", "m"])
def test_max_runtime_rejects_invalid_values(text):
    with pytest.raises(argparse.ArgumentTypeError):
        collect.duration(text)


@pytest.mark.parametrize("step_seconds, steps, stopped, statuses, clock_end", [
    (30, None, [120], ["timeout"], 120),
    (99, 1, [], ["closed"], 99),
])
def test_batch_deadline_stops_the_current_run_and_skips_the_rest(
    tmp_path, monkeypatch, capsys, step_seconds, steps, stopped, statuses, clock_end
):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(collect, "time", SimpleNamespace(
        monotonic=lambda: clock.now, sleep=lambda seconds: setattr(clock, "now", clock.now + seconds),
    ))
    finished = []

    class FakeAgent:
        def __init__(self, url, goal):
            self.trace = Trace(url, goal)

        def run(self):
            taken = 0
            while steps is None or taken < steps:
                clock.now += step_seconds
                taken += 1
                yield {}

        def finish_trace(self, status):
            finished.append(clock.now)
            self.trace.finish(status, 0)

        def close(self):
            self.trace.finish("closed", 0)

    monkeypatch.setattr(collect, "Agent", FakeAgent)
    monkeypatch.setenv("TRACE_DIR", str(tmp_path))
    tasks = tmp_path / "tasks.yaml"
    tasks.write_text("".join(f"- {{id: t{n}, url: u, goal: g, tags: [x]}}\n" for n in range(3)))
    collect.main(["--tasks", str(tasks), "--max-runtime", "100"])
    assert finished == stopped and clock.now == clock_end
    rows = list(csv.DictReader((tmp_path / "summary.csv").open()))
    assert [row["status"] for row in rows] == statuses
    output = capsys.readouterr().out
    assert "skipped 2 runs" in output and re.search(r"skipped\s+2", output)


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


def test_converter_ignores_timing_lines_and_reads_old_traces(tmp_path):
    write_run(tmp_path, "run-old", "DONE", True, ["CLICK", "DONE"])
    write_run(tmp_path, "run-new", "DONE", True, ["CLICK", "DONE"])
    path = tmp_path / "run-new.jsonl"
    decisions = [json.loads(line) for line in path.read_text().splitlines()]
    lines = [
        {"event": "step_start", "step": 1, "t_ms": 0, "url": "u"},
        {"event": "step_failed", "step": 1, "snapshot_ms": 3, "failed_phase": "snapshot", "error": "stale"},
    ]
    for n, decision in enumerate(decisions, 1):
        lines += [
            {"event": "step_start", "step": n, "t_ms": 10 * n, "url": "u"},
            {**decision, "event": "step", "snapshot_ms": 1, "model_ms": 2, "execute_ms": 3, "wait_ms": 4},
        ]
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    counts = traces_to_kev.convert(tmp_path, tmp_path / "kev")
    assert counts["train"] + counts["heldout"] == 4
    assert counts["operation"] == {"CLICK": 2, "DONE": 2}
    kev = tmp_path / "kev"
    records = [json.loads(line) for name in ("train", "heldout") for line in (kev / f"{name}.jsonl").open()]
    assert all(set(r) == {"state", "questions"} for r in records)
