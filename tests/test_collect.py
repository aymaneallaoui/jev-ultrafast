"""Offline contracts for trace collection and conversion. No browser, network, or paid APIs."""

import argparse
import csv
import re
from collections import Counter
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from jev_ultrafast.tracing import Trace
from jev_ultrafast.verifiers import date_forms, echo, flights, hn_story, page
from scripts import collect, summary_by_tag

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
    assert len(tasks) == 95 and len({t["id"] for t in tasks}) == 95
    assert Counter(t["tags"][0] for t in tasks) == {
        "google_flights": 12, "wikipedia": 10, "hackernews": 6, "github": 8, "mdn": 5,
        "openstreetmap": 4, "forms": 21, "youtube": 4, "ebay": 5, "stackoverflow": 2, "arxiv": 2, "amazon": 2,
        "booking": 2, "imdb": 2, "npm": 2, "pypi": 2, "reuters": 2, "weather": 2, "huggingface": 2,
    }
    for task in tasks:
        assert "{" not in task["goal"] and "Stop when" in task["goal"]
        assert (task["verify"] == "flights") == (task["tags"][0] == "google_flights")
        if task["verify"] == "flights":
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
            self.state = {"page": {"url": url}}

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
        "input_tokens_total": "30", "model_tag": "",
    }
    assert rows[1]["status"] == "error" and rows[1]["run_id"] == ""


def test_every_task_has_an_independent_verifier_and_no_password():
    tasks = collect.load_tasks(collect.ROOT / "tasks.yaml", date(2026, 9, 28))
    assert all(t["verify"] in collect.VERIFIERS for t in tasks)
    assert not any("password" in t["goal"].lower() for t in tasks)


def observed(url, text="", actions=()):
    return {"url": url, "text": text, "actions": list(actions)}


def test_page_verifier_decodes_urls_and_checks_text_and_fields():
    form = observed(
        "https://httpbin.org/forms/post",
        actions=[
            {"label": "Customer name:", "value": "Ada Lovelace"},
            {"label": "Medium", "value": "medium", "checked": "true"},
            {"label": "Bacon", "value": "bacon", "checked": "false"},
            {"label": "Dropdown (select) → Two", "kind": "select", "current_value": "Two", "value": "2"},
        ],
    )
    fields = {"Customer name": "ada lovelace", "Medium": True, "Bacon": False, "Dropdown (select)": "Two"}
    assert page(form, url=r"httpbin\.org/forms/post$", fields=fields)["passed"]
    assert not page(form, fields={"Bacon": True})["passed"]
    assert not page(form, fields={"Missing": "x"})["passed"]
    ebay = observed("https://www.ebay.com/sch/i.html?_nkw=film%20camera&rt=nc&_udhi=100")
    assert page(ebay, url=[r"_nkw=[^&]*film camera", r"[?&]_udhi=100(&|$)"])["passed"]
    assert not page(ebay, url=r"[?&]_udhi=1000(&|$)")["passed"]
    wiki = observed("https://en.wikipedia.org/wiki/G%C3%B6del%27s_incompleteness_theorems", "Gödel's theorems")
    assert page(wiki, url=r"wiki/Gödel's_incompleteness_theorems(#|$)", text="godel's")["passed"]


def front_page():
    rows = [("1.\t\n\tFirst story (a.com)", ["vote?id=11&how=up&goto=news", "https://a.com/post/", "from?site=a.com"]),
            ("2.\t\n\tAsk HN: Second", ["vote?id=22&how=up&goto=news", "item?id=22"])]
    actions, guards, node = [], {}, 0
    for scope, hrefs in rows:
        for href in hrefs:
            node += 1
            actions.append({"kind": "click", "node": node, "label": href})
            guards[str(node)] = [node] + [None] * 11 + [href, scope]
    return {"url": "https://news.ycombinator.com/", "actions": actions, "guards": guards}


def test_hn_verifier_follows_the_ranked_story_from_the_initial_front_page():
    initial = front_page()
    assert hn_story(observed("https://a.com/post"), initial=initial, rank=1)["passed"]
    assert hn_story(observed("https://www.a.com/post/"), initial=initial, rank=1)["passed"]
    assert not hn_story(observed("https://b.com/post"), initial=initial, rank=1)["passed"]
    assert hn_story(observed("https://news.ycombinator.com/item?id=22"), initial=initial, rank=2)["passed"]
    thread = "https://news.ycombinator.com/item?id="
    assert hn_story(observed(thread + "11"), initial=initial, rank=1, comments=True)["passed"]
    assert not hn_story(observed(thread + "22"), initial=initial, rank=1, comments=True)["passed"]
    assert not hn_story(observed(thread + "33"), initial=initial, rank=3, comments=True)["passed"]


def test_collector_passes_the_initial_page_to_verifiers_that_need_it(monkeypatch):
    initial, final = front_page(), observed("https://news.ycombinator.com/item?id=11")

    class FakeAgent:
        def __init__(self, url, goal):
            self.state = {"page": initial}
            self.trace = SimpleNamespace(set_verified=lambda value: seen.append(value))

        def run(self):
            return iter(())

        def snapshot(self):
            return {"page": final}

        def close(self):
            pass

    seen = []
    monkeypatch.setattr(collect, "Agent", FakeAgent)
    task = {"id": "hn", "url": initial["url"], "goal": "g", "verify": "hn_story",
            "verify_args": {"rank": 1, "comments": True}}
    collect.run_task(task)
    assert seen == [True]


def test_page_verifier_checks_unlabeled_values_and_checked_counts():
    boxes = observed("https://the-internet.herokuapp.com/checkboxes", actions=[
        {"node": 1, "label": "checkbox", "checked": "true"},
        {"node": 2, "label": "checkbox", "checked": "false"},
        {"node": 3, "label": "Please select → Option 2", "kind": "select", "current_value": "Option 2"},
    ])
    assert page(boxes, values=["Option 2"], checked={"checkbox": 1})["passed"]
    assert not page(boxes, checked={"checkbox": 2})["passed"]
    assert not page(boxes, values=["Option 1"])["passed"]


def test_echo_verifier_requires_the_result_page_and_every_value():
    result = observed("https://httpbin.org/post", '"form": { "custname": "Test User", "size": "medium" }')
    assert echo(result, url=r"httpbin\.org/post$", values=['"custname": "Test User"', '"size": "medium"'])["passed"]
    assert not echo(result, url=r"httpbin\.org/post$", values=['"topping": "bacon"'])["passed"]
    assert not echo(observed("https://httpbin.org/forms/post", result["text"]), url=r"httpbin\.org/post$",
                    values=['"custname": "Test User"'])["passed"]


def test_summary_by_tag_splits_verification_and_counts_usable_steps(tmp_path):
    rows = [
        ("a", "forms;demoqa", "DONE", "3", "true"),
        ("b", "forms", "DONE", "5", ""),
        ("c", "forms", "DONE", "7", "false"),
        ("d", "forms;demoqa", "BLOCKED", "2", "true"),
        ("", "forms", "error", "", ""),
    ]
    with open(tmp_path / "summary.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["run_id", "task_id", "tags", "status", "steps", "verified"])
        writer.writerows([(run, "t", tags, status, steps, verified) for run, tags, status, steps, verified in rows])
    lines = {line["tag"]: line for line in summary_by_tag.table(summary_by_tag.load(tmp_path))}
    forms = lines["forms"]
    assert (forms["runs"], forms["DONE"], forms["BLOCKED"]) == (4, 3, 1)
    assert (forms["verified_true"], forms["verified_false"], forms["verified_null"]) == (2, 1, 1)
    assert forms["usable_steps"] == 8 and lines["all"]["usable_steps"] == 8
    assert lines["demoqa"]["usable_steps"] == 3


def test_summary_gains_the_model_tag_column_without_losing_old_rows(tmp_path):
    path = tmp_path / "summary.csv"
    old_fields = [f for f in collect.SUMMARY_FIELDS if f != "model_tag"]
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=old_fields)
        writer.writeheader()
        writer.writerow({f: "x" for f in old_fields} | {"run_id": "old"})
    collect.append_summary(path, {f: "" for f in collect.SUMMARY_FIELDS} | {"run_id": "new", "model_tag": "jev-08b"})
    rows = list(csv.DictReader(open(path)))
    assert list(rows[0]) == collect.SUMMARY_FIELDS
    assert [(r["run_id"], r["model_tag"]) for r in rows] == [("old", ""), ("new", "jev-08b")]
    assert rows[0]["status"] == "x"


def test_ids_select_tasks_from_a_list_or_file(tmp_path, capsys):
    ids = tmp_path / "ids.txt"
    ids.write_text("# comment\nwikipedia-open-turing\n\nmdn-fetch\n")
    collect.main(["--ids", f"@{ids}", "--dry-run"])
    printed = [line.split("#")[0] for line in capsys.readouterr().out.splitlines()]
    assert printed == ["wikipedia-open-turing", "mdn-fetch"]
    with pytest.raises(SystemExit):
        collect.main(["--ids", "no-such-task", "--dry-run"])


def test_smoke_ids_are_real_tasks():
    tasks = {t["id"] for t in collect.load_tasks(collect.ROOT / "tasks.yaml", TODAY)}
    assert collect.task_ids(f"@{collect.ROOT / 'scripts' / 'smoke_ids.txt'}") <= tasks
