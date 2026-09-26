"""Offline contracts for the audit and the kev converter. No browser, network, or model."""

import csv
import hashlib
import json

from scripts import audit_traces, traces_to_kev

OPERATIONS = {"CLICK": "c", "TYPE_TEXT": "t", "WAIT": "w", "DONE": "d"}
TARGETS = {"1": {"element": "[1] A"}, "2": {"element": "[2] B"}}


def write_run(folder, run_id, status, verified, operations, confidence=0.9, url="https://en.wikipedia.org/wiki/Main_Page"):
    meta = {"goal": "g", "url": url, "status": status, "verified": verified}
    (folder / f"{run_id}.meta.json").write_text(json.dumps(meta))
    lines = []
    for n, operation in enumerate(operations, 1):
        lines.append(json.dumps({
            "event": "step",
            "step": n,
            "request": {
                "state": {"page": {"url": "u", "title": "t", "text": "x" * 2000},
                          "elements": [{"index": "1"}, {"index": "2"}], "recent_actions": []},
                "questions": {
                    "operation": {"type": "choice", "criteria": OPERATIONS},
                    "click_target": {"type": "choice", "criteria": TARGETS},
                    "type_text_target": {"type": "choice", "criteria": {"1": {"element": "[1] A"}}},
                },
            },
            "answers": {"operation": {"choice": operation, "confidence": confidence},
                        "click_target": {"choice": "2"}, "type_text_target": {"choice": "1"}},
        }))
    (folder / f"{run_id}.jsonl").write_text("\n".join(lines) + "\n")


def write_summary(folder, tags):
    with open(folder / "summary.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["run_id", "tags"])
        writer.writerows(tags.items())


def records(folder, name):
    return [json.loads(line) for line in (folder / f"{name}.jsonl").open()]


def test_converter_keeps_the_operation_and_the_executed_target_head_only(tmp_path):
    write_run(tmp_path, "run-pass", "DONE", True, ["CLICK", "TYPE_TEXT", "DONE"])
    write_run(tmp_path, "run-unchecked", "DONE", None, ["CLICK"])
    write_run(tmp_path, "run-failed", "DONE", False, ["CLICK"])
    write_run(tmp_path, "run-blocked", "BLOCKED", None, ["CLICK"])
    assert traces_to_kev.convert(tmp_path, tmp_path / "kev")["skipped_untagged_runs"] == 2
    report = traces_to_kev.convert(tmp_path, tmp_path / "kev", include_untagged=True)
    rows = records(tmp_path / "kev", "train") + records(tmp_path / "kev", "heldout")
    assert report["train"]["records"] + report["heldout"]["records"] == 4
    by_operation = {r["questions"]["operation"]["label"]: r for r in rows}
    assert set(by_operation["CLICK"]["questions"]) == {"operation", "click_target"}
    assert set(by_operation["TYPE_TEXT"]["questions"]) == {"operation", "type_text_target"}
    assert set(by_operation["DONE"]["questions"]) == {"operation"}
    assert by_operation["CLICK"]["questions"]["click_target"]["label"] == "2"
    assert by_operation["CLICK"]["questions"]["click_target"]["src"] == "jev_click_target"
    assert all(len(r["state"]["page"]["text"]) == 1500 for r in rows)
    assert "label" not in json.loads((tmp_path / "run-pass.jsonl").read_text().splitlines()[0])["request"]["questions"]
    assert (traces_to_kev.split("run-pass") == "train") == (int(hashlib.sha256(b"run-pass").hexdigest(), 16) % 100 < 85)


def test_holdout_tags_exclusions_and_nonprogress(tmp_path):
    write_run(tmp_path, "run-wiki", "DONE", True, ["CLICK", "WAIT", "DONE"])
    write_run(tmp_path, "run-map", "DONE", True, ["CLICK", "DONE"], url="https://www.openstreetmap.org/")
    write_summary(tmp_path, {"run-wiki": "wikipedia;section", "run-map": "openstreetmap;search"})
    exclude = tmp_path / "audit.csv"
    exclude.write_text("run_id,step,operation,confidence,reason\nrun-wiki,1,CLICK,0.4,low_confidence\n")
    report = traces_to_kev.convert(tmp_path, tmp_path / "kev", drop_nonprogress=True,
                                   holdout_tags=["openstreetmap"], exclude=exclude)
    assert report["heldout_sites"]["records"] == 2 and report["heldout_sites"]["runs"] == 1
    assert report["train"]["records"] + report["heldout"]["records"] == 1
    assert report["excluded_steps"] == 1


def test_cap_keeps_a_tag_below_its_share_of_training_records(tmp_path, monkeypatch):
    monkeypatch.setattr(traces_to_kev, "split", lambda run_id: "train")
    for n in range(6):
        write_run(tmp_path, f"flight-{n}", "DONE", True, ["CLICK"] * 5)
    write_run(tmp_path, "wiki", "DONE", True, ["CLICK"] * 10)
    write_summary(tmp_path, {**{f"flight-{n}": "google_flights;one_way" for n in range(6)}, "wiki": "wikipedia"})
    report = traces_to_kev.convert(tmp_path, tmp_path / "kev", caps={"google_flights": 0.35})
    flights = report["tag"].get("google_flights", 0)
    assert flights / report["train"]["records"] <= 0.35
    assert report["dropped_runs_by_cap"]["google_flights"] == 5 and flights == 5


def test_converter_ignores_timing_lines_and_reads_old_traces(tmp_path):
    write_run(tmp_path, "run-new", "DONE", True, ["CLICK", "DONE"])
    path = tmp_path / "run-new.jsonl"
    decisions = [json.loads(line) for line in path.read_text().splitlines()]
    old = [{k: v for k, v in d.items() if k != "event"} for d in decisions]
    lines = [{"event": "step_start", "step": 1, "t_ms": 0, "url": "u"},
             {"event": "step_failed", "step": 1, "failed_phase": "snapshot", "error": "stale"}] + decisions
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    write_run(tmp_path, "run-old", "DONE", True, [])
    (tmp_path / "run-old.jsonl").write_text("".join(json.dumps(line) + "\n" for line in old))
    write_summary(tmp_path, {"run-new": "wikipedia", "run-old": "wikipedia"})
    report = traces_to_kev.convert(tmp_path, tmp_path / "kev")
    assert report["train"]["records"] + report["heldout"]["records"] == 4


def test_audit_flags_low_confidence_and_stalls_in_successful_runs_only(tmp_path):
    write_run(tmp_path, "ok", "DONE", True, ["CLICK", "WAIT", "DONE"])
    write_run(tmp_path, "shaky", "DONE", None, ["CLICK"], confidence=0.3)
    write_run(tmp_path, "failed", "DONE", False, ["WAIT"])
    write_run(tmp_path, "blocked", "BLOCKED", None, ["WAIT"], confidence=0.1)
    flagged = {(r["run_id"], r["step"]): r["reason"] for r in audit_traces.audit(tmp_path)}
    assert flagged[("ok", 2)] == "wait_in_successful_run"
    assert flagged[("shaky", 1)] == "low_confidence"
    assert ("failed", 1) not in flagged and ("blocked", 1) not in flagged
