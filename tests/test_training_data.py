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


def test_compare_models_reports_rates_steps_and_latency_per_site(tmp_path):
    from scripts import compare_models

    def collection(name, rows):
        folder = tmp_path / name
        folder.mkdir()
        with open(folder / "summary.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["run_id", "task_id", "tags", "status", "steps", "verified", "jev_latency_p50_ms"])
            writer.writerows(rows)
        return str(folder)

    jev = collection("jev", [("a1", "w", "wikipedia;x", "DONE", "3", "true", "300"),
                             ("a2", "m", "mdn;y", "BLOCKED", "6", "false", "310"),
                             ("a0", "old", "mdn;y", "DONE", "9", "true", "999")])
    kev = collection("kev", [("b1", "w", "wikipedia;x", "DONE", "4", "false", "90"),
                             ("b2", "m", "mdn;y", "DONE", "5", "true", "80")])
    lines = compare_models.compare([("jev", jev, "a1"), ("kev", kev, None)], ids={"w", "m"})
    table = {(line["tag"], line["model"]): line for line in lines}
    assert table[("wikipedia", "jev")]["verified_rate"] == 1.0 and table[("wikipedia", "kev")]["verified_rate"] == 0.0
    assert table[("mdn", "jev")]["done_rate"] == 0.0 and table[("mdn", "kev")]["done_rate"] == 1.0
    assert table[("all", "jev")]["runs"] == 2 and table[("all", "jev")]["median_decision_ms"] == 305.0
    assert table[("all", "kev")]["median_steps"] == 4.5
    assert [line["tag"] for line in lines][-1] == "all"


def test_relabel_blocked_drops_only_hesitations_followed_by_same_page_progress(tmp_path):
    def run(run_id, verified, steps):
        write_run(tmp_path, run_id, "DONE", verified, [op for op, _, _ in steps])
        path = tmp_path / f"{run_id}.jsonl"
        lines = [json.loads(line) for line in path.read_text().splitlines()]
        for line, (_, url, failed) in zip(lines, steps):
            line["request"]["state"]["page"]["url"] = url
            if failed:
                line["failed_phase"] = "execute"
        path.write_text("".join(json.dumps(line) + "\n" for line in lines))

    run("hesitant", True, [("WAIT", "a", False), ("CLICK", "a", False),
                           ("WAIT", "a", False), ("CLICK", "b", False),
                           ("WAIT", "b", False), ("CLICK", "b", True), ("DONE", "b", False)])
    run("unchecked", None, [("WAIT", "a", False), ("CLICK", "a", False), ("DONE", "a", False)])
    write_summary(tmp_path, {"hesitant": "wikipedia", "unchecked": "wikipedia"})
    plain = traces_to_kev.convert(tmp_path, tmp_path / "kev")
    relabeled = traces_to_kev.convert(tmp_path, tmp_path / "kev", relabel_blocked=True)

    def total(report):
        return sum(report[name]["records"] for name in traces_to_kev.FILES)

    assert relabeled["relabeled_hesitations"] == 1
    assert total(plain) - total(relabeled) == 1
    assert plain["relabeled_hesitations"] == 0


def test_converter_merges_sources_and_skips_overridden_steps(tmp_path):
    jev, kev = tmp_path / "raw", tmp_path / "raw-4b"
    jev.mkdir(), kev.mkdir()
    write_run(jev, "jev-run", "DONE", True, ["CLICK", "DONE"])
    write_run(kev, "kev-run", "DONE", True, ["CLICK", "CLICK", "DONE"])
    path = kev / "kev-run.jsonl"
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    lines[1]["loop_guard"] = {"pattern": "cycle"}
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    write_summary(jev, {"jev-run": "wikipedia"})
    write_summary(kev, {"kev-run": "mdn"})
    audit = tmp_path / "audit-4b.csv"
    audit.write_text("run_id,step,operation,confidence,reason\nkev-run,3,DONE,0.4,low_confidence\n")
    report = traces_to_kev.convert([jev, kev], tmp_path / "kev", holdout_tags=["mdn"], exclude=[audit])
    assert report["sources"] == {str(jev): {"runs": 1, "records": 2}, str(kev): {"runs": 1, "records": 1}}
    assert report["overridden_steps"] == 1 and report["excluded_steps"] == 1
    assert report["heldout_sites"]["records"] == 1
