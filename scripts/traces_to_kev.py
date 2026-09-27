"""Turn DONE, not-failed-verification traces into kev labelled requests split by run.

Each record keeps the operation question and the target head of the chosen operation only: the other heads are
speculative answers that were never executed, so they are not labels."""

import argparse
import copy
import csv
import hashlib
import json
import os
import statistics
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

TEXT_LIMIT = 1500
NONPROGRESS = {"WAIT", "BLOCKED"}
PROGRESS = {"CLICK", "TYPE_TEXT"}
FILES = ("train", "heldout", "heldout_sites")


def run_hash(run_id):
    return int(hashlib.sha256(run_id.encode()).hexdigest(), 16)


def split(run_id):
    return "train" if run_hash(run_id) % 100 < 85 else "heldout"


def record(step):
    """A labelled request for the operation and, when it has one, the chosen operation's target."""
    request = copy.deepcopy(step["request"])
    state, questions, answers = request["state"], request["questions"], step["answers"]
    state["page"]["text"] = state["page"]["text"][:TEXT_LIMIT]
    operation = answers["operation"]["choice"]
    kept = {}
    for name in ("operation", f"{operation.lower()}_target"):
        if name in questions and name in answers:
            kept[name] = {**questions[name], "label": answers[name]["choice"], "src": f"jev_{name}"}
    return {"state": state, "questions": kept}


def run_tags(trace_dir):
    path = trace_dir / "summary.csv"
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as file:
        return {row["run_id"]: row["tags"].split(";") for row in csv.DictReader(file) if row["run_id"] and row["tags"]}


def load_exclusions(path):
    if not path:
        return set()
    with open(path, newline="", encoding="utf-8") as file:
        return {(row["run_id"], int(row["step"])) for row in csv.DictReader(file)}


def operation(step):
    return step["answers"].get("operation", {}).get("choice")


def hesitations(steps):
    """Indices of BLOCKED/WAIT decisions directly followed by a CLICK or TYPE_TEXT that executed on the same page."""
    found = set()
    for i, (step, following) in enumerate(zip(steps, steps[1:])):
        if (operation(step) in NONPROGRESS and operation(following) in PROGRESS and not following.get("failed_phase")
                and following["request"]["state"]["page"]["url"] == step["request"]["state"]["page"]["url"]):
            found.add(i)
    return found


def run_records(trace_dir, run_id, drop_nonprogress, excluded, relabel_blocked=False, counts=None):
    steps = []
    for line in (trace_dir / f"{run_id}.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            step = json.loads(line)
            if step.get("event", "step") == "step":
                steps.append(step)
    dropped = hesitations(steps) if relabel_blocked and not drop_nonprogress else set()
    dropped = {i for i in dropped if (run_id, steps[i].get("step")) not in excluded}
    if counts is not None:
        counts["relabeled_hesitations"] = counts.get("relabeled_hesitations", 0) + len(dropped)
    rows = []
    for i, step in enumerate(steps):
        if i in dropped or (run_id, step.get("step")) in excluded:
            continue
        if drop_nonprogress and operation(step) in NONPROGRESS:
            continue
        rows.append(record(step))
    return rows


def cap(runs, tag, fraction):
    """Keep runs carrying `tag` (in hash order) only while their records stay at most `fraction` of the file."""
    capped = [r for r in runs if tag in r["tags"]]
    others = sum(len(r["records"]) for r in runs if tag not in r["tags"])
    budget, kept, used = fraction / (1 - fraction) * others, [], 0
    for run in sorted(capped, key=lambda r: run_hash(r["run_id"])):
        if used + len(run["records"]) <= budget:
            kept.append(run)
            used += len(run["records"])
    dropped = {r["run_id"] for r in capped} - {r["run_id"] for r in kept}
    return [r for r in runs if r["run_id"] not in dropped], len(dropped)


def stats(rows):
    operations = Counter(r["questions"]["operation"]["label"] for r in rows)
    targets = {q["label"] for r in rows for name, q in r["questions"].items() if name != "operation"}
    elements = [len(r["state"]["elements"]) for r in rows]
    chars = [len(json.dumps(r["state"], ensure_ascii=False)) for r in rows]
    return {
        "records": len(rows),
        "operations": dict(operations.most_common()),
        "distinct_target_indices": len(targets),
        "mean_elements": round(statistics.mean(elements), 1) if elements else 0,
        "mean_state_chars": round(statistics.mean(chars)) if chars else 0,
    }


def convert(trace_dir, out, drop_nonprogress=False, holdout_tags=(), caps=None, exclude=None, include_untagged=False,
            relabel_blocked=False):
    trace_dir, out = Path(trace_dir), Path(out)
    tags, excluded = run_tags(trace_dir), load_exclusions(exclude)
    runs = {name: [] for name in FILES}
    hosts, tag_counts, untagged, hesitation_counts = Counter(), Counter(), 0, {}
    for meta_path in sorted(trace_dir.glob("*.meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        run_id = meta_path.name.removesuffix(".meta.json")
        if meta.get("status") != "DONE" or meta.get("verified") not in (True, None):
            continue
        if not (trace_dir / f"{run_id}.jsonl").exists():
            continue
        if run_id not in tags and not include_untagged:
            untagged += 1
            continue
        relabel = relabel_blocked and meta.get("verified") is True
        rows = run_records(trace_dir, run_id, drop_nonprogress, excluded, relabel, hesitation_counts)
        if not rows:
            continue
        run_tag_list = tags.get(run_id, ["unknown"])
        destination = "heldout_sites" if set(run_tag_list) & set(holdout_tags) else split(run_id)
        runs[destination].append({"run_id": run_id, "tags": run_tag_list, "records": rows,
                                  "host": urlparse(meta.get("url", "")).hostname or "unknown"})
    dropped = {}
    for tag, fraction in (caps or {}).items():
        runs["train"], dropped[tag] = cap(runs["train"], tag, fraction)
    out.mkdir(parents=True, exist_ok=True)
    report = {"dropped_runs_by_cap": dropped, "excluded_steps": len(excluded), "skipped_untagged_runs": untagged,
              "relabeled_hesitations": hesitation_counts.get("relabeled_hesitations", 0)}
    for name, members in runs.items():
        rows = [row for run in members for row in run["records"]]
        for run in members:
            hosts[run["host"]] += len(run["records"])
            tag_counts.update({tag: len(run["records"]) for tag in run["tags"]})
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as file:
            file.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        report[name] = {"runs": len(members), **stats(rows)}
    report["tag"], report["host"] = dict(tag_counts.most_common()), dict(hosts.most_common())
    return report


def parse_caps(values):
    caps = {}
    for value in values:
        tag, _, fraction = value.partition("=")
        caps[tag] = float(fraction)
        if not 0 < caps[tag] < 1:
            raise argparse.ArgumentTypeError(f"cap for {tag} must be between 0 and 1")
    return caps


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_dir", nargs="?", default=os.environ.get("TRACE_DIR"))
    parser.add_argument("--out", help="Output folder (default: TRACE_DIR/kev).")
    parser.add_argument("--drop-nonprogress", action="store_true", help="Drop WAIT and BLOCKED decisions.")
    parser.add_argument("--holdout-tags", default="", help="Comma-separated tags sent only to heldout_sites.jsonl.")
    parser.add_argument("--cap-tag", action="append", default=[], metavar="TAG=FRACTION",
                        help="Keep runs with TAG to at most FRACTION of training records.")
    parser.add_argument("--exclude", help="CSV with run_id,step columns (scripts/audit_traces.py output).")
    parser.add_argument("--relabel-blocked", action="store_true",
                        help="In verified DONE runs, drop BLOCKED/WAIT steps that a same-page CLICK/TYPE_TEXT follows.")
    parser.add_argument("--include-untagged", action="store_true",
                        help="Keep runs missing from summary.csv; they cannot be held out or capped by tag.")
    args = parser.parse_args(argv)
    if not args.trace_dir:
        parser.error("Pass a trace directory or set TRACE_DIR.")
    report = convert(args.trace_dir, args.out or Path(args.trace_dir) / "kev", args.drop_nonprogress,
                     [t for t in args.holdout_tags.split(",") if t], parse_caps(args.cap_tag), args.exclude,
                     args.include_untagged, args.relabel_blocked)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
