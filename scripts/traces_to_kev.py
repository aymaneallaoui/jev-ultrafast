"""Turn DONE, not-failed-verification traces into labeled decision records split by run."""

import argparse
import copy
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

TEXT_LIMIT = 1500
NONPROGRESS = {"WAIT", "BLOCKED"}


def split(run_id):
    return "train" if int(hashlib.sha256(run_id.encode()).hexdigest(), 16) % 100 < 85 else "heldout"


def record(step):
    request = copy.deepcopy(step["request"])
    state, questions = request["state"], request["questions"]
    state["page"]["text"] = state["page"]["text"][:TEXT_LIMIT]
    for name, question in questions.items():
        if name in step["answers"]:
            question["label"] = step["answers"][name]["choice"]
    return {"state": state, "questions": questions}


def run_tags(trace_dir):
    path = trace_dir / "summary.csv"
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as file:
        return {row["run_id"]: row["tags"].split(";") for row in csv.DictReader(file) if row["run_id"] and row["tags"]}


def convert(trace_dir, out, drop_nonprogress=False):
    trace_dir, out = Path(trace_dir), Path(out)
    tags = run_tags(trace_dir)
    records = {"train": [], "heldout": []}
    counts = {"operation": Counter(), "tag": Counter(), "host": Counter()}
    for meta_path in sorted(trace_dir.glob("*.meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        run_id = meta_path.name.removesuffix(".meta.json")
        steps_path = trace_dir / f"{run_id}.jsonl"
        if meta.get("status") != "DONE" or meta.get("verified") not in (True, None) or not steps_path.exists():
            continue
        host = urlparse(meta.get("url", "")).hostname or "unknown"
        for line in steps_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            step = json.loads(line)
            operation = step["answers"].get("operation", {}).get("choice")
            if drop_nonprogress and operation in NONPROGRESS:
                continue
            records[split(run_id)].append(record(step))
            counts["operation"][operation] += 1
            counts["tag"].update(tags.get(run_id, ["unknown"]))
            counts["host"][host] += 1
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in records.items():
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as file:
            file.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    return {"train": len(records["train"]), "heldout": len(records["heldout"]), **counts}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_dir", nargs="?", default=os.environ.get("TRACE_DIR"))
    parser.add_argument("--out", help="Output folder (default: TRACE_DIR/kev).")
    parser.add_argument("--drop-nonprogress", action="store_true", help="Drop WAIT and BLOCKED decisions.")
    args = parser.parse_args(argv)
    if not args.trace_dir:
        parser.error("Pass a trace directory or set TRACE_DIR.")
    counts = convert(args.trace_dir, args.out or Path(args.trace_dir) / "kev", args.drop_nonprogress)
    print(f"records  train={counts['train']}  heldout={counts['heldout']}")
    for group in ("operation", "tag", "host"):
        print(f"per {group}:")
        for name, count in counts[group].most_common():
            print(f"  {name:<32}{count}")


if __name__ == "__main__":
    main()
