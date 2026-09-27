"""Group TRACE_DIR/summary.csv by tag, with verification split and usable decision steps."""

import argparse
import csv
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

FIELDS = [
    "tag", "runs", "DONE", "verified_true", "verified_false", "verified_null", "BLOCKED", "error",
    "median_steps", "usable_steps",
]


def usable(row):
    return row["status"] == "DONE" and row["verified"] != "false"


def summarize(tag, rows):
    steps = [int(row["steps"]) for row in rows if row["steps"]]
    return {
        "tag": tag,
        "runs": len(rows),
        "DONE": sum(row["status"] == "DONE" for row in rows),
        "verified_true": sum(row["verified"] == "true" for row in rows),
        "verified_false": sum(row["verified"] == "false" for row in rows),
        "verified_null": sum(row["verified"] not in ("true", "false") for row in rows),
        "BLOCKED": sum(row["status"] == "BLOCKED" for row in rows),
        "error": sum(row["status"] not in ("DONE", "BLOCKED") for row in rows),
        "median_steps": statistics.median(steps) if steps else "",
        "usable_steps": sum(int(row["steps"]) for row in rows if usable(row) and row["steps"]),
    }


def table(rows):
    groups = defaultdict(list)
    for row in rows:
        for tag in filter(None, row["tags"].split(";")):
            groups[tag].append(row)
    lines = [summarize(tag, members) for tag, members in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))]
    return lines + [summarize("all", rows)]


def load(trace_dir, since=None, run_ids=None):
    with open(Path(trace_dir) / "summary.csv", newline="", encoding="utf-8") as file:
        rows = [row for row in csv.DictReader(file) if row["run_id"]]
    if since:
        rows = [row for row in rows if row["run_id"] >= since]
    if run_ids is not None:
        rows = [row for row in rows if row["run_id"] in run_ids]
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_dir", nargs="?", default=os.environ.get("TRACE_DIR"))
    parser.add_argument("--since", help="Only runs whose run_id sorts at or after this, e.g. 20260928T1840.")
    parser.add_argument("--out", type=Path, help="Also write the table as CSV.")
    args = parser.parse_args(argv)
    if not args.trace_dir:
        parser.error("Pass a trace directory or set TRACE_DIR.")
    lines = table(load(args.trace_dir, args.since))
    writer = csv.DictWriter(sys.stdout, fieldnames=FIELDS, delimiter="\t")
    writer.writeheader()
    writer.writerows(lines)
    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as file:
            out = csv.DictWriter(file, fieldnames=FIELDS)
            out.writeheader()
            out.writerows(lines)
    print(f"usable_steps total: {lines[-1]['usable_steps']}")


if __name__ == "__main__":
    main()
