"""Compare decision models end to end from their collections' summary.csv files: per site tag, the DONE rate,
verified rate, median steps, and median decision latency (each run's median TypeSafe/Kev round trip)."""

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect import task_ids  # noqa: E402
from summary_by_tag import load  # noqa: E402

FIELDS = ["tag", "model", "runs", "done_rate", "verified_rate", "median_steps", "median_decision_ms"]


def median(values):
    values = [float(v) for v in values if v not in ("", None)]
    return round(statistics.median(values), 1) if values else ""


def compare(models, ids=None):
    """models: [(label, trace_dir, since)] -> rows per (site tag, model), plus an `all` row per model."""
    lines = []
    for label, trace_dir, since in models:
        rows = load(trace_dir, since)
        if ids:
            rows = [row for row in rows if row["task_id"] in ids]
        groups = defaultdict(list)
        for row in rows:
            groups[row["tags"].split(";")[0]].append(row)
        groups["all"] = rows
        for tag, members in groups.items():
            n = len(members) or 1
            lines.append({
                "tag": tag, "model": label, "runs": len(members),
                "done_rate": round(sum(r["status"] == "DONE" for r in members) / n, 3),
                "verified_rate": round(sum(r["verified"] == "true" for r in members) / n, 3),
                "median_steps": median(r["steps"] for r in members),
                "median_decision_ms": median(r["jev_latency_p50_ms"] for r in members),
            })
    order = {label: n for n, (label, _, _) in enumerate(models)}
    return sorted(lines, key=lambda line: (line["tag"] == "all", line["tag"], order[line["model"]]))


def parse_model(value):
    label, _, rest = value.partition("=")
    trace_dir, _, since = rest.partition("@")
    if not label or not trace_dir:
        raise argparse.ArgumentTypeError("use LABEL=TRACE_DIR or LABEL=TRACE_DIR@SINCE_RUN_ID")
    return label, trace_dir, since or None


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", type=parse_model, required=True, metavar="LABEL=DIR[@SINCE]",
                        help="A collection to compare; @SINCE keeps runs whose run_id sorts at or after it.")
    parser.add_argument("--ids", help="Only these task ids: comma-separated, or @file (e.g. @scripts/smoke_ids.txt).")
    parser.add_argument("--out", type=Path, help="Also write the table as CSV.")
    args = parser.parse_args(argv)
    lines = compare(args.model, task_ids(args.ids) if args.ids else None)
    print("| " + " | ".join(FIELDS) + " |\n|" + "---|" * len(FIELDS))
    for line in lines:
        print("| " + " | ".join(str(line[field]) for field in FIELDS) + " |")
    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(lines)


if __name__ == "__main__":
    main()
