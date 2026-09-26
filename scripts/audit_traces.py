"""List decision steps worth a human look before training: a DONE run's low-confidence operations, and BLOCKED/WAIT
choices in runs that still succeeded. Nothing is relabeled; traces_to_kev.py --exclude drops the listed steps."""

import argparse
import csv
import json
import os
from pathlib import Path

LOW_CONFIDENCE = 0.5
STALL = {"BLOCKED", "WAIT"}


def decisions(path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            step = json.loads(line)
            if step.get("event", "step") == "step" and "answers" in step:
                yield step


def audit(trace_dir, threshold=LOW_CONFIDENCE):
    rows = []
    for meta_path in sorted(Path(trace_dir).glob("*.meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        run_id = meta_path.name.removesuffix(".meta.json")
        steps_path = meta_path.with_name(f"{run_id}.jsonl")
        succeeded = meta.get("status") == "DONE" and meta.get("verified") is not False
        if meta.get("status") != "DONE" or not steps_path.exists():
            continue
        for step in decisions(steps_path):
            operation = step["answers"].get("operation", {})
            choice, confidence = operation.get("choice"), operation.get("confidence")
            reasons = []
            if isinstance(confidence, (int, float)) and confidence < threshold:
                reasons.append("low_confidence")
            if succeeded and choice in STALL:
                reasons.append(f"{choice.lower()}_in_successful_run")
            if reasons:
                rows.append({"run_id": run_id, "step": step["step"], "operation": choice,
                             "confidence": confidence, "reason": ";".join(reasons)})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_dir", nargs="?", default=os.environ.get("TRACE_DIR"))
    parser.add_argument("--out", type=Path, help="Default: TRACE_DIR/audit.csv.")
    parser.add_argument("--threshold", type=float, default=LOW_CONFIDENCE)
    args = parser.parse_args(argv)
    if not args.trace_dir:
        parser.error("Pass a trace directory or set TRACE_DIR.")
    rows = audit(args.trace_dir, args.threshold)
    out = args.out or Path(args.trace_dir) / "audit.csv"
    with open(out, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["run_id", "step", "operation", "confidence", "reason"])
        writer.writeheader()
        writer.writerows(rows)
    reasons = {}
    for row in rows:
        for reason in row["reason"].split(";"):
            reasons[reason] = reasons.get(reason, 0) + 1
    print(f"{len(rows)} steps flagged -> {out}: {reasons}")


if __name__ == "__main__":
    main()
