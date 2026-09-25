"""Run tasks.yaml sequentially with tracing. Live runs make paid API calls; --dry-run does not."""

import argparse
import csv
import os
import re
import statistics
import sys
import time
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import yaml

from jev_ultrafast import Agent
from jev_ultrafast.verifiers import DAYS, MONTHS, flights

ROOT = Path(__file__).resolve().parents[1]
TIMEOUT_S = 120
PAUSE_S = 2
VERIFIERS = {"flights": flights}
SUMMARY_FIELDS = [
    "run_id", "task_id", "tags", "status", "steps", "elapsed_ms", "verified",
    "jev_latency_p50_ms", "jev_latency_max_ms", "input_tokens_total",
]
PLACEHOLDER = re.compile(r"\{([^{}]*)\}")
TOKEN = re.compile(r"(date|weekday)\+(\d+)(?::(.+))?")


def resolve(text, today):
    """Expand {date+N}, {date+N:strftime}, and {weekday+N} relative to today."""

    def replace(match):
        token = TOKEN.fullmatch(match.group(1))
        if not token or (token.group(1) == "weekday" and token.group(3)):
            raise ValueError(f"Unknown placeholder {match.group(0)!r} in {text!r}")
        kind, offset, fmt = token.groups()
        day = today + timedelta(days=int(offset))
        if kind == "weekday":
            return DAYS[day.weekday()]
        return day.strftime(fmt) if fmt else f"{MONTHS[day.month - 1]} {day.day}, {day.year}"

    return PLACEHOLDER.sub(replace, text)


def load_tasks(path, today):
    tasks = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    seen = set()
    for task in tasks:
        missing = {"id", "url", "goal", "tags"} - set(task)
        if missing:
            raise ValueError(f"Task {task.get('id', '?')} is missing {sorted(missing)}")
        if task["id"] in seen:
            raise ValueError(f"Duplicate task id {task['id']}")
        seen.add(task["id"])
        task["goal"] = resolve(task["goal"], today)
        if "verify" in task:
            if task["verify"] not in VERIFIERS:
                raise ValueError(f"Task {task['id']} has unknown verifier {task['verify']!r}")
            arguments = dict(task.get("verify_args", {}))
            for name, target in (("date", "day"), ("return_date", "return_day")):
                if name in arguments:
                    arguments[target] = date.fromisoformat(resolve(arguments.pop(name), today))
            if task["verify"] == "flights" and arguments.get("one_way") is False and "return_day" not in arguments:
                raise ValueError(f"Round-trip task {task['id']} needs verify_args.return_date")
            task["verify_args"] = arguments
    return tasks


def duration(text):
    match = re.fullmatch(r"(\d+)([smh]?)", text.strip())
    if not match or int(match.group(1)) == 0:
        raise argparse.ArgumentTypeError(f"invalid duration {text!r}; use seconds or a number with s, m, or h")
    return int(match.group(1)) * {"": 1, "s": 1, "m": 60, "h": 3600}[match.group(2)]


def run_task(task, batch_deadline=None):
    agent = None
    try:
        agent = Agent(task["url"], task["goal"])
        deadline = time.monotonic() + TIMEOUT_S
        if batch_deadline is not None:
            deadline = min(deadline, batch_deadline)
        for _state in agent.run():
            if time.monotonic() > deadline:
                agent.finish_trace("timeout")
                break
    except Exception as error:
        print(f"  {task['id']}: {type(error).__name__}: {error}", file=sys.stderr)
    finally:
        if agent:
            if "verify" in task:
                try:
                    result = VERIFIERS[task["verify"]](agent.snapshot()["page"], **task["verify_args"])
                    agent.trace.set_verified(result["passed"])
                except Exception as error:
                    print(f"  {task['id']}: verification failed to run: {error}", file=sys.stderr)
            try:
                agent.close()
            except Exception as error:
                print(f"  {task['id']}: close failed: {error}", file=sys.stderr)
    return agent.trace if agent else None


def summary_row(task, trace):
    meta = (trace.meta if trace else None) or {}
    stats = trace.stats() if trace else {}
    return {
        "run_id": trace.run_id if trace else "",
        "task_id": task["id"],
        "tags": ";".join(task["tags"]),
        "status": meta.get("status", "error"),
        "steps": meta.get("steps"),
        "elapsed_ms": meta.get("elapsed_ms"),
        "verified": meta.get("verified"),
        "jev_latency_p50_ms": stats.get("jev_latency_p50_ms"),
        "jev_latency_max_ms": stats.get("jev_latency_max_ms"),
        "input_tokens_total": stats.get("input_tokens_total"),
    }


def cell(value):
    if isinstance(value, bool):
        return str(value).lower()
    return "" if value is None else value


def append_summary(path, row):
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow({key: cell(value) for key, value in row.items()})


def report(rows, skipped=0):
    statuses = Counter(row["status"] for row in rows)
    steps = [row["steps"] for row in rows if row["steps"] is not None]
    table = [
        ("runs", len(rows)),
        ("DONE", statuses["DONE"]),
        ("verified", sum(row["verified"] is True for row in rows)),
        ("BLOCKED", statuses["BLOCKED"]),
        ("error", statuses["error"]),
        ("timeout", statuses["timeout"]),
        ("max_steps", statuses["max_steps"]),
        ("skipped", skipped),
        ("median steps", statistics.median(steps) if steps else "-"),
    ]
    for name, value in table:
        print(f"{name:<14}{value}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default=ROOT / "tasks.yaml", type=Path)
    parser.add_argument("--only", help="Run only tasks carrying this tag.")
    parser.add_argument("--repeat", type=int, help="Runs per task; overrides each task's repeat.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved goals without opening a browser.")
    parser.add_argument("--max-runtime", type=duration, help="Batch budget in seconds, or with s, m, or h (e.g. 90m).")
    args = parser.parse_args(argv)
    tasks = load_tasks(args.tasks, date.today())
    if args.only:
        tasks = [task for task in tasks if args.only in task["tags"]]
    runs = [
        (task, n)
        for task in tasks
        for n in range(1, (task.get("repeat", 1) if args.repeat is None else args.repeat) + 1)
    ]
    if args.dry_run:
        for task, n in runs:
            print(f"{task['id']}#{n}\t{task['goal']}")
        return
    trace_dir = os.environ.get("TRACE_DIR")
    if not trace_dir:
        parser.error("TRACE_DIR must be set; traces and summary.csv are written there.")
    summary = Path(trace_dir) / "summary.csv"
    summary.parent.mkdir(parents=True, exist_ok=True)
    rows, skipped, batch_deadline = [], 0, None
    for index, (task, n) in enumerate(runs):
        if batch_deadline is None and args.max_runtime:
            batch_deadline = time.monotonic() + args.max_runtime
        pause = PAUSE_S if index else 0
        if batch_deadline is not None and time.monotonic() + pause >= batch_deadline:
            skipped = len(runs) - index
            print(f"Batch runtime reached; skipped {skipped} runs.")
            break
        if pause:
            time.sleep(pause)
        row = summary_row(task, run_task(task, batch_deadline))
        append_summary(summary, row)
        rows.append(row)
        print(
            f"{task['id']}#{n}  {row['status']}  steps={row['steps']}  {row['elapsed_ms']} ms  "
            f"verified={cell(row['verified'])}"
        )
    report(rows, skipped)


if __name__ == "__main__":
    main()
