"""Live Google Flights search. Calls TypeSafe; never selects or books a flight."""

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

from jev_ultrafast import Agent
from jev_ultrafast.verifiers import date_forms, flights

URL = "https://www.google.com/travel/flights?hl=en"


def goal(day):
    return (
        f"Find one-way flights from Zurich to London on {date_forms(day)['goal']}, for one adult in economy. "
        "Stop when matching flight options are visible. Do not select or book a flight."
    )


DEPARTURE = date.today() + timedelta(days=14)
GOALS = goal(DEPARTURE)


def verify(page, day=DEPARTURE):
    return flights(page, origin="Zürich", destination="London", day=day)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/flights/latest")
    parser.add_argument("--keep-open", action="store_true")
    parser.add_argument("--date", type=date.fromisoformat, default=DEPARTURE, help="Departure date, YYYY-MM-DD.")
    args = parser.parse_args()
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    agent = Agent(URL, goal(args.date))
    try:
        for state in agent.run():
            last = state["history"][-1] if state["history"] else {}
            print(state["elapsed_ms"], state["status"], last.get("action", ""), flush=True)
    finally:
        state = agent.snapshot()
        state["verification"] = verify(state["page"], args.date)
        agent.trace.set_verified(state["verification"]["passed"])
        (folder / "state.json").write_text(json.dumps(state, indent=2))
        (folder / "session.json").write_text(
            json.dumps({"target": agent.browser.target, "session": agent.browser.session})
        )
        if not args.keep_open:
            agent.close()
    print(json.dumps(state["verification"], indent=2))
    if not state["verification"]["passed"]:
        raise SystemExit("Final page did not satisfy the route/date checks")


if __name__ == "__main__":
    main()
