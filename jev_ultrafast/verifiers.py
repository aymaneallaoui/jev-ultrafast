"""Independent outcome checks on an observed page. They never trust the model's DONE answer."""

import base64
import re
from urllib.parse import parse_qs, urlparse

DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)


def date_forms(day):
    """English date strings as Google Flights renders them, independent of the process locale."""
    weekday, month = DAYS[day.weekday()], MONTHS[day.month - 1]
    return {
        "goal": f"{month} {day.day}, {day.year}",
        "iso": day.isoformat(),
        "departure": f"{weekday[:3]}, {month[:3]} {day.day}",
        "flight": f"{weekday}, {month} {day.day}",
    }


def flights(page, *, origin, destination, day, one_way=True, adults=None):
    """Google Flights search results for one route and departure date."""
    forms = date_forms(day)
    parsed = urlparse(page["url"])
    encoded = parse_qs(parsed.query).get("tfs", [""])[0]
    try:
        date_in_url = forms["iso"].encode() in base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
    except ValueError:
        date_in_url = False
    actions = page["actions"]
    values = {a["label"].strip(): a.get("value") for a in actions}
    flight_labels = [a["label"] for a in actions if "Select flight" in a["label"]]
    trip = "One way" if one_way else "Round trip"
    checks = {
        "search_page": parsed.hostname == "www.google.com" and parsed.path == "/travel/flights/search",
        "one_way" if one_way else "round_trip": values.get(f"Change ticket type. {trip}") == trip,
        "origin": values.get("Where from?") == origin,
        "destination": values.get("Where to?") == destination,
        "date": values.get("Departure") == forms["departure"],
        "year": date_in_url or f"departing {forms['iso']}" in page["text"],
        "results": bool(flight_labels) and all(forms["flight"] in f for f in flight_labels),
    }
    if adults is not None:
        passengers = re.compile(rf"{adults} passengers?\b")
        checks["passengers"] = any(passengers.match(a["label"]) for a in actions if a.get("role") == "button")
    return {"passed": all(checks.values()), "checks": checks, "visible_flights": flight_labels}
