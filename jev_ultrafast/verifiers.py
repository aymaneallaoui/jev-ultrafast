"""Independent outcome checks on an observed page. They never trust the model's DONE answer."""

import base64
import re
import unicodedata
from urllib.parse import parse_qs, unquote_plus, urljoin, urlparse

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


def normalize(text):
    decomposed = unicodedata.normalize("NFKD", text or "")
    return " ".join("".join(c for c in decomposed if not unicodedata.combining(c)).casefold().split())


def city_matches(value, city):
    """Accent- and case-insensitive containment, so "Zürich" matches "Zurich" and "New York, NY" matches "New York"."""
    value, city = normalize(value), normalize(city)
    return bool(value and city) and city in value


def flights(page, *, origin, destination, day, one_way=True, return_day=None, adults=None):
    """Google Flights search results for one route and departure date."""
    if not one_way and return_day is None:
        raise ValueError("Round-trip verification needs return_day")
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
        "origin": city_matches(values.get("Where from?"), origin),
        "destination": city_matches(values.get("Where to?"), destination),
        "date": values.get("Departure") == forms["departure"],
        "year": date_in_url or f"departing {forms['iso']}" in page["text"],
        "results": bool(flight_labels) and all(forms["flight"] in f for f in flight_labels),
    }
    if not one_way:
        checks["return_date"] = values.get("Return") == date_forms(return_day)["departure"]
    if adults is not None:
        counts = [re.search(r"(\d+) passengers?", a["label"]) for a in actions if a.get("role") == "button"]
        checks["passengers"] = adults in [int(match.group(1)) for match in counts if match]
    return {"passed": all(checks.values()), "checks": checks, "visible_flights": flight_labels}


def page(page, *, url=(), text=(), fields=None):
    """Generic checks: URL patterns (decoded, case-insensitive), visible text, and form field values."""
    decoded, visible = unquote_plus(page["url"]), normalize(page["text"])
    patterns = [url] if isinstance(url, str) else url
    needles = [text] if isinstance(text, str) else text
    checks = {f"url:{pattern}": bool(re.search(pattern, decoded, re.I)) for pattern in patterns}
    checks.update({f"text:{needle}": normalize(needle) in visible for needle in needles})
    for label, expected in (fields or {}).items():
        checks[f"field:{label}"] = field_matches(page["actions"], label, expected)
    return {"passed": all(checks.values()), "checks": checks}


def field_matches(actions, label, expected):
    """A field whose label matches holds the value; True/False compare a checkbox or radio's checked state."""
    wanted = normalize(label).rstrip(":")
    for action in actions:
        if normalize(action.get("label", "").split(" → ")[0]).rstrip(":") != wanted:
            continue
        if isinstance(expected, bool):
            if action.get("checked") in ("true", "false"):
                return (action["checked"] == "true") is expected
            continue
        value = action.get("current_value", action.get("value", ""))
        if normalize(value) == normalize(str(expected)):
            return True
    return False


def hn_story(page, *, initial, rank, comments=False):
    """The story at `rank` on the initial front page: its comment thread, or its own link."""
    guards = initial.get("guards", {})
    row = [
        (action, guards[str(action["node"])])
        for action in initial["actions"]
        if action.get("kind") == "click" and guards.get(str(action.get("node")))
        and (guards[str(action["node"])][13] or "").startswith(f"{rank}.\t")
    ]
    hrefs = [guard[12] or "" for _, guard in row]
    story = next((h.split("id=")[1].split("&")[0] for h in hrefs if h.startswith("vote?id=")), None)
    title = next((h for h in hrefs if not h.startswith(("vote?", "from?", "hide?"))), None)
    base = initial["url"]
    if comments:
        expected = f"https://news.ycombinator.com/item?id={story}" if story else None
    else:
        expected = urljoin(base, title) if title else None
    checks = {"story_found": expected is not None, "url": expected is not None and same_url(page["url"], expected)}
    return {"passed": all(checks.values()), "checks": checks, "expected": expected}


def same_url(actual, expected):
    def key(value):
        parsed = urlparse(value)
        return (parsed.hostname or "").removeprefix("www."), parsed.path.rstrip("/"), parsed.query
    return key(actual) == key(expected)
