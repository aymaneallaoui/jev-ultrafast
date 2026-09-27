<img src="docs/banner.svg" alt="Jev Ultrafast · Browser Use × TypeSafe" width="100%" />

# Jev Ultrafast ⚡

> [!IMPORTANT]
> **The Browser Use Cloud waitlist is open.** Get early access to ultrafast browser agents in the cloud.
> **[Join the waitlist →](https://browser-use.com/ultrafast?utm_source=github&utm_medium=readme&utm_campaign=jev-ultrafast)**

**A browser agent with a dynamic, indexed action space.**

Give it one goal. [TypeSafe's Jev](https://docs.typesafe.ai/introduction) picks an operation and an element. A small LLM writes text only when the operation is `TYPE_TEXT`.

**Zürich → London on Google Flights in 7.1 seconds.** One natural-language goal, actual text generation, and loading waits included.

<a href="docs/demo.mp4"><img src="docs/demo.gif" alt="A real Google Flights search at 1× speed, with generated city names and dynamic operation/target decisions" width="100%" /></a>

[Watch the MP4](docs/demo.mp4) · [Measurements](docs/performance.md) · [Read the loop](jev_ultrafast/agent.py)

## The action space

Every observation produces a new element table:

```text
[1] button    Change ticket type · Round trip
[2] combobox  Where from?        · San Francisco
[3] combobox  Where to?          · empty
[4] textbox   Departure          · empty
...
```

The operations are `CLICK`, `TYPE_TEXT`, `SELECT`, `PRESS_ENTER`, `SCROLL_UP`, `SCROLL_DOWN`, `WAIT`, `DONE`, and `BLOCKED`. `PRESS_ENTER` is offered only on a filled text field and submits it once. Only supported operations and targets are offered.

```text
                      one TypeSafe request
                     ┌───────────────────────────┐
page → element table → operation                 │
                     │ click_target              │
                     │ type_text_target          │
                     │ select_target, if present │
                     └─────────────┬─────────────┘
                         use the matching target
                                   │
                    CLICK [7] ─────┤──→ browser
                TYPE_TEXT [3] ─────┘
                          ↓
                   small LLM → text → browser
```

Target questions are speculative. If the operation is `CLICK`, only `click_target` can execute. Two decisions, **one network round trip**. Each target head contains only compatible elements. Native dropdown choices carry an observed element/option index.

There are no site-specific action scripts or prepared field strings in the policy. The Flights example supplies a goal and independently verifies the outcome. The screenshot renderer adds labels afterward; it does not drive the browser.

## Try it

```bash
git clone https://github.com/browser-use/jev-ultrafast.git
cd jev-ultrafast
uv sync
cp .env.example .env
# Add TYPESAFE_API_KEY and TEXT_MODEL_API_KEY.
uv run jev
```

Open **http://127.0.0.1:8766** and click **Start demo → Run automatically**. The inspector shows numbered elements, operation probabilities, target probabilities, and executed actions. **Choose next** pauses before execution.

Chrome connects through [Browser Harness](https://github.com/browser-use/browser-harness), installed by `uv sync`. Run `uv run browser-harness --doctor` if it needs connecting. Allow remote debugging in Chrome when prompted.

Each run drives its own `about:blank` target, opened as a background tab so your visible tab never switches. `JEV_OWN_WINDOW=1` opens that target in its own window instead; use it with a dedicated automation profile such as `jev-chrome`, where Chrome treats a background tab as hidden and throttles menu animations to ~2-3 s. The viewport is `JEV_VIEWPORT_WIDTH`×780 px, default 1480. Upstream used 1120, which the recorded evidence below reflects; Google Flights' two-month calendar is wider than 1120 px, so day buttons past that edge were dropped.

`TEXT_MODEL_API_KEY` is an OpenRouter key in the example configuration. The helper's reply must hold one JSON object with exactly a `text` key; code fences or text around that object are ignored. The current demo uses `inception/mercury-2.5` with reasoning disabled. Gemini, GLM, and DeepSeek can also use the OpenAI-compatible text helper; configure the appropriate model, endpoint, and reasoning setting.

## Use the library

```python
from jev_ultrafast import Agent

with Agent(
    "https://www.google.com/travel/flights?hl=en",
    "Find one-way flights from Zurich to London on September 20, 2026, "
    "for one adult in economy. Stop when matching flight options are visible.",
) as agent:
    for state in agent.run():
        print(state["elapsed_ms"], state["status"])
```

Run with `uv run --env-file .env python your_script.py`. The same policy can run a different task:

```bash
uv run --env-file .env python examples/run.py \
  --url https://en.wikipedia.org/wiki/Main_Page \
  --goal 'Find and open the Wikipedia article about Gödel’s incompleteness theorems.'
```

Set `TRACE_DIR` to write one JSON line per TypeSafe decision to `<run_id>.jsonl`, plus `<run_id>.meta.json` when `run()` ends (`DONE`, `BLOCKED` with an automatic stop's `reason`, `max_steps`, or `error`) or, failing that, as `closed` on `close()`. `trace.set_verified()` records an independent outcome check in the meta file. Each step first writes a `step_start` line (`step`, `t_ms` since the first step, last observed `url`) before any browser or model call, so a stalled step still leaves evidence. Its decision line (`"event": "step"`) is written when the step ends and adds `snapshot_ms`, `model_ms` (including a transient retry), `text_ms` for `TYPE_TEXT`, `execute_ms`, and `wait_ms`; a step that raises records the phases so far plus `failed_phase` and `error`, as a `step_failed` line if no decision was made. Request bodies and text-helper inputs are logged; API keys are not. A transient TypeSafe failure (connection error or invalid response) is retried once after 1 s, before any action, and recorded as `retries` with the raw response; the text helper has a 20-second timeout and one retry. `TYPESAFE_BASE_URL` overrides the TypeSafe endpoint (default `https://api.typesafe.ai`).

`uv run --env-file .env python examples/flights.py --keep-open` performs the flight search, checks the actual route/date/results, and saves its trace. `--date YYYY-MM-DD` sets the departure date (default: 14 days from today). It does not select or book a flight.

## Why it moves

- **One request per decision cycle.** Operation and target heads share the same observed state.
- **No screenshots in the default agent loop.** Jev consumes structured state. The inspector opts into screenshots; the video uses a separate continuous screencast.
- **One browser call per snapshot.** Read visible controls, their names, values, and text atomically. Keep references to the actual DOM nodes.
- **Validate the selected target.** Clicks check the document, form values, target, and nearby context. Animation alone does not force another prediction. Scroll the target into view within any inner scroller, resolve current geometry, and reject covered controls before input. If the same target is refused three times with no successful action in between, the run stops as `BLOCKED` with `blocked_reason`.
- **Wait for useful state.** After typing into a combobox, wait for visible suggestions, capped at 200 ms. Other interactions get at most two animation frames or 50 ms. A click on a control reporting `aria-expanded="false"` then re-reads every 50 ms, for up to 400 ms, until it reports expanded and new elements appear (recorded as `wait_ms`). An empty snapshot is re-read up to 5 times at 100 ms, whether the page is new or previously had elements. A document that is still navigating is re-read every 50 ms for up to 5 s. These reads happen after execution is logged.
- **Keep hidden tabs rendering.** Focus emulation prevents background animation throttling without switching Chrome's visible tab.
- **Send visible text.** Offscreen article bodies and footers do not fill the model context.
- **Reuse an interrupted text request.** A generated value survives a stale-page retry only if the entire text-helper input is unchanged.

Every executed target is resolved from an observed node. The executor rechecks page freshness and click occlusion. Model output never becomes selectors, coordinates, shell commands, or executable JavaScript. Text-helper output must parse as a small JSON object before typing.

## Small enough to read

| File | Job |
| --- | --- |
| [agent.py](jev_ultrafast/agent.py) | The complete loop and text-helper handoff |
| [snapshot.js](jev_ultrafast/snapshot.js) | Atomic DOM snapshot, indexed controls, freshness guards |
| [browser.py](jev_ultrafast/browser.py) | Browser connection, current geometry, execution |
| [model.py](jev_ultrafast/model.py) | Dynamic operation/target heads and text generation |
| [questions.py](jev_ultrafast/questions.py) | Model instructions |
| [demo.py](jev_ultrafast/demo.py) | Local inspector |

## Evidence and limits

The current video is a **7,073 ms** Google Flights run. Timing starts after initial page observation and includes model calls, generated text, browser work, stale decisions, and loading waits. A fresh independent check verifies the one-way setting, Zürich, London, September 20, 2026, and visible flight options. The video plays at 1×, with no opening hold and a 0.5-second final hold.

In six alternating runs with identical models and settings, both versions passed **3/3**. Median task time went from **9.450 s → 7.092 s**, a **25% reduction**; median browser protocol calls went from **1,092 → 101**. This is three repeats of one task on one browser profile, not a general reliability benchmark.

The same policy opened the requested Wikipedia article in **2.798 s** and passed a local hotel search/filter task in **1.896 s**. Runs, failures, source hashes, and measurement boundaries are in [performance.md](docs/performance.md).

A `DONE` choice still requires independent outcome verification. The DOM reader handles common HTML and ARIA controls, not the full accessible-name specification. Open shadow roots are read and hit-tested like the main document; closed shadow roots, frames, canvas, uploads, pop-up tabs, nested scrolling, and arbitrary keyboard widgets remain outside this MVP. Owned tabs share the existing Chrome profile.

## Collecting traces

`scripts/collect.py` runs the goals in [tasks.yaml](tasks.yaml) one after another. It makes live model calls and needs `TRACE_DIR`:

```bash
TRACE_DIR=traces uv run --env-file .env python scripts/collect.py --only wikipedia --repeat 2
uv run python scripts/collect.py --dry-run   # print resolved goals, no browser
```

Goals may use `{date+N}` (today + N days, e.g. `October 12, 2026`), `{date+N:%Y-%m-%d}` (any `strftime` format), and `{weekday+N}`. Unknown placeholders fail when the file loads. Each run has a 120-second budget, checked between steps; `--max-runtime 90m` (seconds, or `s`/`m`/`h`) also caps the batch, stopping the current run at its next step and skipping the rest. Every task names a verifier in `jev_ultrafast/verifiers.py`. `flights` matches city names ignoring accents and case (`Zurich` matches `Zürich, Switzerland`) and checks the return date on round trips. `page` checks decoded URL patterns, visible text, form field values, values held by unlabeled controls, and how many same-labeled boxes are checked. `echo` checks that a form's result page shows every submitted value. `hn_story` resolves the ranked story from the first observed front page and checks its link or comment thread. Every run appends a row to `TRACE_DIR/summary.csv` with status, steps, verification, decision latency, and input tokens.

`uv run python scripts/audit_traces.py traces` lists decisions worth a human look in `traces/audit.csv`: a `DONE` run's operations below 0.5 confidence, and `WAIT`/`BLOCKED` choices in runs that still succeeded. Nothing is relabeled.

`uv run python scripts/traces_to_kev.py traces --drop-nonprogress --exclude traces/audit.csv` keeps tagged `DONE` runs that did not fail verification and writes kev labelled requests to `traces/kev/train.jsonl` and `heldout.jsonl`, split 85/15 by run. Each record keeps the operation question and the executed operation's target head only; the other heads were speculative. `--holdout-tags mdn,openstreetmap` sends those sites' runs to `heldout_sites.jsonl` only, `--cap-tag google_flights=0.35` keeps that tag at most 35% of training records, and runs missing from `summary.csv` are skipped unless `--include-untagged`. `--relabel-blocked` drops, in verified runs, `WAIT`/`BLOCKED` decisions that a same-page `CLICK` or `TYPE_TEXT` follows (hesitations, not correct labels); it has no effect with `--drop-nonprogress`.

`uv run python scripts/summary_by_tag.py traces --since 20260928T1840 --out by-tag.csv` groups `summary.csv` by tag with `verified_true`/`verified_false`/`verified_null` counts and `usable_steps`: decision steps in runs that ended `DONE` without failing verification.

Two optional decision guards, off unless their variables are set: `JEV_LOOP_GUARD=1` swaps a choice for the runner-up target of the same head when the agent loops (the same action twice without a page change, an A,B,A,B cycle over four decisions, or a target refused twice); `JEV_DONE_MIN_CONF` / `JEV_BLOCKED_MIN_CONF` turn a `DONE` / `BLOCKED` whose probability is below the threshold into the next most likely operation, using that operation's own validated target. Trace lines record both as `loop_guard` and `confidence_gate`.

`traces_to_kev.py --source DIR` merges further trace directories (for example `raw` with Jev labels and `raw-4b` with Kev-4B labels), each keeping its own `summary.csv` tags; `--exclude` is repeatable, and steps where a guard replaced the model's choice are skipped because their label is not what executed.

## Comparing decision models

`scripts/serve_local.sh [08b|08b-6144|4b|RUN] [PORT]` serves a fine-tuned Kev checkpoint from `~/kev/runs` with `kev.serve`, which answers the same `/v1/systemone` requests as TypeSafe. `scripts/kev_smoke.sh MODEL [PORT]` then runs the tasks in `scripts/smoke_ids.txt` with `TYPESAFE_BASE_URL` pointed at it, writing traces to `~/jev-traces/kev-smoke-jev-MODEL` so they never mix with TypeSafe labels. `collect.py --model-tag` records the decision model in `summary.csv` (default: `TYPESAFE_MODEL`), and `--ids` limits a run to listed task ids.

`uv run python scripts/compare_models.py --ids @scripts/smoke_ids.txt --model jev=TRACES --model jev-08b=KEV_TRACES` prints DONE rate, verified rate, median steps, and median decision latency per site for each collection.

## Development

```bash
uv run ruff check .
uv run pytest
node --check jev_ultrafast/static/app.js
node --check jev_ultrafast/snapshot.js
uv build
```

Tests are offline. `uv run python scripts/check_guards.py` checks real controls in a local browser without model calls. Live examples and recording scripts make paid API calls. `scripts/record_flights.py <new-folder>` captures original browser timestamps; `scripts/render_demo.py <recording-folder>` renders that verified run at 1× and crops out the Google account strip. Credentials and raw traces stay ignored.

---

[Browser Use](https://github.com/browser-use/browser-use) · [Browser Harness](https://github.com/browser-use/browser-harness) · [TypeSafe speculative fan-out](https://docs.typesafe.ai/patterns/fan-out)
