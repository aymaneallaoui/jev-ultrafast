# Known gaps

Agent-side gaps seen in the 2026-09-28 collections (runs from `20260928T1840` on, commit `86dc22d`). Counts are affected runs over all runs of the task. Model-choice failures (for example Hacker News `BLOCKED` on a correct page) are not listed; verification keeps them out of training data.

## Page load that starts after a click

A click triggers navigation a moment later. The post-action observation still shows the old page, so the model answers `DONE` before the result exists and the verifier fails the run.

| Task | Runs | Example run |
|---|---|---|
| `ebay-keyboard-used` | 2/2 | `20260928T190033-cb51` |
| `ebay-lego-new` | 2/2 | `20260928T190112-d425` |
| `ebay-headphones-new` | 2/2 | `20260928T190153-7687` |
| `github-cpython-issues` | 1/5 | `20260928T192902-89db` |
| `github-rust-issues` | 1/5 | `20260928T192916-a800` |
| `github-cpython-readme` | 5/5 | `20260928T185054-e90f` |

eBay's `Used`/`New` filters are links that navigate after about 150–210 ms; `DONE` followed within that window with no `LH_ItemCondition` in the URL. `github-cpython-readme` also stops on the repository's README tab (`?tab=readme-ov-file`) instead of the file page, so its goal wording is part of the problem.

Likely fix: after a click, watch for a navigation that starts within a short window (for example CDP `Page.frameRequestedNavigation`/`frameStartedLoading`) and observe after it settles.

## Hidden buttons offered as targets (MDN)

After typing into MDN's search dialog, the model clicks a `Search` button that the dialog covers. The pre-click hit-test refuses it three times and the run stops as `BLOCKED` ("Target refused 3 times").

| Task | Runs | Example run |
|---|---|---|
| `mdn-fetch` | 2/5 | `20260928T185248-688c` |
| `mdn-http-404` | 5/5 | `20260928T185332-52e4` |

`mdn-array-map`, `mdn-intersection-observer`, and `mdn-grid-template-columns` were not affected (0/5 each). Likely fix: do not offer controls whose center is covered by another layer (for example an open `aria-modal` dialog), while keeping controls that only need scrolling into view.

## TypeSafe HTTP 400 on the demoqa date picker

After the date is typed, the next decision request is rejected with `Model provider returned HTTP 400; no action executed.` The date is already set, so these runs verify, but they end as `error`.

| Task | Runs | Example run |
|---|---|---|
| `forms-demoqa-date-picker` | 3/3 | `20260928T193721-dbbf` |

The rejected request is not in the trace (the step fails before its decision line is written). Next step: reproduce by recording the request body for a non-transient model error, then find which part of the date picker's state TypeSafe refuses.
