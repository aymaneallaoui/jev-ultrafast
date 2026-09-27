# Results: local decision models for Jev Ultrafast

Measured 2026-09-28 and 2026-09-29. Every number here comes from a recorded run; trace folders and reports are named
next to each table. Laptop: RTX 4090 Laptop GPU, 16 GB. Training and held-out scoring: Modal H100.

## Method

- **Task set.** 28 tasks on 9 sites (`scripts/smoke_ids.txt`), one run per task, 120 s per task. One run per task means
  a difference of one or two tasks between rows is within run-to-run noise.
- **Verified.** Every task has an independent verifier (`jev_ultrafast/verifiers.py`) that checks the final page. A
  `DONE` from the model is not counted as success.
- **Decision latency.** Wall time of the model call for one step, as the agent sees it. Two figures: the median of each
  run's median (short runs weigh as much as long ones) and the median / p90 over all steps.
- **Text helper.** `TYPE_TEXT` values come from a separate text model (Mercury through OpenRouter) in every row unless
  stated. Until the fix in "Text helper" below, one call per run set could stall for 121 s and cost that task.
- **Held-out accuracy.** Agreement with the recorded label of the executed decision, at kev's 8k serving context.
  `heldout` holds unseen runs of training sites; `heldout_sites` holds MDN and OpenStreetMap, which are never in training.
- **Models.** `jev-08b*` and `jev-4b` are Kev checkpoints (Qwen3.5 0.8B / 4B with a pointer head) fine-tuned on Jev's
  recorded decisions. `d1a` and `d1b` are 0.8B models trained on Jev's decisions plus jev-4b's decisions (distillation).

## Task success on the smoke set

| Model | Serving | DONE | Verified | Median steps | Latency, median of run medians | Latency, all steps median / p90 |
|---|---|---|---|---|---|---|
| Jev (TypeSafe, hosted) | API | 23 | 21 | 6 | 308 ms | 313 / 376 ms |
| jev-08b (4096 training context) | laptop, graphs on | 22 | 7 | 4 | 150 ms | 202 / 346 ms |
| jev-08b-6144 | laptop, graphs on | 21 | 11 | 5 | 119 ms | 244 / 323 ms |
| d1a (distilled, 2 epochs, rank 16) | laptop, graphs on | 23 | 13 | 5 | 113 ms | 248 / 362 ms |
| d1b (distilled, 4 epochs, rank 32) | laptop, graphs on | 24 | 16 | 4.5 | 138 ms | 150 / 347 ms |
| jev-4b bf16 | laptop, graphs off | 26 | 26 | 5 | 764 ms | 1,169 / 1,830 ms |
| jev-4b nf4 | laptop, graphs off | 25 | 25 | 5.5 | 1,029 ms | 1,501 / 2,259 ms |
| jev-4b int8 | laptop, graphs off | not run | not run | | | |
| Cascade: d1a, verifier jev-4b nf4 | laptop, two servers | 23 | 21 | 6.5 | 237 ms | 265 / 1,828 ms |

jev-4b int8 was scored on the held-out files and measured for memory and latency on recorded requests (below); the smoke
set was not run with it.

Verified per site:

| Site | Jev | jev-08b-6144 | d1a | d1b | jev-4b bf16 | jev-4b nf4 | Cascade |
|---|---|---|---|---|---|---|---|
| forms | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| youtube | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| wikipedia | 3/3 | 2/3 | 2/3 | 3/3 | 3/3 | 3/3 | 2/3 |
| google_flights | 2/3 | 0/3 | 1/3 | 3/3 | 3/3 | 3/3 | 2/3 |
| github | 2/3 | 1/3 | 2/3 | 1/3 | 3/3 | 3/3 | 3/3 |
| hackernews | 3/3 | 1/3 | 1/3 | 1/3 | 2/3 | 2/3 | 1/3 |
| mdn (never in training) | 1/3 | 0/3 | 0/3 | 1/3 | 3/3 | 2/3 | 2/3 |
| openstreetmap (never in training) | 3/3 | 0/3 | 0/3 | 0/3 | 3/3 | 3/3 | 2/3 |
| ebay | 0/3 | 0/3 | 0/3 | 0/3 | 2/3 | 2/3 | 2/3 |

Traces: `~/jev-traces/smoke-20260928b` (Jev), `kev-smoke-jev-08b`, `kev-smoke-jev-08b-6144`, `kev-smoke-jev-08b-d1a`,
`kev-smoke-jev-08b-d1b`, `kev-smoke-jev-4b`, `kev-smoke-jev-4b-nf4`, `kev-smoke-jev-cascade-d1a-4bnf4`.

## Cascade

The primary (d1a, port 8009, CUDA graphs on) answers every step. The verifier (jev-4b nf4, port 8010) is asked the
identical request when the primary chooses `DONE` or `BLOCKED`, or when its chosen target has probability below 0.5
(`JEV_CASCADE_TARGET_CONF`). The verifier's answer is used. `scripts/serve_cascade.sh` starts both servers.

| Measure | Value |
|---|---|
| Verified | 21 of 28 (d1a alone 13, jev-4b nf4 alone 25) |
| Decision latency, all steps | median 265 ms, p90 1,828 ms |
| Steps answered by the primary alone | 159 of 232, median 135 ms, p90 350 ms |
| Steps that reached the verifier | 73 of 232 (31.5%), median 1,794 ms, p90 2,343 ms |
| Verifier changed the operation | 15 of 73 |
| Verifier errors | 0 |
| Idle GPU memory, both servers | 7,642 MiB (verifier 3,378 MiB, primary 4,264 MiB) |
| Idle GPU memory, whole GPU with desktop and Chrome | 9,722 MiB |
| Peak GPU memory, whole GPU during the run | 14,521 MiB |

- All 73 verifier calls had the reason `done`. The primary never chose `BLOCKED` and never chose a target below 0.5.
- 58 `DONE` decisions for 23 finished runs: a `DONE` on a page that changed since the snapshot is asked again, and each
  repeat costs a verifier call. One YouTube run made 19 verifier calls.
- The 5 `BLOCKED` runs ended on three refused clicks, which the agent decides, not the model, so the verifier never saw them.
- The peak of 14.5 GB leaves little room on a 16 GB GPU; a local text model (1.8 GB) does not fit beside both servers.

## Loop guard and confidence gates

`JEV_LOOP_GUARD=1 JEV_DONE_MIN_CONF=0.9 JEV_BLOCKED_MIN_CONF=0.9`. A gate replaces a `DONE` or `BLOCKED` below the
threshold with the next operation only if that operation's probability is above 0.15.

Thresholds come from the final `DONE` of each ungated smoke run:

| Traces | Lowest probability of a verified DONE | Premature DONE | Of those, below 0.9 with another operation above 0.15 |
|---|---|---|---|
| d1a | 0.999 | 10 | 2 |
| d1b | 1.000 | 8 | 1 |
| jev-4b | 0.984 | 1 | 0 |

| Model | Run | DONE | Verified | Median steps | Guard fired | Gate fired |
|---|---|---|---|---|---|---|
| d1a | no guard, no gates | 23 | 13 | 5 | | |
| d1a | guard and gates | 23 | 12 | 5 | 56 | 104 |
| jev-4b bf16 | no guard, no gates | 26 | 26 | 5 | | |
| jev-4b bf16 | guard and gates | 25 | 25 | 5 | 4 | 1 |

The distilled 0.8B model is confident when it stops too early, so the gates rarely apply; where they apply, the model
repeats the same low-confidence `DONE` every step and the run ends at the step budget instead (3 runs). The task jev-4b
lost under guard and gates timed out in the text helper. Neither guard nor gates changes the result beyond noise.
An earlier jev-4b run with gates at 0.72 and no floor gave 27 of 28 (`kev-smoke-jev-4b-gates`).

Traces: `kev-smoke-jev-08b-d1a-guard-gates`, `kev-smoke-jev-4b-guard-gates`.

## Held-out accuracy

Accuracy / Brier. v1 files carry Jev's labels only (298 and 202 records).

| Model | heldout operation | heldout click target | heldout_sites operation | heldout_sites click target |
|---|---|---|---|---|
| kev-0.8b (not fine-tuned) | 0.570 / 0.691 | 0.126 / 0.949 | 0.589 / 0.701 | 0.268 / 0.907 |
| jev-08b (4096) | 0.859 | 0.824 | 0.876 | 0.625 |
| jev-08b-6144 | 0.926 / 0.081 | 0.918 / 0.132 | 0.911 / 0.164 | 0.625 / 0.616 |
| d1a | 0.990 / 0.018 | 0.943 / 0.096 | 0.911 / 0.123 | 0.714 / 0.462 |
| d1b | 0.997 / 0.006 | 0.956 / 0.081 | 0.985 / 0.017 | 0.482 / 1.023 |
| kev-4b (not fine-tuned) | 0.705 / 0.526 | 0.786 / 0.558 | 0.812 / 0.331 | 0.768 / 0.612 |
| jev-4b bf16 | 0.997 / 0.007 | 0.969 / 0.059 | 1.000 / 0.000 | 0.839 / 0.293 |
| jev-4b int8 | 0.997 / 0.007 | 0.969 / 0.061 | 1.000 / 0.000 | 0.857 / 0.286 |
| jev-4b nf4 | 0.997 / 0.007 | 0.969 / 0.063 | 0.995 / 0.010 | 0.875 / 0.282 |

v2 files mix Jev's and jev-4b's labels (513 and 345 records), all questions:

| Model | v2 heldout | v2 heldout_sites |
|---|---|---|
| jev-08b-6144 | 0.912 / 0.131 | 0.873 / 0.214 |
| d1a | 0.973 / 0.051 | 0.896 / 0.158 |
| d1b | 0.987 / 0.023 | 0.896 / 0.192 |
| jev-4b bf16 (scored partly on its own answers) | 0.993 / 0.014 | 0.984 / 0.030 |

d1b is the better model on sites it was trained on and the worse one on click targets for unseen sites (0.482, below the
model it replaces, and confidently wrong). d1a is the one kept.

Held-out accuracy does not predict task success for the 0.8B models: d1a reaches 0.990 on the operation head and
completes 13 of 28 tasks. What the held-out files do not measure is when to stop; the recorded runs contain one `DONE`
each, always correct.

Reports: `~/kev/runs/modal/` (`v2-*` for the first fine-tunes, `q1-*` for quantization, `d1*` and `ref-*` for distillation).

## Training

`kev.train`, batch 1, accumulation 8, bf16, gradient checkpointing, `--max_state 6144`, seed 0, Modal H100.

| Checkpoint | Base and init | Training file | Records | Epochs | lr | LoRA rank | Steps | Wall time | Peak GPU memory |
|---|---|---|---|---|---|---|---|---|---|
| jev-08b-6144 | kev-0.8b | v1 `train.jsonl` | 930 | 2 | 2e-5 | 16 | 230 | 764 s | 14,493 MiB |
| jev-4b | kev-4b | v1 `train.jsonl` | 930 | 2 | 2e-5 | 16 | | 1,535 s | 35,233 MiB |
| d1a | kev-0.8b | v2 `train.jsonl` | 1,766 | 2 | 2e-5 | 16 | 436 | 1,442 s | 9,181 MiB |
| d1b | kev-0.8b widened to rank 32 | v2 `train.jsonl` | 1,766 | 4 | 3e-5 | 32 | 872 | 2,809 s | 9,395 MiB |

- v1: 253 verified runs of TypeSafe Jev. v2 adds 222 verified runs where jev-4b made the decisions
  (`~/jev-traces/DATASET.md`, `~/jev-traces/reports/item3-report.md`).
- `kev.train` refuses `--init_from` a rank 16 checkpoint with `--lora 32`. For d1b the kev-0.8b adapter was widened
  first: new A rows random, new B columns zero, `lora_alpha` kept at twice the rank. The weight deltas are identical and
  the widened checkpoint scores the same as the original (accuracy 0.4720 both).
- The first d1b attempt was lost at step 440 of 872 when Modal preempted the orchestrating container.

## Serving on the laptop

jev-4b, `KEV_CUDA_GRAPHS=0 KEV_MAX_BATCH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Peak is the server
process on the longest recorded request (11,107 input tokens); latency is over 60 recorded requests.

| Mode | Idle GPU memory | Peak GPU memory | Median latency | p90 latency |
|---|---|---|---|---|
| jev-4b bf16 | 8.81 GiB | 10.59 GiB | 784 ms | 1,385 ms |
| jev-4b int8 | 4.87 GiB | 7.16 GiB | 969 ms | 1,775 ms |
| jev-4b nf4 | 3.30 GiB | 5.60 GiB | 1,035 ms | 1,832 ms |
| d1a bf16, graphs on | 4.16 GiB | 6.21 GiB (during the cascade run) | see smoke table | |

| GPU memory | Recommended | Why |
|---|---|---|
| 16 GB | jev-4b bf16, or the cascade | bf16 is the most accurate and fastest 4B mode; the cascade is 4 times faster at the median and completes 5 fewer tasks |
| 8-12 GB | jev-4b int8 | peaks at 7.2 GiB |
| 6-8 GB | jev-4b nf4 | peaks at 5.6 GiB |

CUDA graphs do not help jev-4b on Jev's requests: the model pass is 762 of 769 ms and a median request is 3,820 tokens,
so the pass is bound by compute, not by kernel launches. With graph buffers sized down (4x4096 bank, 8,192 rows) bf16
idles at 11.4 GiB instead of 9.0 GiB for the same latency, and int8 cannot capture graphs at all. Details:
`docs/kev-serve-memory.md`.

## Text helper

- `TEXT_TIMEOUT_S` (default 20, overridable from the environment) is now a total budget for one helper call, covering
  retries and their pauses; a reply that arrives after the budget is discarded. Before, the timeout applied to each phase
  of each attempt and one call was recorded at 121 s. In the cascade run, made with the fix, the longest helper call
  was 21.6 s.
- Local option: `scripts/serve_text_local.sh` serves `unsloth/Qwen3-1.7B-GGUF`, file `Qwen3-1.7B-Q4_K_M.gguf`, with
  `llama-server`, thinking off, 1,818 MiB of GPU memory. Qwen3-1.7B has no separate "Instruct" release.

Four form tasks with d1a and the local helper (`kev-smoke-jev-08b-d1a-text-local`):

| Task | Verified | Values typed | Helper latency |
|---|---|---|---|
| forms-httpbin-contact | yes | `Ada Lovelace`; Telephone got `Ada Lovelace`, then `555-0100` in a later step; `ada@example.com` | 1,320 ms first call, then 58-82 ms |
| forms-httpbin-toppings | yes | none (no text field) | |
| forms-selenium-select-two | yes | `Jev test` | 70 ms |
| forms-selenium-textarea-three | yes | `Hello from Jev` | 64 ms |

All four tasks verify. One of six values was wrong when first typed (the name in the telephone field); the decision
model chose the field again and the second value was right. Mercury typed all values right the first time, at 1.0 to
1.5 s per call.

## What this shows

1. A fine-tuned 4B model replaces the hosted model on this task set: 26 of 28 against 21 of 28, at 2.5 times the latency
   on a laptop GPU.
2. Quantizing it to 4 bits costs one task and 35% latency and cuts idle memory from 8.8 to 3.3 GiB.
3. A 0.8B model is 5 to 7 times faster than the 4B model and completes half as many tasks. More data from the 4B model
   raises it from 11 to 13 (or 16 with the longer schedule), not to 26.
4. Its failure is stopping early with full confidence. Confidence gates cannot catch that.
5. Asking the 4B model to confirm every stop recovers most of the gap (13 to 21) at a median latency below the hosted
   model's, with a p90 of 1.8 s.

## Open

- Soft-label distillation: train the 0.8B model on jev-4b's probabilities instead of its choices, so it inherits the 4B
  model's calibration. Not started.
- Repeated `DONE` on pages that keep changing multiplies verifier calls in the cascade.
- Agent gaps that no model fixes: `docs/known-gaps.md`.
