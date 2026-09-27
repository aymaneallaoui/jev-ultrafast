# Serving Kev-4B on a 16 GB GPU

Measured 2026-09-29 on an RTX 4090 Laptop GPU (16 GB), serving `~/kev/runs/jev-4b` (Qwen3.5-4B base, LoRA merged, bf16, fused Qwen3.5 kernels). Idle is the server process's memory in `nvidia-smi` after startup, before any request.

| Change (cumulative) | Idle server memory |
|---|---|
| Defaults | Out of memory during startup: 14.5 GiB in use while allocating CUDA graph buffers |
| `KEV_CUDA_GRAPHS=0` | 13.25 GiB |
| `+ PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | 9.89 GiB |
| `+ empty_cache()` after load (kev patch below) | **8.81 GiB** |
| `+ KEV_SERVE_MAX_STATE=4096 KEV_MAX_BATCH=1` (kev patch below) | 8.81 GiB (these cap per-request memory, not idle) |

Where the extra ~5.5 GB came from, measured in-process with PyTorch's counters after `Checkpoint.load()`:

- **CUDA graph buffers, about 5.5 GiB allocated.** `kev.cuda_graphs.CudaGraphs` preallocates a state bank (16 states x 4,096 positions per attention layer), row buffers for 32,768 tokens, and a hidden-state buffer, all sized by module constants (`GRAPH_STATES`, `BANK_WIDTH`, `GRAPH_TOKENS`, `GRAPH_ROWS`, `GRAPH_ROW`). There is no setting to shrink them, only `KEV_CUDA_GRAPHS=0`.
- **About 5.1 GiB reserved but unused after loading.** `merge_and_unload()` and `fuse()` create temporary copies of the weights; PyTorch's caching allocator keeps the freed blocks reserved. With graphs on, the graph buffers reuse that space, so turning graphs off alone saved little. `torch.cuda.empty_cache()` after loading returns it.
- **Not the adapter:** it is merged into the weights (`KEV_MERGE=1`, the default), so there is no second, unmerged copy. The model's weights take about 7.9 GiB.

With `KEV_CUDA_GRAPHS=0 KEV_MAX_BATCH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and no state cap, the longest recorded Jev request (11,107 input tokens) peaked at 11.4 GB for the whole GPU (Chrome and the desktop included) and answered in 0.8-1.7 s warm; a median request takes about 0.75 s without graphs. `scripts/serve_local.sh 4b` sets these three variables.

`KEV_SERVE_MAX_STATE` is not set by default: `encode()` serves with `strict=False`, so a longer state is silently truncated to its first N tokens (dropping later elements and recent actions), not rejected. It was not needed to fit Jev's states on this GPU.

## Proposed upstream change to kev/serve.py

Committed in `~/kev` on branch `jev-serve-memory` (`14f0e47`; a copy of the diff is at `~/jev-traces/tools/kev-serve-memory.patch`):

1. `empty_cache(dev)` right after `ck.load(dev, opts)`: saves ~1.1 GiB idle on Kev-4B even with `expandable_segments`, ~5 GiB without it when graphs are off.
2. `KEV_MAX_BATCH` (default 64) instead of the `MAX_BATCH` constant.
3. `KEV_SERVE_MAX_STATE` (default `SERVE_MAX_STATE`) for the state length passed to `encode()`.

Worth raising upstream as well: a setting to size the CUDA graph buffers (for example `KEV_GRAPH_STATES` / `KEV_BANK_WIDTH` / `KEV_GRAPH_TOKENS`), so small GPUs can keep graphs with smaller buffers instead of turning them off.

## Quantized serving: bf16, int8, nf4

`~/kev` branch `jev-serve-memory` adds `KEV_LOAD_IN_8BIT=1` (bitsandbytes LLM.int8, outlier threshold 6.0) and `KEV_LOAD_IN_4BIT=1` (NF4, double quantization, bf16 compute). The adapter is merged in host memory, then each backbone `Linear` is quantized as it moves to the GPU, so the GPU never holds the bf16 weights; embeddings and the pointer head are not quantized. The fused Qwen3.5 kernels read raw weight tensors, so quantized modes load without them. `scripts/serve_local.sh 4b-int8` and `4b-nf4` select them.

Memory and latency on the laptop GPU (RTX 4090 Laptop, 16 GB), `jev-4b`, `KEV_CUDA_GRAPHS=0 KEV_MAX_BATCH=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Peak is the server process on the longest recorded Jev request (11,107 input tokens); latency is over 60 random recorded requests (seed 0).

| Mode | Idle VRAM | Peak VRAM | Median latency | p90 latency |
|---|---|---|---|---|
| bf16 | 8.81 GiB | 10.59 GiB | 784 ms | 1,385 ms |
| int8 | 4.87 GiB | 7.16 GiB | 969 ms | 1,775 ms |
| nf4 | 3.30 GiB | 5.60 GiB | 1,035 ms | 1,832 ms |

Accuracy and Brier on the held-out files (Modal H100, all three modes in bf16 compute, graphs off, 8k-token context; 298 / 159 / 76 operation / click / type-text questions on heldout, 202 / 56 / 74 on heldout_sites):

| Mode | Held-out | op acc | click acc | type_text acc | Brier (all) | op Brier | click Brier |
|---|---|---|---|---|---|---|---|
| bf16 | heldout | 0.997 | 0.969 | 1.000 | 0.021 | 0.007 | 0.059 |
| bf16 | heldout_sites | 1.000 | 0.839 | 1.000 | 0.049 | 0.000 | 0.293 |
| int8 | heldout | 0.997 | 0.969 | 1.000 | 0.022 | 0.007 | 0.061 |
| int8 | heldout_sites | 1.000 | 0.857 | 1.000 | 0.048 | 0.000 | 0.286 |
| nf4 | heldout | 0.997 | 0.969 | 1.000 | 0.023 | 0.007 | 0.063 |
| nf4 | heldout_sites | 0.995 | 0.875 | 1.000 | 0.054 | 0.010 | 0.282 |

The 28-task smoke set with `4b-nf4`: 25/28 DONE, 25/28 verified, median 5.5 steps, median decision latency 1,029 ms (bf16: 26/28, 26/28, 5 steps, 764 ms). One run per task, so a single task flipping is within run-to-run noise; the differences were on eBay (`ebay-camera-under-100` timed out at 50 steps; `ebay-lego-new` reached the filtered page but answered BLOCKED) and one MDN run that ended DONE without verifying.

| GPU memory | Recommended mode | Why |
|---|---|---|
| 16 GB | bf16 or int8 | bf16 peaks at 10.6 GiB and is fastest; int8 leaves more room for the browser and desktop |
| 8-12 GB | int8 | peaks at 7.2 GiB on the longest request; at 8 GB there is little headroom |
| 6-8 GB | nf4 | peaks at 5.6 GiB; at 6 GB very long pages may still not fit |

Peaks are the server alone; the desktop and Chrome use GPU memory on top (about 1-2 GB on the laptop).
