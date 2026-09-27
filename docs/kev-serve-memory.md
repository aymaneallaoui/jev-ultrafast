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

Local patch in `~/kev` (not committed there; a copy is at `~/jev-traces/tools/kev-serve-memory.patch`):

1. `empty_cache(dev)` right after `ck.load(dev, opts)`: saves ~1.1 GiB idle on Kev-4B even with `expandable_segments`, ~5 GiB without it when graphs are off.
2. `KEV_MAX_BATCH` (default 64) instead of the `MAX_BATCH` constant.
3. `KEV_SERVE_MAX_STATE` (default `SERVE_MAX_STATE`) for the state length passed to `encode()`.

Worth raising upstream as well: a setting to size the CUDA graph buffers (for example `KEV_GRAPH_STATES` / `KEV_BANK_WIDTH` / `KEV_GRAPH_TOKENS`), so small GPUs can keep graphs with smaller buffers instead of turning them off.
