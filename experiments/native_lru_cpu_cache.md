# native_lru_cpu_cache acceptance record

Acceptance: **passed for this scaled synthetic workload**. This does not validate original-length/unscaled traces, exact LRU order, or a StateValue performance benefit.

## Workload and provenance

- Measurement base commit: `aba98b66affbf5fd5b1df4e15b2d09138a25acd5` with implementation changes in the working tree. Every run contains an exact runtime source snapshot and hashes; the compact JSON record preserves those hashes.
- Model: `Qwen/Qwen2.5-1.5B-Instruct`, revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, FP16, one L4 GPU.
- GPU KV budgets: 268,435,456 bytes (256 MiB) and 12,884,901,888 bytes (12 GiB). Actual startup capacities: 9,360 and 449,376 tokens.
- CPU cache: 4,294,967,296 bytes (4 GiB), LMCache LRU. GPU uses unmodified vLLM block-level LRU. No TTL/predictor.
- Source: FreeInference 2026-05-17, CC BY 4.0. One tool-chain segment from each of 24 distinct source sessions, 3–5 requests each, 106 total.
- Input: deterministic Qwen token IDs derived from chained source block IDs, fixed seed 42. One Qwen token per 16-token source block (length scale 1/16), preserving the scaled prefix relationships. Input sizes: 651–6079 tokens.
- Output: exactly 32 tokens per request, greedy sampling, ignore EOS. Generated output does not alter later frozen inputs.
- Scheduling: closed loop, one in-flight request, fixed initial arrival offsets, gaps and arrivals scaled by 0.001. Later timing depends on local completion/queueing.
- Frozen trace workload SHA-256: `11a0a73e1c77632346c93865a54cb3e22a28b23681b0a9dfe56584ce83acbb9b`.
- Every run restarted both services from empty in-memory KV state. Raw artifacts include complete service logs, before/after metrics, sampled metrics, requests and failures.

## Sequential pressure probe

A 4096-token prefix is followed by three unrelated 4096-token pressure prefixes, then the identical A prefix plus a fixed 16-token query suffix. Pressure prefixes total 12,288 full-block tokens, exceeding the small pool. Both capacities succeeded on all 5 requests without active-request preemption.

| GPU pool | GPU capacity tokens | A-repeat GPU hit evidence | A-repeat CPU retrieved tokens | A-repeat TTFT |
|---|---:|---|---:|---:|
| 256 MiB | 9,360 | 0 prefix-hit counter increment | 4,096 | 65.67 ms |
| 12 GiB | 449,376 | 4096 prefix-hit counter increment | 0 | 40.60 ms |

The first probe version repeated the entire block-aligned prompt and the large control retrieved 16 tail tokens despite GPU hits on 4080 tokens. This failed control was retained. Installed vLLM deliberately caps prefix hits below the final request token. The revised fixed-suffix probe removes that confound; the original 4096-token A prefix is identical across the two requests.

## Trace replay and repeat

| Run | Success / attempted | Retrieved tokens | GPU prefix-hit tokens | Preemptions | TTFT p50 / p95 (ms) | Latency p50 / p95 (ms) | Replay wall (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| trace-small-01 | 106/106 | 21,408 | 125,232 | 0 | 41.62 / 282.00 | 458.48 / 748.70 | 85.873 |
| trace-large-01 | 106/106 | 0 | 146,640 | 0 | 35.58 / 279.21 | 455.08 / 742.45 | 85.687 |
| trace-small-02 | 106/106 | 21,408 | 125,232 | 0 | 42.19 / 281.35 | 458.20 / 748.50 | 85.898 |

The large control can fit even the cumulative 212,064-token block-rounded allocation bound that ignores reuse. Small-run GPU prefix hits are lower by exactly 21,408 tokens, matching CPU retrieval. Both same-budget runs completed with the same retrieval count; matching latency values are not required for repeatability.

## Limits and records

- `eviction_count` and `transfer_bytes` are unmeasured (`null`), not zero. Exact block eviction ordering is not verified.
- KV occupancy alone is insufficient: idle cached blocks are on vLLM's recyclable queue and can be excluded from reported occupied-block usage.
- TTFT is client-observed, not pure restore time. Replay wall excludes startup and final metrics settling. One short workload and a same-budget repeat do not establish a general performance improvement.
- Runtime versions: vLLM 0.30.0, LMCache 0.5.5, torch 2.13.0+cu130; Python 3.12.15. All 21 CPU regression tests passed.
- Raw artifacts: `runs/native_lru_cpu_cache/`; frozen plans: `runs/plans/native-lru-probe-v2.json` and `runs/plans/native-lru-trace-v1.json`. These are ignored by Git and must be backed up separately.
- Compact machine-readable results: [native_lru_cpu_cache_results.json](native_lru_cpu_cache_results.json). Procedure: [BASELINE.md](../BASELINE.md).
- After acceptance, both owned services were stopped, ports 5555/8000 were closed, and NVIDIA-SMI reported 0 MiB used.
