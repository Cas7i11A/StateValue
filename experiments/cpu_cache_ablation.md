# 256 MiB CPU-cache ablation: P1 acceptance record

Acceptance: **passed** for the fixed, scaled trace-driven synthetic workload. Both modes preserve GPU prefix caching and native GPU block LRU. This is a CPU-tier system ablation, not a StateValue-policy evaluation.

## Controlled conditions

- GPU KV: 268,435,456 bytes (256 MiB) in both modes; actual startup capacity 9360 tokens in every run.
- CPU arm: LMCacheMPConnector plus a cold 4 GiB LMCache LRU server. No-CPU arm: connector omitted, no LMCache server, CPU KV budget 0. Python packages were unchanged.
- Same Qwen2.5-1.5B-Instruct FP16 revision, runtime source hashes, model length 8192, GPU memory utilization setting 0.7, FlashInfer sampler setting, seed 42 and exactly 32 generated tokens.
- Same frozen 24-session / 106-request trace plan; source block lengths scaled by 1/16, initial arrivals and tool gaps scaled by 0.001, one in-flight request. Closed-loop global interleaving can vary with service time.
- Both arms were rerun on the updated implementation rather than silently comparing different runtime source versions. Four trace runs were cold-started in CPU / no-CPU / no-CPU / CPU order.
- Measurement base commit: `957ccf2717965f5b5335a1af134c6c8a1bf3bf15` with recorded working-tree changes. Exact source snapshots are retained in each run and hashes are in the compact JSON record.
- Trace workload SHA-256: `11a0a73e1c77632346c93865a54cb3e22a28b23681b0a9dfe56584ce83acbb9b`.

## Sequential A → B/C/D → A probe

Both modes completed all 5 identical planned requests without active preemption. The repeated A prefix is 4096 tokens with the fixed 16-token suffix defined in BASELINE.md.

| Mode | A-repeat TTFT (ms) | A-repeat latency (ms) | Total prefill-computed tokens | CPU retrieved tokens |
|---|---:|---:|---:|---:|
| CPU cache | 64.25 | 484.63 | 16,400 | 4096 |
| No CPU cache | 279.08 | 699.16 | 20,496 | N/A |

The CPU arm restored 4096 tokens; the no-CPU arm computed exactly 4096 additional tokens. A-repeat TTFT was about 215 ms lower with the CPU tier. This request-level result includes connector overhead and is not isolated DMA time. Other probe requests may include first-use/JIT overhead; use A-repeat for the clearest mechanism comparison.

## Trace results and repeats

| Run | Success | TTFT p50 / p95 (ms) | Latency p50 / p95 (ms) | Replay wall (s) | Prefill computed | CPU retrieved |
|---|---:|---:|---:|---:|---:|---:|
| trace-cpu-01 | 106/106 | 41.85 / 277.75 | 458.64 / 744.82 | 85.920 | 61,234 | 22,768 |
| trace-no-cpu-01 | 106/106 | 36.70 / 261.69 | 452.08 / 685.21 | 85.891 | 86,418 | N/A |
| trace-cpu-02 | 106/106 | 41.99 / 282.79 | 459.76 / 748.54 | 85.964 | 61,234 | 22,768 |
| trace-no-cpu-02 | 106/106 | 35.90 / 263.31 | 451.35 / 686.45 | 85.883 | 86,418 | N/A |

All four trace runs had zero failures and zero active preemptions. Both modes showed nonzero GPU prefix-cache hits, confirming GPU reuse remained enabled. No-CPU retrieval is inapplicable (`null`), not an observed zero from a missing service log.

CPU-cache runs restored 22,768 tokens across 40 requests each. For those matched request IDs, median paired no-CPU-minus-CPU TTFT differences were +7.62 ms and +8.31 ms in the two comparisons. For all 106 requests, the median paired differences were -5.48 ms and -6.18 ms. Positive values favor CPU caching. Requests are matched by session, step and prompt hash; their actual GPU hit/miss histories need not match under closed-loop replay.

## Interpretation and limits

- Recovery of the long pressure-evicted A prefix is materially faster than recomputation in this controlled probe.
- The scaled trace does **not** show an overall latency win for the CPU tier. The no-CPU arm has lower overall p50/p95 latency in these runs; wall time is nearly unchanged and includes planned waiting. Restored requests can benefit while other requests pay CPU-tier/connector costs. The ablation does not isolate a particular overhead component.
- CPU prefill computation was 61,234 tokens versus 86,418 without CPU caching. GPU-hit counts were 123,872 and 121,456 respectively, reflecting different closed-loop histories. For each run, GPU hits + CPU restores (when applicable) + newly computed prefill equals the 207,874 prompt tokens. Per-request CPU retrieval sums also match the server log totals.
- Two runs per arm are a repeatability check, not a statistical performance claim. First-use/JIT events are retained in the service logs; no latency samples were discarded.
- Input length/time scaling, small model, serial serving and synthetic content limit generalization. No original-scale agent task success, exact eviction order, eviction count or transfer-byte measurement is claimed.
- 25 CPU regression tests passed. Both services were stopped after the experiment; ports 5555/8000 were closed and NVIDIA-SMI reported 0 MiB used.
- Procedure: [CPU_CACHE_ABLATION.md](../CPU_CACHE_ABLATION.md). Compact results: [cpu_cache_ablation_results.json](cpu_cache_ablation_results.json). Raw artifacts: `runs/cpu_cache_ablation/` (ignored by Git; back up separately).
