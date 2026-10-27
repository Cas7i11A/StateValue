# 256 MiB GPU LRU: CPU cache versus recomputation

This control retains vLLM GPU prefix caching and the same explicit 256 MiB GPU
KV pool. `--cpu-cache none` removes the LMCache connector and starts only vLLM.
The existing Python environment is unchanged. There is no CPU KV tier; a missing
GPU prefix must be computed. The default `--cpu-cache lmcache` retains the existing
4 GiB CPU LRU baseline. No cache manager, TTL or StateValue policy is modified.

The two runtime names are `native_lru_cpu_cache` and `native_lru_no_cpu_cache`.
Both consume the **same frozen workload file** from [BASELINE.md](BASELINE.md);
the plan's original baseline label describes its source, not a forced CPU mode.

## Sequential probe

Stop any A/B services first. Each runner invocation starts fresh services and
stops only its own processes. Existing output directories are rejected.

```bash
cd ~/statevalue
.venv/bin/python baseline_run.py \
  --plan runs/plans/native-lru-probe-v2.json \
  --gpu-kv-bytes 268435456 --cpu-cache lmcache --expect pressure \
  --output runs/cpu_cache_ablation/probe-cpu-01

.venv/bin/python baseline_run.py \
  --plan runs/plans/native-lru-probe-v2.json \
  --gpu-kv-bytes 268435456 --cpu-cache none --expect pressure \
  --output runs/cpu_cache_ablation/probe-no-cpu-01

.venv/bin/python cpu_cache_compare.py \
  --cpu runs/cpu_cache_ablation/probe-cpu-01 \
  --no-cpu runs/cpu_cache_ablation/probe-no-cpu-01 \
  --output runs/cpu_cache_ablation/probe-comparison.json
```

The probe repeats the same 4096-token A prefix after three distinct pressure
prefixes. The second A includes the same fixed 16-token suffix in both arms,
avoiding the complete-prompt last-block miss discussed in BASELINE.md.
Compare A-repeat TTFT and latency, rather than inferring recovery benefit from
the median of all five requests. The other requests incur first-use prefill and
possible JIT costs.

## Trace and repeat

Use the existing frozen 24-session / 106-request synthetic trace, seed 42 and
32 generated tokens. Both arms use the same clients and scale. Length is still
scaled by 1/16; initial arrivals and tool gaps by 0.001. Cold KV state does not
mean empty model-weight or disk compilation caches.

```bash
.venv/bin/python baseline_run.py \
  --plan runs/plans/native-lru-trace-v1.json \
  --gpu-kv-bytes 268435456 --cpu-cache lmcache --expect pressure \
  --clients 1 --time-scale 0.001 \
  --output runs/cpu_cache_ablation/trace-cpu-01

.venv/bin/python baseline_run.py \
  --plan runs/plans/native-lru-trace-v1.json \
  --gpu-kv-bytes 268435456 --cpu-cache none --expect pressure \
  --clients 1 --time-scale 0.001 \
  --output runs/cpu_cache_ablation/trace-no-cpu-01
```

Repeat both commands with output names `trace-cpu-02` and `trace-no-cpu-02`, then:

```bash
.venv/bin/python cpu_cache_compare.py \
  --cpu runs/cpu_cache_ablation/trace-cpu-01 \
  --no-cpu runs/cpu_cache_ablation/trace-no-cpu-01 \
  --cpu-repeat runs/cpu_cache_ablation/trace-cpu-02 \
  --no-cpu-repeat runs/cpu_cache_ablation/trace-no-cpu-02 \
  --output runs/cpu_cache_ablation/trace-comparison.json
```

For manual service use, the equivalent mode is:

```bash
STATEVALUE_KV_CACHE_BYTES=268435456 STATEVALUE_CPU_CACHE=none \
  bash statevalue.sh vllm
```

The runner checks ports 5555/8000 are free even for no-CPU runs, ensuring a clean
comparison rather than leaving an unused LMCache process running.

## What is verified

The comparison checks workload hash, model/tokenizer revision, seed, generation
settings, clients, time scale, runtime source hashes, and **actual startup GPU KV
token capacity**, not just requested bytes. The no-CPU arm must retain GPU prefix
caching, omit the connector, and start no LMCache process. Successful requests,
zero active preemptions, and cold process lifecycle are required in both arms.
There is no acceptance requirement that the CPU-cache arm be faster.

`summary.json` adds `prefill_kv_computed_tokens_delta`, taken from the installed
vLLM histogram's `_sum` (new KV tokens computed during prefill, excluding cached
tokens). A missing metric remains `null`. The sequential no-CPU probe requires a
positive measured value. A token delta is not a time or transfer-byte measurement.

In no-CPU runs, `retrieved_tokens` is `null` with
`retrieval_status=not_applicable_cpu_cache_disabled`. This is an inapplicable
measurement, not an observed zero in a nonexistent LMCache log. CPU-cache runs
retain measured log totals. GPU eviction count, exact LRU order and transfer
bytes are still unmeasured.

For serial CPU runs, `requests.csv` also records retrieved tokens from the log
range corresponding to each request. The comparison matches requests by session,
step and prompt hash, and reports both all-request effects and the subset restored
in the CPU run. Positive `no_cpu_minus_cpu` latency differences favor the CPU-cache
system. Closed-loop timing can change global request interleaving and GPU hit/miss
status; the trace subset is therefore not a perfect per-request causal match.
The sequential probe is the clearest recovery-versus-recompute control.

This experiment measures the **net LMCache CPU-tier system effect**, including
connector overhead and extra 4 GiB CPU memory. It does not isolate pure PCIe
transfer latency, establish an LRU-policy advantage, or evaluate StateValue.
Report ordinary/GPU-hit requests as well as restored requests: faster recovery
does not guarantee lower overall latency when CPU-tier overhead affects other
requests. Check service JIT warnings before interpreting isolated latency spikes.

Raw service logs, metrics, frozen inputs and runtime snapshots are saved under
each output directory as before; `runs/` is not backed up by Git. Commit compact
results separately and preserve raw artifacts externally.
