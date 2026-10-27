# native_lru_cpu_cache

This baseline uses unmodified vLLM block-level prefix-cache LRU and LMCache's
4 GiB CPU LRU cache. Active GPU blocks are protected by their reference counts;
idle cached blocks can be recycled under pressure. There is no session-level LRU,
TTL, return-time predictor, or StateValue policy. A CPU hit can restore GPU-missing
KV; otherwise vLLM computes it. No cache-manager implementation is changed.

The commands below use the existing Python environment and service launch script.
`statevalue.sh` additionally accepts `STATEVALUE_KV_CACHE_BYTES`, passed directly
to vLLM's `--kv-cache-memory-bytes`; it does not approximate the pool by reducing
`--gpu-memory-utilization`. Service logs are redirected per experiment with
`STATEVALUE_LOG_DIR`. `STATEVALUE_MODEL_REVISION` pins both model and tokenizer.

## 1. Freeze the sequential probe

The Qwen model and tokenizer must already be cached locally. The preparation tool
resolves the cached model revision and creates a frozen token-ID plan, rather
than reconstructing messages from subsequent model outputs:

```bash
.venv/bin/python baseline_workload.py probe \
  --seed 42 --prompt-tokens 4096 --pushers 3 \
  --output runs/plans/native-lru-probe-v2.json
```

The request sequence is A, B, C, D, A. The first four inputs have 4096 valid,
nonspecial Qwen token IDs. The final A has the **identical 4096-token prefix** plus
a fixed 16-token query suffix; B/C/D have different initial blocks.
The suffix is intentional: vLLM caps a GPU prefix hit below the request's final
token. Repeating an exactly block-aligned complete input can therefore restore
one CPU tail block even without pressure. The suffix lets the whole original A
prefix hit in the large-capacity control and isolates pressure-driven loss.
These are synthetic token sequences, not natural-language agent tasks. All requests
generate exactly 32 tokens (`temperature=0`, fixed seed, `min_tokens=32`,
`max_tokens=32`, `ignore_eos=true`). A `length` finish reason is expected.

For this FP16, single-GPU Qwen model, KV payload is 28,672 bytes/token. A 256 MiB
pool nominally holds about 9360 tokens after block rounding. The probe reads
**actual capacity** from the vLLM startup log and refuses a pressure test if the
12,288 independent B/C/D prefix tokens do not exceed that capacity. The probe's
conservative KV working set is below 4 GiB, so CPU capacity should not be limiting.

## 2. Accept the pressure path and large-capacity control

Stop any existing A/B services before these commands. The runner refuses occupied
ports, starts both services from empty in-memory caches, and stops only the
process groups it created (vLLM first, then LMCache). Run one command at a time:

```bash
.venv/bin/python baseline_run.py \
  --plan runs/plans/native-lru-probe-v2.json \
  --gpu-kv-bytes 268435456 --expect pressure \
  --output runs/native_lru_cpu_cache/probe-small-02

.venv/bin/python baseline_run.py \
  --plan runs/plans/native-lru-probe-v2.json \
  --gpu-kv-bytes 12884901888 --expect no-pressure \
  --output runs/native_lru_cpu_cache/probe-large-02
```

The default probe is strictly serial (`--clients 1`, `--time-scale 1`, both
enforced). Small-pool acceptance requires all 5 requests to succeed, no observed
active-request preemption, B/C/D exceeding capacity, and positive `Retrieved`
tokens in A's second-request log window. The large control requires working-set
fit, GPU prefix reuse and no CPU retrieval. Metrics and log availability are
required; missing observations are not treated as zero.

`Retrieved ... tokens` demonstrates a restore operation, not the precise LRU
eviction order. This probe combines that observation with a fixed identical A prefix,
capacity pressure and no restart/preemption between requests. Exact block eviction
events and ordering are not collected in this version.

## 3. Freeze a trace-driven synthetic replay

Only proceed to trace replay after the probe and control pass.

```bash
.venv/bin/python baseline_workload.py trace \
  data/freeinference/2026-05-17.jsonl.zst \
  --max-sessions 24 --max-requests 5 --seed 42 \
  --tokens-per-source-block 1 \
  --output runs/plans/native-lru-trace-v1.json
```

The raw FreeInference trace has no original text. Requests are generated from its
chained, week-local block IDs. A deterministic mapping of `(seed, week, block ID)`
produces Qwen input token IDs. Shared source prefixes map to identical synthetic
prefixes; consecutive-request prefix lengths are checked exactly and token
collisions that alter them fail preparation. The complete prepared plan is the
workload artifact; **reuse that file for all capacities and future policies**.

The default mapping uses **1 synthetic Qwen token per 16-token source block**,
so lengths are uniformly scaled by 1/16. This accommodates the existing 8192-token
limit. These are source `o200k_base` block lengths, not exact original-model or
Qwen prompt lengths. Partial source blocks are unavailable. Oversized segments
are excluded rather than truncated. To preserve all 16 source length units,
choose `--tokens-per-source-block 16`; more segments will fail the length limit.
Report the scaling and selection alongside results.

One eligible tool-chain segment is chosen per distinct original session. Each
segment has 3–5 requests; user-turn boundaries, model/provider changes, subagents
and overlaps split the chains as in the V1 converter. This is a selected subset,
not full-agent semantic replay. Fixed session arrival offsets and per-step gaps
are retained. Model output is not appended to later requests: future input token
IDs are already frozen, so all policies see the same prompts and reuse structure.

Source: [FreeInference Agentic Trace](https://huggingface.co/datasets/harvardMadsys/freeinference_agentic_trace),
CC BY 4.0. Cite Nixon et al., *From Requests to Sessions: A Large-Scale
Characterization of Human-Driven Agentic Workloads* (2026).

## 4. Replay matched budgets and repeat

Use the same clients and time scale in every run. This initial experiment uses
one in-flight request to avoid active-request preemption. Sessions remain live
while waiting and do not hold a request slot. Queueing and each session's request
order are preserved. This is closed-loop replay: the next request follows local
completion plus its gap, so actual arrivals after the first request vary with
service time. Only the initial arrival plan is fixed across policies.

The example time scale 0.001 applies equally to arrivals and tool gaps. It is a
compressed validation workload, not an unscaled replay of original timing.

```bash
.venv/bin/python baseline_run.py \
  --plan runs/plans/native-lru-trace-v1.json \
  --gpu-kv-bytes 268435456 --expect pressure \
  --clients 1 --time-scale 0.001 \
  --output runs/native_lru_cpu_cache/trace-small-01

.venv/bin/python baseline_run.py \
  --plan runs/plans/native-lru-trace-v1.json \
  --gpu-kv-bytes 12884901888 --expect no-pressure \
  --clients 1 --time-scale 0.001 \
  --output runs/native_lru_cpu_cache/trace-large-01

.venv/bin/python baseline_run.py \
  --plan runs/plans/native-lru-trace-v1.json \
  --gpu-kv-bytes 268435456 --expect pressure \
  --clients 1 --time-scale 0.001 \
  --output runs/native_lru_cpu_cache/trace-small-02

.venv/bin/python baseline_compare.py \
  --small runs/native_lru_cpu_cache/trace-small-01 \
  --large runs/native_lru_cpu_cache/trace-large-01 \
  --repeat runs/native_lru_cpu_cache/trace-small-02 \
  --output runs/native_lru_cpu_cache/trace-comparison.json
```

Every run starts both services cold. The runner does not clear an existing service's
cache or silently kill an unrelated process. Any existing output directory is
rejected. All subprocess startup failures and unsuccessful acceptance checks must
be investigated before interpreting latency differences.

## 5. Records and acceptance limits

Each output directory contains:

| File | Contents |
|---|---|
| `config.json` | Baseline, commit/dirty status, source hashes, seed, workload hash, model/tokenizer revision, GPU/CPU budgets, scheduling and generation settings |
| `workload.json` | Exact frozen requests and planned arrivals/gaps |
| `source/` | Runtime source snapshot, including uncommitted implementation |
| `services/*.log` | Dedicated LMCache and vLLM logs for this cold start |
| `service-lifecycle.json` | Owned process IDs and shutdown status |
| `requests.csv`, `events.jsonl` | Actual ordering, queueing, prompt hashes, fixed token counts, latency, TTFT, tool waits, generated content and failures |
| `metrics-before.prom`, `metrics-after.prom` | Complete raw Prometheus snapshots |
| `metrics.jsonl` | Approximately 1-second samples of KV occupancy, prefix hits, external hits and preemptions; polling failures retained |
| `summary.json` | Success/failure counts, p50/p95 TTFT and latency, replay wall time, retrieved tokens and explicit acceptance checks |
| `failure.json` | Failure details if startup, replay or collection raises an exception |

Replay wall time excludes startup and final metrics settling. The diagnostic
probe includes short log-settling delays; it is not a throughput benchmark.
`Retrieved` tokens are summed from this run's service log; they are not transfer
bytes. TTFT includes client/server overhead, queueing, prefill/restore and first
content delivery. It is not an isolated CPU restore-time measurement.

Eviction count, transfer bytes and exact LRU order remain **unmeasured** (`null`
for numeric fields); no missing metric is reported as zero. Preemption is distinct
from idle-prefix eviction. A pressure run can pass with zero active preemptions.
In this vLLM version, zero-reference cached blocks are on the free/recyclable
queue. The reported KV usage subtracts that free queue, so low or zero occupancy
between requests does **not** show that no reusable prefixes are retained or
evicted. Capacity, reuse and restoration evidence must be interpreted together.
The comparison tool checks matching workload/model/seed/settings/source hashes,
cold starts, measured restoration in the small pool and more restored tokens than
in the large control. The large trace control must fit even the conservative sum
of all request allocations (rounded to 16-token blocks), ignoring any reuse; it
must show GPU prefix hits. Its CPU retrieval count is still recorded rather than
assumed zero, since connector/alignment misses can occur without eviction.
The special probe additionally requires zero large-control CPU restores.
A same-budget repeat is required for
`baseline_acceptance_complete=true`; latency values need not match exactly.

Full raw artifacts under `runs/` are ignored by Git. Preserve them separately;
commit concise experiment notes/tables and the source revision used for each run.
Passing this baseline does not show StateValue improvements, real task success,
or superiority over other policies. Fixed TTL is the next distinct policy to add.
