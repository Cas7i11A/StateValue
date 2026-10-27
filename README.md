# StateValue

**Value-Based KV Retention Across Agent Tool Gaps**

StateValue investigates when an LLM serving system should retain idle KV state on the GPU, offload it to CPU memory, or discard it while an agent waits for an external tool. The planned policy considers tool-return uncertainty, restore/recomputation costs, and GPU memory pressure.
z
The current prototype runs vLLM with an independent LMCache service and a synthetic tool-gap harness. Policy development and evaluation under memory pressure are ongoing.

## Current status

Validated so far:

- Model serving with vLLM on an NVIDIA L4 GPU.
- KV storage through LMCache and restoration after restarting vLLM while keeping LMCache running (`Retrieved 96 tokens` observed).
- A synthetic workload with four concurrent contexts, three requests per context, and two simulated tool waits per context.
- Service startup through `statevalue.sh`, including automatic virtual-environment activation and an HTTP health check.

Not yet validated:

- GPU eviction and CPU restoration under controlled KV memory pressure.
- An end-to-end LRU performance baseline or comparisons against TTL/value-based policies.
- Real coding-agent task success or goodput.

LMCache currently uses LRU for its CPU cache. This setting alone does not establish the project's GPU retention baseline.

## Files

Keep the following files in the project directory on the VM:

```text
statevalue/
├── README.md
├── statevalue.sh
├── setup.sh                          # New-VM environment installation and validation
├── DEPLOY.md                         # Prerequisites and deployment steps
├── tool_gap_harness.py
├── trace_workload.py                 # FreeInference raw-trace converter and plan reader
├── tests/
│   └── test_trace_replay.py           # CPU-only converter and replay checks
├── environment/                      # Historical working-environment snapshots
│   ├── requirements-lmcache-working.txt
│   ├── requirements-deploy.txt        # Direct pins and historical constraints
│   ├── requirements-vllm-working.txt
│   ├── gpu-environment.txt
│   ├── system-info.txt               # Summary with observed/historical provenance
│   ├── verify_deployment.py           # Package/import/CUDA validation
│   └── cuda-toolkit-version.txt
├── .gitignore
├── .venv/                            # Local environment; do not commit
└── runs/                             # Local experiment outputs; do not commit
```

The startup script also supports `harness/tool_gap_harness.py` if the harness is moved into a subdirectory. Environment snapshot files document the working setup; they are not read by the startup script.

Keep the three runtime files (`statevalue.sh`, `tool_gap_harness.py`, and
`trace_workload.py`) at the project root for the documented commands. Tests belong
under `tests/` and environment records under `environment/`. Raw downloaded traces
can live under `data/`; this directory is ignored by Git and is not created until
needed. Run commands from the project root, normally `~/statevalue` on the VM.

Python `__pycache__/` directories can be deleted; Python recreates them when needed.
The downloaded `cuda-keyring_1.1-1_all.deb` installer has been removed: it is not a
runtime dependency and removing the download does not uninstall CUDA. Keep `.venv/`
for the working services and retain important `runs/` outputs separately from Git.
The two existing smoke directories contain different runs; the dated run is cited
below, so neither is treated as a duplicate. Tool-managed hidden directories such
as `.git/`, `.agents/`, `.codex/`, and `.aws/` are outside this cleanup.

## Environment

The existing setup uses:

| Component | Configuration |
|---|---|
| GPU | NVIDIA L4, approximately 23 GiB as reported by NVIDIA-SMI |
| Python | 3.12.15 in the previously validated environment |
| vLLM | 0.30.0 in the previously validated environment |
| Model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Model dtype | FP16 (`half`) |
| CUDA toolkit path | `/usr/local/cuda-13.0` |
| Virtual environment | `~/statevalue/.venv` |
| vLLM address | `127.0.0.1:8000` |
| LMCache address | `127.0.0.1:5555` |

Use the snapshots under `environment/` for the captured software and driver details.
See `environment/system-info.txt` for the consolidated system/runtime summary,
including which values are current observations and which are historical records.
`requirements-vllm-working.txt` captures the earlier vLLM setup;
`requirements-lmcache-working.txt` captures the later LMCache-integrated setup.
These are different historical snapshots, not duplicate installation requirements.
A newly created VM may differ from those snapshots.

**The startup script assumes the working environment already exists.** For a new
VM, follow [DEPLOY.md](DEPLOY.md), inspect `bash setup.sh --dry-run`, then run
`bash setup.sh` after preparing the GPU driver, CUDA 13.0 and system tools.
`bash setup.sh --check` validates an existing environment without installing.
The deployment manifest uses explicit CUDA 13.0 PyTorch selection and historical
dependency constraints; a full clean-GPU-VM deployment has not yet been validated.

## Start services

Start the VM in GCP, connect over SSH, and open three terminals. Keep the two service terminals open while running experiments.

### Terminal A: LMCache

```bash
cd ~/statevalue
bash statevalue.sh lmcache
```

Wait for the LMCache service to start before proceeding.

### Terminal B: vLLM

```bash
cd ~/statevalue
bash statevalue.sh vllm
```

Wait for model loading and engine initialization.

Both service commands automatically display stdout/stderr in the terminal and
save them under `logs/`. Each startup prints its log path and creates a unique
file such as `logs/vllm-20261026T152633Z-Ab12Cd.log` (UTC timestamp).
Python output is unbuffered to reduce logging delays. `Ctrl+C` still stops the
foreground service. Logging applies to services started with the updated script;
it cannot capture output from an already running service retroactively.
The `logs/` directory is ignored by Git; preserve the relevant LMCache and vLLM
logs alongside each experiment's results.

### Terminal C: health check

```bash
cd ~/statevalue
bash statevalue.sh health
```

Expected output:

```text
Ready: HTTP 200
```

The health command checks once and exits. If initialization is still in progress, inspect Terminal B and rerun the check when ready.

For `lmcache`, `vllm`, and `smoke`, the script automatically activates `.venv` and sets `CUDA_HOME` (default: `/usr/local/cuda-13.0`). The health check requires only `curl`.

If the project lives elsewhere, set the directory for each invocation:

```bash
STATEVALUE_DIR=/path/to/statevalue bash /path/to/statevalue/statevalue.sh lmcache
```

## Smoke test

With both services running:

```bash
cd ~/statevalue
bash statevalue.sh smoke
```

The script checks health and invokes the existing harness with:

```text
--clients 4
--rounds 3
--gap-seconds 2
--prefix-lines 32
--max-tokens 64
```

Each context alternates between generation and a simulated tool wait. This produces **12 requests and 8 tool waits** when all contexts complete. The waits are synthetic; no compiler, test runner, or real agent task is executed.

Every invocation creates a fresh output directory named `runs/smoke-<UTC timestamp>-<process ID>`.

| Output file | Contents |
|---|---|
| `config.json` | Workload configuration and runtime metadata |
| `events.jsonl` | Request/tool events and generated text |
| `requests.csv` | Per-request metrics and errors |
| `summary.json` | Aggregate metrics |

To customize the workload, activate the environment and invoke the harness directly. Choose an unused output directory:

```bash
cd ~/statevalue
source .venv/bin/activate

python tool_gap_harness.py \
  --clients 4 \
  --rounds 3 \
  --gap-seconds 2 \
  --prefix-lines 32 \
  --max-tokens 64 \
  --output runs/custom-01
```

## Trace-driven gap replay (v1)

V1 reuses the synthetic prompts and growing conversations from the smoke harness,
but takes session arrival offsets, request counts, and inter-request waits from
[FreeInference Agentic Trace](https://huggingface.co/datasets/harvardMadsys/freeinference_agentic_trace).
It does not replay original text, token lengths, block IDs, or real tools, and does
not implement a retention policy. The dataset is CC BY 4.0; retain its attribution
and cite *From Requests to Sessions: A Large-Scale Characterization of Human-Driven
Agentic Workloads* (Nixon et al., 2026) in research using these traces.
Keep `trace_workload.py` alongside `tool_gap_harness.py` when deploying trace mode,
including if the harness is moved into `harness/`.

Download a raw daily trace (`release/<week>/traces/<day>.jsonl.zst`) from that
dataset, then convert it. V1 accepts raw session JSONL rather than flattened
Parquet tables. Plain `.jsonl` and `.jsonl.gz` use only the Python standard library;
`.jsonl.zst` additionally needs `zstandard` in the active environment (or decompress
the file externally first).

```bash
cd ~/statevalue
source .venv/bin/activate

python trace_workload.py /path/to/day.jsonl.zst \
  --max-sessions 32 \
  --output runs/plans/freeinference-v1.json
```

The converter scans all supplied files, orders eligible tool-chain segments by their first
request time, and selects the earliest `--max-sessions` replay units (default 32). Multiple
input files can be supplied before `--output`. Existing outputs are rejected.
The plan includes source file SHA-256 hashes, selection counts, and exclusion
reasons. Its offsets start at the earliest selected request.

By default, V1 extracts maximal compatible segments with at least two requests
and consecutive source steps. Each transition must be tool-triggered, remain
within one user turn, and follow at least one tool call. A user-turn boundary,
subagent request, overlap, invalid record, or model/provider change breaks the
segment; the converter never bridges across a rejected transition. A source
session can contribute multiple replay units. Each unit starts a fresh synthetic
context and records its original session ID and source steps, so reuse across
segments is not preserved.

Multiple tool calls in a response are treated as one aggregate observed wait
until the next request. Their count and categories are retained; independent tool
timelines and execution are not simulated. Selection metadata identifies the
unit type, source/contributing record counts, replay-unit counts, and rejected
transition reasons. This remains a restricted subset, not a representative sample
of all agent behavior; report these counts with results.

For the original strict filter, add `--selection-mode whole-session`. That mode
rejects a complete session on any incompatible transition or multiple tool calls,
and may yield no eligible sessions on a given day.

The wait after request `i` is:

```text
gap_after_s = (request[i+1].start_ms - request[i].end_ms) / 1000
```

This is an observed request gap, not an isolated measurement of tool execution.
Negative gaps cause exclusion rather than being clamped to zero.

With LMCache and vLLM running and health returning HTTP 200:

```bash
python tool_gap_harness.py \
  --trace-plan runs/plans/freeinference-v1.json \
  --clients 4 \
  --prefix-lines 32 \
  --max-tokens 64 \
  --time-scale 1 \
  --output runs/trace-gap-01
```

In trace mode, `--clients` limits concurrent HTTP requests; sleeping sessions do
not occupy a request slot. All selected sessions have their own scheduled arrival.
`--rounds` and `--gap-seconds` are replaced by the plan. Each session waits for its
local request to complete, sleeps for its planned gap, then queues its next
request. This is closed-loop replay, so later request times depend on local
inference and queueing; they will differ from the original trace.

`--time-scale` multiplies both initial arrival offsets and gaps. Keep it at 1 for
unscaled timing; e.g. 0.1 accelerates both by 10x and changes the offered workload.
Check the plan's offsets before running: a sparse trace may imply long waits.

Results include the usual four output files plus `replay_plan.json`, a copy of the
input plan. `config.json` records the plan hash and time scale. `events.jsonl`
records planned arrivals, arrival lateness, request readiness, and planned/actual
waits. `requests.csv` adds `client_queue_s`, the wait for an HTTP request slot;
this client-side wait is separate from request latency and TTFT. A failed request
ends its session, other sessions continue, and the harness exits nonzero.

The original `bash statevalue.sh smoke` remains available. This new mode alone
does not establish GPU eviction, CPU restoration, or policy performance; collect
service logs and verify those behaviors separately.

CPU-only checks, using a temporary local HTTP server:

```bash
python -m unittest discover -s tests -v
```

## Latest reported smoke result

Run: `runs/smoke-20261026T152633Z-19278` (October 26, 2026).

| Metric | Result |
|---|---:|
| Successful / attempted requests | 12 / 12 |
| Failed requests | 0 |
| Completed contexts | 4 |
| Wall time | 5.847549 s |
| Successful requests per second | 2.052142 |
| Mean successful request latency | 0.575843 s |
| p95 successful request latency | 1.029448 s |
| p95 successful request TTFT | 0.253824 s |
| Length-limited requests | 0 |

This run confirms basic serving and workload execution. It does not establish a performance improvement or demonstrate eviction/restoration under pressure.

## Interpret metrics and cache logs

- **TTFT** is client elapsed time to the first nonempty content chunk. It includes HTTP overhead, queueing, prefill/restore, and first-token delivery; these components are not measured separately.
- Request latency statistics include successful requests only. Inspect failures separately.
- Requests per second is request throughput, not task goodput.
- A `stop` finish reason or zero length-limited requests does not establish semantic correctness.
- With only 12 samples, p95 is a smoke-test diagnostic rather than a stable tail-latency estimate.
- `Stored ... tokens` records storage activity. `Prefetch ... retained keys` does not by itself prove that KV was transferred back to the GPU.
- `Retrieved ... tokens` provides evidence of a restore operation. Its logged duration is not end-to-end request latency.
- When GPU prefixes remain available, later rounds may need no CPU restoration. A successful low-pressure smoke test therefore need not show `Retrieved`.

## Service configuration

`statevalue.sh` preserves the previously working settings:

| Service | Settings |
|---|---|
| LMCache | 4 GB L1 cache, LRU eviction, chunk size 16 |
| vLLM | Maximum model length 8192, GPU memory utilization 0.7, prefix caching enabled |
| Connector | `LMCacheMPConnector`, `kv_both`, local LMCache service on port 5555 |
| Sampler environment | `VLLM_USE_FLASHINFER_SAMPLER=1` |

Inspect the script for the full connector configuration. Both services bind to loopback; run the harness on the VM. No public endpoint is required for this workflow.

## Stop safely

1. Allow the harness to finish, or interrupt it and retain its output for inspection.
2. In Terminal B, press `Ctrl+C` and wait for vLLM to exit.
3. In Terminal A, press `Ctrl+C` and wait for LMCache to exit.
4. Stop the VM in GCP when finished.

Closing SSH alone does not stop the VM. Stopping the services/VM loses their in-memory KV state. Files on the retained persistent disk, including the environment and experiment outputs, remain available. Retained disks and other resources can still incur charges.

## Experiment records and Git

Commit source code, startup scripts, configuration, environment snapshots, and concise experiment notes. Keep environments, downloaded models, credentials, and raw run directories out of Git.

For each recorded experiment, capture:

```bash
git rev-parse HEAD
git status --short
```

Record the commit, whether the working tree was modified, service settings, workload command, output directory, and relevant service logs. The current harness output does not automatically capture all of this provenance.

Suggested `.gitignore` entries:

```gitignore
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
.env
.env.*
!.env.example
credentials*.json
service-account*.json
*.pem
*.key
.cache/
models/
checkpoints/
runs/
logs/
*.log
.DS_Store
```

Ignored experiment outputs are not backed up by `git push`. Preserve important raw results separately; commit selected small tables, plots, and analysis when appropriate.
