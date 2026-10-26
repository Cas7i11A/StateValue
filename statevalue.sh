#!/usr/bin/env bash
# Run on the StateValue VM. Services stay in the foreground.
# Usage: bash statevalue.sh {lmcache|vllm|health|smoke}
# Optional: STATEVALUE_DIR=/path/to/statevalue bash statevalue.sh lmcache
set -euo pipefail

action="${1:-help}"
project_dir="${STATEVALUE_DIR:-$HOME/statevalue}"

usage() {
  cat <<'EOF'
Usage: bash statevalue.sh {lmcache|vllm|health|smoke}

Terminal A: bash statevalue.sh lmcache
Terminal B: bash statevalue.sh vllm
Terminal C: bash statevalue.sh health
Terminal C: bash statevalue.sh smoke

Start the GCP VM and connect via SSH first.
Wait for LMCache to start before starting vLLM.
Run health until it reports HTTP 200 before running experiments.
Stop vLLM with Ctrl+C, then stop LMCache with Ctrl+C.
Finally stop the VM in GCP; closing SSH does not stop the VM.
Smoke uses the existing tool_gap_harness.py; it is not included here.
LMCache and vLLM output is shown in the terminal and saved under logs/.
Override the project directory with STATEVALUE_DIR if needed.
EOF
}

case "$action" in
  help|-h|--help) usage; exit 0 ;;
  lmcache|vllm|health|smoke) ;;
  *) usage >&2; exit 2 ;;
esac

check_health() {
  local code
  if ! code=$(curl --silent --show-error --output /dev/null \
    --write-out '%{http_code}' --connect-timeout 3 --max-time 10 \
    http://127.0.0.1:8000/health); then
    echo 'Health check failed. Check the vLLM terminal.' >&2
    return 1
  fi
  if [[ "$code" != 200 ]]; then
    echo "Not ready: HTTP $code" >&2
    return 1
  fi
  echo 'Ready: HTTP 200'
}

if [[ "$action" == health ]]; then
  check_health
  exit 0
fi

if [[ ! -f "$project_dir/.venv/bin/activate" ]]; then
  echo "Missing virtual environment: $project_dir/.venv" >&2
  exit 1
fi
cd "$project_dir"
source .venv/bin/activate
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export PATH="$CUDA_HOME/bin:$PATH"

if [[ "$action" == lmcache || "$action" == vllm ]]; then
  mkdir -p logs
  log_file=$(mktemp "logs/${action}-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX.log")
  # Keep exec below: the service remains the foreground process and receives
  # Ctrl+C directly. Let tee drain the pipe even when the terminal sends SIGINT.
  exec > >(trap '' INT; exec tee -a "$log_file") 2>&1
  export PYTHONUNBUFFERED=1
  echo "Log: $project_dir/$log_file (UTC filename)"
fi

case "$action" in
  lmcache)
    exec lmcache server \
      --host 127.0.0.1 \
      --port 5555 \
      --l1-size-gb 4 \
      --eviction-policy LRU \
      --chunk-size 16
    ;;
  vllm)
    export VLLM_USE_FLASHINFER_SAMPLER=1
    exec vllm serve Qwen/Qwen2.5-1.5B-Instruct \
      --host 127.0.0.1 \
      --port 8000 \
      --dtype half \
      --max-model-len 8192 \
      --gpu-memory-utilization 0.7 \
      --enable-prefix-caching \
      --kv-transfer-config '{
        "kv_connector": "LMCacheMPConnector",
        "kv_connector_module_path": "lmcache.integration.vllm.lmcache_mp_connector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
          "lmcache.mp.host": "127.0.0.1",
          "lmcache.mp.port": 5555
        }
      }'
    ;;
  smoke)
    check_health
    harness=tool_gap_harness.py
    if [[ ! -f "$harness" && -f harness/tool_gap_harness.py ]]; then
      harness=harness/tool_gap_harness.py
    fi
    if [[ ! -f "$harness" ]]; then
      echo 'Missing tool_gap_harness.py in project root or harness/.' >&2
      exit 1
    fi
    mkdir -p runs
    output="runs/smoke-$(date -u +%Y%m%dT%H%M%SZ)-$$"
    if [[ -e "$output" ]]; then
      echo "Output already exists: $output" >&2
      exit 1
    fi
    echo "Output: $output"
    exec python "$harness" \
      --clients 4 \
      --rounds 3 \
      --gap-seconds 2 \
      --prefix-lines 32 \
      --max-tokens 64 \
      --output "$output"
    ;;
esac
