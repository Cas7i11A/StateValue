#!/usr/bin/env bash
# User-space deployment for the existing StateValue CUDA 13.0 configuration.
set -euo pipefail

project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mode=install
case "${1:-}" in
  '') ;;
  --check) mode=check ;;
  --dry-run) mode=dry-run ;;
  -h|--help)
    cat <<'EOF'
Usage: bash setup.sh [--dry-run|--check]
  default    Check prerequisites, create a NEW .venv, install and validate packages.
  --dry-run  Print installation commands without downloads or filesystem changes.
  --check    Validate an existing .venv and GPU without installing packages.

Prerequisites: Linux x86_64, uv (tested with 0.13.1), working NVIDIA driver,
CUDA toolkit 13.0, build tools, curl, tee and coreutils. See DEPLOY.md.
CUDA_HOME defaults to /usr/local/cuda-13.0; export it when using another path.
The script never installs system packages, drivers or CUDA, and never replaces .venv.
EOF
    exit 0 ;;
  *) echo 'Unknown option. Use bash setup.sh --help.' >&2; exit 2 ;;
esac
if (( $# > 1 )); then
  echo 'Only one option is supported. Use bash setup.sh --help.' >&2
  exit 2
fi

cd "$project_dir"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

if [[ "$mode" == dry-run ]]; then
  printf 'Project: %s\nCUDA_HOME: %s\n' "$project_dir" "$CUDA_HOME"
  printf '%s\n' 'Check Linux x86_64, tools, NVIDIA driver, CUDA 13.0 and absence of .venv.'
  printf '%s\n' 'uv python install 3.12.15' \
    'uv venv --seed --python 3.12.15 .venv' \
    'uv pip install --python .venv/bin/python --torch-backend cu130 -r environment/requirements-deploy.txt' \
    'uv pip check --python .venv/bin/python' \
    '.venv/bin/python environment/verify_deployment.py' \
    'Capture installed packages and system/GPU details in logs/setup-<unique-id>/'
  exit 0
fi

mkdir -p logs
setup_dir=$(mktemp -d "logs/setup-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")
exec > >(trap '' INT; exec tee -a "$setup_dir/setup.log") 2>&1
trap 'code=$?; echo "Setup failed (exit $code). See $setup_dir/setup.log" >&2; exit "$code"' ERR
echo "Setup record: $project_dir/$setup_dir"

fail() { echo "ERROR: $*" >&2; exit 1; }
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || fail 'This manifest targets Linux x86_64.'
for tool in uv nvidia-smi curl tee mktemp date mkdir c++ make pkg-config; do
  command -v "$tool" >/dev/null || fail "Missing tool: $tool. See DEPLOY.md."
done
[[ -x "$CUDA_HOME/bin/nvcc" ]] || fail "Missing $CUDA_HOME/bin/nvcc. Install toolkit 13.0 or set CUDA_HOME."
"$CUDA_HOME/bin/nvcc" --version > "$setup_dir/nvcc.txt"
grep -q 'release 13\.0,' "$setup_dir/nvcc.txt" || fail 'This manifest requires CUDA toolkit 13.0.'
nvidia-smi > "$setup_dir/nvidia-smi.txt" || fail 'NVIDIA GPU/driver unavailable. Fix nvidia-smi before setup.'
cat /etc/os-release > "$setup_dir/os-release.txt"
uname -srmo > "$setup_dir/kernel.txt"
uv --version > "$setup_dir/uv-version.txt"
if command -v free >/dev/null; then free -h > "$setup_dir/memory.txt"; fi
df -h . > "$setup_dir/disk.txt"
if command -v lscpu >/dev/null; then lscpu > "$setup_dir/cpu.txt"; fi
if command -v git >/dev/null && [[ "$(git rev-parse --show-toplevel 2>/dev/null || true)" == "$project_dir" ]]; then
  git rev-parse HEAD > "$setup_dir/git-commit.txt"
  git status --short > "$setup_dir/git-status.txt"
fi
cp environment/requirements-deploy.txt environment/requirements-lmcache-working.txt "$setup_dir/"

if [[ "$mode" == install ]]; then
  [[ ! -e .venv && ! -L .venv ]] || fail '.venv already exists. Use --check, or move it aside manually before a fresh install.'
  uv python install 3.12.15
  uv venv --seed --python 3.12.15 .venv
  uv pip install --python .venv/bin/python --torch-backend cu130 \
    -r environment/requirements-deploy.txt
else
  [[ -x .venv/bin/python ]] || fail 'Missing .venv/bin/python. Run setup without --check on a fresh project.'
fi

uv pip check --python .venv/bin/python
.venv/bin/python --version > "$setup_dir/python-version.txt"
.venv/bin/python environment/verify_deployment.py
uv pip freeze --python .venv/bin/python > "$setup_dir/installed-requirements.txt"
echo 'Environment validation passed. Model serving and KV restoration still need separate checks.'
echo 'Next: start LMCache, start vLLM, run health, then smoke (see DEPLOY.md).'
